#!/usr/bin/env python3
"""Train/evaluate method-specific action heads on frozen VideoMAEv2 aggregates.

This is the CPU stage for separating anonymization-domain shift from actual
loss of action information.  It never runs the VideoMAE encoder and writes to
files separate from the original clean-trained-head results.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_SPLIT = "train_original_r1_allcams"
VAL_SPLIT = "val_original_c1r2"
TEST_SPLITS = ("test_original_c2r2", "test_original_c3r2")
SUPPORTED_METHODS = ("face_blur", "body_blur", "body_pixel", "deepprivacy2")
DEFAULT_SEED = 42


def default_action_root() -> Path:
    project_root = Path(os.environ.get("ACCV_ROOT", Path(__file__).resolve().parents[1]))
    output_root = Path(os.environ.get("PILOT_OUTPUT_ROOT", project_root / "outputs"))
    return output_root / "features" / "action_videomae_v2"


def parse_args() -> argparse.Namespace:
    action_root = default_action_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument(
        "--test-splits",
        default=",".join(TEST_SPLITS),
        help="Comma-separated test splits; use C2 only when that is the completed learned-RGB scope.",
    )
    parser.add_argument("--aggregate-dir", type=Path, default=action_root / "aggregates")
    parser.add_argument("--output-dir", type=Path, default=action_root)
    parser.add_argument("--models-dir", type=Path, default=action_root / "models")
    parser.add_argument(
        "--clean-checkpoint",
        type=Path,
        default=action_root / "models" / "action_videomae_v2_head.pt",
        help="Existing clean-trained head. With --clean-head auto it is reused when present.",
    )
    parser.add_argument("--clean-head", choices=("auto", "reuse", "train"), default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--dry-run",
        "--preflight",
        dest="dry_run",
        action="store_true",
        help="Report every required aggregate and the commands needed to create missing ones.",
    )
    return parser.parse_args()


def parse_methods(raw: str) -> list[str]:
    methods = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise SystemExit(f"Unsupported method(s): {unknown}. Supported: {list(SUPPORTED_METHODS)}")
    if not methods:
        raise SystemExit("--methods must contain at least one method")
    return methods


def safe_method_name(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method)


def aggregate_path(aggregate_dir: Path, method: str, split_name: str) -> Path:
    return aggregate_dir / f"{safe_method_name(method)}__{split_name}.npz"


def inspect_aggregate(path: Path) -> tuple[str, int, int]:
    if not path.exists():
        return "MISSING", 0, 0
    try:
        with np.load(path, allow_pickle=False) as data:
            missing_keys = {"features", "labels"} - set(data.files)
            if missing_keys:
                return "INVALID_KEYS", 0, 0
            features = data["features"]
            labels = data["labels"]
            if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
                return "INVALID_SHAPE", int(len(labels)), int(features.shape[-1] if features.ndim == 2 else 0)
            if len(labels) == 0:
                return "EMPTY", 0, int(features.shape[1])
            return "READY", int(len(labels)), int(features.shape[1])
    except Exception as exc:  # a corrupt/incomplete npz should be visible in preflight
        return f"INVALID:{type(exc).__name__}", 0, 0


def clean_checkpoint_will_be_reused(args: argparse.Namespace) -> bool:
    return args.clean_head == "reuse" or (args.clean_head == "auto" and args.clean_checkpoint.exists())


def required_aggregates(args: argparse.Namespace, methods: list[str]) -> list[dict[str, object]]:
    requirements: list[tuple[str, str, str]] = []
    if not clean_checkpoint_will_be_reused(args):
        requirements.extend(("original", split, "clean head training") for split in (TRAIN_SPLIT, VAL_SPLIT))
    requirements.extend(("original", split, "clean test reference") for split in args.test_splits)
    for method in methods:
        requirements.extend((method, split, "adapted head training") for split in (TRAIN_SPLIT, VAL_SPLIT))
        requirements.extend((method, split, "zero-shot and adapted evaluation") for split in args.test_splits)

    rows = []
    for method, split_name, purpose in requirements:
        path = aggregate_path(args.aggregate_dir, method, split_name)
        status, samples, feature_dim = inspect_aggregate(path)
        rows.append(
            {
                "method": method,
                "split": split_name,
                "purpose": purpose,
                "status": status,
                "samples": samples,
                "feature_dim": feature_dim,
                "path": str(path.resolve()),
            }
        )
    return rows


def print_preflight(args: argparse.Namespace, methods: list[str]) -> bool:
    rows = required_aggregates(args, methods)
    table = pd.DataFrame(rows)
    print("Adapted-utility preflight (head training/evaluation is CPU-only)")
    print(table.to_string(index=False))
    if args.clean_head == "reuse" and not args.clean_checkpoint.exists():
        print(f"\nMISSING clean checkpoint required by --clean-head reuse: {args.clean_checkpoint.resolve()}")

    missing = table[table["status"] != "READY"]
    if missing.empty and not (args.clean_head == "reuse" and not args.clean_checkpoint.exists()):
        print("\nREADY: all required aggregates are present.")
        return True

    print("\nMissing/invalid aggregates (exact paths):")
    for path in missing["path"]:
        print(f"  {path}")
    split_names = list(dict.fromkeys(str(item) for item in missing["split"].tolist()))
    method_names = list(dict.fromkeys(str(item) for item in missing["method"].tolist()))
    print("\nCPU aggregate rebuild (works when the per-video .npy features already exist):")
    print(
        "  python 13_extract_videomae_features.py --aggregate-only "
        f"--methods {','.join(method_names)} --splits {','.join(split_names)} --rebuild-aggregates"
    )
    print("\nIf that produces EMPTY aggregates, the missing per-video VideoMAE features are the GPU prerequisite:")
    print(
        "  python 13_extract_videomae_features.py "
        f"--methods {','.join(method_names)} --splits {','.join(split_names)} "
        "--batch-size 4 --precision auto"
    )
    if any(method == "deepprivacy2" for method in method_names):
        print(
            "  DeepPrivacy2 must first generate anonymized frames for every requested train/val/test clip; "
            "the existing probe-only run covers C2/C3 tests, not the adapted-head train split."
        )
    simple_methods = [method for method in method_names if method in SUPPORTED_METHODS[:3]]
    if simple_methods:
        print(
            "  For missing simple-baseline frames/detections (CPU stages), run: "
            f"python 04_anonymize_baselines_parallel.py --methods {','.join(simple_methods)} --workers 8"
        )
    return False


def load_aggregate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        x = data["features"].astype(np.float32, copy=False)
        y = data["labels"].astype(np.int64, copy=False)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError(f"Invalid or empty aggregate: {path}")
    return x, y


class ActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    top_indices = np.argsort(logits, axis=1)[:, -k:]
    return float(np.mean([truth in row for truth, row in zip(labels, top_indices)])) if len(labels) else 0.0


def mean_class_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    return float(np.mean([np.mean(pred[labels == label] == label) for label in classes])) if len(classes) else 0.0


def predict_logits(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            parts.append(model(xb).cpu().numpy())
    output_dim = model.net[-1].out_features if isinstance(model.net, nn.Sequential) else model.net.out_features
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, output_dim), dtype=np.float32)


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    x_std = (x - mean.reshape(1, -1)) / std.reshape(1, -1)
    logits = predict_logits(model, x_std, batch_size, device)
    pred = logits.argmax(axis=1)
    return {
        "top1": float(np.mean(pred == y)),
        "top5": topk_accuracy(logits, y, 5),
        "mean_class_accuracy": mean_class_accuracy(pred, y),
        "samples": int(len(y)),
    }


def train_head(
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ActionHead, np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_train, y_train = load_aggregate(aggregate_path(args.aggregate_dir, method, TRAIN_SPLIT))
    x_val, y_val = load_aggregate(aggregate_path(args.aggregate_dir, method, VAL_SPLIT))
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(f"Feature dimension mismatch for {method}: train={x_train.shape[1]}, val={x_val.shape[1]}")
    if y_train.min() < 0 or y_train.max() >= args.num_classes:
        raise ValueError(f"Labels for {method} fall outside 0..{args.num_classes - 1}")

    mean = x_train.mean(axis=0).astype(np.float32)
    std = (x_train.std(axis=0) + 1e-6).astype(np.float32)
    x_train = (x_train - mean.reshape(1, -1)) / std.reshape(1, -1)

    model = ActionHead(x_train.shape[1], args.hidden_dim, args.dropout, args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_top1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, x_val, y_val, mean, std, args.batch_size, device)
        history.append(
            {
                "train_method": method,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        print(
            f"{method}: epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f}"
        )
        if float(val_metrics["top1"]) > best_top1:
            best_top1 = float(val_metrics["top1"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"action_videomae_v2_domain_{safe_method_name(method)}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "num_classes": int(args.num_classes),
            "feature_mean": mean,
            "feature_std": std,
            "train_method": method,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "best_epoch": best_epoch,
            "best_val_top1": best_top1,
        },
        checkpoint_path,
    )
    return model, mean, std, history


def load_head(path: Path, device: torch.device) -> tuple[ActionHead, np.ndarray, np.ndarray]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0
        checkpoint = torch.load(path, map_location=device)
    model = ActionHead(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
        int(checkpoint.get("num_classes", 120)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, np.asarray(checkpoint["feature_mean"], dtype=np.float32), np.asarray(checkpoint["feature_std"], dtype=np.float32)


def evaluate_pair(
    train_method: str,
    test_method: str,
    split_name: str,
    evaluation_mode: str,
    model: ActionHead,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    x, y = load_aggregate(aggregate_path(args.aggregate_dir, test_method, split_name))
    if x.shape[1] != len(mean):
        raise ValueError(
            f"Feature dimension mismatch: {train_method} head expects {len(mean)}, "
            f"but {test_method}/{split_name} has {x.shape[1]}"
        )
    return {
        "split": split_name,
        "train_method": train_method,
        "test_method": test_method,
        "evaluation_mode": evaluation_mode,
        **evaluate(model, x, y, mean, std, args.batch_size, device),
    }


def add_comparisons(rows: list[dict[str, object]]) -> None:
    index = {(str(row["split"]), str(row["train_method"]), str(row["test_method"])): row for row in rows}
    for row in rows:
        split_name = str(row["split"])
        test_method = str(row["test_method"])
        clean_reference = index[(split_name, "original", "original")]
        zero_shot = index.get((split_name, "original", test_method), clean_reference)
        clean_top1 = float(clean_reference["top1"])
        zero_top1 = float(zero_shot["top1"])
        top1 = float(row["top1"])
        gain = top1 - zero_top1 if row["evaluation_mode"] == "adapted" else 0.0
        clean_gap = clean_top1 - zero_top1
        row["comparison_baseline"] = f"original->{test_method}"
        row["clean_reference_top1"] = clean_top1
        row["zero_shot_top1"] = zero_top1
        row["top1_gain_vs_zero_shot"] = gain
        row["relative_top1_gain_vs_zero_shot"] = gain / zero_top1 if zero_top1 else np.nan
        row["recovered_clean_gap"] = gain / clean_gap if row["evaluation_mode"] == "adapted" and clean_gap > 0 else np.nan
        row["retained_utility_top1"] = top1 / clean_top1 if clean_top1 else np.nan


def main() -> int:
    args = parse_args()
    args.test_splits = tuple(item.strip() for item in args.test_splits.split(",") if item.strip())
    unknown_splits = sorted(set(args.test_splits) - set(TEST_SPLITS))
    if unknown_splits or not args.test_splits:
        raise SystemExit(f"--test-splits must be a non-empty subset of {TEST_SPLITS}; got {args.test_splits}")
    methods = parse_methods(args.methods)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.num_classes < 2:
        raise SystemExit("epochs, batch-size, and patience must be >=1; num-classes must be >=2")
    ready = print_preflight(args, methods)
    if args.dry_run:
        return 0 if ready else 2
    if not ready:
        print("\nCannot run: satisfy the preflight prerequisites above.", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"\nTraining/evaluation device: {device}")

    all_history: list[dict[str, object]] = []
    if clean_checkpoint_will_be_reused(args):
        clean_model, clean_mean, clean_std = load_head(args.clean_checkpoint, device)
        print(f"Reusing clean head: {args.clean_checkpoint.resolve()}")
    else:
        clean_model, clean_mean, clean_std, history = train_head("original", args, device)
        all_history.extend(history)

    rows: list[dict[str, object]] = []
    for split_name in args.test_splits:
        rows.append(
            evaluate_pair(
                "original", "original", split_name, "clean_reference", clean_model, clean_mean, clean_std, args, device
            )
        )
        for method in methods:
            rows.append(
                evaluate_pair(
                    "original", method, split_name, "zero_shot", clean_model, clean_mean, clean_std, args, device
                )
            )

    for method in methods:
        adapted_model, adapted_mean, adapted_std, history = train_head(method, args, device)
        all_history.extend(history)
        for split_name in args.test_splits:
            rows.append(
                evaluate_pair(
                    method, method, split_name, "adapted", adapted_model, adapted_mean, adapted_std, args, device
                )
            )

    add_comparisons(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "action_videomae_v2_domain_adapted_results.csv"
    history_path = args.output_dir / "action_videomae_v2_domain_adapted_train_history.csv"
    columns = [
        "split",
        "train_method",
        "test_method",
        "evaluation_mode",
        "top1",
        "top5",
        "mean_class_accuracy",
        "samples",
        "comparison_baseline",
        "clean_reference_top1",
        "zero_shot_top1",
        "top1_gain_vs_zero_shot",
        "relative_top1_gain_vs_zero_shot",
        "recovered_clean_gap",
        "retained_utility_top1",
    ]
    results = pd.DataFrame(rows, columns=columns)
    results.to_csv(results_path, index=False)
    pd.DataFrame(all_history).to_csv(history_path, index=False)
    print("\nDomain-adapted VideoMAE utility results")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results: {results_path.resolve()}")
    print(f"History: {history_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
