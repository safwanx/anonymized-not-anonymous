#!/usr/bin/env python3
"""CPU evaluation of PBP utility on clean and AAM-transformed features.

The ``clean`` head is the pipeline step-14 head (zero-shot transfer). The
``pbp_joint`` head is stored in the AAM checkpoint (adapted/deployment utility).
Both heads are evaluated on the same clean and transformed feature matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

import pbp_paths as P

DEFAULT_SPLITS = ("test_original_c2r2", "test_original_c3r2")
RESULT_COLUMNS = (
    "split", "feature_variant", "action_head", "head_training_domain",
    "evaluation_role", "num_features", "expected_features", "feature_coverage",
    "top1", "top5", "mean_class_accuracy", "retained_utility_top1_same_head",
    "feature_source", "aam_checkpoint", "action_head_checkpoint",
)


class CleanActionHead(nn.Module):
    """Architecture saved by pipeline step 14."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return state


def default_aggregates_dir() -> Path:
    output_root = Path(os.environ.get("PILOT_OUTPUT_ROOT", P.PROJECT_ROOT / "outputs"))
    return output_root / "features" / "action_videomae_v2" / "aggregates"


def default_clean_head() -> Path:
    output_root = Path(os.environ.get("PILOT_OUTPUT_ROOT", P.PROJECT_ROOT / "outputs"))
    return output_root / "features" / "action_videomae_v2" / "models" / "action_videomae_v2_head.pt"


def aggregate_path(aggregates_dir: Path, split: str) -> Path:
    return aggregates_dir / f"original__{split}.npz"


def labels_path(split: str) -> Path:
    return P.PBP_LABELS_DIR / f"labels_ntu_{split}.json"


def parse_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_feature_source(split: str, requested: str, aggregates_dir: Path) -> tuple[str, Path, Path | None]:
    aggregate = aggregate_path(aggregates_dir, split)
    h5 = P.feat_h5(split, fb=False)
    labels = labels_path(split)
    if requested == "aggregates":
        return "aggregate_npz", aggregate, None
    if requested == "h5":
        return "pbp_h5", h5, labels
    if aggregate.exists():
        return "aggregate_npz", aggregate, None
    if h5.exists():
        return "pbp_h5", h5, labels
    # Aggregates are the minimal artifacts to copy from SLURM. Prefer their
    # exact missing paths over suggesting an HDF5 export that does not exist.
    return "aggregate_npz", aggregate, None


def inspect_aam_checkpoint(path: Path, require_joint_head: bool) -> list[str]:
    if not path.exists():
        return [f"missing AAM checkpoint: {path}"]
    try:
        state = _torch_load(path)
    except Exception as exc:
        return [f"unreadable AAM checkpoint: {path} ({type(exc).__name__}: {exc})"]
    issues: list[str] = []
    if "fa_model_state_dict" not in state:
        issues.append(f"AAM checkpoint lacks fa_model_state_dict: {path}")
    if state.get("arch", "mlp") != "mlp":
        issues.append(f"unsupported AAM arch={state.get('arch')!r} in {path}; evaluator requires per-vector MLP")
    if require_joint_head and "head_state_dict" not in state:
        issues.append(f"AAM checkpoint lacks jointly trained head_state_dict: {path}")
    return issues


def inspect_clean_head(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing clean action-head checkpoint: {path}"]
    try:
        state = _torch_load(path)
    except Exception as exc:
        return [f"unreadable clean action-head checkpoint: {path} ({type(exc).__name__}: {exc})"]
    required = ("state_dict", "input_dim", "hidden_dim", "dropout", "feature_mean", "feature_std")
    absent = [key for key in required if key not in state]
    return [f"clean action-head checkpoint lacks {', '.join(absent)}: {path}"] if absent else []


def inspect_feature_artifacts(split: str, requested: str, aggregates_dir: Path) -> list[str]:
    source, feature_path, label_file = resolve_feature_source(split, requested, aggregates_dir)
    issues: list[str] = []
    if not feature_path.exists():
        kind = "VideoMAE aggregate" if source == "aggregate_npz" else "PBP VideoMAE HDF5"
        issues.append(f"missing {kind} for {split}: {feature_path}")
    if label_file is not None and not label_file.exists():
        issues.append(f"missing PBP action labels for {split}: {label_file}")
    return issues


def preflight(args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Return readiness and only actionable missing/invalid artifact messages."""
    issues = inspect_aam_checkpoint(args.aam, require_joint_head="pbp_joint" in args.heads)
    if "clean" in args.heads:
        issues.extend(inspect_clean_head(args.clean_head))
    for split in args.splits:
        issues.extend(inspect_feature_artifacts(split, args.feature_source, args.aggregates_dir))
    return not issues, issues


def load_feature_matrix(split: str, requested: str, aggregates_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    source, path, label_file = resolve_feature_source(split, requested, aggregates_dir)
    if source == "aggregate_npz":
        with np.load(path, allow_pickle=False) as data:
            if "features" not in data or "labels" not in data:
                raise ValueError(f"aggregate must contain features and labels arrays: {path}")
            x = data["features"].astype(np.float32, copy=False)
            y = data["labels"].astype(np.int64, copy=False)
    else:
        assert label_file is not None
        labels = json.loads(label_file.read_text(encoding="utf-8"))
        features: list[np.ndarray] = []
        targets: list[int] = []
        with h5py.File(path, "r") as handle:
            for key in sorted(handle.keys()):
                if key not in labels:
                    continue
                value = np.asarray(handle[key][...], dtype=np.float32)
                if value.ndim == 1:
                    vector = value
                elif value.ndim == 2:
                    vector = value.mean(axis=0)
                else:
                    raise ValueError(f"expected [D] or [clips,D] for {key} in {path}; got {value.shape}")
                features.append(vector)
                targets.append(int(labels[key]["action_label"]))
        x = np.stack(features).astype(np.float32, copy=False) if features else np.zeros((0, P.FEATURE_DIM), np.float32)
        y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != P.FEATURE_DIM:
        raise ValueError(f"expected feature matrix [N,{P.FEATURE_DIM}] for {split}; got {x.shape} from {path}")
    if len(x) != len(y):
        raise ValueError(f"feature/label count mismatch for {split}: {len(x)} != {len(y)} in {path}")
    if len(y) and (int(y.min()) < 0 or int(y.max()) >= 120):
        raise ValueError(f"action labels for {split} must be zero-based in [0,119]; found [{y.min()},{y.max()}]")
    return x, y, f"{source}:{path}"


def load_aam(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    state = _torch_load(path)
    if state.get("arch", "mlp") != "mlp":
        raise ValueError("Only the per-vector MLP AAM is valid for mean-pooled [N,768] descriptors")
    MLP, _Transformer = P.import_anonymizer()
    model = MLP(P.FEATURE_DIM, P.FEATURE_DIM)
    model.load_state_dict(state["fa_model_state_dict"])
    model.eval()
    return model, state


def load_clean_action_head(path: Path) -> tuple[nn.Module, np.ndarray, np.ndarray, int]:
    state = _torch_load(path)
    num_classes = int(state.get("num_classes", 120))
    model = CleanActionHead(int(state["input_dim"]), int(state["hidden_dim"]), float(state["dropout"]), num_classes)
    model.load_state_dict(state["state_dict"])
    model.eval()
    mean = np.asarray(state["feature_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(state["feature_std"], dtype=np.float32).reshape(1, -1)
    if mean.shape[1] != P.FEATURE_DIM or std.shape[1] != P.FEATURE_DIM:
        raise ValueError(f"clean head normalization has dimensions mean={mean.shape}, std={std.shape}; expected 768")
    return model, mean, std, num_classes


def load_joint_action_head(state: dict[str, Any]) -> tuple[nn.Module, int]:
    weights = state["head_state_dict"]
    if "weight" not in weights or "bias" not in weights:
        raise ValueError("joint PBP head_state_dict must be a plain Linear with weight and bias")
    weight = weights["weight"]
    if tuple(weight.shape)[1] != P.FEATURE_DIM:
        raise ValueError(f"joint PBP head weight has shape {tuple(weight.shape)}; expected [classes,768]")
    model = nn.Linear(P.FEATURE_DIM, int(weight.shape[0]))
    model.load_state_dict(weights)
    model.eval()
    return model, int(weight.shape[0])


@torch.inference_mode()
def batched_forward(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).float()
        parts.append(model(batch).cpu().numpy())
    if parts:
        return np.concatenate(parts, axis=0)
    width = next((m.out_features for m in reversed(list(model.modules())) if isinstance(m, nn.Linear)), 0)
    return np.zeros((0, width), np.float32)


def metrics(logits: np.ndarray, y: np.ndarray, num_classes: int) -> dict[str, float]:
    if not len(y):
        return {"top1": 0.0, "top5": 0.0, "mean_class_accuracy": 0.0}
    pred = logits.argmax(axis=1)
    k = min(5, logits.shape[1])
    top5_idx = np.argpartition(logits, -k, axis=1)[:, -k:]
    class_scores = [float(np.mean(pred[y == label] == label)) for label in range(num_classes) if np.any(y == label)]
    return {
        "top1": float(np.mean(pred == y)),
        "top5": float(np.mean([truth in guesses for truth, guesses in zip(y, top5_idx)])),
        "mean_class_accuracy": float(np.mean(class_scores)) if class_scores else 0.0,
    }


def expected_count(split: str) -> int:
    path = labels_path(split)
    return len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    aam, aam_state = load_aam(args.aam)
    heads: dict[str, tuple[nn.Module, np.ndarray | None, np.ndarray | None, int, str, Path]] = {}
    if "clean" in args.heads:
        model, mean, std, classes = load_clean_action_head(args.clean_head)
        heads["clean_trained"] = (model, mean, std, classes, "clean_features", args.clean_head)
    if "pbp_joint" in args.heads:
        model, classes = load_joint_action_head(aam_state)
        heads["pbp_joint"] = (model, None, None, classes, "jointly_trained_on_aam_outputs", args.aam)

    rows: list[dict[str, Any]] = []
    for split in args.splits:
        clean_x, y, source = load_feature_matrix(split, args.feature_source, args.aggregates_dir)
        aam_x = batched_forward(aam, clean_x, args.batch_size)
        expected = expected_count(split) or len(clean_x)
        for head_name, (head, mean, std, classes, training_domain, head_path) in heads.items():
            clean_input = (clean_x - mean) / std if mean is not None and std is not None else clean_x
            aam_input = (aam_x - mean) / std if mean is not None and std is not None else aam_x
            clean_metrics = metrics(batched_forward(head, clean_input, args.batch_size), y, classes)
            aam_metrics = metrics(batched_forward(head, aam_input, args.batch_size), y, classes)
            baseline = clean_metrics["top1"]
            for variant, role, values in (
                ("clean", "same-head clean reference", clean_metrics),
                ("pbp_aam", "zero-shot transfer" if head_name == "clean_trained" else "PBP-adapted deployment utility", aam_metrics),
            ):
                rows.append({
                    "split": split, "feature_variant": variant, "action_head": head_name,
                    "head_training_domain": training_domain, "evaluation_role": role,
                    "num_features": len(clean_x), "expected_features": expected,
                    "feature_coverage": len(clean_x) / max(expected, 1), **values,
                    "retained_utility_top1_same_head": values["top1"] / baseline if baseline else "",
                    "feature_source": source, "aam_checkpoint": str(args.aam),
                    "action_head_checkpoint": str(head_path),
                })
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aam", type=Path, default=P.PBP_MODELS_DIR / "pbp_aam_ntu.pth")
    parser.add_argument("--clean-head", type=Path, default=default_clean_head())
    parser.add_argument("--heads", default="clean,pbp_joint", help="Comma-separated: clean,pbp_joint")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--feature-source", choices=("auto", "aggregates", "h5"), default="auto")
    parser.add_argument("--aggregates-dir", type=Path, default=default_aggregates_dir())
    parser.add_argument("--output", type=Path, default=P.PBP_RESULTS_DIR / "pbp_action_utility_results.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--preflight", action="store_true", help="Check artifacts only; do not evaluate or write CSV.")
    args = parser.parse_args(argv)
    args.heads = parse_names(args.heads)
    args.splits = parse_names(args.splits)
    unknown = sorted(set(args.heads) - {"clean", "pbp_joint"})
    if unknown:
        parser.error(f"unknown --heads value(s): {', '.join(unknown)}")
    if not args.heads:
        parser.error("--heads must request clean, pbp_joint, or both")
    if not args.splits:
        parser.error("--splits must not be empty")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ready, issues = preflight(args)
    if not ready:
        print("PBP action-utility preflight: NOT READY", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 2
    print("PBP action-utility preflight: READY")
    if args.preflight:
        return 0
    rows = evaluate(args)
    write_results(args.output, rows)
    for row in rows:
        print(f"{row['split']} | {row['action_head']} | {row['feature_variant']} | "
              f"top1={row['top1']:.4f} top5={row['top5']:.4f} "
              f"mca={row['mean_class_accuracy']:.4f} n={row['num_features']}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
