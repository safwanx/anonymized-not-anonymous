#!/usr/bin/env python3
"""Train a clean RGB-extracted-pose action MLP and evaluate C2/DeepPrivacy2.

This is a compact second utility model based only on YOLO pose trajectories.
It complements VideoMAEv2; it is not presented as an RGB video architecture or
as a full graph-convolutional skeleton model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from action_videomae_common import TEST_C2_SPLIT, TRAIN_SPLIT, load_action_split
from pipeline_common import RANDOM_SEED
from rgb_pose_common import RGB_POSE_MODEL_DIR, RGB_POSE_ROOT, load_pose_feature


class PoseActionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int = 120):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_split(method: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    rows = load_action_split(split)
    feats, labels, names = [], [], []
    for row in rows:
        feature, _stats = load_pose_feature(method, str(row["filename"]))
        if feature is None:
            continue
        feats.append(feature.reshape(-1))
        labels.append(int(row["label"]))
        names.append(str(row["filename"]))
    if not feats:
        return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64), [], len(rows)
    dims = {len(feature) for feature in feats}
    if len(dims) != 1:
        raise ValueError(f"Mixed pose feature dimensions for {method}/{split}: {sorted(dims)}")
    return np.stack(feats).astype(np.float32), np.asarray(labels, np.int64), names, len(rows)


def stratified_holdout(y: np.ndarray, fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    train, val = [], []
    for label in sorted(np.unique(y)):
        indices = np.flatnonzero(y == label)
        n_val = max(1, int(round(len(indices) * fraction)))
        val.extend(indices[:: max(1, len(indices) // n_val)][:n_val])
        val_set = set(val[-n_val:])
        train.extend(index for index in indices if index not in val_set)
    return np.asarray(sorted(train), np.int64), np.asarray(sorted(val), np.int64)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(x), batch_size):
        parts.append(model(torch.from_numpy(x[start : start + batch_size]).float().to(device)).cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros((0, 120), np.float32)


def score(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if not len(y):
        return {"top1": 0.0, "top5": 0.0, "mean_class_accuracy": 0.0}
    pred = logits.argmax(1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    per_class = [float(np.mean(pred[y == c] == c)) for c in np.unique(y)]
    return {
        "top1": float(np.mean(pred == y)),
        "top5": float(np.mean([truth in guesses for truth, guesses in zip(y, top5)])),
        "mean_class_accuracy": float(np.mean(per_class)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="original,deepprivacy2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y, _train_names, train_expected = load_split("original", TRAIN_SPLIT)
    print(f"train original: {len(train_x)}/{train_expected}")
    for method in methods:
        test_x, _test_y, _names, expected = load_split(method, TEST_C2_SPLIT)
        print(f"test {method}: {len(test_x)}/{expected}")
    if args.preflight:
        return 0 if len(train_x) > 0 else 2
    if not len(train_x):
        raise SystemExit("No original training pose features. Run stage 17 with --action-splits first.")

    train_idx, val_idx = stratified_holdout(train_y)
    mean = train_x[train_idx].mean(0).astype(np.float32)
    std = (train_x[train_idx].std(0) + 1e-6).astype(np.float32)
    x_std = (train_x - mean) / std
    model = PoseActionMLP(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(x_std[train_idx], train_y[train_idx], args.batch_size, True, args.seed)
    best_state, best_top1, best_epoch, stale = copy.deepcopy(model.state_dict()), -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step(); losses.append(loss.detach().item())
        val_metrics = score(predict(model, x_std[val_idx], args.batch_size, device), train_y[val_idx])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **val_metrics})
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_top1={val_metrics['top1']:.4f}", flush=True)
        if val_metrics["top1"] > best_top1:
            best_state, best_top1, best_epoch, stale = copy.deepcopy(model.state_dict()), val_metrics["top1"], epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)

    checkpoint = RGB_POSE_MODEL_DIR / "rgb_pose_action_head.pt"
    torch.save({
        "state_dict": model.state_dict(), "input_dim": train_x.shape[1],
        "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "feature_mean": mean, "feature_std": std, "best_epoch": best_epoch,
        "best_val_top1": best_top1, "seed": args.seed, "history": history,
    }, checkpoint)

    rows = []
    for method in methods:
        x, y, _names, expected = load_split(method, TEST_C2_SPLIT)
        if not len(x):
            continue
        if x.shape[1] != train_x.shape[1]:
            raise ValueError(f"Feature dimension mismatch for {method}: {x.shape[1]} vs {train_x.shape[1]}")
        values = score(predict(model, (x - mean) / std, args.batch_size, device), y)
        rows.append({
            "split": TEST_C2_SPLIT, "method": method,
            "utility_model": "rgb_extracted_pose_mlp", "training_domain": "original",
            "features": len(x), "expected": expected, "coverage": len(x) / max(expected, 1),
            **values, "checkpoint": str(checkpoint),
        })
    out = RGB_POSE_ROOT / "rgb_pose_action_utility_results.csv"
    fields = list(rows[0]) if rows else ["split", "method"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['method']}: top1={row['top1']:.4f} top5={row['top5']:.4f} coverage={row['coverage']:.4f}")
    print(f"checkpoint={checkpoint}\nresults={out}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
