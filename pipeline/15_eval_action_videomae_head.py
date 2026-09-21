#!/usr/bin/env python3
"""Evaluate the trained VideoMAE action head on original/anonymized test features."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from action_videomae_common import (
    ACTION_MODELS_DIR,
    ACTION_ROOT,
    DEFAULT_ANON_SPLITS,
    TEST_C2_SPLIT,
    TEST_C3_SPLIT,
    load_action_split,
    load_or_build_feature_matrix,
    mean_class_accuracy,
    parse_csv_arg,
    resolve_methods,
    topk_accuracy,
)
from pipeline_common import DEVICE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="auto")
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--checkpoint-name", default="action_videomae_v2_head.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rebuild-aggregates", action="store_true")
    return parser.parse_args()


class ActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int = 120) -> None:
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


def load_checkpoint(path: str) -> tuple[ActionHead, np.ndarray, np.ndarray]:
    ckpt_path = ACTION_MODELS_DIR / path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing action head checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ActionHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        num_classes=int(checkpoint.get("num_classes", 120)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["feature_mean"].astype(np.float32), checkpoint["feature_std"].astype(np.float32)


def predict_logits(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(DEVICE)
            parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 120), dtype=np.float32)


def evaluate_matrix(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[dict[str, float], np.ndarray]:
    logits = predict_logits(model, x, batch_size)
    pred = logits.argmax(axis=1) if len(logits) else np.asarray([], dtype=np.int64)
    metrics = {
        "top1": float(np.mean(pred == y)) if len(y) else 0.0,
        "top5": topk_accuracy(logits, y, 5),
        "mean_class_accuracy": mean_class_accuracy(pred, y),
        "samples": float(len(y)),
    }
    return metrics, pred


def per_class_rows(method: str, split_name: str, labels: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    manifest = load_action_split(split_name)
    support_by_action = {action: 0 for action in range(1, 121)}
    for row in manifest:
        support_by_action[int(row["action"])] += 1

    rows = []
    for label in range(120):
        mask = labels == label
        rows.append(
            {
                "method": method,
                "split": split_name,
                "action": label + 1,
                "feature_support": int(mask.sum()),
                "manifest_support": int(support_by_action[label + 1]),
                "top1": float(np.mean(pred[mask] == labels[mask])) if mask.any() else np.nan,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    model, mean, std = load_checkpoint(args.checkpoint_name)
    methods = resolve_methods(args.methods, include_original=True)
    splits = DEFAULT_ANON_SPLITS if args.splits.strip().lower() == "auto" else parse_csv_arg(args.splits, [])

    result_rows = []
    class_rows = []
    for split_name in splits:
        original_top1 = None
        split_rows = []
        for method in methods:
            x, y, filenames = load_or_build_feature_matrix(method, split_name, rebuild=args.rebuild_aggregates)
            if len(x) == 0:
                print(f"Skipping {method}/{split_name}: no features.")
                continue
            x = (x - mean.reshape(1, -1)) / std.reshape(1, -1)
            metrics, pred = evaluate_matrix(model, x, y, args.batch_size)
            row = {
                "method": method,
                "split": split_name,
                "num_features": len(filenames),
                **metrics,
            }
            split_rows.append(row)
            class_rows.extend(per_class_rows(method, split_name, y, pred))
            if method == "original":
                original_top1 = metrics["top1"]

        if original_top1 is None:
            original_top1 = next((row["top1"] for row in split_rows if row["method"] == "original"), 0.0)
        for row in split_rows:
            row["retained_utility_top1"] = (
                row["top1"] / original_top1 if original_top1 and row["method"] != "original" else 1.0
            )
            result_rows.append(row)

    df = pd.DataFrame(result_rows)
    per_class = pd.DataFrame(class_rows)
    results_path = ACTION_ROOT / "action_videomae_v2_results.csv"
    per_class_path = ACTION_ROOT / "action_videomae_v2_per_class.csv"
    df.to_csv(results_path, index=False)
    per_class.to_csv(per_class_path, index=False)

    print("=" * 80)
    print("VideoMAE action utility results")
    print("=" * 80)
    if not df.empty:
        print(
            df[
                [
                    "split",
                    "method",
                    "num_features",
                    "top1",
                    "top5",
                    "mean_class_accuracy",
                    "retained_utility_top1",
                ]
            ].to_string(index=False, float_format="{:.4f}".format)
        )
    else:
        print("No action utility results generated.")

    if not df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        plot_df = df.copy()
        plot_df["label"] = plot_df["split"] + " / " + plot_df["method"]
        ax.barh(plot_df["label"], plot_df["top1"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Top-1 action accuracy")
        ax.set_title("VideoMAEv2 Action Utility")
        plt.tight_layout()
        fig_path = ACTION_ROOT / "action_videomae_v2_results.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure: {fig_path}")

    print(f"Results   : {results_path}")
    print(f"Per-class : {per_class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
