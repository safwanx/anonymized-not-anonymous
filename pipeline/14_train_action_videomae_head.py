#!/usr/bin/env python3
"""Train a lightweight 120-class action head on frozen VideoMAEv2 features."""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from action_videomae_common import (
    ACTION_MODELS_DIR,
    ACTION_ROOT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    load_or_build_feature_matrix,
    mean_class_accuracy,
    set_random_seed,
    topk_accuracy,
)
from pipeline_common import DEVICE, RANDOM_SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 trains a linear probe; >0 trains a one-hidden-layer MLP.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--rebuild-aggregates", action="store_true")
    parser.add_argument("--checkpoint-name", default="action_videomae_v2_head.pt")
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


def standardize_train_val(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std, mean.reshape(-1), std.reshape(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=DEVICE == "cuda")


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict[str, float]:
    model.eval()
    logits_parts = []
    with torch.inference_mode():
        for xb, _yb in make_loader(x, y, batch_size=batch_size, shuffle=False):
            logits_parts.append(model(xb.to(DEVICE)).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0) if logits_parts else np.zeros((0, 120), dtype=np.float32)
    pred = logits.argmax(axis=1) if len(logits) else np.asarray([], dtype=np.int64)
    return {
        "top1": float(np.mean(pred == y)) if len(y) else 0.0,
        "top5": topk_accuracy(logits, y, 5),
        "mean_class_accuracy": mean_class_accuracy(pred, y),
        "samples": float(len(y)),
    }


def main() -> int:
    args = parse_args()
    set_random_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    x_train, y_train, _train_files = load_or_build_feature_matrix("original", TRAIN_SPLIT, rebuild=args.rebuild_aggregates)
    x_val, y_val, _val_files = load_or_build_feature_matrix("original", VAL_SPLIT, rebuild=args.rebuild_aggregates)

    if len(x_train) == 0:
        raise RuntimeError("No original train features found. Run 13_extract_videomae_features.py first.")
    if len(x_val) == 0:
        raise RuntimeError("No original validation features found. Run 13_extract_videomae_features.py first.")

    x_train_std, x_val_std, mean, std = standardize_train_val(x_train, x_val)
    model = ActionHead(x_train_std.shape[1], args.hidden_dim, args.dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(x_train_std, y_train, args.batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_top1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, x_val_std, y_val, args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f}"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs; best epoch={best_epoch}")
                break

    model.load_state_dict(best_state)
    final_val = evaluate(model, x_val_std, y_val, args.batch_size)

    ACTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = ACTION_MODELS_DIR / args.checkpoint_name
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(x_train_std.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "num_classes": 120,
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "best_epoch": int(best_epoch),
            "best_val_top1": float(best_top1),
            "final_val": final_val,
            "args": vars(args),
        },
        ckpt_path,
    )

    history_path = ACTION_ROOT / "action_videomae_v2_train_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    elapsed = time.time() - start_time

    print("=" * 80)
    print("VideoMAE action head training complete")
    print("=" * 80)
    print(f"Checkpoint : {ckpt_path}")
    print(f"History    : {history_path}")
    print(f"Best epoch : {best_epoch}")
    print(f"Val metrics: {final_val}")
    print(f"Time       : {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
