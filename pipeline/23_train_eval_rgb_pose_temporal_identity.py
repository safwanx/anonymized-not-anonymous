#!/usr/bin/env python3
"""Train/evaluate a temporal Conv1D identity attacker on RGB pose sequences."""

from __future__ import annotations

import argparse
import copy
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline_common import EVAL_PROTOCOLS, RANDOM_SEED, meta_by_filename
from rgb_pose_common import COCO_KEYPOINTS, RGB_POSE_MODEL_DIR, RGB_POSE_ROOT, filename_to_stem, pose_feature_path


class TemporalPoseCNN(nn.Module):
    def __init__(self, channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def load_sequences(method: str, filenames: list[str]):
    sequences, ids, kept = [], [], []
    for filename in filenames:
        path = pose_feature_path(method, filename_to_stem(filename))
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            feature = data["feature"].astype(np.float32, copy=False)
            frames = int(data["sample_frames"])
        flat_size = frames * COCO_KEYPOINTS * 3
        if frames < 2 or feature.size < flat_size:
            continue
        sequence = feature[:flat_size].reshape(frames, COCO_KEYPOINTS * 3).T
        sequences.append(sequence)
        ids.append(int(meta_by_filename[filename]["person"]))
        kept.append(filename)
    if not sequences:
        return np.zeros((0, COCO_KEYPOINTS * 3, 0), np.float32), np.zeros((0,), np.int64), kept
    shapes = {item.shape for item in sequences}
    if len(shapes) != 1:
        raise ValueError(f"Mixed RGB-pose sequence shapes for {method}: {sorted(shapes)}")
    return np.stack(sequences), np.asarray(ids, np.int64), kept


def stratified_split(y: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_fraction)))
        val.extend(indices[:n_val]); train.extend(indices[n_val:])
    return np.asarray(train, np.int64), np.asarray(val, np.int64)


def loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator,
    )


@torch.inference_mode()
def logits(model, x, batch_size, device):
    model.eval(); parts = []
    for start in range(0, len(x), batch_size):
        parts.append(model(torch.from_numpy(x[start:start + batch_size]).float().to(device)).cpu().numpy())
    width = model.head[-1].out_features
    return np.concatenate(parts) if parts else np.zeros((0, width), np.float32)


def metrics(scores: np.ndarray, y: np.ndarray):
    if not len(y):
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "top5": 0.0}
    pred = scores.argmax(1); top5 = np.argpartition(scores, -min(5, scores.shape[1]), axis=1)[:, -5:]
    per_class = [float(np.mean(pred[y == label] == label)) for label in np.unique(y)]
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.mean(per_class)),
        "top5": float(np.mean([truth in choices for truth, choices in zip(y, top5)])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="crossview_cam2")
    parser.add_argument("--methods", default="original,deepprivacy2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if args.protocol not in EVAL_PROTOCOLS:
        raise SystemExit(f"Unknown protocol: {args.protocol}")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery, probe = EVAL_PROTOCOLS[args.protocol]
    x, person_ids, kept = load_sequences("original", gallery)
    if not len(x):
        raise SystemExit("No original gallery RGB-pose sequences.")
    classes = np.asarray(sorted(np.unique(person_ids)), np.int64)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[int(label)] for label in person_ids], np.int64)
    train_idx, val_idx = stratified_split(y, 0.15, args.seed)
    mean = x[train_idx].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = (x[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    x = (x - mean) / std

    model = TemporalPoseCNN(x.shape[1], len(classes), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(x[train_idx], y[train_idx], args.batch_size, True, args.seed)
    best, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step(); losses.append(loss.detach().item())
        val = metrics(logits(model, x[val_idx], args.batch_size, device), y[val_idx])
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_bal={val['balanced_accuracy']:.4f}", flush=True)
        if val["balanced_accuracy"] > best_score:
            best, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), val["balanced_accuracy"], epoch, 0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best)
    checkpoint = RGB_POSE_MODEL_DIR / f"{args.protocol}_rgb_pose_temporal_cnn.pt"
    torch.save({
        "state_dict": best, "classes": classes, "feature_mean": mean, "feature_std": std,
        "best_epoch": best_epoch, "best_val_balanced_accuracy": best_score, "seed": args.seed,
        "sample_frames": x.shape[2], "train_files": kept,
    }, checkpoint)

    rows = []
    for method in methods:
        test_x, test_ids, test_kept = load_sequences(method, probe)
        known = np.asarray([int(label) in class_to_index for label in test_ids], bool)
        test_x, test_ids = test_x[known], test_ids[known]
        test_y = np.asarray([class_to_index[int(label)] for label in test_ids], np.int64)
        if len(test_x):
            test_x = (test_x - mean) / std
        values = metrics(logits(model, test_x, args.batch_size, device), test_y)
        coverage = len(test_x) / max(len(probe), 1)
        rows.append({
            "attack": "rgb_pose_temporal_cnn_identity", "protocol": args.protocol, "method": method,
            "probe_requested": len(probe), "probe_features": len(test_kept),
            "probe_known_identity": len(test_x), "probe_coverage": coverage,
            **values, "coverage_adjusted_accuracy": values["accuracy"] * coverage,
            "num_gallery_persons": len(classes), "checkpoint": str(checkpoint),
        })
    out = RGB_POSE_ROOT / "rgb_pose_temporal_cnn_identity_results.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['method']}: coverage={row['probe_coverage']:.4f} "
              f"accuracy={row['accuracy']:.4f} balanced={row['balanced_accuracy']:.4f}")
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
