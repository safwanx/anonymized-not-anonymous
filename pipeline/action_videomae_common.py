#!/usr/bin/env python3
"""Shared helpers for the VideoMAE action-utility pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipeline_common import (
    ANON_DIR,
    FEATURES_DIR,
    FRAMES_DIR,
    RANDOM_SEED,
    all_stems,
    anonymization_complete,
    available_anonymization_methods,
    frame_extraction_complete,
    meta,
    meta_by_filename,
)

ACTION_ROOT = FEATURES_DIR / "action_videomae_v2"
ACTION_SPLITS_DIR = ACTION_ROOT / "splits"
VIDEO_FEATURES_DIR = ACTION_ROOT / "video_features"
AGGREGATES_DIR = ACTION_ROOT / "aggregates"
ACTION_MODELS_DIR = ACTION_ROOT / "models"
ACTION_LOGS_DIR = ACTION_ROOT / "logs"

for directory in [
    ACTION_ROOT,
    ACTION_SPLITS_DIR,
    VIDEO_FEATURES_DIR,
    AGGREGATES_DIR,
    ACTION_MODELS_DIR,
    ACTION_LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = "train_original_r1_allcams"
VAL_SPLIT = "val_original_c1r2"
TEST_C2_SPLIT = "test_original_c2r2"
TEST_C3_SPLIT = "test_original_c3r2"

DEFAULT_ORIGINAL_SPLITS = [TRAIN_SPLIT, VAL_SPLIT, TEST_C2_SPLIT, TEST_C3_SPLIT]
DEFAULT_ANON_SPLITS = [TEST_C2_SPLIT, TEST_C3_SPLIT]


SPLIT_DEFINITIONS: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
    TRAIN_SPLIT: (
        "Original RGB training clips: all cameras, replication 1.",
        lambda row: int(row["replication"]) == 1,
    ),
    VAL_SPLIT: (
        "Original RGB validation clips: camera 1, replication 2.",
        lambda row: int(row["camera"]) == 1 and int(row["replication"]) == 2,
    ),
    TEST_C2_SPLIT: (
        "Cross-view test clips: camera 2, replication 2.",
        lambda row: int(row["camera"]) == 2 and int(row["replication"]) == 2,
    ),
    TEST_C3_SPLIT: (
        "Cross-view test clips: camera 3, replication 2.",
        lambda row: int(row["camera"]) == 3 and int(row["replication"]) == 2,
    ),
}


def action_label(action_id: int) -> int:
    """Map NTU action IDs 1..120 to classifier labels 0..119."""
    return int(action_id) - 1


def split_csv_path(split_name: str) -> Path:
    return ACTION_SPLITS_DIR / f"{split_name}.csv"


def split_rows(split_name: str) -> list[dict[str, Any]]:
    if split_name not in SPLIT_DEFINITIONS:
        raise KeyError(f"Unknown action split: {split_name}")

    _description, selector = SPLIT_DEFINITIONS[split_name]
    rows = [row for row in meta if selector(row)]
    rows.sort(key=lambda item: str(item["filename"]))
    return rows


def write_action_splits() -> pd.DataFrame:
    """Write deterministic VideoMAE action split manifests."""
    summaries = []
    fields = [
        "filename",
        "stem",
        "setup",
        "camera",
        "person",
        "replication",
        "action",
        "label",
        "pool",
        "rgb_resolved",
    ]

    for split_name, (description, _selector) in SPLIT_DEFINITIONS.items():
        rows = split_rows(split_name)
        path = split_csv_path(split_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "filename": row["filename"],
                        "stem": row["stem"],
                        "setup": row["setup"],
                        "camera": row["camera"],
                        "person": row["person"],
                        "replication": row["replication"],
                        "action": row["action"],
                        "label": action_label(int(row["action"])),
                        "pool": row["pool"],
                        "rgb_resolved": row["rgb_resolved"],
                    }
                )

        action_counts = pd.Series([int(row["action"]) for row in rows]).value_counts()
        summaries.append(
            {
                "split": split_name,
                "description": description,
                "videos": len(rows),
                "actions": int(action_counts.size),
                "min_per_action": int(action_counts.min()) if not action_counts.empty else 0,
                "max_per_action": int(action_counts.max()) if not action_counts.empty else 0,
                "path": str(path),
            }
        )

    df = pd.DataFrame(summaries)
    df.to_csv(ACTION_ROOT / "action_split_summary.csv", index=False)
    return df


def ensure_action_splits() -> None:
    missing = [name for name in SPLIT_DEFINITIONS if not split_csv_path(name).exists()]
    if missing:
        print("Missing action split manifests; writing them now.")
        write_action_splits()


def load_action_split(split_name: str) -> list[dict[str, Any]]:
    ensure_action_splits()
    path = split_csv_path(split_name)
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ["setup", "camera", "person", "replication", "action", "label"]:
            row[key] = int(row[key])
    return rows


def parse_csv_arg(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() == "auto":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_methods(raw_methods: str | None, include_original: bool = True) -> list[str]:
    if raw_methods is None or raw_methods.strip().lower() == "auto":
        methods = set(available_anonymization_methods())
        if VIDEO_FEATURES_DIR.exists():
            methods.update(path.name for path in VIDEO_FEATURES_DIR.iterdir() if path.is_dir())
        methods.discard("original")
        return (["original"] if include_original else []) + sorted(methods)

    methods = parse_csv_arg(raw_methods, [])
    if include_original and "original" not in methods:
        methods.insert(0, "original")
    return methods


def frame_dir_for_method(method: str, stem: str) -> Path:
    if method == "original":
        return FRAMES_DIR / stem
    return ANON_DIR / method / stem


def method_video_ready(method: str, stem: str) -> bool:
    if method == "original":
        return frame_extraction_complete(stem)
    return anonymization_complete(method, stem)


def safe_method_name(method: str) -> str:
    return method.replace("/", "_").replace("\\", "_").replace(":", "_")


def video_feature_path(method: str, stem: str) -> Path:
    return VIDEO_FEATURES_DIR / safe_method_name(method) / f"{stem}.npy"


def aggregate_path(method: str, split_name: str) -> Path:
    return AGGREGATES_DIR / f"{safe_method_name(method)}__{split_name}.npz"


def expected_splits_for_method(method: str, requested_splits: list[str] | None = None) -> list[str]:
    if requested_splits is not None:
        return requested_splits
    return DEFAULT_ORIGINAL_SPLITS if method == "original" else DEFAULT_ANON_SPLITS


def missing_feature_rows(method: str, split_name: str) -> list[dict[str, Any]]:
    rows = load_action_split(split_name)
    return [
        row
        for row in rows
        if method_video_ready(method, str(row["stem"]))
        and not video_feature_path(method, str(row["stem"])).exists()
    ]


def load_video_feature(method: str, stem: str) -> np.ndarray | None:
    path = video_feature_path(method, stem)
    if not path.exists():
        return None
    return np.load(path).astype(np.float32, copy=False)


def build_feature_matrix(
    method: str,
    split_name: str,
    write_aggregate: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    rows = load_action_split(split_name)
    features = []
    labels = []
    filenames = []
    kept_rows = []

    for row in rows:
        feature = load_video_feature(method, str(row["stem"]))
        if feature is None:
            continue
        features.append(feature.reshape(-1))
        labels.append(int(row["label"]))
        filenames.append(str(row["filename"]))
        kept_rows.append(row)

    if features:
        x = np.stack(features).astype(np.float32, copy=False)
        y = np.asarray(labels, dtype=np.int64)
    else:
        x = np.zeros((0, 0), dtype=np.float32)
        y = np.asarray([], dtype=np.int64)

    df = pd.DataFrame(kept_rows)
    if write_aggregate:
        path = aggregate_path(method, split_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=x.astype(np.float16),
            labels=y,
            filenames=np.asarray(filenames, dtype="U128"),
        )
    return x, y, filenames, df


def load_or_build_feature_matrix(
    method: str,
    split_name: str,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = aggregate_path(method, split_name)
    if path.exists() and not rebuild:
        data = np.load(path, allow_pickle=False)
        return (
            data["features"].astype(np.float32, copy=False),
            data["labels"].astype(np.int64, copy=False),
            [str(item) for item in data["filenames"].tolist()],
        )

    x, y, filenames, _df = build_feature_matrix(method, split_name, write_aggregate=True)
    return x, y, filenames


def summarize_feature_status(methods: list[str], splits: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for split_name in splits:
            split_manifest = load_action_split(split_name)
            ready = sum(1 for row in split_manifest if method_video_ready(method, str(row["stem"])))
            features = sum(1 for row in split_manifest if video_feature_path(method, str(row["stem"])).exists())
            rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "videos": len(split_manifest),
                    "ready_frame_dirs": ready,
                    "features": features,
                    "missing_features": max(len(split_manifest) - features, 0),
                    "aggregate_exists": aggregate_path(method, split_name).exists(),
                }
            )
    return pd.DataFrame(rows)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    top_indices = np.argsort(logits, axis=1)[:, -k:]
    return float(np.mean([truth in row for truth, row in zip(labels, top_indices)]))


def mean_class_accuracy(pred: np.ndarray, labels: np.ndarray, num_classes: int = 120) -> float:
    values = []
    for label in range(num_classes):
        mask = labels == label
        if mask.any():
            values.append(float(np.mean(pred[mask] == labels[mask])))
    return float(np.mean(values)) if values else 0.0


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
