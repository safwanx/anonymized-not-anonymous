#!/usr/bin/env python3
"""Shared helpers for RGB-extracted pose identity leakage experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd

from pipeline_common import (
    ANON_DIR,
    EVAL_PROTOCOLS,
    FEATURES_DIR,
    FRAMES_DIR,
    all_stems,
    anonymization_complete,
    available_anonymization_methods,
    frame_extraction_complete,
    meta_by_filename,
)

RGB_POSE_ROOT = FEATURES_DIR / "rgb_pose_identity"
RGB_POSE_FEATURE_DIR = RGB_POSE_ROOT / "video_features"
RGB_POSE_MODEL_DIR = RGB_POSE_ROOT / "models"
RGB_POSE_LOG_DIR = RGB_POSE_ROOT / "logs"

for directory in [RGB_POSE_ROOT, RGB_POSE_FEATURE_DIR, RGB_POSE_MODEL_DIR, RGB_POSE_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

COCO_KEYPOINTS = 17


def parse_csv_arg(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or raw.strip().lower() == "auto":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_protocols(raw_protocols: str | None) -> list[str]:
    protocols = parse_csv_arg(raw_protocols, list(EVAL_PROTOCOLS.keys()))
    unknown = [protocol for protocol in protocols if protocol not in EVAL_PROTOCOLS]
    if unknown:
        raise KeyError(f"Unknown protocol(s): {unknown}. Known: {sorted(EVAL_PROTOCOLS)}")
    return protocols


def resolve_methods(raw_methods: str | None, include_original: bool = True) -> list[str]:
    if raw_methods is None or raw_methods.strip().lower() == "auto":
        methods = set(available_anonymization_methods())
        if RGB_POSE_FEATURE_DIR.exists():
            methods.update(path.name for path in RGB_POSE_FEATURE_DIR.iterdir() if path.is_dir())
        methods.discard("original")
        return (["original"] if include_original else []) + sorted(methods)
    methods = parse_csv_arg(raw_methods, [])
    if include_original and "original" not in methods:
        methods.insert(0, "original")
    return methods


def safe_method_name(method: str) -> str:
    return method.replace("/", "_").replace("\\", "_").replace(":", "_")


def frame_dir_for_method(method: str, stem: str) -> Path:
    return FRAMES_DIR / stem if method == "original" else ANON_DIR / method / stem


def method_video_ready(method: str, stem: str) -> bool:
    return frame_extraction_complete(stem) if method == "original" else anonymization_complete(method, stem)


def pose_feature_path(method: str, stem: str) -> Path:
    return RGB_POSE_FEATURE_DIR / safe_method_name(method) / f"{stem}.npz"


def protocol_files(protocol: str, role: str) -> list[str]:
    gallery, probe = EVAL_PROTOCOLS[protocol]
    if role == "gallery":
        return list(gallery)
    if role == "probe":
        return list(probe)
    if role == "both":
        return sorted(dict.fromkeys(list(gallery) + list(probe)))
    raise ValueError(f"Unknown role: {role}")


def extraction_targets(
    method: str,
    protocols: list[str],
    include_anonymized_gallery: bool = False,
) -> list[str]:
    filenames: list[str] = []
    for protocol in protocols:
        if method == "original" or include_anonymized_gallery:
            filenames.extend(protocol_files(protocol, "both"))
        else:
            filenames.extend(protocol_files(protocol, "probe"))
    return sorted(dict.fromkeys(filenames))


def filename_to_stem(filename: str) -> str:
    row = meta_by_filename[filename]
    return str(row["stem"])


def load_pose_feature(method: str, filename: str) -> tuple[np.ndarray | None, dict[str, float]]:
    stem = filename_to_stem(filename)
    path = pose_feature_path(method, stem)
    if not path.exists():
        return None, {}
    data = np.load(path, allow_pickle=False)
    feature = data["feature"].astype(np.float32, copy=False)
    stats = {
        "sampled_frames": float(data["sampled_frames"]),
        "valid_pose_frames": float(data["valid_pose_frames"]),
        "pose_frame_coverage": float(data["pose_frame_coverage"]),
        "mean_keypoint_conf": float(data["mean_keypoint_conf"]),
    }
    return feature, stats


def build_xy(method: str, filenames: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    features = []
    person_ids = []
    kept = []
    coverages = []
    keypoint_confs = []

    for filename in filenames:
        if filename not in meta_by_filename:
            continue
        feature, stats = load_pose_feature(method, filename)
        if feature is None:
            continue
        features.append(feature.reshape(-1))
        person_ids.append(int(meta_by_filename[filename]["person"]))
        kept.append(filename)
        if stats:
            coverages.append(stats["pose_frame_coverage"])
            keypoint_confs.append(stats["mean_keypoint_conf"])

    if features:
        dims = [feature.shape[0] for feature in features]
        dim_counts = Counter(dims)
        target_dim = sorted(dim_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
        if len(dim_counts) > 1:
            print(
                f"WARNING: {method} has mixed RGB-pose feature dimensions {dict(dim_counts)}; "
                f"using dim={target_dim}. Re-run 17 with --overwrite to clean stale features."
            )

        keep_idx = [index for index, feature in enumerate(features) if feature.shape[0] == target_dim]
        features = [features[index] for index in keep_idx]
        person_ids = [person_ids[index] for index in keep_idx]
        kept = [kept[index] for index in keep_idx]
        coverages = [coverages[index] for index in keep_idx if index < len(coverages)]
        keypoint_confs = [keypoint_confs[index] for index in keep_idx if index < len(keypoint_confs)]

        x = np.stack(features).astype(np.float32, copy=False)
        y = np.asarray(person_ids, dtype=np.int64)
    else:
        x = np.zeros((0, 0), dtype=np.float32)
        y = np.asarray([], dtype=np.int64)

    summary = {
        "requested": float(len(filenames)),
        "valid": float(len(kept)),
        "video_coverage": len(kept) / max(len(filenames), 1),
        "mean_pose_frame_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_keypoint_conf": float(np.mean(keypoint_confs)) if keypoint_confs else 0.0,
    }
    return x, y, kept, summary


def feature_status(methods: list[str], protocols: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for protocol in protocols:
            for role in ["gallery", "probe"]:
                files = protocol_files(protocol, role)
                ready = 0
                features = 0
                for filename in files:
                    stem = filename_to_stem(filename)
                    ready += int(method_video_ready(method, stem))
                    features += int(pose_feature_path(method, stem).exists())
                rows.append(
                    {
                        "method": method,
                        "protocol": protocol,
                        "role": role,
                        "videos": len(files),
                        "ready_frame_dirs": ready,
                        "pose_features": features,
                    }
                )
    return pd.DataFrame(rows)


def topk_from_proba(proba: np.ndarray, classes: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(labels) == 0 or proba.size == 0:
        return 0.0
    k = min(k, proba.shape[1])
    top_indices = np.argsort(proba, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return float(np.mean([truth in row for truth, row in zip(labels, top_labels)]))
