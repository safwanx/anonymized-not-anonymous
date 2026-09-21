#!/usr/bin/env python3
"""Cache-backed, array-safe GaitBase identity evaluation."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import gait_paths as G
from gaitbase_model import load_gaitbase
from gait_silhouette import extract_sequence

from pipeline_common import (  # noqa: E402
    EVAL_PROTOCOLS,
    FEATURES_DIR,
    FRAMES_DIR,
    SIL_FRAME_STEP,
    available_anonymization_methods,
    compute_reid_metrics,
    frame_root_for_method,
    free_gpu,
    meta_by_filename,
    save_attack_results,
    stem_of,
)


METHOD_ORIGINAL = "original"
CACHE_DIR = FEATURES_DIR / "gait_identity_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=os.environ.get("EVAL_ANON_METHODS", "auto"))
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def parse_methods(raw: str) -> list[str]:
    if raw.strip().lower() == "auto":
        return available_anonymization_methods()
    return [item.strip() for item in raw.split(",") if item.strip() and item != METHOD_ORIGINAL]


def cache_path(method: str, filename: str) -> Path:
    return CACHE_DIR / method / f"{stem_of(filename)}.npz"


def target_items(methods: list[str]) -> list[tuple[str, str]]:
    galleries: set[str] = set()
    probes: set[str] = set()
    for gallery_files, probe_files in EVAL_PROTOCOLS.values():
        galleries.update(gallery_files)
        probes.update(probe_files)
    items = [(METHOD_ORIGINAL, name) for name in sorted(galleries | probes)]
    for method in methods:
        items.extend((method, name) for name in sorted(probes))
    return [item for item in items if item[1] in meta_by_filename]


def extract_one(model, device: torch.device, method: str, filename: str) -> None:
    path = cache_path(method, filename)
    if path.exists():
        return
    stem = str(meta_by_filename[filename]["stem"])
    sequence = extract_sequence(
        stem,
        frame_root_for_method(method),
        frame_step=SIL_FRAME_STEP,
        mask_dir=None,
    )
    if sequence is None:
        embedding = np.zeros((0,), dtype=np.float32)
        sequence_frames = 0
    else:
        embedding = model.embed(sequence, device)
        sequence_frames = int(len(sequence))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embedding=embedding,
        sequence_frames=np.asarray(sequence_frames, dtype=np.int32),
        silhouette_source=np.asarray("otsu_person_crop"),
    )


def run_extract_shard(args: argparse.Namespace, methods: list[str]) -> int:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    items = [
        item
        for index, item in enumerate(target_items(methods))
        if index % args.num_shards == args.shard_index
    ]
    if not args.overwrite_cache:
        items = [item for item in items if not cache_path(*item).exists()]
    print(f"GaitBase shard {args.shard_index}/{args.num_shards}: {len(items)} pending")
    if not items:
        return 0
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_gaitbase(device)
    for method, filename in tqdm(items, desc=f"GaitBase shard {args.shard_index}"):
        path = cache_path(method, filename)
        if args.overwrite_cache and path.exists():
            path.unlink()
        extract_one(model, device, method, filename)
    del model
    gc.collect()
    free_gpu()
    return 0


def load_cached(method: str, filename: str) -> np.ndarray | None:
    path = cache_path(method, filename)
    if not path.is_file():
        return None
    embedding = np.load(path)["embedding"].astype(np.float32, copy=False)
    return embedding if embedding.shape == (4096,) else None


def build_matrix(filenames: list[str], method: str) -> tuple[np.ndarray, list[int], float]:
    features: list[np.ndarray] = []
    person_ids: list[int] = []
    for filename in filenames:
        embedding = load_cached(method, filename)
        if embedding is None:
            continue
        features.append(embedding)
        person_ids.append(int(meta_by_filename[filename]["person"]))
    coverage = len(features) / max(len(filenames), 1)
    if features:
        return np.stack(features), person_ids, coverage
    return np.zeros((0, 4096), dtype=np.float32), [], coverage


def evaluate(methods: list[str]) -> int:
    rows = []
    for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
        gallery_features, gallery_ids, gallery_coverage = build_matrix(
            gallery_files, METHOD_ORIGINAL
        )
        for method in [METHOD_ORIGINAL] + methods:
            probe_features, probe_ids, probe_coverage = build_matrix(probe_files, method)
            metrics = compute_reid_metrics(
                gallery_features, gallery_ids, probe_features, probe_ids
            )
            metrics.update(
                {
                    "attack": "gait_identity",
                    "protocol": protocol,
                    "method": method,
                    "gallery_requested": len(gallery_files),
                    "probe_requested": len(probe_files),
                    "gallery_coverage": gallery_coverage,
                    "probe_coverage": probe_coverage,
                    "coverage_adjusted_rank1": metrics["rank1"] * probe_coverage,
                    "silhouette_source": "otsu_person_crop",
                    "checkpoint_domain": "CASIA-B",
                }
            )
            rows.append(metrics)
    save_attack_results("gait_identity", rows)
    return 0


def main() -> int:
    args = parse_args()
    methods = parse_methods(args.methods)
    print(f"GaitBase methods: {[METHOD_ORIGINAL] + methods}")
    print(f"GaitBase cache  : {CACHE_DIR}")
    if args.evaluate_only:
        return evaluate(methods)
    if args.shard_index < 0:
        raise SystemExit("Use --shard-index for extraction, or --evaluate-only after all shards.")
    return run_extract_shard(args, methods)


if __name__ == "__main__":
    raise SystemExit(main())

