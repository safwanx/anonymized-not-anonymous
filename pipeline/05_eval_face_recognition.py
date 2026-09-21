#!/usr/bin/env python3
"""Evaluate face-recognition privacy leakage with a sharded embedding cache.

InsightFace/ONNXRuntime can leak native host memory over long runs. This stage
therefore extracts embeddings in short subprocess shards, writes one compact
cache file per (method, video), then computes the metrics from cache in the
parent process.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

from pipeline_common import *


METHOD_ORIGINAL = "original"
CACHE_DIR = FEATURES_DIR / "face_recognition_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=os.environ.get("EVAL_ANON_METHODS", "auto"))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("FACE_NUM_SHARDS", "80")))
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def parse_methods(raw: str) -> list[str]:
    if raw.strip().lower() == "auto":
        return available_anonymization_methods()
    return [item.strip() for item in raw.split(",") if item.strip()]


def cache_path(method: str, filename: str) -> Path:
    stem = stem_of(filename)
    return CACHE_DIR / method / f"{stem}.npz"


def protocol_filenames() -> tuple[set[str], set[str]]:
    galleries: set[str] = set()
    probes: set[str] = set()
    for gallery_files, probe_files in EVAL_PROTOCOLS.values():
        galleries.update(gallery_files)
        probes.update(probe_files)
    return galleries, probes


def target_items(methods: list[str]) -> list[tuple[str, str]]:
    galleries, probes = protocol_filenames()
    items: list[tuple[str, str]] = []
    for filename in sorted(galleries | probes):
        if filename in meta_by_filename:
            items.append((METHOD_ORIGINAL, filename))
    for method in methods:
        for filename in sorted(probes):
            if filename in meta_by_filename:
                items.append((method, filename))
    return items


def build_face_rec():
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    cpu_count = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = cpu_count
    session_options.inter_op_num_threads = 1

    cuda_opts = {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "gpu_mem_limit": str(int(os.environ.get("FACE_GPU_MEM_GB", "6")) * 1024**3),
    }
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = [cuda_opts, {}]
    face_rec = FaceAnalysis(
        name="buffalo_l",
        root=ensure_insightface_root("buffalo_l"),
        allowed_modules=["detection", "recognition"],
        providers=providers,
        provider_options=provider_options,
        sess_options=session_options,
    )
    face_rec.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
    return face_rec


def extract_one(face_rec, method: str, filename: str) -> None:
    path = cache_path(method, filename)
    if path.exists():
        return

    row = meta_by_filename.get(filename)
    if row is None:
        return
    stem = row["stem"]
    frames_dir = frame_root_for_method(method) / stem
    path.parent.mkdir(parents=True, exist_ok=True)

    sampled_frames = 0
    face_frame_count = 0
    video_embeddings = []

    if frames_dir.exists():
        for frame_path in sorted(frames_dir.glob("*.jpg"))[::FACE_FRAME_STEP]:
            sampled_frames += 1
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            faces = face_rec.get(image)
            if faces:
                face_frame_count += 1
                best = max(faces, key=lambda item: item.det_score)
                if best.embedding is not None:
                    video_embeddings.append(best.embedding)

    if video_embeddings:
        embedding = np.mean(video_embeddings, axis=0).astype(np.float32)
    else:
        embedding = np.zeros((0,), dtype=np.float32)

    np.savez_compressed(
        path,
        embedding=embedding,
        sampled_frames=np.asarray(sampled_frames, dtype=np.int32),
        face_frame_count=np.asarray(face_frame_count, dtype=np.int32),
    )


def run_extract_shard(args: argparse.Namespace, methods: list[str]) -> int:
    items = target_items(methods)
    if args.overwrite_cache and args.shard_index == 0:
        # Keep overwrite simple and explicit: only shard 0 clears old cache before
        # any extraction starts. For SLURM arrays, use a clean cache instead.
        pass
    shard_items = [
        item
        for index, item in enumerate(items)
        if index % args.num_shards == args.shard_index
    ]
    if not args.overwrite_cache:
        shard_items = [item for item in shard_items if not cache_path(*item).exists()]
    print(
        f"face shard {args.shard_index}/{args.num_shards}: "
        f"{len(shard_items)} of {len(items)} target method-videos",
        flush=True,
    )
    if not shard_items:
        return 0

    face_rec = build_face_rec()
    for method, filename in tqdm(shard_items, desc=f"face shard {args.shard_index}", leave=False):
        if args.overwrite_cache and cache_path(method, filename).exists():
            cache_path(method, filename).unlink()
        extract_one(face_rec, method, filename)

    del face_rec
    gc.collect()
    free_gpu()
    return 0


def run_sharded_extract(args: argparse.Namespace, methods: list[str]) -> int:
    for shard_index in range(args.num_shards):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--methods",
            ",".join(methods),
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(shard_index),
        ]
        if args.overwrite_cache:
            cmd.append("--overwrite-cache")
        print("=" * 80, flush=True)
        print("Running", " ".join(cmd), flush=True)
        print("=" * 80, flush=True)
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=os.environ.copy())
        if result.returncode != 0:
            return result.returncode
    return 0


def load_cached(method: str, filename: str) -> tuple[np.ndarray | None, int, int]:
    path = cache_path(method, filename)
    if not path.exists():
        return None, 0, 0
    data = np.load(path)
    embedding = data["embedding"].astype(np.float32, copy=False)
    sampled_frames = int(data["sampled_frames"])
    face_frame_count = int(data["face_frame_count"])
    if embedding.shape != (512,):
        return None, sampled_frames, face_frame_count
    return embedding, sampled_frames, face_frame_count


def build_matrix(file_list: list[str], method: str) -> tuple[np.ndarray, list[int], dict[str, float]]:
    embeddings = []
    person_ids = []
    sampled_frames = 0
    face_frame_count = 0
    for filename in file_list:
        row = meta_by_filename.get(filename)
        if row is None:
            continue
        embedding, frames, face_frames = load_cached(method, filename)
        sampled_frames += frames
        face_frame_count += face_frames
        if embedding is None:
            continue
        embeddings.append(embedding)
        person_ids.append(int(row["person"]))

    stats = {
        "requested": float(len(file_list)),
        "valid": float(len(embeddings)),
        "video_coverage": len(embeddings) / max(len(file_list), 1),
        "sampled_frames": float(sampled_frames),
        "face_frame_count": float(face_frame_count),
        "face_frame_coverage": face_frame_count / max(sampled_frames, 1),
    }
    if embeddings:
        return np.stack(embeddings), person_ids, stats
    return np.zeros((0, 512), dtype=np.float32), [], stats


def evaluate(methods: list[str]) -> int:
    face_rows = []
    for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
        gallery_feats, gallery_ids, gallery_stats = build_matrix(gallery_files, METHOD_ORIGINAL)
        for method in [METHOD_ORIGINAL] + methods:
            probe_feats, probe_ids, probe_stats = build_matrix(probe_files, method)
            metrics = compute_reid_metrics(gallery_feats, gallery_ids, probe_feats, probe_ids)
            metrics.update(
                {
                    "attack": "face",
                    "protocol": protocol,
                    "method": method,
                    "gallery_requested": len(gallery_files),
                    "probe_requested": len(probe_files),
                    "gallery_coverage": gallery_stats["video_coverage"],
                    "probe_coverage": probe_stats["video_coverage"],
                    "probe_face_frame_coverage": probe_stats["face_frame_coverage"],
                    "coverage_adjusted_rank1": metrics["rank1"] * probe_stats["video_coverage"],
                }
            )
            face_rows.append(metrics)
    save_attack_results("face_recognition", face_rows)
    return 0


def main() -> int:
    args = parse_args()
    methods = parse_methods(args.methods)
    print(f"Face-recognition methods: {[METHOD_ORIGINAL] + methods}", flush=True)
    print(f"Face cache: {CACHE_DIR}", flush=True)

    if args.evaluate_only:
        return evaluate(methods)

    if args.shard_index >= 0:
        return run_extract_shard(args, methods)

    code = run_sharded_extract(args, methods)
    if code != 0:
        return code
    return evaluate(methods)


if __name__ == "__main__":
    raise SystemExit(main())
