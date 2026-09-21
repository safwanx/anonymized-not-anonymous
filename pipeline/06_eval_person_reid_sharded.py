#!/usr/bin/env python3
"""Evaluate OSNet privacy leakage with resumable per-video feature caches."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

from pipeline_common import *


METHOD_ORIGINAL = "original"
CACHE_DIR = FEATURES_DIR / "person_reid_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=os.environ.get("EVAL_ANON_METHODS", "auto"))
    parser.add_argument("--num-shards", type=int, default=40)
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def parse_methods(raw: str) -> list[str]:
    if raw.strip().lower() == "auto":
        return available_anonymization_methods()
    return [item.strip() for item in raw.split(",") if item.strip()]


def cache_path(method: str, filename: str) -> Path:
    return CACHE_DIR / method / f"{stem_of(filename)}.npz"


def protocol_filenames() -> tuple[set[str], set[str]]:
    galleries: set[str] = set()
    probes: set[str] = set()
    for gallery_files, probe_files in EVAL_PROTOCOLS.values():
        galleries.update(gallery_files)
        probes.update(probe_files)
    return galleries, probes


def target_items(methods: list[str]) -> list[tuple[str, str]]:
    galleries, probes = protocol_filenames()
    items = [(METHOD_ORIGINAL, name) for name in sorted(galleries | probes)]
    for method in methods:
        items.extend((method, name) for name in sorted(probes))
    return [item for item in items if item[1] in meta_by_filename]


def build_extractor():
    if str(REPOS / "adversaries" / "deep-person-reid") not in sys.path:
        sys.path.insert(0, str(REPOS / "adversaries" / "deep-person-reid"))
    from torchreid.utils import FeatureExtractor

    weights = (
        MODELS
        / "person_reid_osnet"
        / "kaiyangzhou_osnet"
        / "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    )
    if not weights.is_file():
        raise FileNotFoundError(f"Missing OSNet weights: {weights}")
    return FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=str(weights),
        device=DEVICE,
        verbose=True,
    )


def extract_one(extractor, method: str, filename: str) -> None:
    path = cache_path(method, filename)
    if path.exists():
        return
    row = meta_by_filename.get(filename)
    if row is None:
        return
    stem = str(row["stem"])
    frame_dir = frame_root_for_method(method) / stem
    detections = read_json(DETECT_DIR / f"{stem}.json", {})
    crops: list[np.ndarray] = []
    sampled_frames = 0
    frames_with_person = 0

    for frame_path in sorted(frame_dir.glob("*.jpg"))[::REID_FRAME_STEP]:
        sampled_frames += 1
        persons = detections.get(frame_path.stem, {}).get("persons", [])
        if not persons:
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        best_box = max(persons, key=lambda box: box[4])
        clipped = clip_box(best_box, image.shape[1], image.shape[0])
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        frames_with_person += 1

    if crops:
        with torch.no_grad():
            features = extractor(crops).detach().cpu().numpy().astype(np.float32)
        embedding = features.mean(axis=0).astype(np.float32)
    else:
        embedding = np.zeros((0,), dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embedding=embedding,
        sampled_frames=np.asarray(sampled_frames, dtype=np.int32),
        frames_with_person=np.asarray(frames_with_person, dtype=np.int32),
    )


def run_extract_shard(args: argparse.Namespace, methods: list[str]) -> int:
    items = [
        item
        for index, item in enumerate(target_items(methods))
        if index % args.num_shards == args.shard_index
    ]
    if not args.overwrite_cache:
        items = [item for item in items if not cache_path(*item).exists()]
    print(
        f"OSNet shard {args.shard_index}/{args.num_shards}: {len(items)} pending items",
        flush=True,
    )
    if not items:
        return 0
    extractor = build_extractor()
    for method, filename in tqdm(items, desc=f"OSNet shard {args.shard_index}"):
        path = cache_path(method, filename)
        if args.overwrite_cache and path.exists():
            path.unlink()
        extract_one(extractor, method, filename)
    del extractor
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
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=os.environ.copy())
        if result.returncode != 0:
            return result.returncode
    return 0


def load_cached(method: str, filename: str) -> tuple[np.ndarray | None, int, int]:
    path = cache_path(method, filename)
    if not path.is_file():
        return None, 0, 0
    data = np.load(path)
    embedding = data["embedding"].astype(np.float32, copy=False)
    sampled = int(data["sampled_frames"])
    person_frames = int(data["frames_with_person"])
    if embedding.shape != (512,):
        return None, sampled, person_frames
    return embedding, sampled, person_frames


def build_matrix(
    filenames: list[str], method: str
) -> tuple[np.ndarray, list[int], dict[str, float]]:
    embeddings: list[np.ndarray] = []
    person_ids: list[int] = []
    sampled_frames = 0
    frames_with_person = 0
    for filename in filenames:
        embedding, sampled, person_frames = load_cached(method, filename)
        sampled_frames += sampled
        frames_with_person += person_frames
        if embedding is None:
            continue
        embeddings.append(embedding)
        person_ids.append(int(meta_by_filename[filename]["person"]))
    stats = {
        "video_coverage": len(embeddings) / max(len(filenames), 1),
        "person_frame_coverage": frames_with_person / max(sampled_frames, 1),
    }
    if embeddings:
        return np.stack(embeddings), person_ids, stats
    return np.zeros((0, 512), dtype=np.float32), [], stats


def evaluate(methods: list[str]) -> int:
    rows = []
    for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
        gallery_feats, gallery_ids, gallery_stats = build_matrix(gallery_files, METHOD_ORIGINAL)
        for method in [METHOD_ORIGINAL] + methods:
            probe_feats, probe_ids, probe_stats = build_matrix(probe_files, method)
            metrics = compute_reid_metrics(gallery_feats, gallery_ids, probe_feats, probe_ids)
            metrics.update(
                {
                    "attack": "reid",
                    "protocol": protocol,
                    "method": method,
                    "gallery_requested": len(gallery_files),
                    "probe_requested": len(probe_files),
                    "gallery_coverage": gallery_stats["video_coverage"],
                    "probe_coverage": probe_stats["video_coverage"],
                    "probe_person_frame_coverage": probe_stats["person_frame_coverage"],
                    "coverage_adjusted_rank1": metrics["rank1"]
                    * probe_stats["video_coverage"],
                }
            )
            rows.append(metrics)
    save_attack_results("person_reid", rows)
    return 0


def main() -> int:
    args = parse_args()
    methods = parse_methods(args.methods)
    print(f"OSNet methods: {[METHOD_ORIGINAL] + methods}")
    print(f"OSNet cache  : {CACHE_DIR}")
    if args.evaluate_only:
        return evaluate(methods)
    if args.shard_index >= 0:
        return run_extract_shard(args, methods)
    code = run_sharded_extract(args, methods)
    return code if code != 0 else evaluate(methods)


if __name__ == "__main__":
    raise SystemExit(main())

