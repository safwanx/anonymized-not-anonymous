#!/usr/bin/env python3
"""Generate DeepPrivacy2 anonymized frames for evaluation probes only.

This script is intentionally separate from 04_anonymize_baselines.py because
the full notebook version processes all 21,600 pilot videos. For the paper
extension we only need anonymized probe/test clips: privacy probes plus the
VideoMAE action test clips.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from pipeline_common import (
    ANON_DIR,
    ANON_JPEG_QUALITY,
    DEVICE,
    EVAL_PROTOCOLS,
    FEATURES_DIR,
    FRAMES_DIR,
    LOGS_DIR,
    REPOS,
    append_jsonl,
    count_jpgs,
    expected_frame_count,
    frame_extraction_complete,
    free_gpu,
    meta,
    meta_by_filename,
)


METHOD_NAME = "deepprivacy2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        default="auto",
        help="Comma-separated protocols. Auto uses all privacy protocols.",
    )
    parser.add_argument(
        "--include-action-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also anonymize VideoMAE C2/C3 replication-2 action test clips.",
    )
    parser.add_argument(
        "--action-splits",
        default="",
        help="Optional comma-separated action split manifests to anonymize (for adapted utility training).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug/smoke limit after deterministic sorting. 0 means all targets.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=ANON_JPEG_QUALITY)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--status-tag", default="", help="Optional scope tag for sharded status filenames.")
    return parser.parse_args()


def parse_csv(raw: str, default: list[str]) -> list[str]:
    if raw.strip().lower() == "auto":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def protocol_probe_filenames(protocols: list[str]) -> list[str]:
    filenames: list[str] = []
    for protocol in protocols:
        if protocol not in EVAL_PROTOCOLS:
            raise KeyError(f"Unknown protocol: {protocol}. Known: {sorted(EVAL_PROTOCOLS)}")
        _gallery, probe = EVAL_PROTOCOLS[protocol]
        filenames.extend(probe)
    return filenames


def action_test_filenames() -> list[str]:
    return [
        str(row["filename"])
        for row in meta
        if int(row["camera"]) in {2, 3} and int(row["replication"]) == 2
    ]


def target_filenames(args: argparse.Namespace) -> list[str]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    protocols = parse_csv(args.protocols, list(EVAL_PROTOCOLS.keys()))
    filenames = protocol_probe_filenames(protocols)
    if args.include_action_tests:
        filenames.extend(action_test_filenames())
    if args.action_splits:
        from action_videomae_common import load_action_split

        for split_name in parse_csv(args.action_splits, []):
            filenames.extend(str(row["filename"]) for row in load_action_split(split_name))

    unique = sorted(dict.fromkeys(filename for filename in filenames if filename in meta_by_filename))
    if args.limit > 0:
        unique = unique[: args.limit]
    unique = unique[args.shard_index :: args.num_shards]
    return unique


def filename_to_stem(filename: str) -> str:
    return str(meta_by_filename[filename]["stem"])


def output_dir_for_stem(stem: str) -> Path:
    return ANON_DIR / METHOD_NAME / stem


def deepprivacy2_complete(stem: str) -> bool:
    expected = expected_frame_count(stem)
    return expected > 0 and count_jpgs(output_dir_for_stem(stem)) == expected


def clear_output(stem: str) -> None:
    out_dir = output_dir_for_stem(stem)
    if out_dir.exists():
        shutil.rmtree(out_dir)


def load_deepprivacy2(args: argparse.Namespace):
    if DEVICE != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "DeepPrivacy2 is running without CUDA. Use --allow-cpu only for tiny smoke tests."
        )

    dp2_repo = REPOS / "privacy_methods" / "deep_privacy2"
    dp2_config = dp2_repo / "configs" / "anonymizers" / "FB_cse.py"
    if not dp2_repo.exists():
        raise FileNotFoundError(f"Missing DeepPrivacy2 repo: {dp2_repo}")
    if not dp2_config.exists():
        raise FileNotFoundError(f"Missing DeepPrivacy2 config: {dp2_config}")

    sys.path.insert(0, str(dp2_repo))
    # dp2 loads nested generator configs by repo-relative path
    # (configs/fdh/styleganL.py), so the process must run from the repo root.
    # All pipeline I/O uses absolute paths from pipeline_common, so this is safe.
    os.chdir(dp2_repo)
    from dp2 import utils as dp2_utils
    from tops.config import instantiate as tops_instantiate

    print("Loading DeepPrivacy2 anonymizer...", flush=True)
    cfg = dp2_utils.load_config(str(dp2_config))
    cfg.detector.score_threshold = args.score_threshold
    anonymizer = tops_instantiate(cfg.anonymizer, load_cache=False)
    synthesis_kwargs = {
        "amp": DEVICE == "cuda",
        "multi_modal_truncation": False,
        "truncation_value": 0,
    }
    print("DeepPrivacy2 loaded.", flush=True)
    return dp2_utils, anonymizer, synthesis_kwargs


def anonymize_stem(
    stem: str,
    dp2_utils,
    anonymizer,
    synthesis_kwargs: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    frame_dir = FRAMES_DIR / stem
    out_dir = output_dir_for_stem(stem)
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    expected = expected_frame_count(stem)

    if args.overwrite:
        clear_output(stem)

    out_dir.mkdir(parents=True, exist_ok=True)
    written_before = count_jpgs(out_dir)
    failed_frames = 0
    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]

    for frame_path in frame_paths:
        out_path = out_dir / frame_path.name
        if out_path.exists():
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            failed_frames += 1
            append_jsonl(
                LOGS_DIR / "deepprivacy2_probe_errors.jsonl",
                {"stem": stem, "frame": frame_path.name, "error": "cv2_read_failed"},
            )
            continue

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_t = dp2_utils.im2torch(image_rgb, to_float=False, normalize=False)[0]
            anon_t = anonymizer(image_t, **synthesis_kwargs)
            anon_np = dp2_utils.im2numpy(anon_t)
            anon_bgr = cv2.cvtColor(anon_np, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(out_path), anon_bgr, jpg_params):
                failed_frames += 1
                append_jsonl(
                    LOGS_DIR / "deepprivacy2_probe_errors.jsonl",
                    {"stem": stem, "frame": frame_path.name, "error": "cv2_write_failed"},
                )
        except Exception as exc:
            failed_frames += 1
            append_jsonl(
                LOGS_DIR / "deepprivacy2_probe_errors.jsonl",
                {"stem": stem, "frame": frame_path.name, "error": str(exc)},
            )

    written_after = count_jpgs(out_dir)
    return {
        "stem": stem,
        "expected_frames": expected,
        "written_before": written_before,
        "written_after": written_after,
        "failed_frames": failed_frames,
        "complete": expected > 0 and written_after == expected,
    }


def write_status(
    rows: list[dict[str, object]], target_count: int, args: argparse.Namespace
) -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    scope = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.status_tag.strip())
    prefix = f"deepprivacy2_{scope}_status" if scope else "deepprivacy2_probe_status"
    if args.num_shards == 1:
        status_name = f"{prefix}.csv"
    else:
        status_name = (
            f"{prefix}.shard_{args.shard_index:03d}"
            f"_of_{args.num_shards:03d}.csv"
        )
    status_path = FEATURES_DIR / status_name
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(status_path, index=False)
    complete = int(df["complete"].sum()) if not df.empty and "complete" in df else 0
    print(f"DeepPrivacy2 probe status: {complete}/{target_count} complete")
    print(f"Status CSV: {status_path}")


def main() -> int:
    args = parse_args()
    filenames = target_filenames(args)
    stems = [filename_to_stem(filename) for filename in filenames]

    missing_frames = [stem for stem in stems if not frame_extraction_complete(stem)]
    complete_before = [stem for stem in stems if deepprivacy2_complete(stem)]
    pending = [stem for stem in stems if stem not in set(complete_before)]

    print(f"Method       : {METHOD_NAME}")
    print(f"Shard        : {args.shard_index + 1}/{args.num_shards}")
    print(f"Targets      : {len(stems)} videos")
    print(f"Complete     : {len(complete_before)} videos")
    print(f"Pending      : {len(pending)} videos")
    print(f"Missing frame extraction: {len(missing_frames)} videos")
    print(f"Output root  : {ANON_DIR / METHOD_NAME}")

    if missing_frames:
        print("First missing-frame stems:", missing_frames[:10])
        return 1

    if args.dry_run:
        rows = [
            {
                "stem": stem,
                "expected_frames": expected_frame_count(stem),
                "written_before": count_jpgs(output_dir_for_stem(stem)),
                "written_after": count_jpgs(output_dir_for_stem(stem)),
                "failed_frames": 0,
                "complete": deepprivacy2_complete(stem),
            }
            for stem in stems
        ]
        write_status(rows, len(stems), args)
        return 0

    if not pending and not args.overwrite:
        print("Nothing to anonymize.")
        rows = [
            {
                "stem": stem,
                "expected_frames": expected_frame_count(stem),
                "written_before": count_jpgs(output_dir_for_stem(stem)),
                "written_after": count_jpgs(output_dir_for_stem(stem)),
                "failed_frames": 0,
                "complete": deepprivacy2_complete(stem),
            }
            for stem in stems
        ]
        write_status(rows, len(stems), args)
        return 0

    dp2_utils, anonymizer, synthesis_kwargs = load_deepprivacy2(args)

    start_time = time.time()
    rows = []
    process_stems = stems if args.overwrite else pending
    for stem in tqdm(process_stems, desc="DeepPrivacy2 probes"):
        rows.append(anonymize_stem(stem, dp2_utils, anonymizer, synthesis_kwargs, args))

    if not args.overwrite:
        for stem in complete_before:
            rows.append(
                {
                    "stem": stem,
                    "expected_frames": expected_frame_count(stem),
                    "written_before": count_jpgs(output_dir_for_stem(stem)),
                    "written_after": count_jpgs(output_dir_for_stem(stem)),
                    "failed_frames": 0,
                    "complete": True,
                }
            )

    write_status(rows, len(stems), args)
    free_gpu()
    elapsed = time.time() - start_time
    print(f"DeepPrivacy2 probe anonymization finished in {elapsed / 60:.1f} min")
    return 0 if all(bool(row["complete"]) for row in rows) and len(rows) == len(stems) else 2


if __name__ == "__main__":
    raise SystemExit(main())
