#!/usr/bin/env python3
"""Extract frames for selected VideoMAE train/validation/test splits."""

from __future__ import annotations

import argparse

import cv2
from tqdm.auto import tqdm

from action_videomae_common import load_action_split, parse_csv_arg
from pipeline_common import (
    FRAME_JPEG_QUALITY,
    FRAME_STRIDE,
    FRAMES_DIR,
    count_jpgs,
    frame_extraction_complete,
    frame_manifest_path,
    rgb_paths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, help="Comma-separated action split names.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=FRAME_JPEG_QUALITY)
    return parser.parse_args()


def selected_stems(args: argparse.Namespace) -> list[str]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    stems: list[str] = []
    for split_name in parse_csv_arg(args.splits, []):
        stems.extend(str(row["stem"]) for row in load_action_split(split_name))
    unique = sorted(dict.fromkeys(stems))
    if args.limit > 0:
        unique = unique[: args.limit]
    return unique[args.shard_index :: args.num_shards]


def extract_stem(stem: str, jpeg_quality: int) -> bool:
    video_path = rgb_paths[stem]
    out_dir = FRAMES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("*.jpg"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        write_json(
            frame_manifest_path(stem),
            {"status": "failed", "reason": "VideoCapture open failed", "video": str(video_path)},
        )
        return False

    declared_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected_extracted = (
        (declared_raw_frames + FRAME_STRIDE - 1) // FRAME_STRIDE
        if declared_raw_frames > 0
        else 0
    )
    raw_frames = 0
    extracted = 0
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_frames % FRAME_STRIDE == 0:
            if cv2.imwrite(str(out_dir / f"{raw_frames:06d}.jpg"), frame, params):
                extracted += 1
        raw_frames += 1
    cap.release()
    complete = extracted > 0 and (expected_extracted == 0 or extracted == expected_extracted)
    write_json(
        frame_manifest_path(stem),
        {
            "status": "complete" if complete else "failed",
            "video": str(video_path),
            "raw_frames": raw_frames,
            "declared_raw_frames": declared_raw_frames,
            "expected_extracted_frames": expected_extracted,
            "extracted_frames": extracted,
            "stride": FRAME_STRIDE,
            "jpeg_quality": int(jpeg_quality),
        },
    )
    return complete


def main() -> int:
    args = parse_args()
    stems = selected_stems(args)
    pending = [stem for stem in stems if not frame_extraction_complete(stem)]
    print(f"Shard          : {args.shard_index + 1}/{args.num_shards}")
    print(f"Selected videos: {len(stems)}")
    print(f"Pending        : {len(pending)}")
    failures = []
    for stem in tqdm(pending, desc=f"action frames shard {args.shard_index}"):
        if not extract_stem(stem, args.jpeg_quality):
            failures.append(stem)
    complete = sum(frame_extraction_complete(stem) for stem in stems)
    total_frames = sum(count_jpgs(FRAMES_DIR / stem) for stem in stems)
    print(f"Complete after : {complete}/{len(stems)}")
    print(f"Frames         : {total_frames}")
    if failures:
        print("First failures :", failures[:10])
    return 0 if complete == len(stems) else 2


if __name__ == "__main__":
    raise SystemExit(main())
