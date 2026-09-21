#!/usr/bin/env python3
"""Extract RGB frames for selected evaluation protocols only.

The original stage-02 script covers the complete 21,600-video pilot.  Privacy
evaluation usually needs a smaller, precisely defined gallery/probe subset, so
this companion stage makes that selection explicit while preserving the same
frame layout and completion-manifest format.
"""

from __future__ import annotations

import argparse

import cv2
from tqdm.auto import tqdm

from pipeline_common import (
    EVAL_PROTOCOLS,
    FRAME_JPEG_QUALITY,
    FRAME_STRIDE,
    FRAMES_DIR,
    count_jpgs,
    frame_extraction_complete,
    frame_manifest_path,
    meta_by_filename,
    rgb_paths,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols",
        default="auto",
        help="Comma-separated protocol names. Auto uses the active EVAL_PROTOCOLS.",
    )
    parser.add_argument(
        "--role",
        choices=("gallery", "probe", "both"),
        default="both",
        help="Which side of each selected evaluation protocol to extract.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug limit after deterministic sorting. 0 means all selected videos.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=FRAME_JPEG_QUALITY)
    return parser.parse_args()


def selected_protocols(raw: str) -> list[str]:
    if raw.strip().lower() == "auto":
        return list(EVAL_PROTOCOLS)
    protocols = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in protocols if name not in EVAL_PROTOCOLS]
    if unknown:
        raise KeyError(f"Unknown protocols {unknown}; known: {sorted(EVAL_PROTOCOLS)}")
    return protocols


def selected_stems(args: argparse.Namespace) -> list[str]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    filenames: list[str] = []
    for protocol in selected_protocols(args.protocols):
        gallery, probe = EVAL_PROTOCOLS[protocol]
        if args.role in {"gallery", "both"}:
            filenames.extend(gallery)
        if args.role in {"probe", "both"}:
            filenames.extend(probe)
    unique = sorted(dict.fromkeys(filenames))
    if args.limit > 0:
        unique = unique[: args.limit]
    unique = unique[args.shard_index :: args.num_shards]
    return [str(meta_by_filename[filename]["stem"]) for filename in unique]


def extract_stem(stem: str, jpeg_quality: int) -> dict[str, object]:
    video_path = rgb_paths[stem]
    out_dir = FRAMES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_frame in out_dir.glob("*.jpg"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        row = {
            "status": "failed",
            "reason": "VideoCapture open failed",
            "video": str(video_path),
        }
        write_json(frame_manifest_path(stem), row)
        return row

    declared_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected_extracted = (
        (declared_raw_frames + FRAME_STRIDE - 1) // FRAME_STRIDE
        if declared_raw_frames > 0
        else 0
    )
    raw_frames = 0
    extracted = 0
    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_frames % FRAME_STRIDE == 0:
            out_path = out_dir / f"{raw_frames:06d}.jpg"
            if cv2.imwrite(str(out_path), frame, jpg_params):
                extracted += 1
        raw_frames += 1
    cap.release()

    complete = extracted > 0 and (expected_extracted == 0 or extracted == expected_extracted)
    row = {
        "status": "complete" if complete else "failed",
        "video": str(video_path),
        "raw_frames": raw_frames,
        "declared_raw_frames": declared_raw_frames,
        "expected_extracted_frames": expected_extracted,
        "extracted_frames": extracted,
        "stride": FRAME_STRIDE,
        "jpeg_quality": int(jpeg_quality),
    }
    write_json(frame_manifest_path(stem), row)
    return row


def main() -> int:
    args = parse_args()
    stems = selected_stems(args)
    complete_before = [stem for stem in stems if frame_extraction_complete(stem)]
    pending = [stem for stem in stems if stem not in set(complete_before)]

    print(f"Selected videos : {len(stems)}")
    print(f"Shard           : {args.shard_index + 1}/{args.num_shards}")
    print(f"Complete        : {len(complete_before)}")
    print(f"Pending         : {len(pending)}")
    print(f"Frame root      : {FRAMES_DIR}")
    if args.dry_run:
        return 0

    failures: list[str] = []
    for stem in tqdm(pending, desc="Extracting selected frames"):
        row = extract_stem(stem, args.jpeg_quality)
        if row.get("status") != "complete":
            failures.append(stem)

    complete_after = sum(1 for stem in stems if frame_extraction_complete(stem))
    frame_total = sum(count_jpgs(FRAMES_DIR / stem) for stem in stems)
    print(f"Complete after  : {complete_after}/{len(stems)}")
    print(f"Extracted frames: {frame_total}")
    if failures:
        print("First failures  :", failures[:10])
    return 0 if complete_after == len(stems) else 2


if __name__ == "__main__":
    raise SystemExit(main())
