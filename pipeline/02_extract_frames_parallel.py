#!/usr/bin/env python3
"""Parallel drop-in for 02_extract_frames.py.

Same output as the serial version (frames/<stem>/<NNNNNN>.jpg + manifests under
frames/_manifests/<stem>.json), but each video is decoded in its own worker
process, turning the ~5h single-thread run into well under an hour on a multicore
PC. Resumable: videos with a complete manifest are skipped.

Design note: pipeline_common is imported only inside main() (the parent), so the
spawned workers do NOT re-run its heavy module-level init. Workers are
self-contained (cv2 + stdlib only).

Output location follows the pipeline config:
    frames land in $PILOT_OUTPUT_ROOT/frames  (falls back to $PILOT_ROOT/frames).
To put runtime files on a separate scratch volume:
    export PILOT_OUTPUT_ROOT=/path/to/scratch/accv_outputs

Run:
    python 02_extract_frames_parallel.py --workers 12
    python 02_extract_frames_parallel.py --workers 12 --limit 480   # subset smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2


def extract_one(task: tuple[str, str, str, str, int, int]) -> tuple[str, str, int, int]:
    """Worker: decode one video, write strided JPEGs + manifest. No heavy imports."""
    stem, video_path, out_dir, manifest_path, stride, quality = task
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(
            {"status": "failed", "reason": "VideoCapture open failed", "video": video_path}))
        return (stem, "failed", 0, 0)

    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    raw = extracted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw % stride == 0:
            if cv2.imwrite(str(out / f"{raw:06d}.jpg"), frame, params):
                extracted += 1
        raw += 1
    cap.release()

    status = "complete" if extracted > 0 else "failed"
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps({
        "status": status, "video": video_path, "raw_frames": raw,
        "extracted_frames": extracted, "stride": stride, "jpeg_quality": quality,
    }))
    return (stem, status, raw, extracted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N pending videos (0 = all).")
    args = ap.parse_args()

    import pipeline_common as cfg  # heavy init in parent only

    def cheap_done(stem: str) -> bool:
        """Completeness via manifest only - never opens a video (the serial
        check does, which makes todo-building over 21,600 videos take minutes)."""
        mpath = cfg.frame_manifest_path(stem)
        if not mpath.exists():
            return False
        try:
            m = json.loads(mpath.read_text())
        except Exception:
            return False
        if m.get("status") != "complete" or int(m.get("stride", -1)) != cfg.FRAME_STRIDE:
            return False
        expected = int(m.get("extracted_frames", -1))
        return expected > 0 and cfg.count_jpgs(cfg.FRAMES_DIR / stem) == expected

    print("Scanning for pending videos...", flush=True)
    todo = [s for s in cfg.all_stems if not cheap_done(s)]
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"Frames -> {cfg.FRAMES_DIR}", flush=True)
    print(f"Pending: {len(todo)} / {len(cfg.all_stems)}  workers: {args.workers}  "
          f"stride: {cfg.FRAME_STRIDE}  jpegq: {cfg.FRAME_JPEG_QUALITY}", flush=True)
    if not todo:
        print("Nothing to do.")
        return 0

    tasks = [
        (
            stem,
            str(cfg.rgb_paths[stem]),
            str(cfg.FRAMES_DIR / stem),
            str(cfg.frame_manifest_path(stem)),
            cfg.FRAME_STRIDE,
            cfg.FRAME_JPEG_QUALITY,
        )
        for stem in todo
    ]

    t0 = time.time()
    done = failed = total_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            stem, status, _raw, extracted = fut.result()
            done += 1
            total_frames += extracted
            if status != "complete":
                failed += 1
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(tasks) - done) / rate if rate else 0
                print(f"  [{done}/{len(tasks)}] {total_frames:,} frames | "
                      f"{rate:.1f} vid/s | ETA {eta/60:.1f} min | failed {failed}")

    elapsed = time.time() - t0
    print(f"\nDone: {done} videos in {elapsed/60:.1f} min, {total_frames:,} frames, {failed} failed.")
    if failed:
        print("Re-run to retry failed videos (they have no complete manifest).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
