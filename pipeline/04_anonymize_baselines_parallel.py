#!/usr/bin/env python3
"""Parallel simple anonymization baselines for pilot frames.

This parallelizes the non-DeepPrivacy2 baselines from 04_anonymize_baselines.py:
face_blur, body_blur, body_mask, and body_pixel. Each video/method output is
independent, so process-level parallelism is safe and resumable.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os


def chunk_round_robin(items: list[tuple[str, str]], num_chunks: int) -> list[list[tuple[str, str]]]:
    chunks = [[] for _ in range(num_chunks)]
    for index, item in enumerate(items):
        chunks[index % num_chunks].append(item)
    return chunks


def worker_main(worker_id: int, tasks: list[tuple[str, str]]) -> None:
    import cv2

    from pipeline_common import (
        ANON_DIR,
        ANON_JPEG_QUALITY,
        DETECT_DIR,
        FRAMES_DIR,
        METHODS,
        anonymization_complete,
        detection_complete,
        read_json,
    )

    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, ANON_JPEG_QUALITY]
    completed = 0
    skipped = 0

    print(f"[worker {worker_id}] tasks={len(tasks)} pid={os.getpid()}")
    for task_index, (method_name, stem) in enumerate(tasks, start=1):
        if not detection_complete(stem):
            skipped += 1
            continue
        if anonymization_complete(method_name, stem):
            skipped += 1
            continue

        method_fn = METHODS[method_name]
        detections = read_json(DETECT_DIR / f"{stem}.json", {})
        out_dir = ANON_DIR / method_name / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in sorted((FRAMES_DIR / stem).glob("*.jpg")):
            out_path = out_dir / frame_path.name
            if out_path.exists():
                continue
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            det = detections.get(frame_path.stem, {"persons": [], "faces": []})
            anon_image = method_fn(image, det)
            cv2.imwrite(str(out_path), anon_image, jpg_params)

        completed += 1
        if task_index % 200 == 0:
            print(f"[worker {worker_id}] progress {task_index}/{len(tasks)} completed={completed} skipped={skipped}")

    print(f"[worker {worker_id}] done completed={completed} skipped={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple anonymization baselines in parallel.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ANON_WORKERS", "8")),
        help="Number of CPU worker processes. Default: ANON_WORKERS or 8.",
    )
    parser.add_argument(
        "--methods",
        default=os.environ.get("ANON_METHODS", "face_blur,body_blur,body_mask,body_pixel"),
        help="Comma-separated methods to run.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of video-method tasks for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    from pipeline_common import METHODS, all_stems, anonymization_complete, detection_complete

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown anonymization methods: {unknown}. Available: {sorted(METHODS)}")

    tasks = [
        (method, stem)
        for method in methods
        for stem in all_stems
        if detection_complete(stem) and not anonymization_complete(method, stem)
    ]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"Methods: {methods}")
    print(f"Pending video-method tasks: {len(tasks)}")
    print(f"Workers: {args.workers}")
    if not tasks:
        print("Nothing to anonymize.")
        return 0

    chunks = chunk_round_robin(tasks, args.workers)
    ctx = mp.get_context("spawn")
    processes = []
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        process = ctx.Process(target=worker_main, args=(worker_id, chunk), name=f"anon-worker-{worker_id}")
        process.start()
        processes.append(process)

    exit_code = 0
    for process in processes:
        process.join()
        if process.exitcode != 0:
            print(f"{process.name} failed with exit code {process.exitcode}")
            exit_code = process.exitcode or 1
    return exit_code


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
