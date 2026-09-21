#!/usr/bin/env python3
"""Parallel people/face detection for the pilot frames.

Use one worker process per GPU. Each process loads its own YOLO and InsightFace
models after CUDA_VISIBLE_DEVICES is set, then processes a disjoint chunk of
videos. This avoids sharing GPU model objects across workers.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("At least one GPU id is required")
    return gpus


def chunk_round_robin(items: list[str], num_chunks: int) -> list[list[str]]:
    chunks = [[] for _ in range(num_chunks)]
    for index, item in enumerate(items):
        chunks[index % num_chunks].append(item)
    return chunks


def expand_gpu_workers(gpus: list[str], workers_per_gpu: int) -> list[str]:
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    expanded = []
    for gpu in gpus:
        expanded.extend([gpu] * workers_per_gpu)
    return expanded


def worker_main(worker_id: int, gpu_id: str, stems: list[str]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    import cv2
    from insightface.app import FaceAnalysis
    from ultralytics import YOLO

    from pipeline_common import (
        DETECT_DIR,
        DETECTION_BATCH_SIZE,
        DEVICE,
        FRAMES_DIR,
        LOGS_DIR,
        YOLO_MODEL,
        append_jsonl,
        detection_complete,
        ensure_insightface_root,
        frame_extraction_complete,
        write_json,
    )

    print(f"[worker {worker_id}] gpu={gpu_id} visible_device={os.environ.get('CUDA_VISIBLE_DEVICES')} device={DEVICE}")
    print(f"[worker {worker_id}] loading YOLO: {YOLO_MODEL}")
    yolo = YOLO(YOLO_MODEL)
    yolo.to(DEVICE)

    print(f"[worker {worker_id}] loading InsightFace buffalo_l (detection only)")
    face_app = FaceAnalysis(
        name="buffalo_l",
        root=ensure_insightface_root("buffalo_l"),
        allowed_modules=["detection"],  # we only need face boxes; skip recog/genderage/landmark
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
    print(f"[worker {worker_id}] models loaded; stems={len(stems)}")

    completed = 0
    skipped = 0
    for stem_index, stem in enumerate(stems, start=1):
        if not frame_extraction_complete(stem):
            skipped += 1
            continue
        if detection_complete(stem):
            skipped += 1
            continue

        frames_dir = FRAMES_DIR / stem
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            skipped += 1
            continue

        results = {}
        for batch_start in range(0, len(frame_paths), DETECTION_BATCH_SIZE):
            batch_paths = frame_paths[batch_start : batch_start + DETECTION_BATCH_SIZE]
            loaded = [(path, cv2.imread(str(path))) for path in batch_paths]
            loaded = [(path, image) for path, image in loaded if image is not None]
            if not loaded:
                continue

            paths, images = zip(*loaded)
            yolo_out = yolo.predict(
                list(images),
                classes=[0],
                conf=0.3,
                verbose=False,
                device=DEVICE,
            )

            for frame_path, image, ydet in zip(paths, images, yolo_out):
                persons = []
                if ydet.boxes is not None:
                    for box in ydet.boxes:
                        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                        conf = float(box.conf[0].detach().cpu())
                        persons.append([round(value, 1) for value in xyxy] + [round(conf, 3)])

                faces = []
                try:
                    for face in face_app.get(image):
                        bbox = face.bbox.tolist()
                        faces.append([round(value, 1) for value in bbox] + [round(float(face.det_score), 3)])
                except Exception as exc:
                    append_jsonl(
                        LOGS_DIR / "face_detection_errors.jsonl",
                        {
                            "worker": worker_id,
                            "gpu": gpu_id,
                            "stem": stem,
                            "frame": frame_path.name,
                            "error": str(exc),
                        },
                    )

                results[frame_path.stem] = {"persons": persons, "faces": faces}

        write_json(DETECT_DIR / f"{stem}.json", results)
        completed += 1

        if stem_index % 100 == 0:
            print(f"[worker {worker_id}] progress {stem_index}/{len(stems)} completed={completed} skipped={skipped}")

    print(f"[worker {worker_id}] done completed={completed} skipped={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection with one worker process per GPU.")
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=parse_gpus(os.environ.get("DETECTION_GPUS", "0,1")),
        help="Comma-separated physical GPU ids, e.g. 0,1. Default: DETECTION_GPUS or 0,1.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=int(os.environ.get("DETECTION_WORKERS_PER_GPU", "1")),
        help="Number of independent detector processes per physical GPU. Try 2 when GPU utilization is low.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of pending videos to process, for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Import common only in main after argument parsing; worker processes import it
    # after setting CUDA_VISIBLE_DEVICES.
    from pipeline_common import all_stems, detection_complete, frame_extraction_complete

    todo = [
        stem
        for stem in all_stems
        if frame_extraction_complete(stem) and not detection_complete(stem)
    ]
    if args.limit > 0:
        todo = todo[: args.limit]

    worker_gpus = expand_gpu_workers(args.gpus, args.workers_per_gpu)
    print(f"Detection pending videos: {len(todo)}")
    print(f"Physical GPUs: {args.gpus}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Worker GPU assignment: {worker_gpus}")
    if not todo:
        print("Nothing to detect.")
        return 0

    chunks = chunk_round_robin(todo, len(worker_gpus))
    ctx = mp.get_context("spawn")
    processes = []
    for worker_id, (gpu_id, stems) in enumerate(zip(worker_gpus, chunks)):
        if not stems:
            continue
        process = ctx.Process(target=worker_main, args=(worker_id, gpu_id, stems), name=f"detect-gpu{gpu_id}")
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
