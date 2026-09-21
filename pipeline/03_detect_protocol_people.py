#!/usr/bin/env python3
"""Detect people for selected protocol gallery/probe frames.

This array-safe stage emits the same detection JSON schema as stage 03. Face
boxes are intentionally left empty: ArcFace performs its own face detection,
while OSNet and the gait pipeline only require person boxes.
"""

from __future__ import annotations

import argparse

import cv2
from tqdm.auto import tqdm

from pipeline_common import (
    DETECT_DIR,
    DETECTION_BATCH_SIZE,
    DEVICE,
    EVAL_PROTOCOLS,
    FRAMES_DIR,
    YOLO_MODEL,
    detection_complete,
    frame_extraction_complete,
    meta_by_filename,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DETECTION_BATCH_SIZE)
    parser.add_argument("--model", default=YOLO_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def protocol_names(raw: str) -> list[str]:
    if raw.strip().lower() == "auto":
        return list(EVAL_PROTOCOLS)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in EVAL_PROTOCOLS]
    if unknown:
        raise KeyError(f"Unknown protocols {unknown}; known: {sorted(EVAL_PROTOCOLS)}")
    return names


def selected_stems(args: argparse.Namespace) -> list[str]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    filenames: list[str] = []
    for protocol in protocol_names(args.protocols):
        gallery, probe = EVAL_PROTOCOLS[protocol]
        filenames.extend(gallery)
        filenames.extend(probe)
    unique = sorted(dict.fromkeys(filenames))
    if args.limit > 0:
        unique = unique[: args.limit]
    unique = unique[args.shard_index :: args.num_shards]
    return [str(meta_by_filename[name]["stem"]) for name in unique]


def detect_stem(model, stem: str, batch_size: int) -> None:
    frame_paths = sorted((FRAMES_DIR / stem).glob("*.jpg"))
    results: dict[str, dict[str, object]] = {}
    for batch_start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[batch_start : batch_start + batch_size]
        loaded = [(path, cv2.imread(str(path))) for path in batch_paths]
        loaded = [(path, image) for path, image in loaded if image is not None]
        if not loaded:
            continue
        paths, images = zip(*loaded)
        predictions = model.predict(
            list(images), classes=[0], conf=0.3, verbose=False, device=DEVICE
        )
        for frame_path, prediction in zip(paths, predictions):
            persons = []
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                    confidence = float(box.conf[0].detach().cpu())
                    persons.append(
                        [round(value, 1) for value in xyxy] + [round(confidence, 3)]
                    )
            results[frame_path.stem] = {"persons": persons, "faces": []}
    write_json(DETECT_DIR / f"{stem}.json", results)


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    stems = selected_stems(args)
    pending = [
        stem
        for stem in stems
        if frame_extraction_complete(stem)
        and (args.overwrite or not detection_complete(stem))
    ]
    missing_frames = [stem for stem in stems if not frame_extraction_complete(stem)]
    print(f"Shard          : {args.shard_index + 1}/{args.num_shards}")
    print(f"Selected videos: {len(stems)}")
    print(f"Pending        : {len(pending)}")
    print(f"Missing frames : {len(missing_frames)}")
    print(f"Model          : {args.model}")
    if missing_frames:
        print("First missing  :", missing_frames[:10])
        return 1
    if not pending:
        return 0

    model = YOLO(args.model)
    model.to(DEVICE)
    for stem in tqdm(pending, desc=f"person detection shard {args.shard_index}"):
        detect_stem(model, stem, args.batch_size)

    complete = sum(detection_complete(stem) for stem in stems)
    print(f"Complete after : {complete}/{len(stems)}")
    return 0 if complete == len(stems) else 2


if __name__ == "__main__":
    raise SystemExit(main())

