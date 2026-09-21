#!/usr/bin/env python3
"""Verify and summarize DeepPrivacy2 outputs without loading ML frameworks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--probe-split", default="probe_crossview_cam2.txt")
    parser.add_argument("--method", default="deepprivacy2")
    return parser.parse_args()


def count_jpgs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.jpg")) if path.is_dir() else 0


def main() -> int:
    args = parse_args()
    output_root = Path(os.environ.get("PILOT_OUTPUT_ROOT", args.root / "outputs"))
    split_path = Path(os.environ.get("PILOT_ROOT", args.root / "protocol")) / "splits" / args.probe_split
    manifests = output_root / "frames" / "_manifests"
    anonymized = output_root / "anonymized" / args.method
    features = output_root / "features"
    features.mkdir(parents=True, exist_ok=True)

    filenames = sorted(
        line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    rows: list[dict[str, object]] = []
    for filename in filenames:
        stem = filename.removesuffix("_rgb.avi")
        manifest_path = manifests / f"{stem}.json"
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = int(manifest.get("extracted_frames", 0))
        stride = int(manifest.get("stride", 2))
        video_path = Path(str(manifest.get("video", "")))
        declared_raw = 0
        if video_path.is_file():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                declared_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        container_expected = (
            (declared_raw + stride - 1) // stride if declared_raw > 0 and stride > 0 else expected
        )
        original_count = count_jpgs(output_root / "frames" / stem)
        anonymized_count = count_jpgs(anonymized / stem)
        rows.append(
            {
                "filename": filename,
                "stem": stem,
                "expected_frames": expected,
                "container_raw_frames": declared_raw,
                "container_expected_frames": container_expected,
                "original_frames": original_count,
                "anonymized_frames": anonymized_count,
                "original_complete": container_expected > 0 and original_count == container_expected,
                "complete": container_expected > 0 and anonymized_count == container_expected,
            }
        )

    status_path = features / f"{args.method}_probe_status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    original_complete = sum(bool(row["original_complete"]) for row in rows)
    anonymized_complete = sum(bool(row["complete"]) for row in rows)
    report = {
        "method": args.method,
        "probe_split": args.probe_split,
        "target_videos": len(rows),
        "original_complete": original_complete,
        "anonymized_complete": anonymized_complete,
        "status_csv": str(status_path),
    }
    report_path = features / f"{args.method}_probe_verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if anonymized_complete == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
