#!/usr/bin/env python3
"""Run RGB-pose keypoint extraction, identity training, and identity evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="auto")
    parser.add_argument("--protocols", default="auto")
    parser.add_argument("--pose-model", default="yolo11m-pose.pt")
    parser.add_argument("--sample-frames", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--keypoint-conf", type=float, default=0.15)
    parser.add_argument("--min-valid-frames", type=int, default=4)
    parser.add_argument("--extract-limit", type=int, default=0)
    parser.add_argument(
        "--overwrite-extract",
        action="store_true",
        help="Rebuild existing RGB-pose feature files. Use after changing sample frame count or thresholds.",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def run(root: Path, stage: str, args: list[str]) -> int:
    print("=" * 80, flush=True)
    print("Running", stage, " ".join(args), flush=True)
    print("=" * 80, flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run([sys.executable, str(root / stage), *args], cwd=root, env=env)
    if result.returncode != 0:
        print(f"{stage} failed with exit code {result.returncode}", flush=True)
    return result.returncode


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent

    if not args.skip_extract:
        extract_args = [
            "--methods",
            args.methods,
            "--protocols",
            args.protocols,
            "--pose-model",
            args.pose_model,
            "--sample-frames",
            str(args.sample_frames),
            "--batch-size",
            str(args.batch_size),
            "--conf",
            str(args.conf),
            "--keypoint-conf",
            str(args.keypoint_conf),
            "--min-valid-frames",
            str(args.min_valid_frames),
        ]
        if args.extract_limit > 0:
            extract_args.extend(["--limit", str(args.extract_limit)])
        if args.overwrite_extract:
            extract_args.append("--overwrite")
        code = run(root, "17_extract_rgb_pose_keypoints.py", extract_args)
        if code != 0:
            return code

    if not args.skip_train:
        code = run(root, "18_train_rgb_pose_identity.py", ["--protocols", args.protocols])
        if code != 0:
            return code

    code = run(root, "19_eval_rgb_pose_identity.py", ["--methods", args.methods, "--protocols", args.protocols])
    if code != 0:
        return code

    print("RGB pose identity pipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
