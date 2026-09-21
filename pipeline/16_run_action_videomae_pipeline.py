#!/usr/bin/env python3
"""Run the VideoMAE action-utility pipeline in order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    parser.add_argument("--extract-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
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

    if not args.eval_only:
        code = run(root, "12_prepare_action_videomae_splits.py", [])
        if code != 0:
            return code

    if not args.skip_extract and not args.eval_only:
        extract_args = [
            "--methods",
            args.methods,
            "--batch-size",
            str(args.batch_size),
            "--num-frames",
            str(args.num_frames),
            "--precision",
            args.precision,
        ]
        if args.extract_limit > 0:
            extract_args.extend(["--limit", str(args.extract_limit)])
        code = run(root, "13_extract_videomae_features.py", extract_args)
        if code != 0:
            return code

    if not args.skip_train and not args.eval_only:
        code = run(
            root,
            "14_train_action_videomae_head.py",
            [
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.train_batch_size),
                "--hidden-dim",
                str(args.hidden_dim),
            ],
        )
        if code != 0:
            return code

    code = run(root, "15_eval_action_videomae_head.py", ["--methods", args.methods])
    if code != 0:
        return code

    print("VideoMAE action pipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
