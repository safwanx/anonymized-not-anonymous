#!/usr/bin/env python3
"""Run evaluation and summary stages after anonymization is complete."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


IDENTITY_STAGES = [
    ("05_eval_face_recognition.py", "face_recognition"),
    ("06_eval_person_reid.py", "person_reid"),
]

REMAINING_STAGES = [
    "07_eval_pose_identity.py",
    "08_eval_silhouette_proxy.py",
    "09_eval_action_utility.py",
    "10_write_summary.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-mode",
        choices=["parallel", "sequential"],
        default=os.environ.get("IDENTITY_MODE", "parallel"),
        help="Run face recognition and person re-ID together or one after the other.",
    )
    parser.add_argument(
        "--identity-gpus",
        default=os.environ.get("IDENTITY_GPUS", "0,1"),
        help="Comma-separated physical GPU ids for parallel identity evals. Default: 0,1.",
    )
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="Stop after ArcFace and OSNet; skip legacy pose/silhouette/action proxy stages.",
    )
    return parser.parse_args()


def stage_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def check_required_imports() -> int:
    missing = []
    for module_name in ["gdown"]:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if missing:
        print("Missing Python package(s) required by later stages:", ", ".join(missing), flush=True)
        print("Install them in the active venv, then rerun:", flush=True)
        print(f"{sys.executable} -m pip install {' '.join(missing)}", flush=True)
        return 1
    return 0


def run_stage(root: Path, stage: str, env_extra: dict[str, str] | None = None) -> int:
    gpu = f" on GPU {env_extra['CUDA_VISIBLE_DEVICES']}" if env_extra and "CUDA_VISIBLE_DEVICES" in env_extra else ""
    print("=" * 80, flush=True)
    print(f"Running {stage}{gpu}", flush=True)
    print("=" * 80, flush=True)
    result = subprocess.run([sys.executable, str(root / stage)], cwd=root, env=stage_env(env_extra))
    if result.returncode != 0:
        print(f"{stage} failed with exit code {result.returncode}", flush=True)
    return result.returncode


def run_identity_sequential(root: Path) -> int:
    for stage, _name in IDENTITY_STAGES:
        code = run_stage(root, stage)
        if code != 0:
            return code
    return 0


def run_identity_parallel(root: Path, gpu_ids: list[str]) -> int:
    print("=" * 80, flush=True)
    print("Running identity evaluations in parallel", flush=True)
    print("=" * 80, flush=True)

    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    for index, (stage, name) in enumerate(IDENTITY_STAGES):
        env_extra: dict[str, str] = {}
        if index < len(gpu_ids):
            env_extra["CUDA_VISIBLE_DEVICES"] = gpu_ids[index]
            gpu_label = gpu_ids[index]
        else:
            gpu_label = "default"

        print(f"Starting {stage} ({name}) on GPU {gpu_label}", flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(root / stage)],
            cwd=root,
            env=stage_env(env_extra),
        )
        procs.append((stage, proc))

    failures: list[tuple[str, int]] = []
    for stage, proc in procs:
        code = proc.wait()
        if code != 0:
            failures.append((stage, code))

    if failures:
        for stage, code in failures:
            print(f"{stage} failed with exit code {code}", flush=True)
        return failures[0][1]

    print("Parallel identity evaluations complete.", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent

    code = check_required_imports()
    if code != 0:
        return code

    gpu_ids = [gpu.strip() for gpu in args.identity_gpus.split(",") if gpu.strip()]
    if args.identity_mode == "parallel":
        if len(gpu_ids) < 2:
            print(
                "WARNING: fewer than two identity GPUs configured; "
                "parallel face/re-ID evals may compete for the same GPU.",
                flush=True,
            )
        code = run_identity_parallel(root, gpu_ids)
    else:
        code = run_identity_sequential(root)

    if code != 0:
        return code

    if args.identity_only:
        print("Identity-only run complete; skipped legacy proxy stages.", flush=True)
        return 0

    for stage in REMAINING_STAGES:
        code = run_stage(root, stage)
        if code != 0:
            return code

    print("Remaining stages complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
