#!/usr/bin/env python3
"""Extract frozen VideoMAEv2 features from original and anonymized frame folders.

The script writes one feature file per video, so it can be stopped and resumed.
It also writes compressed per-method/per-split aggregate files used by the
training and evaluation scripts.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from action_videomae_common import (
    ACTION_LOGS_DIR,
    DEFAULT_ANON_SPLITS,
    DEFAULT_ORIGINAL_SPLITS,
    aggregate_path,
    build_feature_matrix,
    expected_splits_for_method,
    frame_dir_for_method,
    load_action_split,
    method_video_ready,
    parse_csv_arg,
    resolve_methods,
    summarize_feature_status,
    video_feature_path,
)
from pipeline_common import DEVICE, RANDOM_SEED, append_jsonl


DEFAULT_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--methods",
        default="auto",
        help="Comma-separated methods. Use auto for original + complete anonymized methods.",
    )
    parser.add_argument(
        "--splits",
        default="auto",
        help="Comma-separated action splits. Auto uses train/val/test for original and test splits for anonymized methods.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per method/split; 0 means all.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Re-extract existing per-video feature files.")
    parser.add_argument("--rebuild-aggregates", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="CPU-only: rebuild aggregates from cached per-video .npy features without loading VideoMAE.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16"],
        default="auto",
        help="Model/input precision. Auto uses fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Per-video feature dtype on disk.",
    )
    return parser.parse_args()


def check_dependencies() -> None:
    missing = []
    for module_name in ["transformers", "timm", "safetensors", "easydict"]:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing package(s): "
            + ", ".join(missing)
            + "\nInstall them in the active venv:\n"
            + "python -m pip install transformers timm safetensors accelerate easydict"
        )


def torch_dtype_for(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.float16 if DEVICE == "cuda" else torch.float32


def load_model(model_id: str, precision: str):
    check_dependencies()
    try:
        from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
    except Exception as exc:
        raise SystemExit(
            "Could not import Hugging Face VideoMAE components. This usually means "
            "the installed transformers build is incompatible with the current Torch build.\n"
            "Known-good fix for this project:\n"
            "python -m pip install --upgrade \"transformers==4.46.3\" "
            "\"tokenizers>=0.20,<0.21\" \"huggingface-hub>=0.23,<1.0\" "
            "timm safetensors accelerate\n"
            f"Original import error: {exc}"
        ) from exc

    dtype = torch_dtype_for(precision)
    print(f"Loading {model_id}")
    print("Using trust_remote_code=True because VideoMAEv2-Base ships custom model code.")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    processor = VideoMAEImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)
    model.eval()
    model.to(device=DEVICE, dtype=dtype)
    return processor, model, dtype


def sample_video_frames(frame_dir: Path, num_frames: int) -> list[np.ndarray] | None:
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        return None

    indices = np.linspace(0, len(frame_paths) - 1, num=num_frames)
    selected = [frame_paths[int(round(index))] for index in indices]

    frames = []
    for frame_path in selected:
        image = cv2.imread(str(frame_path))
        if image is None:
            return None
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


def prepare_pixel_values(processor, videos: list[list[np.ndarray]], dtype: torch.dtype) -> torch.Tensor:
    inputs = processor(videos, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim != 5:
        raise RuntimeError(f"Expected 5D VideoMAE pixel_values, got shape {tuple(pixel_values.shape)}")
    # VideoMAEv2 custom code expects B,C,T,H,W. HF processors commonly emit B,T,C,H,W.
    if pixel_values.shape[1] != 3 and pixel_values.shape[2] == 3:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
    return pixel_values.to(device=DEVICE, dtype=dtype)


def tensor_to_features(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        features = output
    elif hasattr(output, "last_hidden_state"):
        features = output.last_hidden_state
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        features = output.pooler_output
    elif isinstance(output, (tuple, list)) and output:
        features = output[0]
    else:
        raise RuntimeError(f"Could not interpret model output type: {type(output)!r}")

    if features.ndim == 3:
        features = features.mean(dim=1)
    elif features.ndim > 3:
        features = features.flatten(start_dim=1)
    return features


def extract_batch(processor, model, dtype: torch.dtype, videos: list[list[np.ndarray]]) -> np.ndarray:
    pixel_values = prepare_pixel_values(processor, videos, dtype)
    with torch.inference_mode():
        if hasattr(model, "extract_features"):
            output = model.extract_features(pixel_values)
        else:
            output = model(pixel_values=pixel_values)
        features = tensor_to_features(output)
    return features.detach().float().cpu().numpy()


def iter_pending_rows(
    method: str,
    split_name: str,
    overwrite: bool,
    limit: int,
    num_shards: int,
    shard_index: int,
) -> list[dict]:
    # Partition the immutable split before applying resume filters.  Sharding a
    # changing ``pending`` list is unsafe for throttled SLURM arrays: later
    # tasks can see files written by earlier tasks, shift the modulo positions,
    # and silently leave holes in the split.
    rows = load_action_split(split_name)
    if limit > 0:
        rows = rows[:limit]
    rows = rows[shard_index::num_shards]
    pending = []
    for row in rows:
        stem = str(row["stem"])
        if not method_video_ready(method, stem):
            continue
        if video_feature_path(method, stem).exists() and not overwrite:
            continue
        pending.append(row)
    return pending


def save_feature(method: str, stem: str, feature: np.ndarray, save_dtype: str) -> None:
    path = video_feature_path(method, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if save_dtype == "float16" else np.float32
    np.save(path, feature.astype(dtype, copy=False))


def extract_method_split(
    processor,
    model,
    dtype: torch.dtype,
    method: str,
    split_name: str,
    args: argparse.Namespace,
) -> None:
    pending = iter_pending_rows(
        method,
        split_name,
        args.overwrite,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    print(
        f"{method} / {split_name}: pending={len(pending)}, "
        f"aggregate={aggregate_path(method, split_name)}"
    )

    if not pending:
        if args.num_shards == 1 and (
            args.rebuild_aggregates or not aggregate_path(method, split_name).exists()
        ):
            build_feature_matrix(method, split_name, write_aggregate=True)
        return

    start_time = time.time()
    for batch_start in tqdm(range(0, len(pending), args.batch_size), desc=f"{method} {split_name}"):
        batch_rows = pending[batch_start : batch_start + args.batch_size]
        videos = []
        valid_rows = []

        for row in batch_rows:
            stem = str(row["stem"])
            frames = sample_video_frames(frame_dir_for_method(method, stem), args.num_frames)
            if frames is None:
                append_jsonl(
                    ACTION_LOGS_DIR / "videomae_feature_errors.jsonl",
                    {"method": method, "split": split_name, "stem": stem, "error": "missing_or_unreadable_frames"},
                )
                continue
            videos.append(frames)
            valid_rows.append(row)

        if not videos:
            continue

        try:
            features = extract_batch(processor, model, dtype, videos)
        except Exception as exc:
            for row in valid_rows:
                append_jsonl(
                    ACTION_LOGS_DIR / "videomae_feature_errors.jsonl",
                    {"method": method, "split": split_name, "stem": str(row["stem"]), "error": str(exc)},
                )
            continue

        for row, feature in zip(valid_rows, features):
            save_feature(method, str(row["stem"]), feature, args.save_dtype)

    if args.num_shards == 1:
        build_feature_matrix(method, split_name, write_aggregate=True)
    elapsed = time.time() - start_time
    print(f"{method} / {split_name} complete in {elapsed / 60:.1f} min")


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    methods = resolve_methods(args.methods, include_original=True)

    requested_splits = None
    if args.splits.strip().lower() != "auto":
        requested_splits = parse_csv_arg(args.splits, [])

    if args.aggregate_only:
        for method in methods:
            for split_name in expected_splits_for_method(method, requested_splits):
                x, _y, _filenames, _df = build_feature_matrix(method, split_name, write_aggregate=True)
                print(f"{method} / {split_name}: rebuilt aggregate with {len(x)} features at {aggregate_path(method, split_name)}")
        all_splits = sorted(set(DEFAULT_ORIGINAL_SPLITS + DEFAULT_ANON_SPLITS + (requested_splits or [])))
        status = summarize_feature_status(methods, all_splits)
        status.to_csv(ACTION_LOGS_DIR / "videomae_feature_status.csv", index=False)
        print(status.to_string(index=False))
        return 0

    print(f"Shard: {args.shard_index + 1}/{args.num_shards}")
    processor, model, dtype = load_model(args.model_id, args.precision)

    for method in methods:
        splits = expected_splits_for_method(method, requested_splits)
        for split_name in splits:
            extract_method_split(processor, model, dtype, method, split_name, args)

    if args.num_shards > 1:
        print("Shard extraction complete; aggregate/status generation is deferred to the merge job.")
        return 0

    all_splits = sorted(set(DEFAULT_ORIGINAL_SPLITS + DEFAULT_ANON_SPLITS))
    status = summarize_feature_status(methods, all_splits)
    status.to_csv(ACTION_LOGS_DIR / "videomae_feature_status.csv", index=False)
    print(status.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
