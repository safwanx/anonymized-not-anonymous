#!/usr/bin/env python3
"""GPU step: extract a POOL of VideoMAE clip features per video for PBP's privacy loss.

PBP's SSL privacy objective ("fb" frame bank) needs several feature vectors per
video, drawn from different temporal windows, so it can push two views of the
same video apart in the anonymized space. Our main extractor
(pipeline/13_extract_videomae_features.py) mean-pools to ONE vector
per video, which destroys this. This script samples `--pool` distinct 16-frame
windows per video and saves [pool, 768] to <out>/<stem>.npy.

Feed the output dir to export_protocol_features_to_pbp.py --frame-feat-dir.

Requires a GPU realistically (it is the same VideoMAE forward as step 13, just
`pool` times more clips). Run on the SLURM box:

    python pbp_integration/extract_videomae_frame_features.py \
        --split train_original_r1_allcams --pool 10 \
        --out outputs/pbp/frame_features/train
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

import pbp_paths as P
from action_videomae_common import load_action_split, frame_dir_for_method  # noqa: E402
from pipeline_common import DEVICE  # noqa: E402

DEFAULT_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"


def torch_dtype_for(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.float16 if DEVICE == "cuda" else torch.float32


def load_model(model_id: str, precision: str):
    from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor
    dtype = torch_dtype_for(precision)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    proc = VideoMAEImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, config=cfg, trust_remote_code=True)
    model.eval().to(device=DEVICE, dtype=dtype)
    return proc, model, dtype


def sample_window(frame_paths, num_frames, offset):
    """Pick num_frames around a temporal offset in [0,1] of the clip."""
    n = len(frame_paths)
    center = offset * (n - 1)
    half = num_frames / 2
    idx = np.clip(np.round(np.linspace(center - half, center + half, num_frames)), 0, n - 1).astype(int)
    frames = []
    for i in idx:
        img = cv2.imread(str(frame_paths[i]))
        if img is None:
            return None
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


@torch.inference_mode()
def extract(proc, model, videos, dtype):
    inputs = proc(videos, return_tensors="pt")
    pv = inputs["pixel_values"]
    if pv.shape[1] != 3 and pv.shape[2] == 3:
        pv = pv.permute(0, 2, 1, 3, 4).contiguous()
    pv = pv.to(device=DEVICE, dtype=dtype)
    out = model.extract_features(pv) if hasattr(model, "extract_features") else model(pixel_values=pv)
    if isinstance(out, torch.Tensor):
        feats = out
    elif hasattr(out, "last_hidden_state"):
        feats = out.last_hidden_state
    elif isinstance(out, (tuple, list)) and out:
        feats = out[0]
    else:
        raise RuntimeError(f"Could not interpret VideoMAE output type: {type(out)!r}")
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    return feats.float().cpu().numpy()


def extract_in_batches(proc, model, windows, dtype, batch_size):
    parts = []
    for start in range(0, len(windows), batch_size):
        parts.append(extract(proc, model, windows[start : start + batch_size], dtype))
    return np.concatenate(parts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pool", type=int, default=P.FB_FRAME_POOL)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Temporal windows per VideoMAE forward pass.")
    ap.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    if args.pool < 2:
        raise ValueError("--pool must be at least 2 for the privacy-pair loss")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, dtype = load_model(args.model_id, args.precision)
    offsets = np.linspace(0.0, 1.0, args.pool)

    rows = load_action_split(args.split)
    if args.limit > 0:
        rows = rows[: args.limit]
    # Stable sharding precedes the resume filter; this remains correct when a
    # throttled array starts later shards after earlier shards have written data.
    rows = rows[args.shard_index :: args.num_shards]
    pending = [r for r in rows if args.overwrite or not (out_dir / f"{r['stem']}.npy").exists()]
    print(f"device={DEVICE} dtype={dtype} shard={args.shard_index + 1}/{args.num_shards}")
    print(f"assigned={len(rows)} pending={len(pending)} pool={args.pool} window_batch={args.batch_size}")

    started = time.time()
    written = missing = failed = 0
    for r in tqdm(pending, desc=f"frame-bank {args.split}"):
        stem = str(r["stem"])
        dst = out_dir / f"{stem}.npy"
        frame_paths = sorted(frame_dir_for_method("original", stem).glob("*.jpg"))
        if not frame_paths:
            missing += 1
            continue
        windows = [w for off in offsets if (w := sample_window(frame_paths, args.num_frames, off))]
        if len(windows) < 2:
            missing += 1
            continue
        try:
            feats = extract_in_batches(proc, model, windows, dtype, args.batch_size)
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, feats.astype(np.float16))
            tmp.replace(dst)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] {stem}: {type(exc).__name__}: {exc}", flush=True)
    elapsed = (time.time() - started) / 60.0
    print(f"done -> {out_dir}; written={written} missing={missing} failed={failed} minutes={elapsed:.1f}")
    return 0 if failed == 0 and missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
