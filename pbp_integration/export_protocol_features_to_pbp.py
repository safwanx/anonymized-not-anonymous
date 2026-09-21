#!/usr/bin/env python3
"""Export NTU120-Privacy-21K VideoMAE features into the HDF5 format PBP expects.

PBP training reads, per dataset split:
    features_ntu_<split>_mae.h5         key=<ntu_filename> -> [num_clips, 768]
    features_ntu_<split>_mae_fb<P>.h5   key=<ntu_filename> -> [P, 768]   (frame bank)
plus we emit a labels JSON (filename -> {action_label, person}) so our focused
trainer does not depend on PBP's hardcoded NTU subject/gender split logic.

The clip-level file is built from our cached per-split aggregates (CPU only).
The frame-bank file requires per-frame VideoMAE features, produced by
extract_videomae_frame_features.py (GPU). If that source is absent the fb file
is skipped with a warning; PBP's SSL privacy loss needs it, the AR/recon path
does not.

Run (CPU):
    python pbp_integration/export_protocol_features_to_pbp.py \
        --splits train_original_r1_allcams,test_original_c2r2
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

import pbp_paths as P

# our pipeline helpers (path injected by pbp_paths)
from action_videomae_common import (  # noqa: E402
    load_action_split,
    load_or_build_feature_matrix,
    load_video_feature,
)


def export_clip_level(method: str, split: str) -> int:
    """Write features_ntu_<split>_mae.h5 from the cached aggregate. Returns count."""
    feats, _labels, filenames = load_or_build_feature_matrix(method, split)
    out = P.feat_h5(split, fb=False)
    if feats.shape[0] == 0:
        print(f"  [WARN] no features for {method}/{split}; aggregate missing on this machine.")
        return 0
    with h5py.File(out, "w") as h:
        for vec, name in zip(feats, filenames):
            # PBP expects [num_clips, dim]; our descriptor is a single clip.
            h.create_dataset(name, data=vec.reshape(1, -1).astype(np.float32))
    print(f"  wrote {out.name}: {len(filenames)} videos")
    return len(filenames)


def export_labels(split: str) -> None:
    rows = load_action_split(split)
    labels = {
        str(r["filename"]): {"action_label": int(r["label"]), "person": int(r["person"])}
        for r in rows
    }
    out = P.PBP_LABELS_DIR / f"labels_ntu_{split}.json"
    out.write_text(json.dumps(labels, indent=0))
    print(f"  wrote {out.name}: {len(labels)} entries")


def export_frame_bank(split: str, frame_feat_dir: str) -> int:
    """Write the frame-bank fb<P>.h5 from per-frame VideoMAE features (GPU-made).

    Expects frame_feat_dir/<stem>.npy of shape [num_frames, 768]. We sample/pad
    to exactly FB_FRAME_POOL frames per video so PBP's loader can draw 2 of P.
    """
    from pathlib import Path

    src = Path(frame_feat_dir)
    if not src.exists():
        print(f"  [SKIP] frame-feature source not found: {src} (GPU step not run yet)")
        return 0

    rows = load_action_split(split)
    out = P.feat_h5(split, fb=True)
    written = 0
    pool = P.FB_FRAME_POOL
    with h5py.File(out, "w") as h:
        for r in rows:
            stem, name = str(r["stem"]), str(r["filename"])
            f = src / f"{stem}.npy"
            if not f.exists():
                continue
            arr = np.load(f).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # sample/pad to exactly `pool` rows
            if arr.shape[0] >= pool:
                idx = np.linspace(0, arr.shape[0] - 1, pool).round().astype(int)
                arr = arr[idx]
            else:
                reps = int(np.ceil(pool / arr.shape[0]))
                arr = np.tile(arr, (reps, 1))[:pool]
            h.create_dataset(name, data=arr)
            written += 1
    print(f"  wrote {out.name}: {written} videos (pool={pool})")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="original",
                    help="Feature source method (original for clean AAM training).")
    ap.add_argument("--splits", default="train_original_r1_allcams,test_original_c2r2",
                    help="Comma-separated action splits to export.")
    ap.add_argument("--frame-feat-dir", default="",
                    help="Dir of per-video [num_frames,768] npy for the fb privacy loss (GPU output).")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"PBP feature export -> {P.PBP_FEAT_DIR}")
    for split in splits:
        print(f"[{split}]")
        export_clip_level(args.method, split)
        export_labels(split)
        if args.frame_feat_dir:
            export_frame_bank(split, args.frame_feat_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
