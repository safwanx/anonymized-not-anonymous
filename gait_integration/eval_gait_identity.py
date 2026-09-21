#!/usr/bin/env python3
"""Gait identity adversary (OpenGait GaitBase) over NTU120-Privacy-21K.

Replaces the weak silhouette/motion proxy (pipeline/08) with a real
clip-level gait recognizer. For each protocol (cross-view C2/C3, cross-setup,
cross-range) and each anonymization method, we:
  1. extract an ordered silhouette sequence per video (gait_silhouette),
  2. embed it with GaitBase -> 4096-d descriptor,
  3. cosine-retrieve probes against the clean (original) gallery,
reusing pipeline_common.compute_reid_metrics + coverage so the numbers sit in
the same table as face / re-ID. Gallery is always built from original frames;
probes from the (possibly anonymized) method frames.

GPU recommended (GaitBase forward per video). Result -> features/gait_identity_results.csv.

Run on SLURM:
    python gait_integration/eval_gait_identity.py --methods auto
    python gait_integration/eval_gait_identity.py --methods original,face_blur,body_blur,body_pixel,deepprivacy2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

import gait_paths as G
from gaitbase_model import load_gaitbase
from gait_silhouette import extract_sequence

from pipeline_common import (  # noqa: E402
    EVAL_PROTOCOLS,
    FRAMES_DIR,
    SIL_FRAME_STEP,
    SIL_MAX_EVAL,
    available_anonymization_methods,
    compute_reid_metrics,
    files_to_ids,
    frame_root_for_method,
    free_gpu,
    meta_by_filename,
    safe_limit,
    save_attack_results,
)


def gait_embeddings(model, device, file_list, frame_root, desc, max_videos, mask_dir):
    selected = safe_limit(file_list, max_videos) if max_videos else file_list
    feats, ids, valid = [], [], 0
    for filename in tqdm(selected, desc=desc, leave=False):
        row = meta_by_filename.get(filename)
        if row is None:
            continue
        seq = extract_sequence(str(row["stem"]), frame_root, frame_step=SIL_FRAME_STEP, mask_dir=mask_dir)
        if seq is None:
            continue
        feats.append(model.embed(seq, device))
        ids.append(int(row["person"]))
        valid += 1
    coverage = valid / max(len(selected), 1)
    if feats:
        return np.stack(feats), ids, coverage
    return np.zeros((0, 4096), np.float32), [], coverage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", default="auto",
                    help="auto = original + available anonymized methods, or a CSV list.")
    ap.add_argument("--mask-root", default="",
                    help="Optional YOLO-seg mask root from extract_silhouette_masks.py. "
                         "Per-method subdirs (<root>/<method>/<stem>/<frame>.png) are used; "
                         "gallery always uses <root>/original.")
    ap.add_argument("--max-videos", type=int, default=SIL_MAX_EVAL,
                    help="Cap probes/gallery per protocol (0 = all). Matches SIL_MAX_EVAL default.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from pathlib import Path
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    mask_root = Path(args.mask_root) if args.mask_root else None

    def mask_dir_for(method: str):
        return (mask_root / method) if mask_root else None

    if args.methods.strip().lower() == "auto":
        methods = ["original"] + available_anonymization_methods()
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"device={device}  methods={methods}  max_videos={args.max_videos}")

    model = load_gaitbase(device)

    rows = []
    for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
        g_feats, g_ids, g_cov = gait_embeddings(
            model, device, gallery_files, FRAMES_DIR, f"gait gallery {protocol}",
            args.max_videos, mask_dir_for("original"))
        for method in methods:
            p_feats, p_ids, p_cov = gait_embeddings(
                model, device, probe_files, frame_root_for_method(method),
                f"gait probe {protocol} {method}", args.max_videos, mask_dir_for(method))
            m = compute_reid_metrics(g_feats, g_ids, p_feats, p_ids)
            m.update({
                "attack": "gait_identity",
                "protocol": protocol,
                "method": method,
                "gallery_requested": len(gallery_files),
                "probe_requested": len(probe_files),
                "gallery_coverage": g_cov,
                "probe_coverage": p_cov,
                "coverage_adjusted_rank1": m["rank1"] * p_cov,
            })
            rows.append(m)

    save_attack_results("gait_identity", rows)
    free_gpu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
