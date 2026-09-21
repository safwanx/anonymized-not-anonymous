#!/usr/bin/env python3
"""Feature-space identity-linkability attack for the latent anonymization paradigm.

Privacy Beyond Pixels anonymizes VideoMAE *features*, not pixels, so our
pixel-space attackers (ArcFace / OSNet / pose) do not apply. The matched attack
is in feature space: an adversary holds a clean gallery of VideoMAE descriptors
and receives anonymized probe descriptors, then does cosine retrieval, exactly
like the re-ID attacker but on the same features used for action utility.

For each protocol (cross-view C2/C3, cross-setup, cross-range) we report, on the
SAME gallery (clean, original features):
    method=original   : probe = clean features        (leakage upper bound / sanity)
    method=pbp_aam     : probe = AAM(clean features)    (the released anonymization)

Metrics reuse pipeline_common.compute_reid_metrics (Rank-1/5/mAP) + coverage,
so the numbers sit in the same table as the pixel-space adversaries.

This script reads per-video features via action_videomae_common.load_video_feature
('original'), so it needs the cached VideoMAE per-video .npy on the running
machine. The AAM checkpoint comes from train_pbp_aam_ntu.py.

Run (CPU fine):
    python pbp_integration/eval_pbp_feature_linkability.py \
        --aam outputs/pbp/saved_models/pbp_aam_ntu.pth
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import pbp_paths as P

# heavy module-level init (loads metadata, splits); needs the pilot present.
from pipeline_common import (  # noqa: E402
    EVAL_PROTOCOLS,
    compute_reid_metrics,
    files_to_ids,
    save_attack_results,
)
from action_videomae_common import load_video_feature  # noqa: E402


def load_aam(ckpt_path: str | None):
    """Return (transform_fn). If ckpt is None, identity (clean-feature baseline)."""
    if not ckpt_path:
        return lambda x: x
    state = torch.load(ckpt_path, map_location="cpu")
    MLP, Transformer = P.import_anonymizer()
    arch = state.get("arch", "mlp")
    model = MLP(P.FEATURE_DIM, P.FEATURE_DIM) if arch == "mlp" else Transformer(P.FEATURE_DIM, 8, 3)
    model.load_state_dict(state["fa_model_state_dict"])
    model.eval()

    @torch.no_grad()
    def transform(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).float()
        return model(t).numpy()

    return transform


def feature_matrix(file_list, transform):
    """Build [N,768] feature matrix + matching ids for files that have a feature."""
    feats, ids = [], []
    for filename in file_list:
        stem = filename.replace("_rgb.avi", "")
        vec = load_video_feature("original", stem)
        if vec is None:
            continue
        feats.append(vec.reshape(-1))
        ids.append(filename)
    if not feats:
        return np.zeros((0, P.FEATURE_DIM), np.float32), [], 0
    x = np.stack(feats).astype(np.float32)
    x = transform(x)
    person_ids = files_to_ids(ids)
    return x, person_ids, len(feats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aam", default="", help="AAM checkpoint; omit to run only the clean baseline.")
    args = ap.parse_args()

    identity = load_aam(None)
    anon = load_aam(args.aam) if args.aam else None

    rows = []
    for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
        g_feats, g_ids, _ = feature_matrix(gallery_files, identity)   # gallery always clean
        for method, tf in [("original", identity)] + ([("pbp_aam", anon)] if anon else []):
            p_feats, p_ids, p_have = feature_matrix(probe_files, tf)
            m = compute_reid_metrics(g_feats, g_ids, p_feats, p_ids)
            coverage = p_have / max(len(probe_files), 1)
            m.update({
                "attack": "feature_linkability",
                "protocol": protocol,
                "method": method,
                "probe_requested": len(probe_files),
                "probe_coverage": coverage,
                "coverage_adjusted_rank1": m["rank1"] * coverage,
            })
            rows.append(m)

    save_attack_results("feature_linkability", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
