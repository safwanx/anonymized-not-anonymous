"""Shared path/config glue between our NTU120-Privacy-21K pipeline and the
Privacy Beyond Pixels (PBP) repo.

PBP operates entirely on precomputed VideoMAE features in HDF5, keyed by NTU
video name (S###C###P###R###A###), each value shaped [num_clips, 768]. Our
pipeline (pipeline/13_extract_videomae_features.py) already produces
one 768-d mean-pooled descriptor per video; we re-package those into the format
PBP expects.

This module only resolves paths and imports; it does not require a GPU and does
not import torch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- repo locations ---------------------------------------------------------
# Code, protocol manifests, external dependencies, and runtime outputs are separate.
PROJECT_ROOT = Path(os.environ.get("ACCV_ROOT", Path(__file__).resolve().parents[1]))
PIPELINE_STEPS_DIR = PROJECT_ROOT / "pipeline"
CACHE_ROOT = Path(os.environ.get("ACCV_CACHE", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "accv"))
REPOS = Path(os.environ.get("REPOS_DIR", CACHE_ROOT / "repos"))
PBP_REPO = REPOS / "privacy_methods" / "PrivacyBeyondPixels"

# Point the pipeline at the real protocol if the caller did not. Search common
# data locations so the PBP scripts "just work" from either drive.
if not os.environ.get("PILOT_ROOT"):
    for _cand in (
        PROJECT_ROOT / "protocol",
        PROJECT_ROOT / "pilot",
    ):
        if (_cand / "metadata.csv").exists():
            os.environ["PILOT_ROOT"] = str(_cand)
            break

# Make our pipeline helpers importable (pipeline_common, action_videomae_common).
if str(PIPELINE_STEPS_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_STEPS_DIR))

# --- where the PBP-format artifacts live ------------------------------------
# Self-contained under the project so a SLURM run can point ACCV_ROOT at it.
PBP_WORK = Path(os.environ.get("PBP_WORK", Path(os.environ.get("PILOT_OUTPUT_ROOT", PROJECT_ROOT / "outputs")) / "pbp"))
PBP_FEAT_DIR = PBP_WORK / "ntu_features"      # features_ntu_<split>_mae[ _fb<pool> ].h5
PBP_LABELS_DIR = PBP_WORK / "ntu_labels"      # protocol split membership + id/action labels
PBP_MODELS_DIR = PBP_WORK / "saved_models"    # trained AAM checkpoints
PBP_RESULTS_DIR = PBP_WORK / "results"        # feature-space attack + utility CSVs

for _d in (PBP_FEAT_DIR, PBP_LABELS_DIR, PBP_MODELS_DIR, PBP_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# VideoMAE descriptor size used throughout the project.
FEATURE_DIM = 768

# Frame-bank pool size for PBP's SSL privacy loss (matches params_fa fb_frame_pool).
FB_FRAME_POOL = int(os.environ.get("PBP_FB_FRAME_POOL", "10"))


def feat_h5(split: str, fb: bool = False) -> Path:
    """Path of the PBP feature HDF5 for a split. fb=True => frame-bank file."""
    suffix = f"_fb{FB_FRAME_POOL}" if fb else ""
    return PBP_FEAT_DIR / f"features_ntu_{split}_mae{suffix}.h5"


def import_anonymizer():
    """Import PBP's AAM classes without pulling in tridet/mgfn dependencies.

    model_loaders.load_fa imports tridet + mgfn (heavy, GPU-CUDA-ext heavy). The
    anonymizer itself lives in models/anonymizer.py with no such deps, so we add
    the repo root to sys.path and import that module directly.
    """
    if str(PBP_REPO) not in sys.path:
        sys.path.insert(0, str(PBP_REPO))
    from models.anonymizer import MLP, TransformerAnonymizer  # type: ignore

    return MLP, TransformerAnonymizer
