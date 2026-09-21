"""Synthetic CPU smoke test for eval_pbp_action_utility.py."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def _identity_aam_state() -> dict[str, torch.Tensor]:
    eye = torch.eye(768)
    return {
        "fc1.weight": eye.clone(),
        "fc1.bias": torch.zeros(768),
        "fc2.weight": eye.clone(),
    }


def test_synthetic_clean_and_aam_utility(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    aggregates = tmp_path / "aggregates"
    aggregates.mkdir()
    splits = ("test_original_c2r2", "test_original_c3r2")
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    features = np.zeros((len(labels), 768), dtype=np.float32)
    features[np.arange(len(labels)), labels] = 1.0
    for split in splits:
        np.savez_compressed(aggregates / f"original__{split}.npz", features=features, labels=labels)

    classifier_weight = torch.zeros((120, 768))
    classifier_weight[0, 0] = classifier_weight[1, 1] = classifier_weight[2, 2] = 10.0
    classifier_bias = torch.full((120,), -1.0)

    aam_path = tmp_path / "aam.pth"
    torch.save({
        "arch": "mlp",
        "fa_model_state_dict": _identity_aam_state(),
        "head_state_dict": {"weight": classifier_weight, "bias": classifier_bias},
    }, aam_path)

    clean_head_path = tmp_path / "clean_head.pt"
    torch.save({
        "state_dict": {"net.weight": classifier_weight, "net.bias": classifier_bias},
        "input_dim": 768,
        "hidden_dim": 0,
        "dropout": 0.2,
        "num_classes": 120,
        "feature_mean": np.zeros(768, dtype=np.float32),
        "feature_std": np.ones(768, dtype=np.float32),
    }, clean_head_path)

    output = tmp_path / "pbp_action_utility_results.csv"
    env = os.environ.copy()
    env["PBP_WORK"] = str(tmp_path / "pbp_work")
    proc = subprocess.run(
        [
            sys.executable, str(root / "pbp_integration" / "eval_pbp_action_utility.py"),
            "--aam", str(aam_path), "--clean-head", str(clean_head_path),
            "--aggregates-dir", str(aggregates), "--output", str(output),
        ],
        cwd=root, env=env, text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {(row["action_head"], row["feature_variant"]) for row in rows} == {
        ("clean_trained", "clean"), ("clean_trained", "pbp_aam"),
        ("pbp_joint", "clean"), ("pbp_joint", "pbp_aam"),
    }
    assert all(float(row["top1"]) == 1.0 for row in rows)
    assert all(row["feature_source"].startswith("aggregate_npz:") for row in rows)
