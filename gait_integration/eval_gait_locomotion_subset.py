#!/usr/bin/env python3
"""Evaluate cached GaitBase identity descriptors on locomotion-dominant actions.

The subset is a sensitivity analysis, not a claim of canonical gait recognition:
A026 hopping, A027 jump-up, A042 staggering, A043 falling, A059 walking
towards another person, and A060 walking apart.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from eval_gait_identity_sharded import METHOD_ORIGINAL, load_cached
from pipeline_common import EVAL_PROTOCOLS, FEATURES_DIR, compute_reid_metrics, meta_by_filename


DEFAULT_ACTIONS = (26, 27, 42, 43, 59, 60)


def matrix(filenames: list[str], method: str, actions: set[int]):
    selected = [name for name in filenames if int(meta_by_filename[name]["action"]) in actions]
    features, ids = [], []
    for filename in selected:
        feature = load_cached(method, filename)
        if feature is None:
            continue
        features.append(feature)
        ids.append(int(meta_by_filename[filename]["person"]))
    x = np.stack(features) if features else np.zeros((0, 4096), np.float32)
    return x, ids, len(selected), len(features) / max(len(selected), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=os.environ.get("EVAL_ANON_METHODS", "deepprivacy2"))
    parser.add_argument("--actions", default=",".join(map(str, DEFAULT_ACTIONS)))
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip() and item.strip() != METHOD_ORIGINAL]
    actions = {int(item) for item in args.actions.split(",") if item.strip()}
    rows = []
    for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
        gallery_x, gallery_ids, gallery_requested, gallery_coverage = matrix(gallery_files, METHOD_ORIGINAL, actions)
        for method in [METHOD_ORIGINAL] + methods:
            probe_x, probe_ids, probe_requested, probe_coverage = matrix(probe_files, method, actions)
            values = compute_reid_metrics(gallery_x, gallery_ids, probe_x, probe_ids)
            rows.append({
                "attack": "gait_identity_locomotion_dominant", "protocol": protocol,
                "method": method, "action_ids": ",".join(map(str, sorted(actions))),
                "gallery_requested": gallery_requested, "probe_requested": probe_requested,
                "gallery_coverage": gallery_coverage, "probe_coverage": probe_coverage,
                **values, "coverage_adjusted_rank1": values["rank1"] * probe_coverage,
                "silhouette_source": "otsu_person_crop", "checkpoint_domain": "CASIA-B",
                "scope_note": "locomotion-dominant sensitivity subset; not canonical gait-only data",
            })
    path = FEATURES_DIR / "gait_identity_locomotion_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['protocol']} {row['method']}: n={row['num_probe']:.0f}/{row['probe_requested']} "
              f"coverage={row['probe_coverage']:.3f} rank1={row['rank1']:.3f}")
    print(f"Results: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
