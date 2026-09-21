#!/usr/bin/env python3
"""Evaluate RGB-extracted pose identity leakage on original/anonymized probes."""

from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from rgb_pose_common import (
    RGB_POSE_MODEL_DIR,
    RGB_POSE_ROOT,
    build_xy,
    protocol_files,
    resolve_methods,
    resolve_protocols,
    topk_from_proba,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="auto")
    parser.add_argument("--protocols", default="auto")
    return parser.parse_args()


def load_protocol_model(protocol: str) -> dict:
    path = RGB_POSE_MODEL_DIR / f"{protocol}_rgb_pose_identity.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing RGB pose identity model for {protocol}: {path}")
    return joblib.load(path)


def main() -> int:
    args = parse_args()
    methods = resolve_methods(args.methods, include_original=True)
    protocols = resolve_protocols(args.protocols)
    rows = []

    for protocol in protocols:
        payload = load_protocol_model(protocol)
        clf = payload["classifier"]
        encoder = payload["encoder"]
        probe_files = protocol_files(protocol, "probe")
        chance = 1 / max(len(encoder.classes_), 1)

        for method in methods:
            x_test, y_test, kept, stats = build_xy(method, probe_files)
            if len(x_test) == 0:
                print(f"Skipping {protocol}/{method}: no pose features.")
                continue

            known = np.isin(y_test, encoder.classes_)
            x_known = x_test[known]
            y_known = y_test[known]
            if len(x_known) == 0:
                print(f"Skipping {protocol}/{method}: no probe identities known from gallery.")
                continue

            y_enc = encoder.transform(y_known)
            pred_enc = clf.predict(x_known)
            pred_ids = encoder.inverse_transform(pred_enc)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(x_known)
                top5 = topk_from_proba(proba, encoder.classes_, y_known, k=5)
            else:
                top5 = 0.0

            acc = accuracy_score(y_known, pred_ids)
            bal_acc = balanced_accuracy_score(y_enc, pred_enc)
            rows.append(
                {
                    "attack": "rgb_pose_identity",
                    "protocol": protocol,
                    "method": method,
                    "probe_requested": len(probe_files),
                    "probe_features": len(kept),
                    "probe_known_identity": int(known.sum()),
                    "probe_coverage": stats["video_coverage"],
                    "mean_pose_frame_coverage": stats["mean_pose_frame_coverage"],
                    "mean_keypoint_conf": stats["mean_keypoint_conf"],
                    "num_gallery_persons": len(encoder.classes_),
                    "accuracy": acc,
                    "top5": top5,
                    "balanced_accuracy": bal_acc,
                    "coverage_adjusted_accuracy": acc * stats["video_coverage"],
                    "chance": chance,
                }
            )

    df = pd.DataFrame(rows)
    path = RGB_POSE_ROOT / "rgb_pose_identity_results.csv"
    df.to_csv(path, index=False)

    print("=" * 80)
    print("RGB pose identity results")
    print("=" * 80)
    if not df.empty:
        print(
            df[
                [
                    "protocol",
                    "method",
                    "probe_features",
                    "probe_coverage",
                    "accuracy",
                    "top5",
                    "balanced_accuracy",
                    "coverage_adjusted_accuracy",
                    "chance",
                ]
            ].to_string(index=False, float_format="{:.3f}".format)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = df.copy()
        plot_df["label"] = plot_df["protocol"] + " / " + plot_df["method"]
        ax.barh(plot_df["label"], plot_df["coverage_adjusted_accuracy"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Coverage-adjusted identity accuracy")
        ax.set_title("RGB-Extracted Pose Identity Leakage")
        plt.tight_layout()
        fig_path = RGB_POSE_ROOT / "rgb_pose_identity_results.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure: {fig_path}")
    else:
        print("No RGB pose identity results generated.")

    print(f"Results: {path}")
    return 0 if not df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
