#!/usr/bin/env python3
"""Train RGB-extracted pose identity adversaries on original gallery features."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pipeline_common import RANDOM_SEED
from rgb_pose_common import RGB_POSE_MODEL_DIR, RGB_POSE_ROOT, build_xy, protocol_files, resolve_protocols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", default="auto")
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--hidden", default="512,256", help="Comma-separated MLP hidden sizes.")
    parser.add_argument("--alpha", type=float, default=1e-4)
    return parser.parse_args()


def parse_hidden(raw: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return tuple(values) if values else (512, 256)


def main() -> int:
    args = parse_args()
    protocols = resolve_protocols(args.protocols)
    hidden = parse_hidden(args.hidden)
    rows = []

    for protocol in protocols:
        gallery_files = protocol_files(protocol, "gallery")
        x_train, y_train, kept, stats = build_xy("original", gallery_files)
        if len(x_train) == 0:
            print(f"Skipping {protocol}: no original gallery pose features.")
            continue

        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y_train)
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=args.alpha,
                max_iter=args.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_SEED,
                verbose=False,
            ),
        )
        clf.fit(x_train, y_enc)

        model_path = RGB_POSE_MODEL_DIR / f"{protocol}_rgb_pose_identity.joblib"
        joblib.dump(
            {
                "protocol": protocol,
                "classifier": clf,
                "encoder": encoder,
                "train_files": kept,
                "train_stats": stats,
                "hidden": hidden,
                "max_iter": args.max_iter,
            },
            model_path,
        )

        rows.append(
            {
                "protocol": protocol,
                "model_path": str(model_path),
                "train_requested": len(gallery_files),
                "train_valid": len(kept),
                "train_coverage": stats["video_coverage"],
                "mean_pose_frame_coverage": stats["mean_pose_frame_coverage"],
                "mean_keypoint_conf": stats["mean_keypoint_conf"],
                "num_persons": len(encoder.classes_),
            }
        )
        print(f"{protocol}: trained on {len(kept)}/{len(gallery_files)} gallery videos -> {model_path}")

    df = pd.DataFrame(rows)
    path = RGB_POSE_ROOT / "rgb_pose_identity_train_summary.csv"
    df.to_csv(path, index=False)
    print("=" * 80)
    print("RGB pose identity training summary")
    print("=" * 80)
    if not df.empty:
        print(df.to_string(index=False, float_format="{:.3f}".format))
    else:
        print("No RGB pose identity models were trained.")
    print(f"Summary: {path}")
    return 0 if not df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
