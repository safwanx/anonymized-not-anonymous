#!/usr/bin/env python3
"""Emit per-protocol coverage/conditional/end-to-end privacy attack results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "baselines"
OUT = ROOT / "outputs" / "analysis"


def retrieval_rows(filename: str, attack: str) -> pd.DataFrame:
    frame = pd.read_csv(FINAL / filename)
    return pd.DataFrame(
        {
            "attack": attack,
            "protocol": frame["protocol"],
            "method": frame["method"],
            "conditional_metric": "rank1_given_descriptor",
            "conditional_success": frame["rank1"],
            "descriptor_coverage": frame["probe_coverage"],
            "end_to_end_success": frame["coverage_adjusted_rank1"],
        }
    )


def pose_rows() -> pd.DataFrame:
    frame = pd.read_csv(FINAL / "rgb_pose_identity_results.csv")
    return pd.DataFrame(
        {
            "attack": "rgb_pose",
            "protocol": frame["protocol"],
            "method": frame["method"],
            "conditional_metric": "identity_accuracy_given_descriptor",
            "conditional_success": frame["accuracy"],
            "descriptor_coverage": frame["probe_coverage"],
            "end_to_end_success": frame["coverage_adjusted_accuracy"],
        }
    )


def reid_rows() -> pd.DataFrame:
    frame = pd.read_csv(FINAL / "person_reid_results.csv")
    coverage = frame["probe_coverage"] if "probe_coverage" in frame else 1.0
    return pd.DataFrame(
        {
            "attack": "person_reid",
            "protocol": frame["protocol"],
            "method": frame["method"],
            "conditional_metric": "rank1_given_descriptor",
            "conditional_success": frame["rank1"],
            "descriptor_coverage": coverage,
            "end_to_end_success": frame["rank1"] * coverage,
        }
    )


def write_summary(frame: pd.DataFrame) -> None:
    heavy = frame[frame["method"] == "body_pixel"].copy()
    heavy["conditional_pct"] = 100 * heavy["conditional_success"]
    heavy["coverage_pct"] = 100 * heavy["descriptor_coverage"]
    heavy["end_to_end_pct"] = 100 * heavy["end_to_end_success"]
    lines = [
        "# Coverage decomposition",
        "",
        "Each row reports success among probes with a valid descriptor, descriptor coverage, and their product (end-to-end attack success). Values below are percentages for `body_pixel`.",
        "",
        "| Attack | Protocol | Conditional success | Coverage | End-to-end success |",
        "|---|---|---:|---:|---:|",
    ]
    for row in heavy.itertuples(index=False):
        lines.append(
            f"| {row.attack} | {row.protocol} | {row.conditional_pct:.2f}% | "
            f"{row.coverage_pct:.2f}% | {row.end_to_end_pct:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Interpretation: body pixelation's near-zero end-to-end face, pose, and temporal-silhouette scores are often dominated by descriptor extraction failure. Re-ID remains fully covered and therefore measures reduced discriminability directly.",
            "",
            "The conditional pose metric here is ordinary identity accuracy because that is the quantity multiplied by coverage in the submitted evaluator. Balanced accuracy and its corrected action-conditioned null are reported separately in `pose_corrected_null.csv`.",
        ]
    )
    (OUT / "coverage_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            retrieval_rows("face_recognition_results.csv", "face"),
            reid_rows(),
            retrieval_rows("gait_identity_results.csv", "temporal_silhouette_gaitbase"),
            pose_rows(),
        ],
        ignore_index=True,
    )
    frame["product_check_error"] = (
        frame["conditional_success"] * frame["descriptor_coverage"] - frame["end_to_end_success"]
    ).abs()
    if frame["product_check_error"].max() > 1e-9:
        raise RuntimeError("Coverage-adjusted results do not equal conditional success x coverage")
    frame.to_csv(OUT / "coverage_decomposition.csv", index=False)
    write_summary(frame)
    print(f"Wrote {len(frame)} rows to {OUT / 'coverage_decomposition.csv'}")


if __name__ == "__main__":
    main()
