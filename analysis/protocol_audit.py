#!/usr/bin/env python3
"""CPU-only protocol audit for pose chance controls and gallery-size claims.

The script uses only Python's standard library. It derives split composition from
the shipped filename lists, joins the published result CSVs, and writes new
artifacts without modifying any source result.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FILENAME_RE = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<person>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})"
)

PROTOCOLS = {
    "crossview_cam2": ("gallery_crossview.txt", "probe_crossview_cam2.txt"),
    "crossview_cam3": ("gallery_crossview.txt", "probe_crossview_cam3.txt"),
    "crosssetup": ("gallery_crosssetup.txt", "probe_crosssetup.txt"),
    "crossrange": ("gallery_crossrange.txt", "probe_crossrange.txt"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        name = raw.strip()
        if not name:
            continue
        match = FILENAME_RE.search(name)
        if not match:
            raise ValueError(f"Cannot parse NTU filename in {path}: {name}")
        record: dict[str, int | str] = {"filename": name}
        record.update({key: int(value) for key, value in match.groupdict().items()})
        records.append(record)
    return records


def fmt_ids(ids: Iterable[int]) -> str:
    return " ".join(f"P{value:03d}" for value in sorted(ids))


def macro_expected_chance(
    probe: list[dict[str, int | str]], candidate_count_by_action: dict[int, int]
) -> float:
    """Expected balanced accuracy of action-aware uniform guessing.

    Balanced accuracy is macro recall. We therefore first average 1/K(action)
    within each true identity and then average identities, rather than weighting
    identities with more probe samples more heavily.
    """
    chance_by_person: dict[int, list[float]] = defaultdict(list)
    for item in probe:
        action = int(item["action"])
        person = int(item["person"])
        chance_by_person[person].append(1.0 / candidate_count_by_action[action])
    per_person = [sum(values) / len(values) for values in chance_by_person.values()]
    return sum(per_person) / len(per_person)


def micro_expected_chance(
    probe: list[dict[str, int | str]], candidate_count_by_action: dict[int, int]
) -> float:
    values = [1.0 / candidate_count_by_action[int(item["action"])] for item in probe]
    return sum(values) / len(values)


def metadata_index(metadata_path: Path | None) -> dict[str, dict[str, str]] | None:
    if metadata_path is None:
        return None
    rows = read_csv(metadata_path)
    index = {row["filename"]: row for row in rows}
    if len(index) != len(rows):
        raise AssertionError(f"Duplicate filenames in {metadata_path}")
    return index


def validate_against_metadata(
    records: list[dict[str, int | str]], index: dict[str, dict[str, str]] | None
) -> None:
    if index is None:
        return
    for record in records:
        name = str(record["filename"])
        if name not in index:
            raise AssertionError(f"Split filename absent from metadata.csv: {name}")
        metadata = index[name]
        for field in ("setup", "camera", "person", "replication", "action"):
            if int(metadata[field]) != int(record[field]):
                raise AssertionError(f"Filename/metadata {field} mismatch for {name}")


def split_audit(
    split_dir: Path, metadata_path: Path | None = None
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    index = metadata_index(metadata_path)
    gallery_rows: list[dict[str, object]] = []
    chances: dict[str, dict[str, float]] = {}
    for protocol, (gallery_name, probe_name) in PROTOCOLS.items():
        gallery = read_split(split_dir / gallery_name)
        probe = read_split(split_dir / probe_name)
        validate_against_metadata(gallery, index)
        validate_against_metadata(probe, index)
        gallery_ids = {int(item["person"]) for item in gallery}
        probe_ids = {int(item["person"]) for item in probe}
        if gallery_ids != probe_ids:
            raise AssertionError(f"Gallery/probe identities differ for {protocol}")

        naive = 1.0 / len(gallery_ids)
        gallery_actions = {int(item["action"]) for item in gallery}
        probe_actions = {int(item["action"]) for item in probe}

        if gallery_actions.isdisjoint(probe_actions):
            # Cross-range is deliberately A001-A060 -> A061-A120. Every bridge
            # identity spans both ranges, so knowing the range removes no identity.
            candidate_count = {action: len(gallery_ids) for action in probe_actions}
            condition = "action_range_only_disjoint_exact_actions"
        else:
            ids_by_action: dict[int, set[int]] = defaultdict(set)
            for item in gallery:
                ids_by_action[int(item["action"])].add(int(item["person"]))
            missing = probe_actions - ids_by_action.keys()
            if missing:
                raise AssertionError(f"Missing gallery action candidates for {protocol}: {missing}")
            candidate_count = {action: len(ids_by_action[action]) for action in probe_actions}
            condition = "exact_action"

        macro = macro_expected_chance(probe, candidate_count)
        micro = micro_expected_chance(probe, candidate_count)
        distribution: dict[int, int] = defaultdict(int)
        for value in candidate_count.values():
            distribution[value] += 1
        candidate_summary = "; ".join(
            f"{n_actions} actions x {n_ids} IDs" for n_ids, n_actions in sorted(distribution.items())
        )

        gallery_rows.append(
            {
                "protocol": protocol,
                "gallery_samples": len(gallery),
                "probe_samples": len(probe),
                "gallery_identity_count": len(gallery_ids),
                "gallery_identity_ids": fmt_ids(gallery_ids),
                "gallery_action_min": min(gallery_actions),
                "gallery_action_max": max(gallery_actions),
                "probe_action_min": min(probe_actions),
                "probe_action_max": max(probe_actions),
                "action_conditioning": condition,
                "candidate_pool_distribution": candidate_summary,
                "naive_chance": naive,
                "action_conditioned_chance_balanced": macro,
                "action_conditioned_chance_sample_weighted": micro,
            }
        )
        chances[protocol] = {"naive": naive, "macro": macro, "micro": micro}
    return gallery_rows, chances


def pose_audit(
    pose_path: Path, chances: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in read_csv(pose_path):
        protocol = source["protocol"]
        observed = float(source["balanced_accuracy"])
        coverage = float(source["probe_coverage"])
        conditioned = chances[protocol]["macro"]
        rows.append(
            {
                "protocol": protocol,
                "method": source["method"],
                "gallery_identity_count": int(source["num_gallery_persons"]),
                "probe_requested": int(source["probe_requested"]),
                "probe_features": int(source["probe_features"]),
                "probe_coverage": coverage,
                "observed_balanced_accuracy": observed,
                "naive_chance": chances[protocol]["naive"],
                "observed_over_naive": observed / chances[protocol]["naive"],
                "action_conditioned_chance_balanced": conditioned,
                "observed_over_action_conditioned": observed / conditioned,
                "point_estimate_above_action_conditioned": observed > conditioned,
                "chance_scope": (
                    "full_requested_split_approximation_for_covered_subset"
                    if coverage < 0.999
                    else "full_requested_split"
                ),
                "permutation_test": "unavailable_no_per_sample_predictions_or_features",
            }
        )
    return rows


def protocol_results(
    reid_path: Path, pose_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in read_csv(reid_path):
        rows.append(
            {
                "adversary": "reid",
                "protocol": source["protocol"],
                "method": source["method"],
                "metric": "rank1",
                "value": float(source["rank1"]),
                "coverage": float(source["probe_coverage"]),
                "naive_chance": "",
                "action_conditioned_chance": "",
                "interpretation_guardrail": (
                    "3-identity diagnostic; fixed clothing/session; do not generalize magnitude"
                    if source["protocol"] == "crossrange"
                    else "report per protocol; do not average with different gallery sizes"
                ),
            }
        )
    for source in pose_rows:
        rows.append(
            {
                "adversary": "rgb_pose_identity",
                "protocol": source["protocol"],
                "method": source["method"],
                "metric": "balanced_accuracy",
                "value": source["observed_balanced_accuracy"],
                "coverage": source["probe_coverage"],
                "naive_chance": source["naive_chance"],
                "action_conditioned_chance": source["action_conditioned_chance_balanced"],
                "interpretation_guardrail": (
                    "3-identity diagnostic; no transferable leakage-magnitude claim"
                    if source["protocol"] == "crossrange"
                    else "use action-conditioned null; significance unavailable"
                ),
            }
        )
    return rows


def extrapolation_audit(fit_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in read_csv(fit_path):
        max_measured = int(source["max_pool_measured"])
        full_pool = int(float(source["full_pool"]))
        split = source["split"]
        value = float(source["extrapolated_rank1_full_pool"])
        rows.append(
            {
                "adversary": source["adversary"],
                "split": split,
                "max_empirical_pool": max_measured,
                "rank1_at_max_empirical_pool": float(source["rank1_at_max_pool"]),
                "fit_r2_in_observed_range": float(source["fit_r2"]),
                "extrapolated_pool": full_pool,
                "extrapolation_factor": full_pool / max_measured,
                "extrapolated_rank1": value,
                "chance_at_extrapolated_pool": 1.0 / full_pool,
                "extrapolated_over_chance": value * full_pool,
                "claim_scope": (
                    "matches manuscript cross-view range"
                    if split.startswith("crossview")
                    else "not part of manuscript cross-view range; especially long extrapolation"
                ),
                "evidence_type": "log-fit extrapolation; not an empirical 106-identity result",
            }
        )
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def make_markdown(
    gallery_rows: list[dict[str, object]],
    pose_rows: list[dict[str, object]],
    protocol_rows: list[dict[str, object]],
    extrap_rows: list[dict[str, object]],
) -> str:
    original_pose = [row for row in pose_rows if row["method"] == "original"]
    original_reid = {
        str(row["protocol"]): row
        for row in protocol_rows
        if row["adversary"] == "reid" and row["method"] == "original"
    }
    lines = [
        "# Protocol audit: pose chance and identity-pool sensitivity",
        "",
        "Generated reproducibly from `protocol/metadata.csv`, the fixed split files, and existing final CSVs with "
        "`python analysis/protocol_audit.py`.",
        "",
        "## Split and chance audit",
        "",
        "| Protocol | Gallery IDs | Exact IDs | Naive chance | Action-conditioned BA null | Sample-weighted null |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in gallery_rows:
        lines.append(
            f"| {row['protocol']} | {row['gallery_identity_count']} | "
            f"{row['gallery_identity_ids']} | {pct(float(row['naive_chance']))} | "
            f"{pct(float(row['action_conditioned_chance_balanced']))} | "
            f"{pct(float(row['action_conditioned_chance_sample_weighted']))} |"
        )
    lines.extend(
        [
            "",
            "For cross-view, actions A001-A060 have 16 eligible identities and A061-A120 have 25; "
            "the overlap consists of bridge identities. For cross-setup the corresponding pools are 11 and 8. "
            "Thus action/action-range information narrows the identity candidate set before any identity-bearing pose is used. "
            "The primary null above is class-balanced because the reported pose metric is balanced accuracy; the sample-weighted "
            "null is included to make the averaging convention explicit.",
            "",
            "Cross-range is different: gallery A001-A060 and probe A061-A120 are disjoint, and all three bridge identities span "
            "both ranges. Action-range knowledge therefore leaves all three candidates and the null remains 33.33%.",
            "",
            "## Original RGB pose result under the corrected null",
            "",
            "| Protocol | Observed balanced accuracy | Naive ratio | Corrected null | Corrected ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in original_pose:
        lines.append(
            f"| {row['protocol']} | {pct(float(row['observed_balanced_accuracy']))} | "
            f"{float(row['observed_over_naive']):.2f}x | "
            f"{pct(float(row['action_conditioned_chance_balanced']))} | "
            f"{float(row['observed_over_action_conditioned']):.2f}x |"
        )
    lines.extend(
        [
            "",
            "Only cross-view C2 clearly exceeds the corrected action-aware point null for original RGB (9.88% vs 4.86%, 2.03x). "
            "C3 (4.21% vs 4.86%), cross-setup (9.63% vs 10.51%), and cross-range (30.28% vs 33.33%) do not. "
            "Among the other full-coverage methods, face blur and body blur also exceed the point null on C2. Body blur is only "
            "marginally above it on cross-setup (11.02% vs 10.51%), which should not be called evidence without a valid test.",
            "",
            "An action-stratified permutation p-value cannot be reconstructed: no per-sample pose predictions, logits, or features "
            "are present locally. Aggregate accuracies are insufficient to preserve the dependence among identity, action, and "
            "prediction. For body pixelation, the metadata-only corrected null is additionally approximate because the identities/actions "
            "of the covered subset are unavailable.",
            "",
            "## Identity-pool and extrapolation audit",
            "",
            "The shipped galleries contain 38 identities for each cross-view split, 18 for cross-setup, and 3 for cross-range. "
            "Re-ID and pose values should be shown per protocol rather than averaged across these incomparable candidate sets. "
            "Cross-range is a diagnostic stress test only: its three identities, fixed within-session clothing, and 33.33% chance "
            "make its leakage magnitude unsuitable for transfer claims.",
            "",
            "| Protocol | Gallery IDs | Original re-ID Rank-1 | Original pose BA | Corrected pose null |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    gallery_by_protocol = {str(row["protocol"]): row for row in gallery_rows}
    for pose in original_pose:
        protocol = str(pose["protocol"])
        lines.append(
            f"| {protocol} | {gallery_by_protocol[protocol]['gallery_identity_count']} | "
            f"{pct(float(original_reid[protocol]['value']))} | "
            f"{pct(float(pose['observed_balanced_accuracy']))} | "
            f"{pct(float(pose['action_conditioned_chance_balanced']))} |"
        )
    lines.extend(
        [
            "",
            "The CSV `protocol_results_unaveraged.csv` contains the corresponding per-method rows. The apparently high "
            "cross-range re-ID Rank-1 (62.08%) must be read against only three candidates and the clothing/session confound.",
            "",
            "| Adversary | Split | Largest empirical N | Rank-1 there | Extrapolated N | Extrapolated Rank-1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in extrap_rows:
        lines.append(
            f"| {row['adversary']} | {row['split']} | {row['max_empirical_pool']} | "
            f"{float(row['rank1_at_max_empirical_pool']):.3f} | {row['extrapolated_pool']} | "
            f"{float(row['extrapolated_rank1']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The manuscript's 0.74-0.81 re-ID and 0.37-0.78 face ranges correctly select the two cross-view 106-identity "
            "log-fit projections. They are not measurements on 106 people. The empirical evidence is identity subsampling up to "
            "N=38 (cross-view; the full N=38 point has one realization) or N=15 (cross-setup); N=106 is extrapolation by 2.79x "
            "or 7.07x in pool size. In-range R-squared does not validate the assumed log trend outside the observed range.",
            "",
            "## Rebuttal-ready wording",
            "",
            "> We thank the reviewers for identifying the small-pool and pose-control issues. Re-auditing the fixed splits, we find "
            "that action range itself narrows the cross-view identity pool (16 identities for A001-A060 and 25 for A061-A120), "
            "raising the class-balanced pose null from 2.63% to 4.86%; for cross-setup it rises from 5.56% to 10.51%. Under this "
            "stronger null, original RGB pose remains above the point null on C2 (9.88%, 2.03x), but not on C3, cross-setup, or "
            "cross-range, so we narrow our pose claim accordingly. We also now characterize cross-range as a three-identity diagnostic "
            "stress test and clarify that the 106-identity values are log-fit projections from empirical subsampling (up to 38 identities), "
            "not measurements on a 106-person gallery.",
            "",
        ]
    )
    return "\n".join(lines)


def run(repo_root: Path, output_dir: Path) -> None:
    split_dir = repo_root / "protocol" / "splits"
    result_dir = repo_root / "results" / "baselines"
    gallery_rows, chances = split_audit(split_dir, repo_root / "protocol" / "metadata.csv")

    expected_counts = {
        "crossview_cam2": 38,
        "crossview_cam3": 38,
        "crosssetup": 18,
        "crossrange": 3,
    }
    actual_counts = {row["protocol"]: row["gallery_identity_count"] for row in gallery_rows}
    if actual_counts != expected_counts:
        raise AssertionError(f"Unexpected gallery identity counts: {actual_counts}")

    pose_rows = pose_audit(result_dir / "rgb_pose_identity_results.csv", chances)
    protocol_rows = protocol_results(result_dir / "person_reid_results.csv", pose_rows)
    extrap_rows = extrapolation_audit(result_dir / "identity_pool_scaling_fit.csv")

    write_csv(
        output_dir / "gallery_and_chance_audit.csv",
        gallery_rows,
        list(gallery_rows[0]),
    )
    write_csv(output_dir / "pose_corrected_null.csv", pose_rows, list(pose_rows[0]))
    write_csv(output_dir / "protocol_results_unaveraged.csv", protocol_rows, list(protocol_rows[0]))
    write_csv(output_dir / "identity_pool_extrapolation_audit.csv", extrap_rows, list(extrap_rows[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cpu_audit_summary.md").write_text(
        make_markdown(gallery_rows, pose_rows, protocol_rows, extrap_rows), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "outputs" / "analysis",
    )
    args = parser.parse_args()
    run(args.repo_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
