#!/usr/bin/env python3
"""Prepare deterministic action-utility splits for the VideoMAE pipeline."""

from __future__ import annotations

from action_videomae_common import ACTION_ROOT, write_action_splits


def main() -> int:
    summary = write_action_splits()
    print("=" * 80)
    print("VideoMAE action split manifests written")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nOutput root: {ACTION_ROOT}")

    bad = summary[summary["actions"] != 120]
    if not bad.empty:
        print("\nWARNING: at least one split does not contain all 120 actions:")
        print(bad[["split", "actions", "videos"]].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
