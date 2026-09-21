# Optional local experiment artifacts

This directory retains trained experiment heads/adapters and six aggregate
VideoMAEv2 feature arrays. No raw video, frames, skeletons, or downloaded
pretrained backbones are included.

- `features/action_videomae_v2/`: Original/DeepPrivacy2 train, validation, and C2
  aggregates; clean and DP2-adapted action heads.
- `features/rgb_pose_identity/`: C2 MLP/temporal identity heads and pose action head.
- `pbp/saved_models/`: NTU-trained PBP adapter and joint utility head.

Only this README and the integrity manifest belong in the public source release.
Binaries remain local and are excluded by `.gitignore`; they may contain
dataset-derived information and are not automatically safe to redistribute.
Their absence in a fresh checkout is expected. Published CSVs are in `results/`.

Inspect retained C2 adaptation inputs without recomputing features:

```bash
PILOT_OUTPUT_ROOT="$PWD/artifacts" python pipeline/21_run_adapted_action_utility.py \
  --methods deepprivacy2 --test-splits test_original_c2r2 --dry-run
```

For PBP C2 utility, use the explicit artifact paths in
`pbp_integration/README.md`. Keep new outputs outside published result tables.
