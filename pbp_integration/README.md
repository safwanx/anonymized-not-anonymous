# Privacy Beyond Pixels integration

PBP anonymizes VideoMAEv2 **features**, not RGB pixels. We evaluate feature-space
identity linkability separately from the image-based attacks. The NTU integration
uses upstream anonymizer modules and the privacy loss, without upstream TAD/AD
co-training. Its temporal-window feature bank is required for the privacy term;
`--fb-weight 0` is a reconstruction/utility ablation, not the full objective.

The pipeline order is:

1. `export_protocol_features_to_pbp.py`: export mean-pooled descriptors and labels.
2. `extract_videomae_frame_features.py`: extract a pool of temporal-window features
   on GPU; rerun the exporter with `--frame-feat-dir` for the privacy loss input.
3. `train_pbp_aam_ntu.py`: train the NTU adapter and joint action head.
4. `eval_pbp_feature_linkability.py`: clean-gallery/anonymized-probe retrieval.
5. `eval_pbp_action_utility.py`: clean-trained and PBP-joint utility heads.

External code is resolved through `REPOS_DIR`; new artifacts go to `PBP_WORK`
(default `outputs/pbp`). Saved published results are in `results/pbp/` and
`results/c2/feature_linkability_results.csv`.

For local retained C2 artifacts, first check availability:

```bash
python pbp_integration/eval_pbp_action_utility.py --preflight \
  --splits test_original_c2r2 \
  --aam artifacts/pbp/saved_models/pbp_aam_ntu.pth \
  --clean-head artifacts/features/action_videomae_v2/models/action_videomae_v2_head.pt \
  --aggregates-dir artifacts/features/action_videomae_v2/aggregates
```

Remove `--preflight` to evaluate, optionally setting `--output` to a new CSV path.
This requires the optional local binaries and pinned upstream PBP source; they
are not bundled in the public source package. Only C2 aggregates are retained,
so explicitly selecting C2 is important.

`clean_trained` evaluates zero-shot transfer under the original clean-trained
head. `pbp_joint` evaluates the jointly trained deployment head; its clean row is
a same-head diagnostic, not a separately trained clean baseline.

Synthetic CPU regression test (with PyTorch, NumPy, h5py, pytest and PBP source):

```bash
python -m pytest -q -p no:cacheprovider pbp_integration/test_eval_pbp_action_utility.py
```
