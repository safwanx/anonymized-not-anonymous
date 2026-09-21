# GaitBase identity adversary

The integration applies the CASIA-B-trained OpenGait GaitBase checkpoint to NTU
temporal silhouettes. It measures cross-dataset identity leakage, not a CASIA-B
benchmark or a gait model trained on NTU. The gallery is clean; probes use the
selected anonymization method.

Obtain the pinned OpenGait source using `tools/fetch_sources.py` and set
`GAITBASE_CKPT` to the separately obtained checkpoint. Frames and person detections
must be present before evaluation. New integration outputs default to
`outputs/gait`; numerical results use the pipeline's `outputs/features/`.

```bash
python gait_integration/eval_gait_identity.py \
  --methods original,deepprivacy2 --max-videos 0
```

`--max-videos 0` requests the full selected protocol. The default capped run is
only a diagnostic. By default silhouettes are extracted using Otsu thresholding
inside person crops. `extract_silhouette_masks.py` and `--mask-root` support a
separate segmentation-mask variant; do not switch silhouette sources silently
when comparing against saved results.

`eval_gait_identity_sharded.py` supports parallel feature extraction/evaluation.
`eval_gait_locomotion_subset.py` evaluates the locomotion subset; that subset's
sample count and population differ from the full C2 protocol. Retained tables
are in `results/baselines/` and `results/c2/`.
