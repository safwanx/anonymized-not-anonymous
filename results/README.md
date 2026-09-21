# Saved experimental evidence

These compact CSV/JSON files are retained measurements, not automatically updated
when an experiment runs. New runs write to `outputs/`. Preserve the distinction
between original multi-protocol results and the matched C2 follow-up experiments.

| Evidence | Location | Interpretation |
| --- | --- | --- |
| Four-protocol RGB baselines and action utility | `baselines/` | Original, face blur, body blur, body pixelation. |
| Matched clean/DP2 identity attacks | `c2/`, `c2/rgb_pose_identity/` | C2 only; use coverage with conditional success. |
| DP2 clean-head and adapted action utility | `c2/action_videomae_v2/` | Same clean reference; adaptation is a separate head. |
| Temporal pose and locomotion controls | `c2/rgb_pose_identity/`, `c2/gait_identity_locomotion_results.csv` | Restricted endpoints/populations, not pooled protocols. |
| PBP feature linkability | `c2/feature_linkability_results.csv` | Feature-space attack; not RGB privacy equivalence. |
| PBP utility and training history | `pbp/` | Clean-trained and jointly trained heads use distinct references. |
| Corrected pose null and coverage decomposition | `analysis/` | Derived from released splits and baseline tables. |
| DP2 synthesis time | `runtime.csv` | Four A4500 shards; includes one already-cached video. |

The runtime rows sum to 650.5 GPU-minutes (10.84 GPU-hours); this is synthesis from
already-extracted frames, not full pipeline time or parallel wall time. Source-log
SHA-256 values preserve provenance after the verbose logs were removed.

`baselines/identity_pool_scaling_fit.csv` retains historical fitted projections
only so the extrapolation audit is reproducible. Projections beyond 38 identities
are **not measured experiments**; the manuscript's corrected plot shows measured
pools only. `pose_identity_results.csv` is the native-skeleton proxy, not the
anonymization-aware RGB-pose result. `silhouette_proxy_results.csv` is not GaitBase.

`MANIFEST.sha256` checks the retained numerical files. Absolute machine paths in
path-only CSV fields were made portable; numerical cells were not changed.
