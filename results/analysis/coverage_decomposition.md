# Coverage decomposition

Each row reports success among probes with a valid descriptor, descriptor coverage, and their product (end-to-end attack success). Values below are percentages for `body_pixel`.

| Attack | Protocol | Conditional success | Coverage | End-to-end success |
|---|---|---:|---:|---:|
| face | crossview_cam2 | 0.65% | 4.25% | 0.03% |
| face | crossview_cam3 | 0.83% | 3.36% | 0.03% |
| face | crosssetup | 10.81% | 3.25% | 0.35% |
| face | crossrange | 30.00% | 4.17% | 1.25% |
| person_reid | crossview_cam2 | 49.42% | 100.00% | 49.42% |
| person_reid | crossview_cam3 | 35.69% | 100.00% | 35.69% |
| person_reid | crosssetup | 26.67% | 100.00% | 26.67% |
| person_reid | crossrange | 58.33% | 100.00% | 58.33% |
| temporal_silhouette_gaitbase | crossview_cam2 | 0.00% | 0.31% | 0.00% |
| temporal_silhouette_gaitbase | crossview_cam3 | 8.33% | 0.67% | 0.06% |
| temporal_silhouette_gaitbase | crosssetup | 0.00% | 0.00% | 0.00% |
| temporal_silhouette_gaitbase | crossrange | 0.00% | 0.00% | 0.00% |
| rgb_pose | crossview_cam2 | 0.77% | 21.64% | 0.17% |
| rgb_pose | crossview_cam3 | 7.60% | 22.31% | 1.69% |
| rgb_pose | crosssetup | 8.00% | 26.32% | 2.11% |
| rgb_pose | crossrange | 12.12% | 13.75% | 1.67% |

Interpretation: body pixelation's near-zero end-to-end face, pose, and temporal-silhouette scores are often dominated by descriptor extraction failure. Re-ID remains fully covered and therefore measures reduced discriminability directly.

The conditional pose metric here is ordinary identity accuracy because that is the quantity multiplied by coverage in the submitted evaluator. Balanced accuracy and its corrected action-conditioned null are reported separately in `pose_corrected_null.csv`.
