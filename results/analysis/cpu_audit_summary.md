# Protocol audit: pose chance and identity-pool sensitivity

Generated reproducibly from `protocol/metadata.csv`, the fixed split files, and existing final CSVs with `python analysis/protocol_audit.py`.

## Split and chance audit

| Protocol | Gallery IDs | Exact IDs | Naive chance | Action-conditioned BA null | Sample-weighted null |
|---|---:|---|---:|---:|---:|
| crossview_cam2 | 38 | P001 P002 P006 P007 P008 P011 P015 P016 P017 P018 P019 P025 P026 P027 P028 P038 P041 P043 P044 P048 P049 P051 P067 P074 P075 P076 P080 P081 P085 P088 P089 P090 P091 P092 P093 P094 P095 P096 | 2.63% | 4.86% | 5.01% |
| crossview_cam3 | 38 | P001 P002 P006 P007 P008 P011 P015 P016 P017 P018 P019 P025 P026 P027 P028 P038 P041 P043 P044 P048 P049 P051 P067 P074 P075 P076 P080 P081 P085 P088 P089 P090 P091 P092 P093 P094 P095 P096 | 2.63% | 4.86% | 5.01% |
| crosssetup | 18 | P001 P007 P008 P015 P016 P017 P018 P019 P025 P027 P028 P041 P043 P044 P048 P067 P075 P088 | 5.56% | 10.51% | 10.53% |
| crossrange | 3 | P006 P008 P011 | 33.33% | 33.33% | 33.33% |

For cross-view, actions A001-A060 have 16 eligible identities and A061-A120 have 25; the overlap consists of bridge identities. For cross-setup the corresponding pools are 11 and 8. Thus action/action-range information narrows the identity candidate set before any identity-bearing pose is used. The primary null above is class-balanced because the reported pose metric is balanced accuracy; the sample-weighted null is included to make the averaging convention explicit.

Cross-range is different: gallery A001-A060 and probe A061-A120 are disjoint, and all three bridge identities span both ranges. Action-range knowledge therefore leaves all three candidates and the null remains 33.33%.

## Original RGB pose result under the corrected null

| Protocol | Observed balanced accuracy | Naive ratio | Corrected null | Corrected ratio |
|---|---:|---:|---:|---:|
| crossview_cam2 | 9.88% | 3.75x | 4.86% | 2.03x |
| crossview_cam3 | 4.21% | 1.60x | 4.86% | 0.87x |
| crosssetup | 9.63% | 1.73x | 10.51% | 0.92x |
| crossrange | 30.28% | 0.91x | 33.33% | 0.91x |

Only cross-view C2 clearly exceeds the corrected action-aware point null for original RGB (9.88% vs 4.86%, 2.03x). C3 (4.21% vs 4.86%), cross-setup (9.63% vs 10.51%), and cross-range (30.28% vs 33.33%) do not. Among the other full-coverage methods, face blur and body blur also exceed the point null on C2. Body blur is only marginally above it on cross-setup (11.02% vs 10.51%), which should not be called evidence without a valid test.

An action-stratified permutation p-value cannot be reconstructed: no per-sample pose predictions, logits, or features are present locally. Aggregate accuracies are insufficient to preserve the dependence among identity, action, and prediction. For body pixelation, the metadata-only corrected null is additionally approximate because the identities/actions of the covered subset are unavailable.

## Identity-pool and extrapolation audit

The shipped galleries contain 38 identities for each cross-view split, 18 for cross-setup, and 3 for cross-range. Re-ID and pose values should be shown per protocol rather than averaged across these incomparable candidate sets. Cross-range is a diagnostic stress test only: its three identities, fixed within-session clothing, and 33.33% chance make its leakage magnitude unsuitable for transfer claims.

| Protocol | Gallery IDs | Original re-ID Rank-1 | Original pose BA | Corrected pose null |
|---|---:|---:|---:|---:|
| crossview_cam2 | 38 | 85.31% | 9.88% | 4.86% |
| crossview_cam3 | 38 | 80.61% | 4.21% | 4.86% |
| crosssetup | 18 | 45.26% | 9.63% | 10.51% |
| crossrange | 3 | 62.08% | 30.28% | 33.33% |

The CSV `protocol_results_unaveraged.csv` contains the corresponding per-method rows. The apparently high cross-range re-ID Rank-1 (62.08%) must be read against only three candidates and the clothing/session confound.

| Adversary | Split | Largest empirical N | Rank-1 there | Extrapolated N | Extrapolated Rank-1 |
|---|---|---:|---:|---:|---:|
| reid | crossview_cam2 | 38 | 0.853 | 106 | 0.808 |
| reid | crossview_cam3 | 38 | 0.806 | 106 | 0.743 |
| reid | crosssetup | 15 | 0.470 | 106 | 0.095 |
| face | crossview_cam2 | 38 | 0.832 | 106 | 0.776 |
| face | crossview_cam3 | 38 | 0.524 | 106 | 0.372 |
| face | crosssetup | 15 | 0.897 | 106 | 0.767 |

The manuscript's 0.74-0.81 re-ID and 0.37-0.78 face ranges correctly select the two cross-view 106-identity log-fit projections. They are not measurements on 106 people. The empirical evidence is identity subsampling up to N=38 (cross-view; the full N=38 point has one realization) or N=15 (cross-setup); N=106 is extrapolation by 2.79x or 7.07x in pool size. In-range R-squared does not validate the assumed log trend outside the observed range.

## Rebuttal-ready wording

> We thank the reviewers for identifying the small-pool and pose-control issues. Re-auditing the fixed splits, we find that action range itself narrows the cross-view identity pool (16 identities for A001-A060 and 25 for A061-A120), raising the class-balanced pose null from 2.63% to 4.86%; for cross-setup it rises from 5.56% to 10.51%. Under this stronger null, original RGB pose remains above the point null on C2 (9.88%, 2.03x), but not on C3, cross-setup, or cross-range, so we narrow our pose claim accordingly. We also now characterize cross-range as a three-identity diagnostic stress test and clarify that the 106-identity values are log-fit projections from empirical subsampling (up to 38 identities), not measurements on a 106-person gallery.
