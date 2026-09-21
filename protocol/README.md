# NTU120-Privacy-21K protocol

`metadata.csv` identifies 21,600 NTU RGB+D 120 videos: 9,000 in pool A,
11,880 in pool B, and 720 bridge samples. The selection contains 38 identities
and 120 actions. `splits/` supplies the exact fixed filename lists.

Gallery sizes are 38 for cross-view C2/C3, 18 for cross-setup, and 3 for
cross-range. The small cross-range pool is a diagnostic, not broad evidence
about large-population anonymity. The CPU audit derives these counts directly
from filename membership.

`rgb_path` and `skeleton_path` are relative locators, **not included files**.
`has_skeleton` records availability when the protocol was constructed, not current
disk availability. The dataset is not redistributed. Obtain it under NTU's terms
and configure `NTU_ROOT` to reproduce the pixel-based experiments.

The metadata checksum reflects its portable path fields; split-file checksums
are unchanged. Tests validate every row's encoded setup, camera, person,
replication, and action against its filename, plus the exact pool counts.

Do not replace these fixed manifests with whatever subset happens to exist on a
local machine.
