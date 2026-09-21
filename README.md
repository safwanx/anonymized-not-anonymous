# Anonymized but Not Anonymous

**Multi-Cue Identity Leakage in Privacy-Preserving Action Recognition**

Safwan Nabeel · Farah AlShiha · Muzammil Behzad  
ACCV 2026

Evaluation code, protocol manifests, and saved results for our study
of identity leakage after video anonymization. We evaluate face, appearance,
RGB-extracted pose, and temporal-silhouette attacks alongside action utility.
DeepPrivacy2 is an image anonymizer; **it does not provide differential privacy**.
Privacy Beyond Pixels is evaluated separately in feature space.

[Results](results/README.md) · [Reproduction](docs/REPRODUCING.md) ·
[Dependencies](environments/README.md)

## Quick start: no dataset or GPU required

With Python 3.10 or newer, from this directory:

```bash
python -m unittest discover -s analysis -p 'test_*.py' -v
python tools/check_project.py
```

These checks validate protocol composition, corrected chance controls, saved
result integrity, and repository structure. They do not rerun GPU experiments.
Regenerate the CPU audit separately with:

```bash
python analysis/protocol_audit.py --output-dir outputs/protocol_audit
```

## Layout

| Directory | Contents |
| --- | --- |
| `pipeline/` | Extraction, anonymization, attack, and utility stages. |
| `gait_integration/`, `pbp_integration/` | GaitBase and PBP integrations. |
| `analysis/` | Protocol/chance audit, coverage decomposition, output verification. |
| `protocol/` | Filename metadata and fixed splits; no videos or skeletons. |
| `results/` | Baseline, matched-C2, PBP, analysis, and synthesis-timing tables. |
| `jobs/` | Portable SLURM experiment templates. |
| `environments/` | Dependencies, pinned external sources, compatibility patches. |
| `tools/` | Protocol reconstruction, dependency setup, checks, job submission. |
| `artifacts/` | Optional local heads and aggregates; binaries excluded from release. |

New experiments write to `outputs/`, not the published results. External sources
and model downloads default to `~/.cache/accv/`, outside this checkout.

## Data and availability

NTU RGB+D 120 videos, skeletons, frames, detection caches, and pretrained weight
downloads are **not included**. Obtain them from their owners under the relevant
terms. Saved CSVs support table/protocol checks without these inputs. Full
pixel-based reproduction requires restoring them; a fresh end-to-end GPU
installation has not been validated for this release layout.

Local `artifacts/` retains six Original/DeepPrivacy2 train, validation, and C2
aggregate files and six trained heads/adapters—not a complete set of all methods
or C3 aggregates. Dataset-derived binaries are excluded from the public source
package. See [artifact notes](artifacts/README.md).

## Citation and license

See [CITATION.cff](CITATION.cff) for the author/title metadata. Add the official
paper URL/DOI when available; none is fabricated here.

Original code is released under the [MIT License](LICENSE). Dataset media,
pretrained weights, external code, and publisher/template materials are not
relicensed by that file. See [third-party notices](THIRD_PARTY.md).

We thank King Fahd University of Petroleum and Minerals (KFUPM) for its support.
