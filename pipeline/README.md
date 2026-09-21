# Evaluation pipeline

Run from the repository root with your experiment environment activated. Shared
configuration is in `pipeline_common.py`; data/weight setup and path overrides
are documented in `../docs/REPRODUCING.md`.

| Stages | Role |
| --- | --- |
| `01_*`, `02_*`, `03_*` | Validate metadata, extract frames, detect people/faces. |
| `04_*` | RGB blur/pixelation and DeepPrivacy2 synthesis. |
| `05_*`, `06_*` | Face and appearance identity attacks. |
| `07_*`, `08_*`, `09_*` | Native-skeleton/silhouette/lightweight utility proxies; not the primary RGB-pose/GaitBase/VideoMAEv2 endpoints. |
| `12_*`–`16_*` | VideoMAEv2 splits, frozen features, clean utility head, evaluation. |
| `17_*`–`20_*` | RGB pose extraction, MLP identity training, evaluation. |
| `21_*` | Gallery-size sensitivity and method-adapted action utility. |
| `22_*`, `23_*` | Pose action utility and temporal pose identity control. |

The numerical prefixes group stages; they are not a command to run every script
in lexical order. Parallel/sharded variants are alternatives, not additional
required passes. GaitBase and PBP have sibling integration directories.

Typical baseline preparation, after restoring licensed inputs and model weights:

```bash
python pipeline/01_validate_metadata.py
python pipeline/02_extract_frames.py
python pipeline/03_detect_people_faces_parallel.py --gpus 0 --limit 20
```

The limited detection command is a smoke test, not a complete experiment.
Remove the limit and choose anonymization/evaluation stages deliberately. GPU
commands must run on a compute node, not a shared login node.

VideoMAEv2 utility uses stages 12–15. RGB-pose identity uses stages 17–19. The
DeepPrivacy2 adapted head additionally requires method-matched train/validation
features; C2 probe features alone are insufficient. Run
`21_run_adapted_action_utility.py --dry-run` to inspect missing inputs first.

New results are written under `outputs/features/`, leaving `results/` unchanged.
Shared pipeline imports validate the raw RGB/skeleton paths and require restored
media, even for many stage-level `--help` commands. They may also create empty
runtime folders. Use the standalone `analysis/` commands for data-free checks;
model-backed extraction is not part of that lightweight CPU test.
