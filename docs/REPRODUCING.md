# Reproducing the evaluation

## What can be checked without data?

The standard-library tests in `analysis/` reconstruct gallery sizes and
action-conditioned guessing baselines from the supplied protocol. The manuscript
table builder checks coverage products, sample counts, and nominal Wilson
intervals against saved CSVs. These are evidence checks, not a new independent
replication of the underlying experiments.

`results/baselines/` covers the original four protocols. `results/c2/` contains
the matched C2 learned-anonymization and control experiments. Do not silently
combine these populations or treat the three-identity cross-range diagnostic as
large-gallery evidence. See `results/README.md` for the table mapping.

## Paths

Run commands from the repository root with the appropriate Python environment
activated. `ACCV_ROOT` is normally inferred from the source location. Optional
overrides are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PILOT_ROOT` | `protocol/` | Metadata and split manifests. |
| `NTU_ROOT` | repository root | Your separately obtained NTU media location. |
| `PILOT_OUTPUT_ROOT` | `outputs/` | New frames, features, and evaluation outputs. |
| `ACCV_CACHE` | `~/.cache/accv` | External source and model cache. |
| `REPOS_DIR` | `$ACCV_CACHE/repos` | External implementations. |
| `MODELS_DIR` | `$ACCV_CACHE/models` | Separately obtained pretrained models. |
| `PBP_WORK`, `GAIT_WORK` | `outputs/pbp`, `outputs/gait` | Integration runtime outputs. |
| `GAITBASE_CKPT` | model-cache search | Explicit CASIA-B GaitBase checkpoint. |
| `TECTONIC` | executable on PATH, then local cache | Paper compiler. |

`PILOT_*` environment names are retained as API names; the supplied protocol
folder is now named `protocol/`. Metadata records portable dataset filenames.
The released splits are fixed; do not overwrite them.

## GPU reproduction

1. Obtain the NTU RGB+D 120 data and required model weights under their owners'
   terms. Point `NTU_ROOT` to the dataset and restore weights into `MODELS_DIR`.
2. Follow `environments/README.md` to prepare the appropriate runtime. Prepopulate
   caches on a network-enabled machine before using offline SLURM templates.
3. Validate metadata and extract the required original frames (`pipeline/01_*`,
   `02_*`). Detection (`03_*`) precedes RGB anonymization (`04_*`).
4. Run the endpoint-specific feature extraction and evaluation: ArcFace, OSNet,
   RGB pose, GaitBase, or VideoMAEv2. Each needs the appropriate gallery and probe
   inputs; requesting only anonymized probes is not sufficient.
5. For DP2-adapted action utility, also synthesize training and validation data,
   extract features, and run `21_run_adapted_action_utility.py`. Probe-only synthesis
   cannot train an adapted head. For PBP, use its separate feature-space workflow.

The supplied SLURM scripts capture the completed C2 run configurations, but their
partition and Python activation are intentionally site-independent. For example:

```bash
python tools/submit_job.py --partition YOUR_GPU_PARTITION --dry-run jobs/extract_c2_frames.sbatch
```

Remove `--dry-run` only after inspecting inputs, GPU resources, and output paths.
The wrapper creates `outputs/logs/` before submission. Use SLURM dependencies
between completed arrays; inspect completeness before starting dependent stages.
No job is submitted by any setup or validation command in this repository.

## Storage and interpretation

Generate only the splits and endpoints needed: intermediate original/anonymized
frames can consume hundreds of GiB. Keep result CSVs, split manifests, and trained
heads; delete regenerable frame caches only after validating completed outputs.
No automatic deletion is performed by the public tools.

DP2 means DeepPrivacy2, not differential privacy. Low attack success is evidence
against the tested attack, not a guarantee of anonymity. Report descriptor
coverage with conditional success. The paper's Wilson intervals are nominal
binomial intervals over covered probes, not identity-clustered or multi-seed
uncertainty estimates. PBP releases features, so pixel-space attacks do not
provide a matched comparison for its output.
