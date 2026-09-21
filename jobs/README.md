# SLURM templates

Activate your experiment environment and restore the data/model inputs first.
Submit from any directory using `tools/submit_job.py` with an explicit partition.
The wrapper resolves the repository root and creates a log directory before
calling `sbatch`; no username, site-specific partition, or conda path is embedded.

```bash
python tools/submit_job.py --partition YOUR_GPU_PARTITION --dry-run jobs/extract_c2_frames.sbatch
```

Inspect resource requests and prerequisites, then remove `--dry-run` when ready.
Pass a SLURM dependency using `--dependency afterok:JOB_ID`. Submission does not
imply completion: inspect each stage's completeness checks before consuming it.
`accv_dp2_env.sh` preserves caller overrides, including `REPOS_DIR`, `MODELS_DIR`,
`PILOT_OUTPUT_ROOT`, `PBP_WORK`, and `GAIT_WORK`.

The experiment templates are useful reproducibility code and are retained. Old
execution logs and cleanup-only jobs are not part of the release. Fresh logs are
runtime output under `outputs/logs/`, excluded from the public source package.
