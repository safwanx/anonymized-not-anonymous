#!/usr/bin/env bash
#SBATCH --job-name=accv-dp2-check
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"
python -u "$ACCV_ROOT/analysis/verify_deepprivacy2_outputs.py"
