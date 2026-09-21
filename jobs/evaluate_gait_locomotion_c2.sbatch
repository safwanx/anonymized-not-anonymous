#!/usr/bin/env bash
#SBATCH --job-name=accv-gait-locomotion
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/gait_integration"
python -u eval_gait_locomotion_subset.py --methods deepprivacy2
