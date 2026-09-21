#!/usr/bin/env bash
#SBATCH --job-name=accv-gait-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"
export EVAL_ANON_METHODS="deepprivacy2"
cd "$ACCV_ROOT/gait_integration"
python -u eval_gait_identity_sharded.py --methods deepprivacy2 --evaluate-only
