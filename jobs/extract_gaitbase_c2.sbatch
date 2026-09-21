#!/usr/bin/env bash
#SBATCH --job-name=accv-gaitbase
#SBATCH --array=0-31%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"
export EVAL_ANON_METHODS="deepprivacy2"

cd "$ACCV_ROOT/gait_integration"
python -u eval_gait_identity_sharded.py \
  --methods deepprivacy2 \
  --num-shards 32 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"

