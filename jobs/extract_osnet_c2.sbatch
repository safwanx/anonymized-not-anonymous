#!/usr/bin/env bash
#SBATCH --job-name=accv-osnet
#SBATCH --array=0-39%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"
export EVAL_ANON_METHODS="deepprivacy2"

cd "$ACCV_ROOT/pipeline"
python -u 06_eval_person_reid_sharded.py \
  --methods deepprivacy2 \
  --num-shards 40 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
