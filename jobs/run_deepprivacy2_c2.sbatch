#!/usr/bin/env bash
#SBATCH --job-name=accv-dp2-c2
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

# Populate model caches before submitting offline synthesis jobs.

num_shards="${SLURM_ARRAY_TASK_COUNT:-4}"
shard_index="${SLURM_ARRAY_TASK_ID:-0}"

cd "$ACCV_ROOT/pipeline"
python -u 04_deepprivacy2_probe_only.py \
  --protocols crossview_cam2 \
  --no-include-action-tests \
  --num-shards "$num_shards" \
  --shard-index "$shard_index"
