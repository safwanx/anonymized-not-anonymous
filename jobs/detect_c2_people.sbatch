#!/usr/bin/env bash
#SBATCH --job-name=accv-c2-detect
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

num_shards="${SLURM_ARRAY_TASK_COUNT:-4}"
shard_index="${SLURM_ARRAY_TASK_ID:-0}"

cd "$ACCV_ROOT/pipeline"
python -u 03_detect_protocol_people.py \
  --protocols crossview_cam2 \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --batch-size 16 \
  --model "$YOLO_MODEL"
