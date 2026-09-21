#!/usr/bin/env bash
#SBATCH --job-name=accv-c2-frames
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

num_shards="${SLURM_ARRAY_TASK_COUNT:-4}"
shard_index="${SLURM_ARRAY_TASK_ID:-0}"
frame_role="${FRAME_ROLE:-both}"

cd "$ACCV_ROOT/pipeline"
python -u 02_extract_protocol_frames.py \
  --protocols crossview_cam2 \
  --role "$frame_role" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index"
