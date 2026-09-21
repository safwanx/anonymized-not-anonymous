#!/usr/bin/env bash
#SBATCH --job-name=accv-action-frames
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 02_extract_action_frames.py \
  --splits train_original_r1_allcams,val_original_c1r2 \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
