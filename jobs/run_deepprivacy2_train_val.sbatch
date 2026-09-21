#!/usr/bin/env bash
#SBATCH --job-name=accv-dp2-train
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 04_deepprivacy2_probe_only.py \
  --protocols "" \
  --no-include-action-tests \
  --action-splits train_original_r1_allcams,val_original_c1r2 \
  --status-tag train_val \
  --num-shards 4 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
