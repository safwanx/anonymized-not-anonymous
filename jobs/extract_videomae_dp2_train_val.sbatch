#!/usr/bin/env bash
#SBATCH --job-name=accv-videomae-dp2-train
#SBATCH --array=0-7%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"
export HF_HOME="$MODELS_DIR/huggingface/cache"
export HF_HUB_OFFLINE="1"
unset HF_HUB_ENABLE_HF_TRANSFER

cd "$ACCV_ROOT/pipeline"
python -u 13_extract_videomae_features.py \
  --model-id "$MODELS_DIR/huggingface/OpenGVLab--VideoMAEv2-Base" \
  --methods deepprivacy2 \
  --splits train_original_r1_allcams,val_original_c1r2 \
  --batch-size 4 \
  --num-frames 16 \
  --precision auto \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
