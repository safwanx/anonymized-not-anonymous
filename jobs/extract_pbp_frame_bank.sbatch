#!/usr/bin/env bash
#SBATCH --job-name=accv-pbp-bank
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

cd "$ACCV_ROOT"
python -u pbp_integration/extract_videomae_frame_features.py \
  --split train_original_r1_allcams \
  --pool 10 \
  --num-frames 16 \
  --batch-size 4 \
  --precision auto \
  --model-id "$MODELS_DIR/huggingface/OpenGVLab--VideoMAEv2-Base" \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --out "$PBP_WORK/frame_features/train"
