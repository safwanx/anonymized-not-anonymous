#!/usr/bin/env bash
#SBATCH --job-name=accv-arcface
#SBATCH --array=0-79%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"
export EVAL_ANON_METHODS="deepprivacy2"
export INSIGHTFACE_RUNTIME_ROOT="$PILOT_OUTPUT_ROOT/model_runtime/insightface"
export FACE_GPU_MEM_GB="6"

cd "$ACCV_ROOT/pipeline"
python -u 05_eval_face_recognition.py \
  --methods deepprivacy2 \
  --num-shards 80 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
