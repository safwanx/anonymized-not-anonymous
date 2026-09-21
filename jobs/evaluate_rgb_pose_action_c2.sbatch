#!/usr/bin/env bash
#SBATCH --job-name=accv-pose-action-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 22_train_eval_rgb_pose_action.py \
  --methods original,deepprivacy2 \
  --epochs 80 \
  --patience 12 \
  --seed 42
