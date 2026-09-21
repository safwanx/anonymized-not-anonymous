#!/usr/bin/env bash
#SBATCH --job-name=accv-pose-temporal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 23_train_eval_rgb_pose_temporal_identity.py \
  --protocol crossview_cam2 \
  --methods original,deepprivacy2 \
  --epochs 80 \
  --patience 10 \
  --seed 42
