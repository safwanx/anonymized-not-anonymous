#!/usr/bin/env bash
#SBATCH --job-name=accv-rgb-pose-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 18_train_rgb_pose_identity.py --protocols crossview_cam2
python -u 19_eval_rgb_pose_identity.py \
  --methods original,deepprivacy2 \
  --protocols crossview_cam2
