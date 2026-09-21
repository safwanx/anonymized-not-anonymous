#!/usr/bin/env bash
#SBATCH --job-name=accv-rgb-pose
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 17_extract_rgb_pose_keypoints.py \
  --methods original,deepprivacy2 \
  --protocols crossview_cam2 \
  --pose-model "$MODELS_DIR/ultralytics/yolo11m-pose.pt" \
  --sample-frames 32 \
  --batch-size 32 \
  --min-valid-frames 2 \
  --keypoint-conf 0.05 \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
