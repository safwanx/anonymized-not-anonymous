#!/usr/bin/env bash
#SBATCH --job-name=accv-pbp-c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT"
python -u pbp_integration/export_protocol_features_to_pbp.py \
  --splits train_original_r1_allcams \
  --frame-feat-dir "$PBP_WORK/frame_features/train"
python -u pbp_integration/export_protocol_features_to_pbp.py \
  --splits test_original_c2r2
python -u pbp_integration/train_pbp_aam_ntu.py \
  --train-split train_original_r1_allcams \
  --epochs 100 \
  --batch-size 256 \
  --seed 1337 \
  --run-id pbp_aam_ntu
python -u pbp_integration/eval_pbp_feature_linkability.py \
  --aam "$PBP_WORK/saved_models/pbp_aam_ntu.pth"
python -u pbp_integration/eval_pbp_action_utility.py \
  --aam "$PBP_WORK/saved_models/pbp_aam_ntu.pth" \
  --splits test_original_c2r2
