#!/usr/bin/env bash
#SBATCH --job-name=accv-dp2-adapted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 13_extract_videomae_features.py \
  --aggregate-only \
  --methods deepprivacy2 \
  --splits train_original_r1_allcams,val_original_c1r2,test_original_c2r2 \
  --rebuild-aggregates
python -u 21_run_adapted_action_utility.py \
  --methods deepprivacy2 \
  --test-splits test_original_c2r2 \
  --device cuda \
  --epochs 80
