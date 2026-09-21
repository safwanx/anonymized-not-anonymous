#!/usr/bin/env bash
#SBATCH --job-name=accv-videomae-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
source "${ACCV_ROOT:-${SLURM_SUBMIT_DIR:?Submit using tools/submit_job.py}}/jobs/accv_dp2_env.sh"

cd "$ACCV_ROOT/pipeline"
python -u 13_extract_videomae_features.py \
  --aggregate-only \
  --methods original,deepprivacy2 \
  --splits train_original_r1_allcams,val_original_c1r2,test_original_c2r2 \
  --rebuild-aggregates
python -u 14_train_action_videomae_head.py --epochs 80
python -u 15_eval_action_videomae_head.py \
  --methods original,deepprivacy2 \
  --splits test_original_c2r2 \
  --rebuild-aggregates
