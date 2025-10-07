#!/usr/bin/env bash
#SBATCH --job-name=allophant
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# --- user vars ---
CONDA_ENV="allophant"
PROJ="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1"
DATASET="buckeye"
MODEL_ID="allophant"
# call this script like sbatch run_baseline_inference.sh --dataset D --model M to override defaults
# ------------------

set -euo pipefail

# Activate env
source /u/sbharadwaj/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$PROJ"

srun -u python baselines/run_inference.py \
  --dataset "${DATASET}" \
  --model "${MODEL_ID}" \
  --num_workers 15 "$@"