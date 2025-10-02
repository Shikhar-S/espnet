#!/usr/bin/env bash
#SBATCH --job-name=powsminference
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out

# --- user vars ---
CONDA_ENV="powsm2"
PROJ="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1"
ESPEAK_LIB="/work/nvme/bbjs/sbharadwaj/powsm/espeak-ng/installed/lib/libespeak-ng.so.1.1.51"
DATASET="buckeye"
MODEL_ID="facebook/wav2vec2-lv-60-espeak-cv-ft"
# call this script like sbatch run_baseline_inference.sh --dataset D --model M to override defaults
# ------------------

set -euo pipefail

# Activate env
source /u/sbharadwaj/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Inference requires phonemizer + panphon==0.20
export PHONEMIZER_ESPEAK_LIBRARY="${ESPEAK_LIB}"

cd "$PROJ"

srun -u python baselines/run_inference.py \
  --dataset "${DATASET}" \
  --model "${MODEL_ID}" "$@"