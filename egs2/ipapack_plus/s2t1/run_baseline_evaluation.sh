#!/usr/bin/env bash
#SBATCH --job-name=powsmeval
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

# --- user vars ---
CONDA_ENV="powsm_evals"
PROJ="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1"
DATASET="buckeye"
MODEL_ID="zipa"

# # call this script like sbatch run_baseline_evaluation.sh --dataset D --model M to override defaults
# for M in 'facebook/wav2vec2-lv-60-espeak-cv-ft' 'facebook/wav2vec2-xlsr-53-espeak-cv-ft' \
#     'ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns' 'allophant' 'allosaurus'; do

#     for D in 'aishell' 'buckeye' 'cv' 'doreco' 'epadb' 'fleurs' \
#             'fleurs_indv' 'kazakh' 'l2arctic' 'librispeech' \
#             'mls_dutch' 'mls_french' 'mls_german' \
#             'mls_italian' 'mls_polish' 'mls_portuguese' \
#             'mls_spanish' 'southengland' 'speechoceannotth' \
#             'tamil' 'tusom2021' 'voxangeles'; do
#             sbatch run_baseline_evaluation.sh --dataset $D --model $M
#     done 
# done

# ------------------

set -euo pipefail

source /u/sbharadwaj/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Evaluation requires panphon==0.22.* (no phonemizer/espeak needed)
python - <<'PY'
import pkgutil, sys
import subprocess
try:
    import panphon
    assert panphon.__version__.startswith("0.22.1"), f"need panphon 0.22.1, got {panphon.__version__}"
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "panphon==0.22.1"])
PY

cd "$PROJ"

srun -u python baselines/evaluate.py \
  --dataset "${DATASET}" --model "${MODEL_ID}" --workers $SLURM_CPUS_PER_TASK "$@"