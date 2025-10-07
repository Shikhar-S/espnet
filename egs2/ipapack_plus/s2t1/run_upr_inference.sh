#!/usr/bin/env bash
#SBATCH --job-name=powsminference
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# --- user vars ---
CONDA_ENV="powsmesp"
PROJ="/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1"
ESPEAK_LIB="/work/nvme/bbjs/sbharadwaj/powsm/espeak-ng/installed/lib/libespeak-ng.so.1.1.51"
MODEL_ID="powsm_ctc3_beam1"

# DATASET="l2arctic_perceived"
# DATASET="southengland"
# DATASET="buckeye"
# DATASET="voxangeles"
# DATASET="tusom2021"
# DATASET="epadb"
# DATASETS=("southengland" "buckeye" "voxangeles" "tusom2021" "epadb")
DATASET="doreco"

# call this script like sbatch run_baseline_inference.sh --dataset D --model M to override defaults
# ------------------

# ---- POWSM VARS ----
POWSM_BPE_MODEL=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/data/token_list/bpe_unigram40000_mega/bpe.model

############################################
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/exp_bigpr/s2t_train_s2t_transformer_mask_norm_raw_bpe40000/valid.acc.ave_5best.till40epoch.pth
# POWSM_BPE_MODEL=/work/nvme/bbjs/sbharadwaj/powsm/espnet/egs2/ipapack_plus/s2t1/data/token_list/bpe_unigram40000/bpe.model

# # standard model
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_raw_bpe40000_mega/valid.acc.ave_5best.till45epoch.pth
# CKPT_TAG=0926

# # standard model but only last checkpoint instead of averaging 5
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_raw_bpe40000_mega/45epoch.pth
# CKPT_TAG=0926ep45

##### Increasing ctc weight curriculum ####
# finetuned with ctc=0.5, till 50 ep
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega/50epoch.pth
# CKPT_TAG=0926ftctc5ep50

# # then finetuned with ctc=0.7, till 55 ep
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega/55epoch.pth
# CKPT_TAG=0926ftctc7ep55

# ##### OPPOSITE CURRICULUM #####
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ctc7_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ctc7_raw_bpe40000_mega/45epoch.pth
# CKPT_TAG=0926ft_decreasingctc7_ep45

# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ctc7_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_ctc7_raw_bpe40000_mega/valid.acc.ave_5best.till45epoch.pth
# CKPT_TAG=0926ft_decreasingctc7_av4ep45


# #### RANDOM CTC WEIGHTS ####
# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_randctc_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_randctc_raw_bpe40000_mega/valid.acc.ave_5best.till45epoch.pth
# CKPT_TAG=0926ftrandomctc_av5

# POWSM_S2T_TRAIN_CONFIG=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_randctc_raw_bpe40000_mega/config.yaml
# POWSM_MODEL_FILE=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_0926/s2t_train_s2t_transformer_deltaai_panphon_randctc_raw_bpe40000_mega/45epoch.pth
# CKPT_TAG=0926ftrandomctc_ep45
############################################

###### 1k exp 1 task models #####
# BASE_EXP_DIR=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_1task_1k
# EXP_100M=s2t_1kexp_panphon_100m_raw_bpe40000_mega
# EXP_300M=s2t_1kexp_panphon_raw_bpe40000_mega

# POWSM_S2T_TRAIN_CONFIG=${BASE_EXP_DIR}/${EXP_300M}/config.yaml
# POWSM_MODEL_FILE=${BASE_EXP_DIR}/${EXP_300M}/45epoch.pth
# CKPT_TAG=1task_1kexp300m

# POWSM_S2T_TRAIN_CONFIG=${BASE_EXP_DIR}/${EXP_100M}/config.yaml
# POWSM_MODEL_FILE=${BASE_EXP_DIR}/${EXP_100M}/45epoch.pth
# CKPT_TAG=1task_1kexp100m

##### 4 task 1k exp models #####

# BASE_EXP_DIR=/work/nvme/bbjs/cli22/espnet/egs2/ipapack_plus/s2t1/exp_dtai_1k

# POWSM_S2T_TRAIN_CONFIG=${BASE_EXP_DIR}/${EXP_100M}/config.yaml
# POWSM_MODEL_FILE=${BASE_EXP_DIR}/${EXP_100M}/45epoch.pth
# CKPT_TAG=1kexp100m

# POWSM_S2T_TRAIN_CONFIG=${BASE_EXP_DIR}/${EXP_300M}/config.yaml
# POWSM_MODEL_FILE=${BASE_EXP_DIR}/${EXP_300M}/45epoch.pth
# CKPT_TAG=1kexp300m
############################################


BEAM_SIZE=1
CTC_WEIGHT=3

MODEL_ID=powsm_${CKPT_TAG}_ctc${CTC_WEIGHT}_beam${BEAM_SIZE}
# --------------------

set -euo pipefail

# Activate env
source /u/sbharadwaj/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Inference requires phonemizer + panphon==0.20
export PHONEMIZER_ESPEAK_LIBRARY="${ESPEAK_LIB}"

cd "$PROJ"

POWSM_S2T_TRAIN_CONFIG=dummy
POWSM_MODEL_FILE=dummy
echo "Using model ID: ${MODEL_ID} with config: ${POWSM_S2T_TRAIN_CONFIG} and model file: ${POWSM_MODEL_FILE}"

srun -u python baselines/run_inference.py \
    --dataset "${DATASET}" \
    --model "${MODEL_ID}" \
    --num_workers 20 \
    --s2t_train_config "${POWSM_S2T_TRAIN_CONFIG}" \
    --s2t_model_file "${POWSM_MODEL_FILE}" \
    --bpemodel "${POWSM_BPE_MODEL}" \
    --beam_size ${BEAM_SIZE} \
    --ctc_weight 0.${CTC_WEIGHT} "$@"