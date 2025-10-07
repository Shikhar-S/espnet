# DATASETS=(l2arctic_perceived tusom2021 voxangeles)

#### Multi-task 100M and 350M ####
# BASE_PATHS=(exp_dtai_2task_1k) 
# SUB_PATHS=(s2t_1kexp_panphon_100m_raw_bpe40000_mega s2t_1kexp_panphon_raw_bpe40000_mega)

# BASE_PATHS=(exp_dtai_1task exp_dtai_2task)
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_raw_bpe40000_mega)

# BASE_PATHS=(exp_dtai_0926 exp_dtai_2task exp_dtai_1task)
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_100m_raw_bpe40000_mega) # averaged over 5 best checkpoints till 45 epochs

# CHECKPOINT_NAME=valid.acc.ave_5best.till45epoch.pth
# CTC_WEIGHT=3
####################################################################################
###### CTC curriculum models #####
# BASE_PATHS=(exp_dtai_0926)

# # Decreasing
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_ctc7_raw_bpe40000_mega)
# CHECKPOINT_NAME=valid.acc.ave_5best.till45epoch.pth
# # CHECKPOINT_NAME=45epoch.pth
# CTC_WEIGHT=7

# # Increasing
# ##### Increasing ctc weight curriculum ####
# # finetuned with ctc=0.5, till 50 ep
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega)
# CHECKPOINT_NAME=50epoch.pth
# CTC_WEIGHT=5

# # then finetuned with ctc=0.7, till 55 ep
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega)
# CHECKPOINT_NAME=55epoch.pth
# CTC_WEIGHT=7
################################################################################

# 1B models
# BASE_PATHS=(exp_dtai_1task exp_dtai_2task exp_dtai_0926)
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_1b_raw_bpe40000_mega)
# CHECKPOINT_NAME=32epoch.pth
# CTC_WEIGHT=3
################################################################################

# DATASETS=(mls_italian mls_polish voxangeles tusom2021 southengland epadb)
# BASE_PATHS=(exp_dtai_0926)
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_ftctc53_raw_bpe40000_mega)
# CHECKPOINT_NAME=50epoch.pth
# CTC_WEIGHT=3

# BASE_PATHS=(exp_dtai_0926)
# SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega)
# CHECKPOINT_NAME=45epoch.pth
# CTC_WEIGHT=3

BASE_PATHS=(exp_dtai_0926)
SUB_PATHS=(s2t_train_s2t_transformer_deltaai_panphon_ftctc57_raw_bpe40000_mega)
CHECKPOINT_NAME=50epoch.pth
CTC_WEIGHT=3


DATASETS=(mls_italian mls_polish voxangeles tusom2021 southengland epadb l2arctic_perceived tusom2021 voxangeles)

BEAM_SIZE=1
for base_path in ${BASE_PATHS[@]}; do
    for sub_path in ${SUB_PATHS[@]}; do
        for D in ${DATASETS[@]};do 
            # echo "Running inference for dataset: $D, base_path: $base_path, sub_path: $sub_path"
            POWSM_S2T_TRAIN_CONFIG=${base_path}/${sub_path}/config.yaml
            POWSM_MODEL_FILE=${base_path}/${sub_path}/${CHECKPOINT_NAME}
            CKPT_TAG=${base_path}_${sub_path}_ck50 #5avtill45

            # ls $POWSM_MODEL_FILE
            # ls $POWSM_S2T_TRAIN_CONFIG
            MODEL_ID=powsm_${CKPT_TAG}_infctc${CTC_WEIGHT}_beam${BEAM_SIZE}
            sbatch --job-name new_powsminference run_upr_inference.sh --dataset $D \
                --model "${MODEL_ID}" \
                --num_workers 20 \
                --s2t_train_config "${POWSM_S2T_TRAIN_CONFIG}" \
                --s2t_model_file "${POWSM_MODEL_FILE}" \
                --beam_size ${BEAM_SIZE} \
                --ctc_weight 0.${CTC_WEIGHT} "$@"
        done
    done
done