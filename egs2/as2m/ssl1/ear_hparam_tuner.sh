#!/bin/bash
set -e
set -u
set -o pipefail

# Default Hyperparameters
TOTAL_STEPS=400000

use_wandb=true

generate_deepspeed_config() {
    local learning_rate=$1
    local warmup_steps=$2

    cat <<EOF
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": ${learning_rate},
      "betas": [0.9, 0.98],
      "eps": 1e-12,
      "weight_decay": 1.0e-2,
      "adam_w_mode": true
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_type": "linear",
      "total_num_steps": ${TOTAL_STEPS},
      "warmup_num_steps": ${warmup_steps},
      "warmup_max_lr": ${learning_rate},
      "warmup_min_lr": 1.0e-6
    }
  },
  "wall_clock_breakdown": false,
  "steps_per_print": 3000
}
EOF
}


# wandb_project=EARlarge.PT
# wandb_project=BEATsTokenizerPT 1. change SSL tag here, 2. change ngpu to 3 or 4, 3. change model size in run_ear
# beats_iter2_large2.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000

# LARGE-2kvocab
# TOTAL_STEPS=800000
# for LEARNING_RATE in 1.0e-4; do
#   for WARMUP_STEPS in 40000; do
#     for BATCH_BINS in 1600000; do
#       ITER=0
#       SSL_TAG="large${ITER}.2kvocab.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
#       N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

#       deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
#       deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
#       echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
      
#       ./run_beatsv2_large_2kvocab.sh --ngpu 4 --ssl_tag "${SSL_TAG}" \
#           --train_start_iter ${ITER} --train_stop_iter ${ITER} \
#           --stage 7 --stop_stage 7 \
#           --n_targets 2048 \
#           --train_config conf/ear_large2k.yaml \
#           --tokenizer_train_config conf/tok_ear_large2k.yaml \
#           --tokenizer_inference_config conf/tokenizer_inf_ear_large2k.yaml \
#           --tokenizer_inference_batch_size 40 \
#           --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
#           --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar" &
#       # sleep 5s
#     done
#   done
# done

# wait


# # LARGE-4kvocab
# TOTAL_STEPS=800000
# for LEARNING_RATE in 1.0e-4; do
#   for WARMUP_STEPS in 40000; do
#     for BATCH_BINS in 1600000; do
#       ITER=0
#       SSL_TAG="large${ITER}.4kvocab.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
#       N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

#       deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
#       deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
#       # iter 1 tokenizers
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_tokenizer_iter1_base_tok.tune_lr1.0e-3_warmup40000_bins800000_totalsteps400000/epoch3.pt
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter1_large_tok.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter2_large_tok2.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt
#       echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
      
#       ./run_ear.sh --ngpu 4 --ssl_tag "${SSL_TAG}" \
#           --train_start_iter ${ITER} --train_stop_iter ${ITER} \
#           --stage 7 --stop_stage 7 \
#           --n_targets 4096 \
#           --train_config conf/ear_large4k.yaml \
#           --tokenizer_train_config conf/tok_ear_large4k.yaml \
#           --tokenizer_inference_config conf/tokenizer_inf_ear_large4k.yaml \
#           --tokenizer_inference_batch_size 40 \
#           --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
#           --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar" &
#       # sleep 5s
#     done
#   done
# done

# wait


# LARGE
# for LEARNING_RATE in 1.0e-4; do
#   for WARMUP_STEPS in 40000; do
#     for BATCH_BINS in 1600000; do
#       ITER=0
#       SSL_TAG="large${ITER}.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
#       N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

#       deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
#       deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
#       # iter 1 tokenizers
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_tokenizer_iter1_base_tok.tune_lr1.0e-3_warmup40000_bins800000_totalsteps400000/epoch3.pt
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter1_large_tok.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter2_large_tok2.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt
#       echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"

#       ./run_ear.sh --ngpu 4 --ssl_tag "${SSL_TAG}" \
#           --external_tokenizer_model ${external_tokenizer_model} \
#           --train_start_iter ${ITER} --train_stop_iter ${ITER} \
#           --n_targets 4096 \
#           --train_config conf/ear_large.yaml \
#           --tokenizer_train_config conf/tok_ear_large.yaml \
#           --tokenizer_inference_batch_size 500 \
#           --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
#           --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar" &
#       # sleep 5s
#     done
#   done
# done

# wait


# # # LARGE
# for LEARNING_RATE in 1.0e-4; do
#   for WARMUP_STEPS in 40000; do
#     for BATCH_BINS in 1600000; do
#       for mixup in 5 10 20 40; do
#         ITER=2
#         SSL_TAG="large${ITER}.mixup${mixup}.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
#         N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

#         deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
#         deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
#         # iter 1 tokenizers
#         # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_tokenizer_iter1_base_tok.tune_lr1.0e-3_warmup40000_bins800000_totalsteps400000/epoch3.pt
#         # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter1_large_tok.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt
#         external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter2_large_tok2.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt

#         echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
        
#         cp conf/ear_large.yaml conf/ear_large.mixup${mixup}.yaml
#         sed -i "s/mixup_probability: 0.0/mixup_probability: 0.${mixup}/g" conf/ear_large.mixup${mixup}.yaml

#         ./run_ear.sh --ngpu 4 --ssl_tag "${SSL_TAG}" \
#             --external_tokenizer_model ${external_tokenizer_model} \
#             --train_start_iter ${ITER} --train_stop_iter ${ITER} \
#             --train_config conf/ear_large.mixup${mixup}.yaml \
#             --tokenizer_train_config conf/tok_ear_large.yaml \
#             --tokenizer_inference_batch_size 500 \
#             --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
#             --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar" &
#         # sleep 5s
#       done
#     done
#   done
# done

# wait

# TOTAL_STEPS=400000
# # BASE
# for LEARNING_RATE in 5e-4; do
#   for WARMUP_STEPS in 40000; do
#     for BATCH_BINS in 1600000; do

#       SSL_TAG="base2.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
#       N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

#       deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
#       deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
#       # iter 1
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_tokenizer_iter1_base_tok.tune_lr1.0e-3_warmup40000_bins800000_totalsteps400000/epoch3.pt
#       # external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter1_base_tok.tune_lr5.0e-4_warmup40000_bins800000_totalsteps100000/epoch10.pt
      
#       # iter 2
#       external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_tokenizer_iter2_base_tok2.tune_lr5.0e-4_warmup40000_bins800000_totalsteps100000/epoch10.pt
#       tokenizer_inf_config=conf/tokenizer_inf_base_100k_steps.yaml

#       echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"

#       ./run_ear.sh --ngpu 3 --ssl_tag "${SSL_TAG}" \
#           --external_tokenizer_model ${external_tokenizer_model} --train_start_iter 2 --train_stop_iter 2 \
#           --tokenizer_inference_config ${tokenizer_inf_config} \
#           --train_config conf/ear_base.yaml \
#           --tokenizer_train_config conf/tok_ear_base.yaml \
#           --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
#           --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar"
#       # sleep 5s

#     done
#   done
# done