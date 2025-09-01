#!/usr/bin/env bash
set -euo pipefail

asr_speech_fold_length=4800 # 480000/16000 = 30 seconds
inference_ckpt=valid.acc.best

./asr.sh \
    --feats_normalize uttmvn \
    --stage 11 \
    --stop_stage 12 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 32 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type char \
    --use_lm false \
    --inference_args "--ctc_weight 0.0" \
    --train_set train \
    --valid_set dev \
    --test_sets "dev test" \
    --asr_config conf/train_hubert_transformer.yaml \
    --asr_speech_fold_length ${asr_speech_fold_length} \
    --inference_asr_model ${inference_ckpt}.pth