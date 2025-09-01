#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# asr_config="conf/tuning/train_asr_conformer6_n_fft400_hop_length160.yaml"
# decode_config="conf/decode_asr.yaml"

# ./asr.sh \
#     --lang "hi" \
#     --stage 1 \
#     --asr_config $asr_config \
#     --inference_config $decode_config \
#     --use_lm false \
#     --train_set "train100" \
#     --valid_set "dev" \
#     --test_sets "test" \
#     --token_type char  "$@"


asr_config="conf/train_hubert_transformer.yaml"
inference_ckpt=valid.acc.best

./asr.sh \
    --feats_normalize uttmvn \
    --lang hi \
    --stage 11 \
    --ngpu 1 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 1 \
    --stop_stage 12 \
    --asr_config "$asr_config" \
    --use_lm false \
    --train_set "train100" \
    --valid_set "dev" \
    --test_sets "dev test" \
    --inference_asr_model "${inference_ckpt}.pth" \
    --token_type char  "$@"


# ./asr.sh \
#     --max_wav_duration 30 \
#     --inference_args "--ctc_weight 0.0" \
#     --asr_speech_fold_length ${asr_speech_fold_length} \