# !/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

train_config=conf/train_delay_asr.yaml
inference_config=conf/decode_asr.yaml

./speechlm.sh \
    --task "aac" \
    --data_name clotho_v2 \
    --stage 7 \
    --audio_format wav \
    --stop_stage 7 \
    --nj 32 \
    --train_set development \
    --valid_set validation \
    --test_sets "validation evaluation" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch # --token_list_dir data/token_list/asr_vocab

#TODO: change token list to correct path