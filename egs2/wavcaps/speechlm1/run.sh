# !/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

train_config=conf/train_delay_asr.yaml
inference_config=conf/decode_asr.yaml

./speechlm.sh \
    --task "aac_codecssl" \
    --data_name wavcaps \
    --stage 5 \
    --stop_stage 5 \
    --ngpu 1 \
    --audio_format flac.ark \
    --nj 64 \
    --train_set wavcaps_train \
    --valid_set "" \
    --test_sets "" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --ssl_checkpoint_path exp/kmeans_xues/38epoch.pth --ssl_kmeans_path exp/kmeans_xues/km_5000.mdl --ssl_nlayer 16 \
    --codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch # --token_list_dir data/token_list/asr_vocab


# audio_format was wav for codec tokens
