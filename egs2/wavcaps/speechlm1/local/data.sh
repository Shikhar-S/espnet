#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0
python=python3
stage=1
stop_stage=100

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh || exit 1
. ./cmd.sh || exit 1


# We always use wavcaps data for creating training set.
# Wavcaps consists of 4 datasets: AudioSet_SL, BBC_Sound_Effects, FreeSound, SoundBible
# Clotho_v2 and AudioCap are always used for evaluation.
# In the zero-shot setting, we only use wavcaps data for training, followed by evaluation on clotho_v2 and Audiocap test sets.
# In the full fine-tune setting we use clotho_v2 and audiocap for training in addition to wavcap. Evaluation are as ususall on both clotho_v2 and audiocap test sets.
# To keep data preparation modular we have three different python scripts data_prep_wavcaps.py, data_prep_clotho.py, data_prep_audiocap.py. These create {dataset}/{split}/{text,wav.scp} files in the Kaldi-speechlm format.
# The data_prep_wavcaps.py script is always run, while the other two are run only if the setting is not zero-shot.


if [ -z "${WAVCAPS}" ]; then
    log "Fill the value of 'WAVCAPS' of db.sh"
    exit 1
fi

SETTING=${SETTING:-"zero-shot"} # zero-shot or full-fine-tune

TRAIN_DATASETS=(wavcaps)
if [ ${SETTING} == "full-fine-tune" ]; then
    echo "Running data preparation for full-fine-tune setting: wavcaps, clotho_v2, audiocap"
    TRAIN_DATASETS+=(clotho_v2 audiocap)
elif [ ${SETTING} == "zero-shot" ]; then
    echo "Running data preparation for zero-shot setting: wavcaps"
else
    log "Invalid setting ${SETTING}. Choose either zero-shot or full-fine-tune"
    exit 1
fi

DEV_SETS=(clotho_v2 audiocap)
TEST_SETS=(clotho_v2 audiocap)

SPLITS=(train dev test)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"

    if [ ! -d ${WAVCAPS} ]; then
        echo Cannot find WAVCAPS root! Exiting...
        exit 1
    fi

    if [ ${SETTING} == "full-fine-tune" ]; then
        if [ ! -d ${CLOTHO_V2} ]; then
            echo Cannot find CLOTHO root! Exiting...
            exit 1
        fi

        if [ ! -d ${AUDIOCAP} ]; then
            echo Cannot find AUDIOCAP root! Exiting...
            exit 1
        fi
    fi
    
    # Prepare data in the Kaldi-speechlm format, two files:
    # text, wav.scp
    # ${cmd} "JOB=1:${nj}" "${logdir}/format_wav_scp.JOB.log" \
    ${python} local/data_prep_wavcaps.py ${WAVCAPS}

    if [ ${SETTING} == "full-fine-tune" ]; then
        python3 local/data_prep_clotho.py ${CLOTHO_V2}
        python3 local/data_prep_audiocap.py ${AUDIOCAP}
        # Combine the training data from all datasets
        mkdir -p "data/all_train"
        for dataset in ${TRAIN_DATASETS[@]}; do
            cat data/${dataset}_train/text >> data/all_train/text
            cat data/${dataset}_train/wav.scp >> data/all_train/wav.scp
        done
    fi

    # Sort all data
    for dir_names in $(ls data); do
        for f in wav.scp text; do
            if [ -f data/${dir_names}/${f} ]; then
                echo "Sorting data/${dir_names}/${f}"
                sort data/${dir_names}/${f} -o data/${dir_names}/${f}
            fi
        done
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"