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

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${PLACES_HI}" ]; then
    log "Fill the value of 'PLACES_HI' of db.sh"
    exit 1
fi

log "stage 1: Data preparation"

## DOWNLOAD DATA if PLACES_HI is set to downloads
if [ "${PLACES_HI}" == "downloads" ]; then
    # If there is no argument, the default download directory is set to currentdir/downloads
    if [ $# -ne 1 ]; then
        PLACES_HI_ROOT_DIR="$(pwd)/downloads"
        log "Using the default download directory: ${PLACES_HI_ROOT_DIR}"
    else
        PLACES_HI_ROOT_DIR="$1/downloads"
        log "Using the specified download directory: ${PLACES_HI_ROOT_DIR}"
    fi

    if [ ! -e "${PLACES_HI_ROOT_DIR}/download_done" ]; then
        log "stage 1: Data preparation"
        echo "Downlaoding PLACES_HI spoken caption dataset into ${PLACES_HI_ROOT_DIR}."
        mkdir -p "${PLACES_HI_ROOT_DIR}"
        cd ${PLACES_HI_ROOT_DIR}
        wget https://data.csail.mit.edu/placesaudio/PlacesHindi100k.tar.gz
        echo "Expanding"
        tar -xvf PlacesHindi100k.tar.gz
        touch "${PLACES_HI_ROOT_DIR}/download_done"
    else
        echo "Places dataset is already downloaded. ${PLACES_HI_ROOT_DIR}/download_done exists."
    fi
    PLACES_HI_ROOT_DIR="${PLACES_HI_ROOT_DIR}/PlacesHindi100k"
else
    PLACES_HI_ROOT_DIR=${PLACES_HI}
    log "Using the specified data directory: ${PLACES_HI_ROOT_DIR}"
fi

## PREPARE DATA
if [ ! -d ${PLACES_HI_ROOT_DIR} ]; then
    echo Cannot find ${PLACES_HI_ROOT_DIR} directory! Exiting...
    exit 1
fi

SPLITS=(train dev test)
for split_name in "${SPLITS[@]}"; do
    mkdir -p "data/${split_name}"
done

# Prepare data in the Kaldi format, including three files:
# text, wav.scp, utt2spk
python3 local/data_prep_places_hindi.py ${PLACES_HI_ROOT_DIR}

# SORT ALL
for split_name in "${SPLITS[@]}"; do
    for f in wav.scp utt2spk; do
        sort data/${split_name}/${f} -o data/${split_name}/${f}
    done
    # Sort all text files
    if [ -f data/${split_name}/text ]; then
        sort data/${split_name}/text -o data/${split_name}/text
    fi
    echo "Running spk2utt for ${split_name}"
    utils/utt2spk_to_spk2utt.pl data/${split_name}/utt2spk > "data/${split_name}/spk2utt"
done

log "Successfully finished. [elapsed=${SECONDS}s]"
