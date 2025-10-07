#!/bin/bash

# Datasets
datasets=(
  aishell buckeye cv doreco epadb fleurs fleurs_indv kazakh l2arctic librispeech
  mls_dutch mls_french mls_german mls_italian mls_polish mls_portuguese mls_spanish
  southengland speechoceannotth tamil tusom2021 voxangeles
)

datasets=(l2arctic_perceived)

# Models
models=(
  allophant
  facebook/wav2vec2-lv-60-espeak-cv-ft
  facebook/wav2vec2-xlsr-53-espeak-cv-ft
  ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
  allosaurus
)

# Loop through all dataset–model combinations
for D in "${datasets[@]}"; do
  for M in "${models[@]}"; do
    echo "Submitting job for dataset=$D, model=$M"
    sbatch run_baseline_inference.sh --dataset "$D" --model "$M" --num_workers 5
  done
done