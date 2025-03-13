#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"prs-eth/marigold-v1-0"}
subfolder=${2:-"eval"}

python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 15 \
    --ensemble_size 5 \
    --processing_res 512 \
    --dataset_config config/dataset/data_middlebury_test.yaml \
    --output_dir output/${subfolder}/middlebury_test/prediction \
