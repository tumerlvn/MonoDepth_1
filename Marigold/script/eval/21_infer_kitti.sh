#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-v1-0"}
subfolder=${2:-"eval"}

python infer.py  \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $BASE_DATA_DIR \
    --denoise_steps 10 \
    --ensemble_size 1 \
    --processing_res 256 \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --output_dir output/${subfolder}/kitti_eigen_test/prediction \
