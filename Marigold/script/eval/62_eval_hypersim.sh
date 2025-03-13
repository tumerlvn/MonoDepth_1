#!/usr/bin/env bash
set -e
set -x

subfolder=${1:-"eval"}

python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_hypersim_val.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/hypersim_val/prediction \
    --output_dir output/${subfolder}/hypersim_val/eval_metric \
