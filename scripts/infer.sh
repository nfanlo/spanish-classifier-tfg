#!/bin/bash

source ${HOME}/dev/spanish-classifier-tfg/.venv/bin/activate

MODEL_PATH=${1:-/Users/fperez/dev/data/spanishclassfier_exp/tweet-sa-spanish-distilbert-base-uncased-ep_4-lr_5e-5-msl_72-bs_8/best_model}

shift

MSL=72
TRAIN_BS=8

export EXP_NAME=tweet-sa-spanish-distilbert-base-uncased-ep_4-lr_5e-5-msl_72-bs_8
export OUTPUT_DIR="${HOME}/dev/data/spanishclassfier_exp/${EXP_NAME}"
export LOG_DEST=${OUTPUT_DIR}

echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

infer_cli \
        --transformed_data_dir "/home3/fperez/yk/data/yahoo/Database/transformed_datasets_athlete_desc" \
        --log_level "debug" \
        --transformed_data_dir "${TMPDIR}" \
        --dataset_config_name 60-20-20 \
        --limited_record_count -1 \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path ${MODEL_PATH} \
        --include_token_type_ids False \
        --test_split_name test \
        --problem_type single_label_classification \
        --target_labels_column_name labels \
        --label_names labels \
        --max_seq_length ${MSL} \
