#!/bin/bash

source ${HOME}/dev/spanish-classifier-tfg/.venv/bin/activate

MSL=72

export OUTPUT_DIR="${HOME}/dev/data/spanishclassfier_exp/"
export LOG_DEST=${OUTPUT_DIR}

echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

CLEAN_DS="${CLEAN_DS:-false}"

evaluator_cli \
        --log_level "debug" \
        --transformed_data_dir "${TMPDIR}" \
        --dataset_config_name 60-20-20 \
        --use_cleaned ${CLEAN_DS} \
        --limited_record_count -1 \
        --output_dir "${OUTPUT_DIR}" \
        --include_token_type_ids False \
        --test_split_name test \
        --problem_type single_label_classification \
        --target_labels_column_name labels \
        --label_names labels \
        --max_seq_length ${MSL} \
