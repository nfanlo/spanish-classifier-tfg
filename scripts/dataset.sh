#!/bin/bash

source ${HOME}/dev/spanish-classifier-tfg/.venv/bin/activate

USE_CACHED_DS="${1:-true}"
shift

echo "Use cached ds: ${USE_CACHED_DS}"

dataset_cli \
    --raw_data_dir "${HOME}/dev/spanish-classifier-tfg/dataset" \
    --transformed_data_dir "${TMPDIR}" \
    --limited_record_count -1 \
    --dataset_config_name 60-20-20 \
    --files_have_header true \
    --target_labels_column_name labels \
    --perform_cleanup true \
    --use_cached_ds ${USE_CACHED_DS}