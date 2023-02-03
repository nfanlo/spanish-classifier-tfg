#!/bin/bash

# shellcheck source=/dev/null
source "${HOME}"/dev/spanish-classifier-tfg/.venv/bin/activate

MODEL_ID=${1:-"distilbert-base-uncased"}

case ${MODEL_ID} in
    distil)
        MODEL="distilbert-base-uncased";;
    distilmulti)
        MODEL="distilbert-base-multilingual-cased";;
    distilbeto)
        MODEL="dccuchile/distilbert-base-spanish-uncased";;
    *)
        MODEL=${MODEL_ID};;
esac
shift

EPOCHS="${EPOCHS:-4}"

echo "Using ${EPOCHS} epochs"

LR="${LR:-5e-5}"

echo "Using ${LR} as learning rate"

MSL=72
TRAIN_BS=8

CLEAN_DS="${CLEAN_DS:-false}"

echo "Using cleaned DS? ${CLEAN_DS}"

SUB_DIR=$(echo "${MODEL}" | sed -r 's/\//-/')-finetuned-with-spanish-tweets-clf
echo Subdir: "${SUB_DIR}"
if [ "${CLEAN_DS}" = true ] ; then
    SUB_DIR=${SUB_DIR}-cleaned-ds
fi

PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

echo "PUSH TO HUB? ${PUSH_TO_HUB}"
HF_HUB_USER=francisco-perez-sorrosal
HF_HUB_ID=${HF_HUB_USER}/${SUB_DIR}

DS_CONFIG="${DS_CONFIG:-60-20-20}"

echo "DS Config: ${DS_CONFIG}"

# Overfitting parameters

DROPOUT="${DROPOUT:0.1}"

echo "DROPOUT: ${DROPOUT}"

DISTIL_LAYERS="${DISTIL_LAYERS:6}"

echo "Distilbert Layers: ${DISTIL_LAYERS}"

export EXP_NAME=ep_${EPOCHS}-lr_${LR}-msl_${MSL}-bs_${TRAIN_BS}-ds_config_${DS_CONFIG}-nl_${DISTIL_LAYERS}-do_${DROPOUT}
export OUTPUT_DIR="${HOME}/dev/data/spanishclassfier_exp/${SUB_DIR}/${EXP_NAME}"
export LOG_DEST=${OUTPUT_DIR}

echo "Output dir: ${OUTPUT_DIR}"

train_cli \
        --log_level "debug" \
        --transformed_data_dir "${TMPDIR}" \
        --dataset_config_name "${DS_CONFIG}" \
        --use_cleaned "${CLEAN_DS}" \
        --limited_record_count -1 \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path "${MODEL}" \
        --include_token_type_ids False \
        --dev_split_name dev \
        --problem_type single_label_classification \
        --target_labels_column_name labels \
        --label_names labels \
        --max_seq_length ${MSL} \
        --num_train_epoch "${EPOCHS}" \
        --learning_rate "${LR}" \
        --dropout "${DROPOUT}" \
        --distil_layers "${DISTIL_LAYERS}" \
        --per_device_train_batch_size ${TRAIN_BS} \
        --per_device_eval_batch_size $(( 4*TRAIN_BS )) \
        --logging_strategy epoch \
        --logging_steps 500 \
        --evaluation_strategy epoch \
        --eval_steps 500 \
        --save_strategy epoch \
        --save_steps 500 \
        --metric_for_best_model f1 \
        --load_best_model_at_end true \
        --early_stopping_patience 3 \
        --resume_from_checkpoint false \
        --save_total_limit 2 \
        --push_to_hub "${PUSH_TO_HUB}" \
        --hub_model_id "${HF_HUB_ID}" \
        --hub_strategy checkpoint \
