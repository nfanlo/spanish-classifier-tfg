#!/bin/bash

# shellcheck disable=SC1091
source "${HOME}"/dev/spanish-classifier-tfg/.venv/bin/activate

# /Users/fperez/dev/data/spanishclassfier_exp/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds/ep_2-lr_5e-5-msl_72-bs_8-ds_config_80-10-10-nl_5-do_0.2/

EXP_NAME=$1
shift

MSL=72
TRAIN_BS=8

MODEL_ID=${1:-"distilbert-base-uncased"}

case ${MODEL_ID} in
    distil)
        MODEL="distilbert-base-uncased";;
    distilmulti)
        MODEL="distilbert-base-multilingual-cased";;
    distilbeto)
        MODEL="dccuchile/distilbert-base-spanish-uncased";;
    distilbetomldoc)
        MODEL="dccuchile/distilbert-base-spanish-uncased-finetuned-mldoc";;
    *)
        MODEL=${MODEL_ID};;
esac
shift

IS_LOCAL="${IS_LOCAL:-true}"

echo "Is local model? ${IS_LOCAL}"

CLEAN_DS="${CLEAN_DS:-false}"

echo "Using cleaned DS? ${CLEAN_DS}"

DS_CONFIG="${DS_CONFIG:-60-20-20}"

echo "DS Config: ${DS_CONFIG}"

SUB_DIR=$(echo "${MODEL}" | sed -r 's/\//-/')-finetuned-with-spanish-tweets-clf
echo Subdir: "${SUB_DIR}"
if [ "${CLEAN_DS}" = true ] ; then
    SUB_DIR=${SUB_DIR}-cleaned-ds
fi

OUTPUT_DIR="${HOME}/dev/data/spanishclassfier_exp/${SUB_DIR}/${EXP_NAME}"
if [ "${CLEAN_DS}" = true ] ; then
    SUB_DIR=${SUB_DIR}-cleaned-ds
fi

if [ "${IS_LOCAL}" = true ] ; then
    MODEL_PATH=${OUTPUT_DIR}/best_model
else
    MODEL_PATH=${MODEL_ID}
fi

export LOG_DEST=${OUTPUT_DIR}
TEST_SPLIT=${TEST_SPLIT:-"test"}

echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

infer_cli \
        --log_level "debug" \
        --transformed_data_dir "${TMPDIR}" \
        --dataset_config_name "${DS_CONFIG}" \
        --use_cleaned "${CLEAN_DS}" \
        --limited_record_count -1 \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path "${MODEL_PATH}" \
        --include_token_type_ids False \
        --test_split_name "${TEST_SPLIT}" \
        --problem_type single_label_classification \
        --target_labels_column_name labels \
        --label_names labels \
        --max_seq_length ${MSL} \
        --per_device_eval_batch_size $(( 4*TRAIN_BS )) \
