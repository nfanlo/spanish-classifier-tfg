#!/bin/bash

source ${HOME}/dev/spanish-classifier-tfg/.venv/bin/activate

MODEL_ID=${1:-"distilbert-base-uncased"}

case ${MODEL_ID} in
    distil)
        MODEL="distilbert-base-uncased";;
    distilmulti)
        MODEL="distilbert-base-multilingual-cased";;
    distilbeto)
        MODEL="dccuchile/distilbert-base-spanish-uncased ";;
    *)
        MODEL=${MODEL_ID};;
esac
shift

EPOCHS=4
LR=5e-5
MSL=72
TRAIN_BS=8

export EXP_NAME=tweet-sa-spanish-${MODEL}-ep_${EPOCHS}-lr_${LR}-msl_${MSL}-bs_${TRAIN_BS}
export OUTPUT_DIR="${HOME}/dev/data/spanishclassfier_exp/${EXP_NAME}"
export LOG_DEST=${OUTPUT_DIR}

echo "Output dir: ${OUTPUT_DIR}"

train_cli \
        --transformed_data_dir "/home3/fperez/yk/data/yahoo/Database/transformed_datasets_athlete_desc" \
        --log_level "debug" \
        --transformed_data_dir "${TMPDIR}" \
        --dataset_config_name 60-20-20 \
        --limited_record_count -1 \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path ${MODEL} \
        --include_token_type_ids False \
        --dev_split_name dev \
        --problem_type single_label_classification \
        --target_labels_column_name labels \
        --label_names labels \
        --max_seq_length ${MSL} \
        --num_train_epoch ${EPOCHS} \
        --learning_rate ${LR} \
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
        --save_total_limit 2 \
