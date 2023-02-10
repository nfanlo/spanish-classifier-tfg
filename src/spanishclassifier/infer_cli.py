import os
import time
from pprint import pformat

import evaluate
from datasets import load_from_disk
from evaluate import evaluator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)

from spanishclassifier import logger
from spanishclassifier.utils.cli import (
    InferencingPipelineArguments,
    InferPipelineArguments,
    get_cli_arguments,
)
from spanishclassifier.utils.metrics import ConfiguredMetric


def main():
    logger.info("*" * 100)
    logger.info("Inferer".center(100))
    logger.info("*" * 100)

    args: InferencingPipelineArguments = get_cli_arguments(
        (InferPipelineArguments, TrainingArguments), InferencingPipelineArguments, True
    )

    ### Tokenization
    logger.info(f"Loading tokenizer from {args.pipeline.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.pipeline.model_name_or_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"]
        )  # , padding="max_length", truncation=True, max_length=args.pipeline.max_seq_length)

    start_t = time.time()
    transformed_ds_filename = (
        args.pipeline.dataset_config_name
        if not args.pipeline.use_cleaned_ds
        else f"{args.pipeline.dataset_config_name}-cleaned"
    )
    dataset_dir = os.path.join(args.pipeline.transformed_data_dir, transformed_ds_filename)
    logger.info(f"Loading and tokenizing eval (dev/test) dataset splits from {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    end_load_t = time.time()
    logger.info(f"Time to load dataset: {end_load_t-start_t}")
    logger.info(f"Dataset info:\n{ds}")

    logger.info(
        f"Sample of 10 transformed examples from test ds ({args.pipeline.test_split_name} split) \
        {' (cleaned):' if args.pipeline.use_cleaned_ds else ':'}\n{pformat(ds[args.pipeline.test_split_name][:10], width=200)}"
    )

    test_examples = len(ds[args.pipeline.test_split_name])
    if args.pipeline.limited_record_count != -1:
        logger.warning(
            f"Limiting the train set to only {args.pipeline.limited_record_count} training examples! MAYBE YOU ARE TESTING STUFF???"
        )
        test_examples = args.pipeline.limited_record_count

    test_tokenized_ds = (
        ds[args.pipeline.test_split_name].map(tokenize_function, batched=True).select(range(test_examples))
    )
    logger.info(f"Time to tokenize test dataset: {time.time()-end_load_t}")
    logger.info(f"Tokenized sample:\n{test_tokenized_ds[:1]}")

    model = AutoModelForSequenceClassification.from_pretrained(args.pipeline.model_name_or_path)

    ### Metrics setup
    metrics = evaluate.combine(
        [
            evaluate.load("accuracy"),
            ConfiguredMetric(evaluate.load("f1"), average="macro"),
            ConfiguredMetric(evaluate.load("precision"), average="macro"),
            ConfiguredMetric(evaluate.load("recall"), average="macro"),
        ]
    )

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    ds_label_info = ds[args.pipeline.test_split_name].features[args.pipeline.target_labels_column_name]

    task_evaluator = evaluator("text-classification")

    test_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=test_tokenized_ds,
        metric=metrics,
        label_column=args.pipeline.target_labels_column_name,
        label_mapping=ds_label_info._str2int,
    )
    logger.info(f"Test metrics results\n{test_results}")


if __name__ == "__main__":
    main()
