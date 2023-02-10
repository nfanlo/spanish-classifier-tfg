import json
import os
import time
from distutils.dir_util import copy_tree
from pprint import pformat

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_from_disk
from torch.nn.functional import cross_entropy, softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from spanishclassifier import logger
from spanishclassifier.utils.cli import (
    TrainingPipelineArguments,
    TrainPipelineArguments,
    get_cli_arguments,
)
from spanishclassifier.utils.metrics import ConfiguredMetric
from spanishclassifier.utils.model import build_model_extra_config

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)


class MetricsSaverCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics = metrics
        logger.info(f"Storing metrics: {metrics}")

    def on_save(self, args, state, control, **kwargs):
        json_string = json.dumps(self.metrics, indent=2, sort_keys=True) + "\n"
        json_metrics_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", "eval_metrics.json")
        logger.info(f"Saving eval metrics on {json_metrics_path}")
        with open(json_metrics_path, "w", encoding="utf-8") as f:
            f.write(json_string)


def main():
    logger.info("*" * 100)
    logger.info("Trainer".center(100))
    logger.info("*" * 100)

    args: TrainingPipelineArguments = get_cli_arguments(
        (TrainPipelineArguments, TrainingArguments), TrainingPipelineArguments, True
    )

    ### Tokenization
    tokenizer_config = {
        "padding": "max_length",
        "truncation": True,
        "max_length": args.pipeline.max_seq_length,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.pipeline.model_name_or_path, **tokenizer_config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=args.pipeline.max_seq_length
        )

    # import datasets
    # datasets.disable_caching()
    start_t = time.time()
    transformed_ds_filename = (
        args.pipeline.dataset_config_name
        if not args.pipeline.use_cleaned_ds
        else f"{args.pipeline.dataset_config_name}-cleaned"
    )
    dataset_dir = os.path.join(args.pipeline.transformed_data_dir, transformed_ds_filename)
    logger.info(f"Loading and tokenizing train/dev datasets from {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    end_load_t = time.time()
    logger.info(f"Time to load dataset: {end_load_t-start_t}")
    logger.info(f"Dataset info:\n{ds}")

    logger.info(
        f"Sample of 10 transformed examples from train ds{' (cleaned):' if args.pipeline.use_cleaned_ds else ':'}\n{pformat(ds['train'][:10], width=200)}"
    )

    train_examples = len(ds[args.pipeline.train_split_name])
    if args.pipeline.limited_record_count != -1:
        logger.warning(
            f"Limiting the train set to only {args.pipeline.limited_record_count} training examples! MAYBE YOU ARE TESTING STUFF???"
        )
        train_examples = args.pipeline.limited_record_count

    train_tokenized_ds = (
        ds[args.pipeline.train_split_name]
        .map(tokenize_function, batched=True)
        .shuffle(seed=42)
        .select(range(train_examples))
    )
    dev_tokenized_ds = (
        ds[args.pipeline.dev_split_name]
        .map(tokenize_function, batched=True)
        .select(
            range(
                train_examples
                if train_examples < len(ds[args.pipeline.dev_split_name])
                else len(ds[args.pipeline.dev_split_name])
            )
        )
    )
    logger.info(f"Time to tokenize train/dev datasets: {time.time()-end_load_t}")
    logger.info(f"Tokenized sample:\n{train_tokenized_ds[:1]}")

    model_extra_config = build_model_extra_config(args, args.pipeline.train_split_name, ds)
    config = AutoConfig.from_pretrained(args.pipeline.model_name_or_path, **model_extra_config)
    config.dropout = args.pipeline.dropout  # - 0.2
    config.attention_dropout = args.pipeline.dropout  # - 0.2
    config.seq_classif_dropout = args.pipeline.dropout
    config.n_layers = args.pipeline.distil_layers

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pipeline.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )

    logger.info(f"Model config\n{model.config}")

    logger.info(f"Model\n{model}")

    ### Metrics setup
    # f1 = evaluate.load("f1")
    # metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    metrics = evaluate.combine(
        [
            evaluate.load("accuracy"),
            ConfiguredMetric(evaluate.load("f1"), average="macro"),
            ConfiguredMetric(evaluate.load("precision"), average="macro"),
            ConfiguredMetric(evaluate.load("recall"), average="macro"),
        ]
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics_args = {
            "f1": {"average": "macro"},
            "precision": {"average": "macro"},
            "recall": {"average": "macro"},
        }
        return metrics.compute(predictions=predictions, references=labels)  # , **metrics_args)

    callbacks = [MetricsSaverCallback]
    if args.pipeline.early_stopping_patience > 0:
        logger.info(f"Adding Early Stopping Callback with patience of {args.pipeline.early_stopping_patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.pipeline.early_stopping_patience))

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=args.train,
        train_dataset=train_tokenized_ds,
        eval_dataset=dev_tokenized_ds,
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    resume_from_checkpoint = args.train.resume_from_checkpoint
    if (
        resume_from_checkpoint is None
        or args.train.resume_from_checkpoint == "false"
        or args.train.resume_from_checkpoint == "False"
    ):
        resume_from_checkpoint = False
    if resume_from_checkpoint == "true" or args.train.resume_from_checkpoint == "True":
        resume_from_checkpoint = True

    logger.warning(f"Resuming from the last checkpoint: {resume_from_checkpoint}")
    training_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info(f"Training output:\n{pformat(training_output.metrics)}")

    def forward_pass_with_label(batch):
        inputs = {k: v.to(trainer.model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            output = trainer.model(**inputs)
            # logger.info(output.logits)
            probabilities = torch.softmax(output.logits, dim=-1)
            # logger.info(probabilities)
            pred_label = torch.argmax(output.logits, axis=-1)
            loss = cross_entropy(
                output.logits, batch[args.pipeline.target_labels_column_name].to(trainer.model.device), reduction="none"
            )
        return {
            "loss": loss.cpu().numpy(),
            "predicted_labels": pred_label.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
        }

    def label_int2str(row):
        return dev_tokenized_ds.features["labels"].int2str(row)

    dev_tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_tokenized_ds_with_predictions = dev_tokenized_ds.map(
        forward_pass_with_label, batched=True, batch_size=32, load_from_cache_file=False
    )

    dev_tokenized_ds_with_predictions.set_format("pandas")
    cols = ["text", "labels", "predicted_labels", "loss", "probabilities"]
    logger.info(dev_tokenized_ds_with_predictions[:3])
    dev_df = dev_tokenized_ds_with_predictions[:][cols]
    dev_df["label"] = dev_df["labels"].apply(label_int2str)
    dev_df["predicted_label"] = dev_df["predicted_labels"].apply(label_int2str)
    logger.info(f"Examples with Highest losses:\n{dev_df.sort_values('loss', ascending=False).head(30)}")
    logger.info(f"Examples with Smallest losses:\n{dev_df.sort_values('loss', ascending=True).head(30)}")

    if not args.train.push_to_hub and trainer.state.best_model_checkpoint is not None:
        best_model_dest_path = f"{args.train.output_dir}/best_model"
        logger.info(f"Copying best model from {trainer.state.best_model_checkpoint} to {best_model_dest_path}")
        copy_tree(trainer.state.best_model_checkpoint, best_model_dest_path)
        dev_df.to_csv(os.path.join(best_model_dest_path, "dev_predictions.tsv"), sep="\t")
        # logger.info(f"Saving tokenizer to {best_model_dest_path}")
        # tokenizer.save_pretrained(best_model_dest_path)

    if args.train.push_to_hub:
        logger.info("Pushing to hub")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
