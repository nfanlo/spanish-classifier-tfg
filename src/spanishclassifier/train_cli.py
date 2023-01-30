import dataclasses
import evaluate
import json
import numpy as np
import os
import time

from distutils.dir_util import copy_tree
from dataclasses import dataclass, field
from datasets import load_from_disk, DatasetDict
from evaluate import CombinedEvaluations
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, TrainingArguments, Trainer, TrainerCallback
from typing import Dict, Optional

from spanishclassifier import logger
from spanishclassifier.utils.cli import get_cli_arguments, TrainingPipelineArguments, TrainPipelineArguments
from spanishclassifier.utils.model import build_model_extra_config
from spanishclassifier.utils.metrics import ConfiguredMetric


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


# def build_model_extra_config(args: TrainingPipelineArguments, dataset: DatasetDict) -> Dict:
#     logger.info(f"DS features:\n{dataset[args.pipeline.train_split_name].features}")
#     ds_label_info = dataset[args.pipeline.train_split_name].features[args.pipeline.target_labels_column_name]

#     model_extra_config = {
#         "problem_type": args.pipeline.problem_type,
#         "id2label": {v: k for k,v in ds_label_info._str2int.items()},
#         "label2id": ds_label_info._str2int,
#         "num_labels": ds_label_info.num_classes,
#     }
#     logger.info(f"Model extra-config:\n\n{model_extra_config}")
#     return model_extra_config


def main():
    logger.info("*" * 100)
    logger.info("Trainer".center(100))
    logger.info("*" * 100)

    args: TrainingPipelineArguments = get_cli_arguments((TrainPipelineArguments, TrainingArguments), TrainingPipelineArguments, True)

    ### Tokenization
    tokenizer_config = {
        "padding": "max_length",
        "truncation": True,
        "max_length": args.pipeline.max_seq_length,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.pipeline.model_name_or_path, **tokenizer_config)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.pipeline.max_seq_length)

    # import datasets
    # datasets.disable_caching()
    start_t = time.time()
    dataset_dir = os.path.join(args.pipeline.transformed_data_dir, args.pipeline.dataset_config_name)
    logger.info(f"Loading and tokenizing train/dev datasets from {dataset_dir}")
    ds = load_from_disk(dataset_dir)
    end_load_t = time.time()
    logger.info(f"Time to load dataset: {end_load_t-start_t}")
    logger.info(f"Dataset info:\n{ds}")
    train_examples = len(ds[args.pipeline.train_split_name])
    if args.pipeline.limited_record_count != -1:
        logger.warning(f"Limiting the train set to only {args.pipeline.limited_record_count} training examples! MAYBE YOU ARE TESTING STUFF???")
        train_examples = args.pipeline.limited_record_count

    train_tokenized_ds = ds[args.pipeline.train_split_name].map(tokenize_function, batched=True).shuffle(seed=42).select(range(train_examples))
    dev_tokenized_ds = ds[args.pipeline.dev_split_name].map(tokenize_function, batched=True).select(range(train_examples if train_examples < len(ds[args.pipeline.dev_split_name]) else len(ds[args.pipeline.dev_split_name])))
    logger.info(f"Time to tokenize train/dev datasets: {time.time()-end_load_t}")
    logger.info(f"Tokenized sample:\n{train_tokenized_ds[:1]}")

    model_extra_config = build_model_extra_config(args, args.pipeline.train_split_name, ds)
    model = AutoModelForSequenceClassification.from_pretrained(args.pipeline.model_name_or_path, **model_extra_config)

    ### Metrics setup
    # f1 = evaluate.load("f1")
    # metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    metrics = evaluate.combine([
        evaluate.load("accuracy"), 
        ConfiguredMetric(evaluate.load("f1"), average="macro"),
        ConfiguredMetric(evaluate.load("precision"), average="macro"),
        ConfiguredMetric(evaluate.load("recall"), average="macro")
    ])


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics_args = {
            "f1": {
                "average": "macro"
            },
            "precision": {
                "average": "macro"
            },
            "recall": {
                "average": "macro"
            },
        }
        return metrics.compute(predictions=predictions, references=labels) #, **metrics_args)

    callbacks = [MetricsSaverCallback]
    if args.pipeline.early_stopping_patience > 0:
        logger.info(f"Adding Early Stopping Callback with patience of {args.pipeline.early_stopping_patience}")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.pipeline.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args.train,
        train_dataset=train_tokenized_ds,
        eval_dataset=dev_tokenized_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    if trainer.state.best_model_checkpoint is not None:
        best_model_dest_path = f"{args.train.output_dir}/best_model"
        logger.info(f"Copying best model from {trainer.state.best_model_checkpoint} to {best_model_dest_path}")
        copy_tree(trainer.state.best_model_checkpoint, best_model_dest_path)
        logger.info(f"Saving tokenizer to {best_model_dest_path}")
        tokenizer.save_pretrained(best_model_dest_path)
        

if __name__ == "__main__":
    main()