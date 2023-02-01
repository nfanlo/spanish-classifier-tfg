import evaluate
import os
import pandas as pd
import time

from datasets import load_from_disk
from evaluate import evaluator
from evaluate.visualization import radar_plot
from pprint import pprint, pformat
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, pipeline

from spanishclassifier import logger
from spanishclassifier.utils.cli import get_cli_arguments, InferPipelineArguments, InferencingPipelineArguments
from spanishclassifier.utils.metrics import ConfiguredMetric


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def main():
    logger.info("*" * 100)
    logger.info("Evaluator".center(100))
    logger.info("*" * 100)

    args: InferencingPipelineArguments = get_cli_arguments((InferPipelineArguments, TrainingArguments), InferencingPipelineArguments, True)
    
    transformed_datasets = {}

    start_t = time.time()
    # Load the two possible datasets (dirty and cleaned)
    for transformed_ds_filename in [args.pipeline.dataset_config_name, f"{args.pipeline.dataset_config_name}-cleaned"]:
        dataset_dir = os.path.join(args.pipeline.transformed_data_dir, transformed_ds_filename)
        logger.info(f"Loading and tokenizing train/dev datasets from {dataset_dir}")
        ds = load_from_disk(dataset_dir)
        if not dataset_dir.endswith("-cleaned"):
            transformed_datasets['dirty'] = ds
            logger.info(f"Dataset info (dirty):\n{ds}")
        else:
            transformed_datasets['cleaned'] = ds
            logger.info(f"Dataset info (cleaned):\n{ds}")
        logger.info(f"Sample of 10 transformed examples from test ds{' (cleaned):' if args.pipeline.use_cleaned_ds else ':'}\n{pformat(ds[args.pipeline.test_split_name][:10], width=200)}")
    end_load_t = time.time()
    logger.info(f"Time to load datasets: {end_load_t-start_t}")

    ### Metrics setup
    metrics = evaluate.combine([
        evaluate.load("accuracy"), 
        ConfiguredMetric(evaluate.load("f1"), average="macro"),
        ConfiguredMetric(evaluate.load("precision"), average="macro"),
        ConfiguredMetric(evaluate.load("recall"), average="macro")
    ])

    models = [
        "francisco-perez-sorrosal/distilbert-base-uncased-finetuned-with-spanish-tweets-clf",
        "francisco-perez-sorrosal/distilbert-base-multilingual-cased-finetuned-with-spanish-tweets-clf",
        "francisco-perez-sorrosal/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf",
        "francisco-perez-sorrosal/distilbert-base-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds",
        "francisco-perez-sorrosal/distilbert-base-multilingual-cased-finetuned-with-spanish-tweets-clf-cleaned-ds",
        "francisco-perez-sorrosal/dccuchile-distilbert-base-spanish-uncased-finetuned-with-spanish-tweets-clf-cleaned-ds",
    ]

    ds_label_info = ds[args.pipeline.test_split_name].features[args.pipeline.target_labels_column_name]
    
    task_evaluator = evaluator("text-classification")

    results = []
    for model in models:
        transformed_ds_id = "dirty" if not model.endswith("-cleaned-ds") else "cleaned"
        logger.info(f"Evaluating {model} with {args.pipeline.dataset_config_name} ({transformed_ds_id}) dataset")
        test_results = task_evaluator.compute(
            model_or_pipeline=model,
            data=transformed_datasets[transformed_ds_id][args.pipeline.test_split_name],
            metric=metrics,
            label_column=args.pipeline.target_labels_column_name,
            label_mapping=ds_label_info._str2int
        )
        logger.info(f"Test metrics results:\n{pformat(test_results)}")
        results.append(test_results)

    df = pd.DataFrame(results, index=models)
    df[["accuracy", "f1", "precision", "recall", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]
    logger.info(f"Results:\n{df.head(10)}")
    
    plot = radar_plot(data=results, model_names=[m.split("/")[1] for m in models], invert_range=["latency_in_seconds"], config={"legend_loc": "lower right"})
    radarplot_path = os.path.join(args.train.output_dir, "spanish-tweet-models-radarplot.png")
    logger.info(f"Saving Radar plot to: {radarplot_path}")
    plot.savefig(radarplot_path, bbox_inches='tight')


if __name__ == "__main__":
    main()