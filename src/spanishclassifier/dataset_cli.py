import dataclasses
import json
import os

import datasets
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass, field
from datasets import load_dataset, Dataset  # type: ignore
from transformers import AutoTokenizer
from transformers.hf_argparser import HfArgumentParser
from typing import List, Optional

from spanishclassifier import logger
from spanishclassifier.utils.filters import clean_html, remove_twitter_handles, remove_urls

@dataclass
class DatasetPipelineArguments:

    raw_data_dir: str = field(
        metadata={"help": "The base directory that contains the raw data"},
        default="./dataset"
    )

    dataset_config_name: str = field(
        metadata={"help": "The particular configuration of the dataset to use"},
        default="60-20-20"
    )

    trainsformed_data_dir: str = field(
        metadata={"help": "The place where to save the transformed dataset"},
        default=os.environ["TMPDIR"]
    )

    files_have_header: bool = field(
        metadata={"help": "If the original raw train/dev/test files have a header."},
        default=True,
    )

    perform_cleanup: bool = field(
        metadata={"help": "Whether or not to perform the cleanup of the dataset splits."},
        default=False,
    )

    process_only_split: Optional[str] = field(
        metadata={
            "help": "If specified, process only the split name specified. If None (default) all splits will be processed."
        },
        default=None,
    )

    train_split_name: Optional[str] = field(
        metadata={"help": "The name of file containing the train raw data to load in 'raw_data_dir'"},
        default="train",
    )

    dev_split_name: Optional[str] = field(
        metadata={"help": "The name of file containing the dev raw data to load in 'raw_data_dir'"},
        default="dev",
    )

    test_split_name: Optional[str] = field(
        metadata={"help": "The name of file containing the test raw data to load in 'raw_data_dir'"},
        default="test",
    )

    limited_record_count: Optional[int] = field(
        metadata={
            "help": "The number of records to process from the datases (default is -1, which means all records). Useful for testing/debug"
        },
        default=-1,
    )

    use_cached_ds: bool = field(
        metadata={"help": "Whether or not to use the cached dataset."},
        default=True,
    )

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)


@dataclass
class DatasetProcessingPipelineArguments:
    pipeline: DatasetPipelineArguments

    def __repr__(self):
        wrapper_rep = f"=" * 100
        wrapper_rep += f"\nCLI Arguments\n"
        wrapper_rep += f"=" * 100
        wrapper_rep += f"\nOur Pipeline-Related:\n{self.pipeline.to_json_string()}"
        return wrapper_rep


def get_cli_arguments(print_args: bool = False) -> DatasetProcessingPipelineArguments:
    parser = HfArgumentParser((DatasetPipelineArguments))

    dsp_args, remaining_strings = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cli_args = DatasetProcessingPipelineArguments(dsp_args)
    if print_args:
        logger.info(cli_args)
        logger.info(f"Non-processed Args:\n{remaining_strings}") if remaining_strings else None
    return cli_args

def generate_basic_ds_stats(args, ds, transformed_ds_dir):
    logger.info("\n" + "*" * 100 + "\nBasic Dataset stats\n" + "*" * 100)
    ds.set_format(type='pandas')
    train_df = ds[args.pipeline.train_split_name][:]
    dev_df = ds[args.pipeline.dev_split_name][:]
    test_df = ds[args.pipeline.test_split_name][:]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    for dfn, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        df['label_name'] = df['labels'].apply(lambda r: ds[args.pipeline.train_split_name].features['labels'].int2str(r))
        df['label_name'].value_counts(ascending=True).plot.barh()
        plt.title(f"Class Freq ({dfn} split)")
        path = os.path.join(transformed_ds_dir, f"{dfn}_class_freq.png")
        logger.info(f"Saving class freq plot to: {path}")
        plt.savefig(path)
        plt.close()

        df['words_per_tweet'] = df['text'].str.split().apply(len)
        df.boxplot("words_per_tweet", by='label_name', grid=False, showfliers=False)
        plt.title(f"Words per Tweet ({dfn} split)")
        path = os.path.join(transformed_ds_dir, f"{dfn}_tweet_length_in_words.png")
        logger.info(f"Saving tweet length (words) plot to: {path}")
        plt.savefig(path)
        plt.close()

        df['tokens_per_tweet'] = df['text'].apply(lambda t: len(tokenizer(t)['input_ids']))
        df.boxplot("tokens_per_tweet", by='label_name', grid=False, showfliers=False)
        plt.title(f"Tokens per Tweet ({dfn} split)")
        path = os.path.join(transformed_ds_dir, f"{dfn}_tweet_length_in_tokens.png")
        logger.info(f"Saving tweet length (tokens) plot to: {path}")
        plt.savefig(path)
        plt.close()

        logger.info(f"{dfn} dataframe sample:\n{df.head(5)}")


def main():

    args: DatasetProcessingPipelineArguments = get_cli_arguments(True)

    logger.info(f"Arguments passed: {args}")

    config_kwargs = {
        "data_dir": args.pipeline.raw_data_dir,
        "files_have_header": args.pipeline.files_have_header,
        'train_file': "train.csv",
        'dev_file': "dev.csv",
        'test_file': "test.csv",
        'train_split_name': args.pipeline.train_split_name,
        'dev_split_name': args.pipeline.dev_split_name,
        'test_split_name': args.pipeline.test_split_name, 
        'limited_record_count': args.pipeline.limited_record_count,
    }
    logger.info("Loading dataset")
    if not args.pipeline.use_cached_ds:
        datasets.disable_caching()
    ds = load_dataset("./src/spanishclassifier/dataset.py", args.pipeline.dataset_config_name, **config_kwargs)
    logger.info(f"Dataset summary:\n{ds}")
    logger.info(f"Dataset labels: {ds['train'].features['labels']}")
    if args.pipeline.perform_cleanup:
        logger.info("Cleaning datasets:\n1) Removing Twittwer handles\n2) Removing urls\n3) Removing html codes")
        ds = ds.map(remove_twitter_handles, load_from_cache_file=args.pipeline.use_cached_ds)
        ds = ds.map(remove_urls, load_from_cache_file=args.pipeline.use_cached_ds)
        ds = ds.map(clean_html, load_from_cache_file=args.pipeline.use_cached_ds)

    logger.info(f"Dataset samples from train split: {ds['train'][:13]}")

    transformed_ds_dir=os.path.join(args.pipeline.trainsformed_data_dir, args.pipeline.dataset_config_name)
    logger.info(f"Saving dataset in {transformed_ds_dir}")
    ds.save_to_disk(transformed_ds_dir)

    generate_basic_ds_stats(args, ds, transformed_ds_dir)



if __name__ == "__main__":
    main()
