import dataclasses
import json
import os

from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import Any, TypeVar, Sequence, Optional

from spanishclassifier import logger


T = TypeVar('T')


@dataclass
class TrainPipelineArguments:

    # Data/Model related parameters

    transformed_data_dir: str = field(metadata={"help": "The dir where the transformed data is located."}, default=os.environ["TMPDIR"])

    dataset_config_name: str = field(
        metadata={"help": "The particular configuration of the dataset to use"},
        default="60-20-20"
    )

    limited_record_count: Optional[int] = field(
        metadata={
            "help": "The number of records to process from the datases (default is -1, which means all records). Useful for testing/debug"
        },
        default=-1,
    )

    train_split_name: str = field(metadata={"help": "Name of the dataset to train on"}, default="train")

    dev_split_name: Optional[str] = field(metadata={"help": "Name of the dataset to validate on"}, default="dev")

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pre-trained model (from huggingface.co/models) or shortcut name selected in the list"
            # + ", ".join(ALL_PRETRAINED_CONFIG_ARCHIVE_MAP)
        },
        default="distilbert-base-uncased",
    )

    problem_type: str = field(
        metadata={"help": "regression | single_label_classification | multi_label_classification"},
        default="multi_label_classification",
    )

    # Model-input-related parameters

    include_token_type_ids: bool = field(
        metadata={"help": "Whether to include or not token_type_ids in the tensors to pass to the model"}, default=False
    )

    target_labels_column_name: str = field(
        metadata={"help": "The name of the column containing the labels."}, default="labels"
    )

    max_seq_length: int = field(
        default=172,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    early_stopping_patience: int = field(
        metadata={"help": "Early stopping patience (0, means no early stopping)"},
        default=3,
    )

    metrics_classification_threshold: Optional[float] = field(
        metadata={"help": "The threshold for metrics classification"},
        default=None,
    )

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)


@dataclass
class TrainingPipelineArguments:
    pipeline: TrainPipelineArguments
    train: TrainingArguments

    def __repr__(self):
        wrapper_rep = f"\n\n"
        wrapper_rep += f"=" * 100
        wrapper_rep += f"\nCLI Arguments\n"
        wrapper_rep += f"=" * 100
        wrapper_rep += f"\Pipeline-Related:\n{self.pipeline.to_json_string()}"
        wrapper_rep += f"\nHF-Related:\n{self.train.to_json_string()}"
        return wrapper_rep


@dataclass
class InferPipelineArguments:

    # Data/Model related parameters

    transformed_data_dir: str = field(metadata={"help": "The dir where the transformed data is located."}, default=os.environ["TMPDIR"])

    dataset_config_name: str = field(
        metadata={"help": "The particular configuration of the dataset to use"},
        default="60-20-20"
    )

    limited_record_count: Optional[int] = field(
        metadata={
            "help": "The number of records to process from the datases (default is -1, which means all records). Useful for testing/debug"
        },
        default=-1,
    )

    test_split_name: str = field(metadata={"help": "Name of the dataset to apply inference on"}, default="test")

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pre-trained model (from huggingface.co/models) or shortcut name selected in the list"
            # + ", ".join(ALL_PRETRAINED_CONFIG_ARCHIVE_MAP)
        },
        default="distilbert-base-uncased",
    )

    problem_type: str = field(
        metadata={"help": "regression | single_label_classification | multi_label_classification"},
        default="multi_label_classification",
    )

    # Model-input-related parameters

    include_token_type_ids: bool = field(
        metadata={"help": "Whether to include or not token_type_ids in the tensors to pass to the model"}, default=False
    )

    target_labels_column_name: str = field(
        metadata={"help": "The name of the column containing the labels."}, default="labels"
    )

    max_seq_length: int = field(
        default=72,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    metrics_classification_threshold: Optional[float] = field(
        metadata={"help": "The threshold for metrics classification"},
        default=None,
    )

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)


@dataclass
class InferencingPipelineArguments:
    pipeline: InferPipelineArguments
    train: TrainingArguments

    def __repr__(self):
        wrapper_rep = f"\n\n"
        wrapper_rep += f"=" * 100
        wrapper_rep += f"\nCLI Arguments\n"
        wrapper_rep += f"=" * 100
        wrapper_rep += f"\Pipeline-Related:\n{self.pipeline.to_json_string()}"
        wrapper_rep += f"\nHF-Related:\n{self.train.to_json_string()}"
        return wrapper_rep

def get_cli_arguments(dataclasses_to_parse: Sequence[Any], args_class: type[T], print_args: bool = False) -> T:
    parser = HfArgumentParser(dataclasses_to_parse)

    pipeline_args, training_args, remaining_strings = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    cli_args = args_class(pipeline_args, training_args)
    if print_args:
        logger.info(cli_args)
        logger.info(f"Non-processed Args:\n{remaining_strings}") if remaining_strings else None
    return cli_args
