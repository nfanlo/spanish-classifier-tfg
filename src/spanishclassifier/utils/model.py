from typing import Dict, Union

from datasets import DatasetDict

from spanishclassifier import logger
from spanishclassifier.utils.cli import (
    InferencingPipelineArguments,
    TrainingPipelineArguments,
)


def build_model_extra_config(
    args: Union[TrainingPipelineArguments, InferencingPipelineArguments], split_name: str, dataset: DatasetDict
) -> Dict:
    logger.info(f"DS features:\n{dataset[split_name].features}")
    ds_label_info = dataset[split_name].features[args.pipeline.target_labels_column_name]

    model_extra_config = {
        "problem_type": args.pipeline.problem_type,
        "id2label": {v: k for k, v in ds_label_info._str2int.items()},
        "label2id": ds_label_info._str2int,
        "num_labels": ds_label_info.num_classes,
    }
    logger.info(f"Model extra-config:\n\n{model_extra_config}")
    return model_extra_config
