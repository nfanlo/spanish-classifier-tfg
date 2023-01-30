# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spanish Tweet Dataset"""

import csv
import os
import sys

import datasets
from typing import Optional

csv.field_size_limit(sys.maxsize)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Tweet dataset},
author={Dunno.},
year={2022}
}
"""

_DESCRIPTION = """\
This is the Spanish Tweet Dataset used for sentiment analisys.
"""
_HOMEPAGE = ""
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    "first_domain": "https://",
    "second_domain": "https://",
}

_CLASS_NAMES = ["P", "NEU", "N"]


logger = datasets.utils.logging.get_logger()

class TweetDSConfig(datasets.BuilderConfig):

    def __init__(self, features=["text", "labels"], label_classes=_CLASS_NAMES, **kwargs):
        """BuilderConfig for SuperGLUE.

        Args:
        features: *list[string]*, list of the features that will appear in the
            feature dict. Should not include "label".
        data_url: *string*, url to download the zip file from.
        citation: *string*, citation for the data set.
        data_partition: *string*, url for information about the data set.
        label_classes: *list[string]*, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
        **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.2: Fixed non-nondeterminism in ReCoRD.
        # 1.0.1: Change from the pre-release trial version of SuperGLUE (v1.9) to
        #        the full release (v2.0).
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.2: Initial version.
        super().__init__(**kwargs)
        self.features = features
        self.label_classes = label_classes

    # @property
    # def features(self):  # noqa
    #     feat_dict = {
    #         "text": datasets.Value("string"),
    #         "labels": datasets.Sequence(
    #             datasets.ClassLabel(num_classes=3, names=_CLASS_NAMES)
    #         ),  # I coded this as a multiclass/multilabel just in case
    #     }
    #     return feat_dict


DEFAULT_FILES = {
    "train": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv",
}


class TweetDataset(datasets.GeneratorBasedBuilder):
    """Tweet Dataset"""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        TweetDSConfig(name="60-20-20", version=VERSION, description="60-20-20 partition"),
        TweetDSConfig(name="70-15-15", version=VERSION, description="70-15-15 partition"),
    ]

    DEFAULT_CONFIG_NAME = (
        "60-20-20"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def __init__(
        self,
        data_dir,
        files_have_header,
        train_file,
        dev_file,
        test_file,
        train_split_name,
        dev_split_name,
        test_split_name,
        limited_record_count,
        *args,
        **kwargs,
    ):
        super(TweetDataset, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        if not self.data_dir:
            raise ValueError("data_dir argument should be specified!")

        self.files_have_header = files_have_header
        self.train_file = train_file if train_file else DEFAULT_FILES["train"]
        self.dev_file = dev_file if dev_file else DEFAULT_FILES["dev"]
        self.test_file = test_file if test_file else DEFAULT_FILES["test"]
        self.train_split_name = train_split_name if train_split_name else "train"
        self.dev_split_name = dev_split_name if dev_split_name else "dev"
        self.test_split_name = test_split_name if test_split_name else "test"
        self.limited_record_count = limited_record_count

        logger.info(f"Data dir: {self.data_dir}")
        logger.info(f"Do files have header: {self.files_have_header}")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "labels": datasets.ClassLabel(num_classes=len(_CLASS_NAMES), names=_CLASS_NAMES)
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _get_split_generator(
        self,
        split: datasets.NamedSplit,
        file_path: str,
        split_name: Optional[str] = None,
    ):
        return datasets.SplitGenerator(
            name=str(split),
            gen_kwargs={  # These kwargs will be passed to _generate_examples
                "filepath": file_path,
                "split": split_name if split_name else str(split),
            },
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            self._get_split_generator(
                datasets.Split.TRAIN, os.path.join(self.data_dir, self.config.name, self.train_file), self.train_split_name
            ),
            self._get_split_generator(
                datasets.NamedSplit(self.dev_split_name), os.path.join(self.data_dir, self.config.name, self.dev_file), self.dev_split_name
            ),
            self._get_split_generator(
                datasets.Split.TEST, os.path.join(self.data_dir, self.config.name, self.test_file), self.test_split_name
            ),
        ]

    def _generate_examples(
        self,
        filepath,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        logger.info(f"Generating examples for {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            columns = self.config.features
            logger.info(f"Header columns {columns}")
            reader = csv.DictReader(f, delimiter=",", fieldnames=columns)
            for id_, row in enumerate(reader):
                if self.files_have_header and id_ == 0:
                    logger.info(f"Header skipped: {row}")
                    continue
                # print(row)
                keys_dict = {}
                for k in row.keys():
                    keys_dict[k] = row[k]
                yield id_, keys_dict

                if id_ == (self.limited_record_count - 1):
                    logger.warning(
                        f"Just generated {self.limited_record_count} records! So probably you are debugging!"
                    )
                    break
