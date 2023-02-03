import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
from fast_ml.model_development import train_valid_test_split


def data_split(src_file, train_size=0.6, dev_size=0.2, test_size=0.2):
    print(f"Splitting data from file {src_file}")
    df = pd.read_csv(src_file)
    return train_valid_test_split(
        df, target="sentiment", train_size=train_size, valid_size=dev_size, test_size=test_size
    )


def save_split(dest_dir, file_names: List, X: List, Y: List):
    for file_name, x, y in zip(file_names, X, Y):
        dest = Path(dest_dir, file_name)
        print(f"Saving Dataset {dest}")
        df = pd.concat([x, y], axis=1)
        print(f"{df.info()}")
        print(f"{df.head(10)}")
        df.to_csv(dest, index=False)


def main():
    parser = argparse.ArgumentParser(description="Dataset splitter")

    parser.add_argument("--train_size", help="Percentage of the train set", type=float, default=0.6)

    parser.add_argument("--dev_size", help="Percentage of the dev set", type=float, default=0.2)

    parser.add_argument("--test_size", help="Percentage of the test set", type=float, default=0.2)

    parser.add_argument(
        "--src_file",
        help="Source data file to split",
        type=str,
        default="/Users/nfanlo/dev/spanish-classifier-tfg/dataset/original-combined/all.csv",
    )

    parser.add_argument(
        "--dest_dir",
        help="Destination directory where to save the split",
        type=str,
        default="/Users/nfanlo/dev/spanish-classifier-tfg/dataset/60-20-20",
    )

    args = parser.parse_args()
    print(args)

    isExist = os.path.exists(args.dest_dir)
    if not isExist:
        print(f"Creating directory {args.dest_dir}")
        os.makedirs(args.dest_dir)

    X_train, y_train, X_dev, y_dev, X_test, y_test = data_split(
        args.src_file, train_size=args.train_size, dev_size=args.dev_size, test_size=args.test_size
    )
    save_split(args.dest_dir, ["train.csv", "dev.csv", "test.csv"], [X_train, X_dev, X_test], [y_train, y_dev, y_test])


if __name__ == "__main__":
    main()
