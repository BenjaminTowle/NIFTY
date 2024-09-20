# Script to generate stats about the dataset

import pandas as pd
from datasets import Dataset

def read_dataset(path: str):
    cols = ["messages", "responses", "cls_labels", "suggestions", "acts"]
    dataset = Dataset.from_pandas(pd.read_csv(path, sep="\t", names=cols))
    return dataset

def filter_dataset(dataset: Dataset):
    return dataset.filter(lambda x: x["cls_labels"] == 0)

def main():
    dataset = "multiwoz"
    paths = [
        f"data/{dataset}-train.tsv", 
        f"data/{dataset}-valid.tsv", 
        f"data/{dataset}-test.tsv"
    ]

    train, valid, test = [read_dataset(path) for path in paths]
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    num_labels = max(train["acts"] + valid["acts"] + test["acts"]) + 1
    print(f"Number of labels: {num_labels}")

    train = filter_dataset(train)
    valid = filter_dataset(valid)
    test = filter_dataset(test)
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    num_labels = max(train["acts"] + valid["acts"] + test["acts"]) + 1
    print(f"Number of labels: {num_labels}")

if __name__ == "__main__":
    main()
