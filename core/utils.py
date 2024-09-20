import torch
import random
import numpy as np
import torch
import pandas as pd
from datasets import Dataset

def load_dataset(path: str, tokenize_fn: callable = None, batch_size: int = 32):
    cols = ["messages", "responses", "cls_labels", "suggestions", "acts"]
    dataset = Dataset.from_pandas(pd.read_csv(path, sep="\t", names=cols))
    if tokenize_fn is not None:
        dataset = dataset.map(tokenize_fn, batched=True, batch_size=batch_size)

    return dataset


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
