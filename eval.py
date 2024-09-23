import argparse
import os
from datasets import disable_caching

from core.utils import set_random_seed, load_dataset
from core.engine import get_engine, ENGINES

import torch

# Check if we have a GPU
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("Using the CPU! Exiting...")
    exit()

disable_caching()

model_type = "bb"
dataset = "multiwoz"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tsv_path", type=str, default=f"data/{dataset}-valid.tsv")
    parser.add_argument("--generator_load_path", type=str, default=f"models/{model_type}-{dataset}")
    parser.add_argument("--classifier_load_path", type=str, default=f"models/cls-{dataset}-action")
    parser.add_argument("--evaluator_load_path", type=str, default=f"models/intent-detect-{dataset}")
    parser.add_argument("--intent_predict_load_path", type=str, default=f"models/cls-{dataset}-intent")
    parser.add_argument("--method", type=str, default="nifty-intent", choices=list(ENGINES.keys()))
    parser.add_argument("--max_message_length", type=int, default=64)
    parser.add_argument("--max_response_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=0.5)

    return parser.parse_args()


def get_task(path: str):
    if "multiwoz" in path:
        return "multiwoz"
    elif "dstc8" in path:
        return "dstc8"
    else:
        raise ValueError(f"Unknown task for path: {path}")


def preprocess(batch, tokenizer, args):

    enc_inputs = tokenizer(
        batch["messages"], max_length=args.max_message_length, 
        padding="max_length", truncation=True
    )

    input_ids = enc_inputs["input_ids"]
    attention_mask = enc_inputs["attention_mask"]

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prefix": [" "] * len(batch["messages"])
    }

    return data


def _main(args):
    method = get_engine(args.method)(args)

    dataset = load_dataset(args.test_tsv_path,  lambda x: preprocess(x, method.tokenizer, args))
    dataset = dataset.filter(lambda x: x["cls_labels"] == 0)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print(f"Dataset size: {len(dataset)}")

    if not os.path.isdir("results"):
        os.makedirs("results")
    task = get_task(args.test_tsv_path)

    fpath = f"results/{args.method}-{model_type}-{task}-{args.alpha}.tsv"
    
    with open(fpath, "w", encoding="utf-8") as f:
        dataset = dataset.map(
            lambda x: method.predict(x, f), batched=True, batch_size=1
        )


def main():
    args = parse_args()
    set_random_seed(args.seed)
    _main(args)


 


if __name__ == "__main__":
    main()
