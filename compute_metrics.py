import argparse
import pandas as pd
import re

from transformers import pipeline
from rouge_score import rouge_scorer
import evaluate


def parse_args():
    dataset = "dstc8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=dataset)
    parser.add_argument("--results_path", type=str, default=f"results/nifty-intent-bb-{dataset}.tsv")
    parser.add_argument("--evaluator_load_path", type=str, default=f"models/intent-detect-{dataset}")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
bleu = evaluate.load("bleu")


def rouge(targets, preds):
    scores = []
    for t, p in zip(targets, preds):
        scores.append(scorer.score(t, p)["rougeL"].fmeasure)

    return scores

def intent_accuracy(targets, preds, args):
    evaluator = pipeline(
        "text-classification", model=args.evaluator_load_path, 
        tokenizer=args.evaluator_load_path, device=args.device
    )

    predicted_acts = []
    for prediction in preds:
        predicted_act = evaluator(prediction)[0]["label"]
        predicted_act = int(re.search(r"LABEL_(\d+)", predicted_act).group(1))
        predicted_acts.append(predicted_act)
    accs = [int(pred == act) for pred, act in zip(predicted_acts, targets)]

    return accs


def compute_individual_metrics(args, path: str):
    results = pd.read_csv(
        path, sep="\t", 
        names=["messages", "predictions", "targets", "acts"]
    )

    accs = intent_accuracy(results["acts"], results["predictions"], args)
    print(f"Model: {path} -- DA Accuracy: {sum(accs) / len(accs)}")

    _rouge = rouge(results["targets"], results["predictions"])
    print(f"Model: {path} -- Rouge: {sum(_rouge) / len(_rouge)}")

    return {"rouge": _rouge}

def main():
    args = parse_args()
    metrics = compute_individual_metrics(args, args.results_path)


if __name__ == "__main__":
    main()
