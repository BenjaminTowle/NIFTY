import argparse
import os
from datasets import disable_caching
from random import randint
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from core.utils import set_random_seed, load_dataset

os.environ["WANDB_DISABLED"] = "true"
disable_caching()

############################################
# Hyperparameters
############################################

MODEL_NAME = "distilbert-base-uncased"
#MODEL_NAME = "models/cls-multiwoz-intent"
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
SEED = 0
MAX_LENGTH = 128
MAX_REPLY_LENGTH = 64

def parse_args():
    dataset = "dstc8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv_path", type=str, default=f"data/{dataset}-train.tsv")
    parser.add_argument("--valid_tsv_path", type=str, default=f"data/{dataset}-test.tsv")
    parser.add_argument("--model_load_path", type=str, default=MODEL_NAME)
    parser.add_argument("--model_save_path", type=str, default=f"models/cls-{dataset}")
    parser.add_argument("--task", type=str, default="action", choices=["intent", "action"])

    # Probably don't need to changes these args
    parser.add_argument("--per_device_train_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    
    parse_args, _ = parser.parse_known_args()
    
    return parse_args


def preprocess_action(batch, tokenizer, max_length=MAX_LENGTH):
    batch["suggestions"] = [S.split("//") for S in batch["suggestions"]]
    responses = [r.split() if r is not None else "Good bye .".split() for r in batch["responses"]]  # A couple of None responses in dstc8
    #responses = [r.split() for r in batch["responses"]]
    split_idxs = [randint(1, len(ids)) for ids in responses]
    responses = [" ".join(r[:idx]) for r, idx in zip(responses, split_idxs)]
    sep = tokenizer.sep_token
    texts = [r + sep + sep.join(s) for r, s in zip(responses, batch["suggestions"])]
    tokens = tokenizer(texts, truncation=True, max_length=max_length)

    return {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "labels": batch["cls_labels"],
        "lengths": split_idxs
    }

def preprocess_intent(batch, tokenizer, max_length=MAX_LENGTH):
    #responses = batch["responses"]
    #batch = {k: [v for v, r in zip(batch[k], responses) if r is not None] for k in batch.keys()}
    #if None in responses:
     #   print("None in responses")
    
    responses = [r.split() if r is not None else "Good bye .".split() for r in batch["responses"]]  # A couple of None responses in dstc8
    split_idxs = [randint(1, len(ids)) for ids in responses]

    # Split resources between predicting from only message and from message + prefix
    binaries = [randint(0, 1) for _ in range(len(responses))]
    responses = [" ".join(r[:idx]) if b == 1 else " " for r, idx, b in zip(responses, split_idxs, binaries)]
    sep = tokenizer.sep_token
    texts = [m + sep + r for m, r in zip(batch["messages"], responses)]
    tokens = tokenizer(texts, truncation=True, max_length=max_length)

    return {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "labels": batch["acts"],
        "lengths": split_idxs
    }


def compute_metrics_action(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_intent(pred):
    # Multilabel classification
    labels = pred.label_ids
    preds = pred.predictions.argsort(axis=-1)[:, -3:]
    r1 = (preds[:, -1] == labels).mean()
    r3 = []
    for i in range(len(preds)):
        if labels[i] in preds[i]:
            r3.append(1)
        else:
            r3.append(0)
    r3 = sum(r3) / len(r3)

    return {"r@1": r1, "r@3": r3}

def main():
    args = parse_args()
    args.model_save_path += f"-{args.task}"
    set_random_seed(args.seed)

    preprocess = preprocess_action if args.task == "action" else preprocess_intent
    compute_metrics = compute_metrics_action if args.task == "action" else compute_metrics_intent

    tokenizer = AutoTokenizer.from_pretrained(args.model_load_path)
    tokenize_fn = lambda batch: preprocess(batch, tokenizer, args.max_length)
    train_dataset = load_dataset(args.train_tsv_path, tokenize_fn)
    valid_dataset = load_dataset(args.valid_tsv_path, tokenize_fn)

    if args.task == "intent":
        num_labels = max(train_dataset["acts"] + valid_dataset["acts"]) + 1
    else:
        num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(args.model_load_path, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if not os.path.isdir("models"):
        os.makedirs("models")

    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        save_steps=2000,
        num_train_epochs=5,
        save_total_limit=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.mode == "eval":
        print(trainer.evaluate(valid_dataset))
        return

    trainer.train()

    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)

if __name__ == "__main__":
    main()
