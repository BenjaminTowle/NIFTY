import argparse
import os
from datasets import disable_caching
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
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
SEED = 0
MAX_LENGTH = 128
MAX_REPLY_LENGTH = 64

def parse_args():
    dataset = "dstc8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv_path", type=str, default=f"data/{dataset}-train.tsv")
    parser.add_argument("--valid_tsv_path", type=str, default=f"data/{dataset}-valid.tsv")
    parser.add_argument("--model_load_path", type=str, default=MODEL_NAME)
    parser.add_argument("--model_save_path", type=str, default=f"models/intent-detect-{dataset}")

    # Probably don't need to changes these args
    parser.add_argument("--per_device_train_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    
    parse_args, _ = parser.parse_known_args()
    
    return parse_args
   

def preprocess(batch, tokenizer, max_length=MAX_LENGTH):
    batch["responses"] = [r if r is not None else "Good bye ." for r in batch["responses"]]
    tokens = tokenizer(batch["responses"], truncation=True, max_length=max_length)

    return {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "labels": batch["acts"],
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    
    return {"accuracy": accuracy}


def main():
    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_load_path)
    tokenize_fn = lambda batch: preprocess(batch, tokenizer, args.max_length)
    train_dataset = load_dataset(args.train_tsv_path, tokenize_fn)
    valid_dataset = load_dataset(args.valid_tsv_path, tokenize_fn)

    # Remove all samples that have empty acts column
    train_dataset = train_dataset.filter(lambda x: x["acts"] is not None)
    valid_dataset = valid_dataset.filter(lambda x: x["acts"] is not None)
    
    num_labels = max(train_dataset["acts"] + valid_dataset["acts"]) + 1
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

    trainer.train()

    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)

if __name__ == "__main__":
    main()
