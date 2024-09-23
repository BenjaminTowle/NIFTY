import argparse
import os
from datasets import disable_caching
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback, 
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM
)
from peft import LoraConfig, TaskType, get_peft_model

from core.utils import set_random_seed, load_dataset


disable_caching()


############################################
# Hyperparameters
############################################

MODEL_NAMES = {
    "bb": "facebook/blenderbot-400M-distill",
    "t5": "google-t5/t5-base",
}
MODEL2TARGET_MODULES = {
    "bb": ["q_proj", "v_proj"],
    "t5": ["q", "v"]
}


#MODEL_NAME = "facebook/blenderbot-400M-distill"
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
SEED = 0
MAX_MESSAGE_LENGTH = 64
MAX_REPLY_LENGTH = 64

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Change as needed
dataset = "dstc8"
model_type = "bb"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv_path", type=str, default=f"data/{dataset}-train.tsv")
    parser.add_argument("--valid_tsv_path", type=str, default=f"data/{dataset}-valid.tsv")
    parser.add_argument("--model_load_path", type=str, default=MODEL_NAMES[model_type])
    parser.add_argument("--model_save_path", type=str, default=f"models/{model_type}-{dataset}")

    # Probably don't need to changes these args
    parser.add_argument("--per_device_train_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_message_length", type=int, default=MAX_MESSAGE_LENGTH)
    parser.add_argument("--max_reply_length", type=int, default=MAX_REPLY_LENGTH)
    
    parse_args, _ = parser.parse_known_args()
    
    return parse_args


class CustomCallback(TrainerCallback):
    # Sanity check to see if the model is working

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        super().__init__()

    def on_log(self, *args, **kwargs):
        message = "Hello how are you doing today?"
        inputs = self.tokenizer([message], return_tensors="pt")
        inputs = {k: v.to(kwargs["model"].device) for k, v in inputs.items()}
        
        kwargs["model"].eval()
        output = kwargs["model"].generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            num_beams=10,
        )
        kwargs["model"].train()
        print(self.tokenizer.decode(output[0], skip_special_tokens=False))


def tokenize(batch, tokenizer, max_message_length=MAX_MESSAGE_LENGTH, max_reply_length=MAX_REPLY_LENGTH):
    prompt_tokens = tokenizer(batch["messages"], max_length=max_message_length, truncation=True)
    completion_tokens = tokenizer(batch["responses"], max_length=max_reply_length, truncation=True)

    return {
        "input_ids": prompt_tokens["input_ids"],
        "attention_mask": prompt_tokens["attention_mask"],
        "labels": completion_tokens["input_ids"],
        "cls_labels": batch["cls_labels"],
    }


def main():
    args = parse_args()
    set_random_seed(args.seed)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_load_path, cache_dir="P:\.hf_cache")
    print(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=LORA_R, 
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=MODEL2TARGET_MODULES[model_type])
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_load_path)
    tokenize_fn = lambda batch: tokenize(batch, tokenizer, args.max_message_length, args.max_reply_length)
    train_dataset = load_dataset(args.train_tsv_path, tokenize_fn)
    valid_dataset = load_dataset(args.valid_tsv_path, tokenize_fn)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    if not os.path.isdir("models"):
        os.makedirs("models")

    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        save_steps=2000,
        num_train_epochs=1,
        save_total_limit=1,
        evaluation_strategy="epoch",
        logging_steps=100,
        logging_strategy="steps",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        run_name=model_type,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[CustomCallback(tokenizer)]
    )

    trainer.train()

    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)

if __name__ == "__main__":
    main()
