# classifier guidance logits classifier
import torch
from transformers.generation.logits_process import LogitsProcessor
from typing import Union

class ClassifierGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(self, classifier, cls_tokenizer, gpt_tokenizer, alpha=1.0, **kwargs):
        self.classifier = classifier
        self.cls_tokenizer = cls_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
        self.alpha = alpha
        
        self.prompt = None
        self.label_idx = None

        super().__init__(**kwargs)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def set_label_idx(self, label_idx: Union[int, list]):
        self.label_idx = label_idx

    def get_texts(self, texts):
        texts = [text + self.cls_tokenizer.sep_token + self.prompt for text in texts]
        return texts

    def __call__(self, input_ids, scores):
        # Scores are batch * num_beams x vocab_size
        
        # Apply classifier to topk possible continuations
        topk = torch.topk(scores, k=10, dim=-1)  # bsz x k

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < topk[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, -float("inf"))

        # Generate new input_ids and attention_mask
        new_input_ids = input_ids.unsqueeze(1).expand(-1, topk[0].shape[-1], -1)  # bsz x k x seq_len
        new_input_ids = torch.cat([
            new_input_ids,
            topk.indices.unsqueeze(-1),
        ], dim=-1)
        new_input_ids = new_input_ids.view(-1, new_input_ids.shape[-1])  # (bsz * k) x seq_len

        texts = self.gpt_tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)
        texts = self.get_texts(texts)
        new_inputs = self.cls_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        cls_scores = self.classifier(
            input_ids=new_inputs.input_ids.to(self.classifier.device),
            attention_mask=new_inputs.attention_mask.to(self.classifier.device),
        )

        scores = self.adjust_scores(scores, topk, cls_scores, self.label_idx)
        
        return scores

    def adjust_scores(self, scores, topk, cls_scores, label_idx, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        cls_scores = torch.log_softmax(cls_scores.logits, dim=-1)[:, label_idx]  # (bsz * k) x num_labels
        cls_scores = cls_scores.view(-1, topk[0].shape[-1])  # bsz x k
        scores = torch.log_softmax(scores, dim=-1)

        # Compute new scores
        for i in range(topk.indices.shape[0]):
            for j in range(topk.indices.shape[1]): 
                scores[i, topk.indices[i, j]] = scores[i, topk.indices[i, j]] + alpha * cls_scores[i, j]

        return scores

class ClassifierGuidanceIntentLogitsProcessor(ClassifierGuidanceLogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = None

    def get_texts(self, texts):
        texts = [self.message + self.cls_tokenizer.sep_token + text for text in texts]
        return texts



import pickle

id2str = {
    "0": "affirm",
    "1": "affirm intent",
    "2": "confirm",
    "3": "goodbye",
    "4": "inform",
    "5": "inform count",
    "6": "inform intent",
    "7": "negate",
    "8": "negate intent",
    "9": "notify failure",
    "10": "notify success",
    "11": "offer",
    "12": "offer intent",
    "13": "request",
    "14": "requests alts",
    "15": "req more",
    "16": "select",
    "17": "thank you"
}

class UnlikelihoodDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, gpt_tokenizer, alpha=1.0, rejected_intents=[], task="dstc8", **kwargs):
        self.gpt_tokenizer = gpt_tokenizer
        self.alpha = alpha

        self.id2act = pickle.load(open(f"data/{task}-id2act.pkl", "rb"))
        if task == "multiwoz":
            self.rejected_intents = " ".join([self.id2act[i] for i in rejected_intents])

        elif task == "dstc8":
            rejected_intents = [self.id2act[i] for i in rejected_intents]
            rejected_intents = [id2str[s] for i in rejected_intents for s in i.split(" ")]
            self.rejected_intents = " ".join(rejected_intents)

        self.reject_token_ids = self.gpt_tokenizer(self.rejected_intents)["input_ids"][:-1]
        
        super().__init__(**kwargs)

    def get_texts(self, texts):
        texts = [text + self.cls_tokenizer.sep_token + self.prompt for text in texts]
        return texts

    def __call__(self, input_ids, scores):
        # Scores are batch * num_beams x vocab_size
        base_scores = torch.softmax(scores, dim=-1)
        adj_scores = base_scores.clone()
        adj_scores[:, self.reject_token_ids] = 0.0
        scores = (1 - self.alpha) * base_scores + self.alpha * adj_scores
        
        return scores
