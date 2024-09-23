import abc
import argparse
import numpy as np
import re

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    LogitsProcessorList,
    pipeline
)

from .classifier_guidance import (
    ClassifierGuidanceIntentLogitsProcessor, 
    ClassifierGuidanceLogitsProcessor,
    UnlikelihoodDecodingLogitsProcessor
)


class Engine(abc.ABC):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.args = args
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.generator_load_path, cache_dir="P:\.hf_cache").to(args.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_load_path, cache_dir="P:\.hf_cache")
        super().__init__()

    def _generate(self, batch, **kwargs):
        outputs = self.model.generate(
            input_ids=batch["input_ids"].to(self.model.device), 
            attention_mask=batch["attention_mask"].to(self.model.device), 
            logits_processor=kwargs.get("logits_processor_list", None),
            num_beams=5,
            num_return_sequences=kwargs.get("num_return_sequences", 1),
        )
        return outputs

    def generate(self, batch, **kwargs):
        return self._generate(batch, **kwargs)

    def _generate_completion(self, batch, **kwargs):
        outputs = self.model.generate(
            input_ids=batch["input_ids"].to(self.model.device), 
            attention_mask=batch["attention_mask"].to(self.model.device), 
            decoder_input_ids=batch["decoder_input_ids"].to(self.model.device),
            max_new_tokens=6,
            logits_processor=kwargs.get("logits_processor_list", None),
            num_beams=1,
            num_return_sequences=kwargs.get("num_return_sequences", 1)
        )
        return outputs

    @staticmethod
    def write(batch, response, f):
        f.write(f'{batch["messages"][0]}\t{response}\t{batch["responses"][0]}\t{batch["acts"][0]}\n')

    def predict(self, batch, f):
        output = self._predict(batch)
        response = self.tokenizer.decode(output, skip_special_tokens=True)
        self.write(batch, response, f)

    @abc.abstractmethod
    def _predict(self, batch):
        pass


class BaselineEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        super().__init__(args)

    def _predict(self, batch):
        outputs = self.generate(batch)
        return outputs[0]


class RulesBasedEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.evaluator = pipeline(
            "text-classification", model=args.evaluator_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device
        )
        
        super().__init__(args)

    def _predict(self, batch):
        sugg_intents = self.evaluator(batch["suggestions"][0].split("//"))
        sugg_intents = [s["label"] for s in sugg_intents]

        outputs = self.generate(batch, num_return_sequences=5)
        candidates = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        cands_intents = self.evaluator(candidates)
        cands_intents = [s["label"] for s in cands_intents]
        for i, intent in enumerate(cands_intents):
            if intent not in sugg_intents:
                return outputs[i]

        return outputs[0]

        
class RerankerActionEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.reranker = pipeline(
            "text-classification", 
            model=args.classifier_load_path, 
            tokenizer=args.classifier_load_path, 
            device=args.device,
            return_all_scores=True
        )
        super().__init__(args)

    def _predict(self, batch):
        outputs = self.generate(batch, num_return_sequences=5)
        candidates = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        sep = self.reranker.tokenizer.sep_token
        texts = [r + sep + sep.join(batch["suggestions"][0].split("//")) for r in candidates]
        reranked_candidates = self.reranker(texts)
        idx = np.argmax([cand[0]["score"] for cand in reranked_candidates])
        return outputs[idx]

        
class RerankerIntentEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.evaluator = pipeline(
            "text-classification", model=args.evaluator_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device
        )
        self.intent_predictor = pipeline(
            "text-classification", model=args.intent_predict_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device,
            return_all_scores=True
        )
        super().__init__(args)

    def _predict(self, batch):
        sugg_intents = self.evaluator(batch["suggestions"][0].split("//"))
        sugg_intents = [s["label"] for s in sugg_intents]
        
        next_intents = self.intent_predictor(batch["messages"][0])[0]
        for intent in next_intents:
            if intent["label"] in sugg_intents:
                intent["score"] = 0.0
        idx = np.argmax([intent["score"] for intent in next_intents])

        outputs = self.generate(batch, num_return_sequences=5)
        candidates = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        reranked_candidates = self.intent_predictor(candidates)
        idx = np.argmax([cand[idx]["score"] for cand in reranked_candidates])

        return outputs[idx]


class NiftyActionEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        super().__init__(args) 
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            args.classifier_load_path).to(args.device)
        self.classifier.eval()
        self.cls_tokenizer = AutoTokenizer.from_pretrained(
            args.classifier_load_path)
        self.logits_processor = ClassifierGuidanceLogitsProcessor(
            self.classifier, self.cls_tokenizer, self.tokenizer) 

    def _predict(self, batch):
        prompt = self.logits_processor.cls_tokenizer.sep_token.join(batch["suggestions"][0].split("//"))
        self.logits_processor.set_prompt(prompt)
        self.logits_processor.set_label_idx(batch["cls_labels"][0])
        logits_processor_list = LogitsProcessorList([self.logits_processor])
        self.logits_processor.do_stop = False

        outputs = self.generate(batch, logits_processor_list=logits_processor_list)
        return outputs[0]

class NiftyIntentEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.evaluator = pipeline(
            "text-classification", model=args.evaluator_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device
        )
        self.intent_predictor = pipeline(
            "text-classification", model=args.intent_predict_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device,
            return_all_scores=True
        )

        self.accs = []
        
        super().__init__(args)

        self.logits_processor = ClassifierGuidanceIntentLogitsProcessor(
            self.intent_predictor.model, self.intent_predictor.tokenizer, 
            self.tokenizer, alpha=args.alpha
        )

    def _predict(self, batch):
        sugg_intents = self.evaluator(batch["suggestions"][0].split("//"))
        sugg_intents = [s["label"] for s in sugg_intents]
        
        next_intents = self.intent_predictor(batch["messages"][0])[0]
        for intent in next_intents:
            if intent["label"] in sugg_intents:
                intent["score"] = 0.0
        idx = np.argmax([intent["score"] for intent in next_intents])

        self.logits_processor.set_label_idx(idx)
        self.logits_processor.message = batch["messages"][0]
        logits_processor_list = LogitsProcessorList([self.logits_processor])
        self.logits_processor.do_stop = False
        outputs = self.generate(batch, logits_processor_list=logits_processor_list)
        
        return outputs[0]


class OracleEngine(Engine):
    # Sanity check by using ground truth intent
    def __init__(self, args: argparse.ArgumentParser) -> None:

        super().__init__(args)

        self.intent_predictor = pipeline(
            "text-classification", model=args.intent_predict_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device,
            return_all_scores=True
        )

        self.logits_processor = ClassifierGuidanceIntentLogitsProcessor(
            self.intent_predictor.model, self.intent_predictor.tokenizer, 
            self.tokenizer,
        )

    def _predict(self, batch):
        self.logits_processor.set_label_idx(batch["acts"][0])
        self.logits_processor.message = batch["messages"][0]
        logits_processor_list = LogitsProcessorList([self.logits_processor])
        self.logits_processor.do_stop = False

        outputs = self.generate(batch, logits_processor_list=logits_processor_list)
        return outputs[0]


class UnlikelihoodDecodingEngine(Engine):
    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.evaluator = pipeline(
            "text-classification", model=args.evaluator_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device
        )
        self.intent_predictor = pipeline(
            "text-classification", model=args.intent_predict_load_path, 
            tokenizer="distilbert-base-uncased", device=args.device,
            return_all_scores=True
        )

        self.accs = []
        
        super().__init__(args)

    def _predict(self, batch):
        sugg_intents = self.evaluator(batch["suggestions"][0].split("//"))
        sugg_intents = [s["label"] for s in sugg_intents]
        sugg_intents = [int(re.search(r"LABEL_(\d+)", s).group(1)) for s in sugg_intents]

        self.logits_processor = UnlikelihoodDecodingLogitsProcessor( 
            self.tokenizer, alpha=0.5, rejected_intents=sugg_intents
        )

        logits_processor_list = LogitsProcessorList([self.logits_processor])

        outputs = self.generate(batch, logits_processor_list=logits_processor_list)
        
        return outputs[0]




ENGINES = {
    "baseline": BaselineEngine,
    "reranker-action": RerankerActionEngine,
    "reranker-intent": RerankerIntentEngine,
    "nifty-action": NiftyActionEngine,
    "nifty-intent": NiftyIntentEngine,
    "oracle": OracleEngine,
    "unlikelihood": UnlikelihoodDecodingEngine,
    "rules-based": RulesBasedEngine
}


def get_engine(engine: str):
    return ENGINES[engine]
