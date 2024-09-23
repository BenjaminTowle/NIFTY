Official code for the EMNLP 2024 accepted paper: *Enhancing AI Assisted Writing with One-Shot Implicit Negative Feedback*.


## Setup
The code is designed to run on the MultiWOZ v2.2 and SGD / DSTC8 datasets. To implement your own dataset, you should create a `.tsv` file, with the columns corresponding to: *message, response, user action, suggestions, intent*, and place it in the `data/` directory.  Intent should be represented as an integer, and suggestions should be a single string, separated by `//`.

The code is tested in Python 3.8.12.  To install prerequisites, run:
```
pip install -r requirements.txt
```

## Project structure
The main directory of the project contains the different scripts that need to be run to recreate the results from the paper. The `core` directory contains the implementations for classifier guidance and the different methods reported on in this paper. The smart reply code can be cloned from `https://github.com/BenjaminTowle/STAR`, but for convenience we already include the preprocessed files in the `data/` directory.


## Training
The project requires three separate models to be trained: an unconditional generator, a classifier to predict user actions / intents, and an evaluator to detect intents. To train the generator, run for example:

```
python train_generator.py --train_tsv_path PATH/TO/YOUR/TRAIN/DATA \
    --valid_tsv_path PATH/TO/YOUR/VALID/DATA \
    --model_load_path facebook/blenderbot-400M-distill \
    --model_save_path models/bb-multiwoz \
```

To train the classifier run:
```
python train_classifier.py --train_tsv_path PATH/TO/YOUR/TRAIN/DATA \
    --valid_tsv_path PATH/TO/YOUR/VALID/DATA \
    --model_load_path distilbert-base-uncased \
```

To train the evaluator run:
```
python train_evaluator.py --train_tsv_path PATH/TO/YOUR/TRAIN/DATA \
    --valid_tsv_path PATH/TO/YOUR/VALID/DATA \
    --model_load_path distilbert-base-uncased \
    --model_save_path models/intent-detect-multiwoz \
```

## Testing
The testing pipeline comprises two stages. Firstly, a `eval.py` file generates predictions for each of the test samples. See `core/engine.py` for a list of possible methods that can be run.
```
python eval.py --method nifty-intent \
    --test_tsv_path PATH/TO/YOUR/TEST/DATA \
    --generator_load_path PATH/TO/YOUR/GENERATOR \
    --classifier_load_path PATH/TO/YOUR/CLASSIFIER \
    --evaluator_load_path PATH/TO/YOUR/EVALUATOR \
```
Finally, we use `compute_metrics.py` to evaluate the predictions. Note that the location for where the predictions are stored is automatically known by the script.
```
python compute_metrics.py --evaluator_load_path PATH/TO/YOUR/EVALUATOR \
```

## Results
### MultiWOZ

| Base       | Methods         | ROUGE-L  | R@1      |
|------------|-----------------|----------|----------|
| BlenderBot | Baseline        | 25.0     | 15.6     |
|            | Unlikelihood    | 25.3     | 15.0     |
|            | Rules-based     | 25.2     | 17.5     |
|            | Reranker Action | 24.9     | 15.4     |
|            | Reranker Intent | 25.2     | 17.9     |
|            | NIFTY Action    | 25.3     | 18.1     |
|            | NIFTY Intent    | **27.2** | **26.1** |
| T5         | Baseline        | 27.7     | 16.4     |
|            | Unlikelihood    | 25.3     | 11.1     |
|            | Rules-based     | 27.8     | 18.0     |
|            | Reranker Action | 24.8     | 12.8     |
|            | Reranker Intent | 27.6     | 17.1     |
|            | NIFTY Action    | 24.8     | 12.7     |
|            | NIFTY Intent    | **29.0** | **28.5** |


### Schema-guided Dialog

| Base       | Methods         | ROUGE-L  | R@1      |
|------------|-----------------|----------|----------|
| BlenderBot | Baseline        | 25.0     | 15.6     |
|            | Unlikelihood    | 25.3     | 15.0     |
|            | Rules-based     | 25.2     | 17.5     |
|            | Reranker Action | 24.9     | 15.4     |
|            | Reranker Intent | 25.2     | 17.9     |
|            | NIFTY Action    | 25.3     | 18.1     |
|            | NIFTY Intent    | **27.2** | **26.1** |
| T5         | Baseline        | 27.7     | 16.4     |
|            | Unlikelihood    | 25.3     | 11.1     |
|            | Rules-based     | 27.8     | 18.0     |
|            | Reranker Action | 24.8     | 12.8     |
|            | Reranker Intent | 27.6     | 17.1     |
|            | NIFTY Action    | 24.8     | 12.7     |
|            | NIFTY Intent    | **29.0** | **28.5** |


## Citation