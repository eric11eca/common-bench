# ML Project 2: Human-centerd Commonsense Benchmark
EPFL Machine Learning course project 2. Associated with NLP Lab. Commonsense reasoning benchmark and probing for large language models.

## Installation

```
python -r requirement.txt
```
You need a wandb credentials.

## Run

```
bash run.sh
```

### For running setup

Change the configurations below depend on your need. 

```
DATASET="socialiqa"
TASK="socialiqa"
MODEL_TYPE="opt"
MODEL_NAME_OR_PATH="facebook/opt-66b"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
N_GPU=8
```
cf.

* MODEL? model_class_registry = {
    "t5": AutoModelForSeq2SeqLM,
    "opt": AutoModelForCausalLM,
    "bloom": AutoModelForCausalLM,
    "gpt": AutoModelForCausalLM
}

`MODEL_NAME_OR_PATH` can be a model path that you saved the model before (saved_)
