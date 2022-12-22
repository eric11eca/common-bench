# ML Project 2: Human-centerd Commonsense Benchmark
EPFL Machine Learning course project 2. Associated with NLP Lab. Commonsense reasoning benchmark and probing for large language models.

## Installation

```
python -r requirement.txt
```

We tested our python codes on the interactive mode of RunAI. Please look through if you are new user of [RunAI](https://github.com/sori424/runLLM).

## Run

```
bash run.sh
```

For running setup, you can change the configurations below. 

```
DATASET="socialiqa"
TASK="socialiqa"
MODEL_TYPE="opt"
MODEL_NAME_OR_PATH="facebook/opt-66b"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
N_GPU=8
```

cf. MODEL_TYPE? 
["t5", "opt", "bloom", "gpt"]

MODEL_NAME_OR_PATH?
Can be your volume directory with model checkpoints (.bin) or you can also directly download by passing argument (e.g., 'facebook/opt-66b').
