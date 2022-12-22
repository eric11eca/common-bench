# ML Project 2: Human-centerd Commonsense Benchmark
EPFL Machine Learning course project 2. Associated with NLP Lab. Commonsense reasoning benchmark and probing for large language models.

### Baseline Models
We employed [T5](https://arxiv.org/pdf/1910.10683.pdf) based models.
* [UnifiedQA](https://arxiv.org/abs/2005.00700)
* [Macaw](https://arxiv.org/abs/2109.02593)
* [FLAN](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)
* [T0++](https://huggingface.co/bigscience/T0pp)

Also, we employed large language models.
* [OPT66B](https://huggingface.co/facebook/opt-66b/tree/main)
* [GPT3](https://openai.com/api/)

### Human-centered Commonsense Benchmark
We employed 5 different commonsense benchmarks from social interaction to ethical judgment that human could face in every real-life. 
* [Theory of Mind Task Dataset](https://arxiv.org/abs/1808.09352)
* [Social Interaction QA](https://arxiv.org/abs/1904.09728)
* [Complementary Commonsense](https://arxiv.org/abs/2106.00969)
* [SCRUPLES](https://paperswithcode.com/paper/scruples-a-corpus-of-community-ethical)
* [COmmonsense Dataset Adversarially-authored by Humans](https://arxiv.org/abs/1904.04365)

## Installation

```
pip install -r requirement.txt
```

We tested our python codes on the interactive mode of RunAI @ EPFL cluster. Please look through if you are new user of [RunAI](https://github.com/sori424/runLLM).

# WANDB dataset/model versioning and loading
This repo is designed to work with wandb for dataset and model
versioning, experimental visualization, etc.. Assuming that you have a
[**wandb**](https://wandb.ai/home) account you first need to set your
*WANDB_API_KEY*
```bash
export WANDB_API_KEY=XXXXXXXXXXXXXXXX
```
In the code above you can then specify: `--wandb_entity`,
`--wandb_project` (the target project), `--wandb_name` (name of
experiment), `--wandb_data` (for automatic loading of data),
`--wandb_model` (for automatic loading of models). In **RunAI** wandb
can be used by adding `WANDB_API_KEY` to the `env` variables in the
yaml script (see below). You need a [wandb](https://wandb.ai/site) account. 

## Quickstart

```
bash run.sh
```

For running setup, you can change the configurations below. 

```
DATASET="socialiqa" 
TASK="socialiqa"
MODEL_TYPE="opt" # ["t5", "opt", "bloom", "gpt"]
MODEL_NAME_OR_PATH="facebook/opt-66b" # volume directory with model checkpoints (.bin) or hugginface download ('facebook/opt-66b').
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
N_GPU=8
```

