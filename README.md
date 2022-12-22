# ML Project 2: Human-centerd Commonsense Benchmark
EPFL Machine Learning course project 2. Associated with NLP Lab. Commonsense reasoning benchmark and probing for large language models.

### Baseline Models:
We employed [T5](https://arxiv.org/pdf/1910.10683.pdf) based models.
* [UnifiedQA](https://arxiv.org/abs/2005.00700)
* [Macaw](https://arxiv.org/abs/2109.02593)
* [FLAN](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)
* [T0++](https://huggingface.co/bigscience/T0pp)

Also, we employed large language models.
* [OPT66B](https://huggingface.co/facebook/opt-66b/tree/main)
* [GPT3](https://openai.com/api/)

### Human-centered Commonsense Benchmark

* [Theory of Mind Task Dataset](https://arxiv.org/abs/1808.09352)
* [Social Interaction QA](https://arxiv.org/abs/1904.09728)
* [Complementary Commonsense](https://arxiv.org/abs/2106.00969)
* [SCRUPLES](https://paperswithcode.com/paper/scruples-a-corpus-of-community-ethical)
* [COmmonsense Dataset Adversarially-authored by Humans](https://arxiv.org/abs/1904.04365)

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
MODEL_TYPE="opt" # ["t5", "opt", "bloom", "gpt"]
MODEL_NAME_OR_PATH="facebook/opt-66b" # volume directory with model checkpoints (.bin) or hugginface download ('facebook/opt-66b').
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
N_GPU=8
```
