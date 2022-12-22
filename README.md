
#  ML Project 2: Human-centerd Commonsense Benchmark

EPFL Machine Learning course project 2. Associated with NLP Lab. Commonsense reasoning benchmark and probing for large language models.

###  Baseline Models

We employed [T5](https://arxiv.org/pdf/1910.10683.pdf) based models.

* [UnifiedQA](https://arxiv.org/abs/2005.00700)

* [Macaw](https://arxiv.org/abs/2109.02593)

* [FLAN](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)

* [T0++](https://huggingface.co/bigscience/T0pp)

Also, we employed large language models.

* [OPT66B](https://huggingface.co/facebook/opt-66b/tree/main)

* [GPT3](https://openai.com/api/)

###  Human-centered Commonsense Benchmark

We employed 5 different commonsense benchmarks from social interaction to ethical judgment that human could face in every real-life.

* [Theory of Mind Task Dataset](https://arxiv.org/abs/1808.09352)

* [Social Interaction QA](https://arxiv.org/abs/1904.09728)

* [Complementary Commonsense](https://arxiv.org/abs/2106.00969)

* [SCRUPLES](https://paperswithcode.com/paper/scruples-a-corpus-of-community-ethical)

* [COmmonsense Dataset Adversarially-authored by Humans](https://arxiv.org/abs/1904.04365)

##  Installation

```

pip install -r requirement.txt

```

We tested our python codes on the interactive mode of RunAI @ EPFL cluster. Please look through if you are new user of [RunAI](https://github.com/sori424/runLLM).

####  WANDB dataset/model versioning and loading

This repo is designed to work with wandb for dataset and model versioning, experimental visualization, etc.. Assuming that you have a [**wandb**](https://wandb.ai/home) account you first need to set your *WANDB_API_KEY*

```bash

export WANDB_API_KEY=XXXXXXXXXXXXXXXX

```

In the code above you can then specify: `--wandb_entity`, `--wandb_project` (the target project), `--wandb_name` (name of experiment), `--wandb_data` (for automatic loading of data), `--wandb_model` (for automatic loading of models). In **RunAI** wandb can be used by adding `WANDB_API_KEY` to the `env` variables. 

##  Quickstart
To run the code, simply execute the main bash script:
```

bash run.sh

```

For running setup, you can change the configurations below.

```

DATASET="socialiqa"

TASK="socialiqa"

MODEL_TYPE="opt" <-- select from ["t5", "opt", "bloom", "gpt"]

MODEL_NAME_OR_PATH="facebook/opt-66b" <-- volume directory with model checkpoints (.bin) or hugginface download ('facebook/opt-66b').

TRAIN_BATCH_SIZE=4   <-- training batch size

PREDICT_BATCH_SIZE=1 <-- prediction batch size

N_GPU=8 <-- number of GPUs to use

```

## In-context Learning

To run the code for vinalla **In-context Learning**, first modify the running command in `run.sh`:
```

accelerate launch main.py \
	--do_inference \
	--dataset ${DATASET} \
	--task ${TASK} \
	--model_type ${MODEL_TYPE} \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--predict_batch_size ${PREDICT_BATCH_SIZE} \
	--wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-icl-4-rand \
	--n_gpu ${N_GPU} \
	--max_data 0 \
	--do_icl \			<-- **Add this flag**
	--num_examples 2	<-- **Number of demonstrations used**
	
```
Then, execute the script. To use examples pre-selected  by the KNN method, modify the running command:
```

accelerate launch main.py \
	--do_inference \
	--dataset ${DATASET} \
	--task ${TASK} \
	--model_type ${MODEL_TYPE} \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--predict_batch_size ${PREDICT_BATCH_SIZE} \
	--wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-icl-4-rand \
	--n_gpu ${N_GPU} \
	--max_data 0 \
	--do_icl \			
	--num_examples 2	
	--search 			<-- **Add this flag**
	--encoder simcse	<-- **Name of the sentence encoder for embedding**
```
Then, execute the script.

## KNN Example Selection
```

python dynamic_icl.py \
	--dataset $DATASET_NAME \
	--task $TASK_NAME \
	--encoder_name simcse \ <-- nli_mean or simcse
	--metric cosine \	<-- cosine or euclidean
	--num_neighbors 16
	
```
The output file will be under the name `$DATA_DIR/$DATASET/train_$ENCODER_NAME.json`
