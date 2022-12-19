#!/bin/bash

DATASET="socialiqa"
TASK="socialiqa"
MODEL_TYPE="gpt"
MODEL_NAME_OR_PATH="gptj"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=2
N_GPU=4

mkdir -p data/${DATASET}
wandb artifact get epfl_nlp_phd/data-collection/${DATASET}:v0 --root data/${DATASET}

accelerate launch main.py \
    --do_inference \
    --dataset ${DATASET} \
    --task ${TASK} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-icl \
    --n_gpu ${N_GPU} \
    --do_icl \
    --num_examples 4 \
    --search \
    --encoder nli_mean 