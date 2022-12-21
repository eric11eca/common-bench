#!/bin/bash

DATASET="socialiqa"
TASK="socialiqa"
MODEL_TYPE="opt"
MODEL_NAME_OR_PATH="gpt3"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=16
N_GPU=8

mkdir -p data/${DATASET}
wandb artifact get epfl_nlp_phd/data-collection/${DATASET}:latest --root data/${DATASET}

accelerate launch main.py \
    --do_inference \
    --dataset ${DATASET} \
    --task ${TASK} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-eval \
    --n_gpu ${N_GPU} \
    --max_data 0