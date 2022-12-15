#!/bin/bash

DATASET="tomi"
TASK="tomi"
MODEL_TYPE="t5"
MODEL_NAME_OR_PATH="macaw-11b"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
N_GPU=1

mkdir -p data/${DATASET}
wandb artifact get epfl_nlp_phd/data-collection/${DATASET}:v0 --root data/${DATASET}

accelerate launch main.py \
    --do_inference \
    --dataset ${DATASET} \
    --task ${TASK} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-eval \
    --n_gpu ${N_GPU}