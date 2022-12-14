#!/bin/bash

DATASET="tomi"
TASK="tomi"
MODEL_TYPE="t5"
MODEL_NAME_OR_PATH="macaw-11b"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=1
N_GPU=4

python main.py \
    --do_eval \
    --dataset ${DATASET} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --learning_rate 3e-5 \
    --wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-eval \
    --n_gpu ${N_GPU}