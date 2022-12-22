#!/bin/bash

DATASET="scruples-dilemma"
TASK="dilemma"
MODEL_TYPE="gpt"
MODEL_NAME_OR_PATH="gpt3"
TRAIN_BATCH_SIZE=4
PREDICT_BATCH_SIZE=32
N_GPU=8

accelerate launch main.py \
    --do_inference \
    --dataset ${DATASET} \
    --task ${TASK} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --predict_batch_size ${PREDICT_BATCH_SIZE} \
    --wandb_name ${MODEL_NAME_OR_PATH}-${DATASET}-eval \
    --n_gpu ${N_GPU} \
    --max_data 0 \
    # --do_icl \
    # --num_examples 2 \
    # --search \
    # --encoder_name simcse