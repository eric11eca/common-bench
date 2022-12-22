#!/bin/bash

DATASET="com2sense"
TASK="com2sense"
MODEL_TYPE="gpt"
MODEL_NAME_OR_PATH="gpt3"
PREDICT_BATCH_SIZE=32
N_GPU=4

mkdir -p data/${DATASET}
wandb artifact get epfl_nlp_phd/data-collection/${DATASET}:latest --root data/${DATASET}


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
    # --search \
    # --encoder nli_mean 