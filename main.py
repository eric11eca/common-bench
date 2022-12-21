from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import logging

import random
import numpy as np
import torch

from runner import run
from inference import *


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--train_name", default="train")
    parser.add_argument("--dev_name", default="dev")
    parser.add_argument("--test_name", default="test")
    parser.add_argument("--output_dir", default="output",
                        type=str, required=False)

    parser.add_argument("--dataset", default="socialiqa")
    parser.add_argument("--task", default="socialiqa")

    parser.add_argument("--model_name_or_path",
                        default="macaw-3b", required=False)
    parser.add_argument("--model_type",
                        default="t5", required=False)

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_inference", action='store_true')
    parser.add_argument("--do_icl", action='store_true',
                        help="do in-context learning")
    parser.add_argument("--num_examples", default=4, type=int,
                        help="Number of examples to use for in-context learning")
    parser.add_argument("--search", action='store_true',
                        help="search for in-context learning examples")
    parser.add_argument("--encoder", default="simcl",
                        help="encoder to use for search")

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.08, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--callback_monitor', type=str, default='val_acc',
                        help='The metric to monitor for early stopping')
    parser.add_argument('--patience', type=int, default=4,
                        help='The number of epochs to wait before early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='The number of workers to use for data loading')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='The number of gpus to use')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='path to checkpoint')

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=10,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--wandb_api_key", type=str, default="9edee5b624841e10c88fcf161d8dc54d4efbee29",
                        help="The particular wandb api key to use [default='']")
    parser.add_argument('--wandb_entity', type=str, default='epfl_nlp_phd',
                        help="The particular wandb entity to use [default='']")
    parser.add_argument('--wandb_project', type=str, default='common-bench',
                        help="The particular wandb project to use [default='']")
    parser.add_argument('--wandb_name', type=str,
                        default='macaw-3b-socialiqa-eval')
    parser.add_argument('--wandb_data', type=str,
                        default='')
    parser.add_argument("--wandb_note",
                        dest="wandb_note",
                        default='empty',
                        type=str,
                        help="The note to use for the wandb [default='empty']")
    parser.add_argument("--wandb_model",
                        dest="wandb_model",
                        default='',
                        type=str,
                        help="Specifies a location to an existing wandb model [default='']")

    # for gpt3 api
    parser.add_argument("--gpt3_key",
                        type=str,
                        required=False)

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                                  logging.StreamHandler()])
    logger = logging.getLogger("common_bench.cli_maml")
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{args.output_dir}/{timestr}"
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir
    
    if args.model_name_or_path == 'gpt3':
        run_gpt3(args)

    elif args.do_inference:
        run_acclerate(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
