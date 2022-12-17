# usage:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#

import json
import time
import os
import logging
import wandb
import deepspeed
import torch
import torch.distributed as dist

from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from argparse import ArgumentParser

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode

from common_bench.dataset import CommonDataset
from common_bench.model import TranslationOutput
from common_bench.utils.py_io import *

util_logger = logging.getLogger(
    'common_bench.bloom_ds_inference'
)


tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8",
                        "microsoft/bloom-deepspeed-inference-fp16"]

parser = ArgumentParser()

parser.add_argument("--data_dir", default="data")
parser.add_argument("--train_name", default="train")
parser.add_argument("--dev_name", default="dev")
parser.add_argument("--test_name", default="test")
parser.add_argument("--output_dir", default="output",
                    type=str, required=False)

parser.add_argument("--dataset", default="tomi")
parser.add_argument("--task", default="tomi")
parser.add_argument("--model_type",
                    default="bloom", required=False)
parser.add_argument("--predict_batch_size", default=1, type=int,
                    help="Batch size per GPU/CPU for evaluation.")

parser.add_argument("--model_name_or_path", required=True,
                    type=str, help="model_name")
parser.add_argument("--dtype", type=str, help="float16 or int8",
                    choices=["int8", "float16"], default="float16")
parser.add_argument("--local_rank", required=False,
                    type=int, help="used by dist launchers")
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--wandb_entity', type=str, default='epfl_nlp_phd')
parser.add_argument('--wandb_project', type=str, default='common-bench')
parser.add_argument('--wandb_name', type=str,
                    default='bloom-ds-inference-tomi')
args = parser.parse_args()

timestr = time.strftime("%Y%m%d-%H%M%S")
run_dir = f"{args.output_dir}/{timestr}"
os.makedirs(run_dir, exist_ok=True)
args.run_dir = run_dir

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


# Model loading and instantiating on GPUs
def get_repo_root(model_name_or_path):
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")

    if rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            ignore_patterns=["*.safetensors"],
        )

    dist.barrier()
    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        ignore_patterns=["*.safetensors"],
    )


def get_checkpoint_files(model_name_or_path):
    cached_repo_dir = get_repo_root(model_name_or_path)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(
        cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


model_name = args.model_name_or_path
infer_dtype = args.dtype

tp_presharded_mode = True if model_name in tp_presharded_models else False

# print(get_checkpoint_files(model_name))

print_rank0(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
config = AutoConfig.from_pretrained(model_name)


dtype = torch.bfloat16 if model_name in [
    "bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16


# use one of these args to `init_inference`
# 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
# 2. replace_with_kernel_inject is the faster one (fast fused kernels)
kernel_inject = True
# kernel_inject = False

if kernel_inject:
    dtype = torch.float16
else:
    dtype = torch.bfloat16

# IMPORTANT: Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=dtype, device="meta"):
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16)

model = model.eval()


checkpoints_json = "checkpoints.json"


def write_checkpoints_json():
    checkpoint_files = get_checkpoint_files(model_name)
    if rank == 0:
        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))


if kernel_inject:
    kwargs = dict(replace_with_kernel_inject=True)
else:
    kwargs = dict(injection_policy={BloomBlock: (
        "self_attention.dense", "mlp.dense_4h_to_h")})

repo_root = get_repo_root(model_name)
if tp_presharded_mode:
    checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
else:
    write_checkpoints_json()
    dist.barrier()

model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    base_dir=repo_root,
    dtype=getattr(torch, infer_dtype),
    checkpoint=checkpoints_json,
    **kwargs,
)

model = model.module

print_rank0("*** Starting to Generation Pipeline")

generate_kwargs = dict(
    do_sample=False,
    num_beams=5,
    num_return_sequences=1,
    num_beam_groups=1,
    temperature=0.7,
    early_stopping=True,
)

print_rank0(f"Generate args {generate_kwargs}")


def init_wandb(args):
    runner = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name
    )
    return runner


def load_data(args, tokenizer):
    """Function responsible for loading the data for the model runner.

    :param args: the arguments for the model runner
    :param tokenizer: the tokenizer for the model runner
    """
    test_data = CommonDataset(
        util_logger,
        args,
        tokenizer,
        args.data_dir,
        data_type="test",
        is_training=False
    )

    dataloader = test_data.load_dataloader()
    return dataloader


def output_parser_metrics(raw_output):
    """Function responsible for parsing the raw_output and computing particular
    metrics from the model runner output.

    :param raw_output: the raw output created by the model runner
    :rtype: tuple
    """
    metrics = {}
    sout = TranslationOutput.from_output(raw_output)
    scores = sout.compute_metrics()
    metrics.update(scores)
    return (sout, metrics)


def evaluate_output(output, wandb_runner, out_file=None, metric_file=None):
    """Method for generating output produced during training and/or evaluation.

    :param output: the output generated by runner
    :param wandb_runner: the wandb runner for logging
    :param out_file: the file to write the output to
    :param metric_file: the file to write the metrics to
    :return: the metrics
    """
    sout, metrics = output_parser_metrics(output)
    if out_file:
        out_dir = Path(out_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for instance in sout:
            outputs.append(instance)
        write_json(outputs, out_file)

        artifact = wandb.Artifact(f"test_eval_out", type='dataset')
        artifact.add_file(out_file)
        wandb_runner.log_artifact(artifact)

    if metric_file:
        out_dir = Path(metric_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(metrics, metric_file)

        artifact = wandb.Artifact(f"test_metrics", type='dataset')
        artifact.add_file(metric_file)
        wandb_runner.log_artifact(artifact)

    return metrics


def generate(print_out):
    device = torch.cuda.current_device()
    output_length = 0
    answers = [data for data in print_out['answer']]
    for answer in answers:
        out_ids = tokenizer(answer, return_tensors="pt").input_ids
        output_length = max(output_length, out_ids.size(1))

    input_length = 0
    questions = [data for data in print_out['question']]
    for question in questions:
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        input_length = max(input_length, input_ids.size(1))

    input_ids = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=input_length,
        return_tensors="pt"
    ).input_ids.to(device)

    greedy_outputs = model.generate(
        input_ids.to(device),
        max_new_tokens=output_length,
        **generate_kwargs
    )

    outputs = tokenizer.batch_decode(
        greedy_outputs,
        skip_special_tokens=True
    )

    clean_outputs = [gen.replace(q, "") for q, gen in zip(questions, outputs)]

    return clean_outputs


dataloader = load_data(args, tokenizer)
input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
]

warm_up_out = {
    "question": input_sentences,
    "answer": ["good"] * len(input_sentences)
}

print_rank0("*** Running generate warmup")
_ = generate(warm_up_out)

print_rank0("*** Running generate")
output_all = []
for batch in tqdm(dataloader):
    print_out = batch["print_out"]
    question = [p for p in print_out['question']]
    pipe_out = generate(print_out)
    print_out["gen_out"] = pipe_out
    output_all.append({"print_out": print_out})


out_file_name = f"test_eval_out.json"
metirc_file_name = f"test_metrics.json"

if rank == 0:
    wandb_runner = init_wandb(args)
    metrics_out = evaluate_output(
        output_all,
        wandb_runner,
        f"{args.run_dir}/{out_file_name}",
        f"{args.run_dir}/{metirc_file_name}"
    )

    wandb_runner.log(metrics_out)

    print("Inference Finished ==== Metrics: ")
    pprint(metrics_out)
    wandb_runner.finish()
