import os
import torch
import wandb
import logging

from tqdm import tqdm
from pathlib import Path
from pprint import pprint

from accelerate import infer_auto_device_map, init_empty_weights

from transformers import pipeline
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import Text2TextGenerationPipeline
from transformers.tokenization_utils import TruncationStrategy

from common_bench.dataset import CommonDataset
from common_bench.model import TranslationOutput
from common_bench.utils.py_io import *

util_logger = logging.getLogger(
    'common_bench.runner'
)

model_path_hf = {
    "flan-t5": ("google/flan-t5-xxl", "chenz16/flan-xxl-sharded-fp16"),
    "t0pp": ("bigscience/T0pp", "chenz16/T0pp-11b-sharded-fp16"),
    "unified-qa": ("allenai/unifiedqa-v2-t5-11b-1251000", "chenz16/unifiedqa-11b-sharded-fp16"),
    "gptj": "EleutherAI/gpt-j-6B",
    "macaw-11b": ("allenai/macaw-11b", "chenz16/macaw-11b-sharded-fp16"),
    "bloom-3b": ("bigscience/bloom-3b", "sharded-bloom-3b"),
    "bloom-1b": ("bigscience/bloom-1b7", "sharded-bloom-1b7")
}

model_class_registry = {
    "t5": AutoModelForSeq2SeqLM,
    "opt": AutoModelForCausalLM,
    "bloom": AutoModelForCausalLM,
    "gpt": AutoModelForCausalLM
}


def init_wandb(args):
    runner = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name
    )
    return runner


def load_model(model_name, local_name, model_class):
    """Function responsible for loading the model for the model runner.

    :param args: the arguments for the model runner
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)

    # device_map = infer_auto_device_map(model, dtype="float16")
    # device_map["lm_head"] = 0

    model = model_class.from_pretrained(
        local_name,
        local_files_only=False,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder="offload",
        offload_state_dict=True
    )
    model = model.eval()
    return model, tokenizer, config


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


def parse_checkpoint_path(args):
    model_class = model_class_registry[args.model_type]
    hf_name = model_path_hf[args.model_name_or_path]
    if isinstance(hf_name, tuple):
        model_name = hf_name[0]
        local_name = hf_name[1]
    else:
        model_name = hf_name
        local_name = hf_name
    return model_name, local_name, model_class


class Text2TextGenerator(Text2TextGenerationPipeline):
    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        inputs = self._parse_and_tokenize(
            inputs, truncation=truncation, **kwargs)
        inputs = inputs.to("cuda:0")
        return inputs


def run_acclerate(args):
    wandb_runner = init_wandb(args)
    torch.set_grad_enabled(False)

    model_name, local_name, model_class = parse_checkpoint_path(args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataloader = load_data(args, tokenizer)

    model, tokenizer, _ = load_model(model_name, local_name, model_class)

    generator = pipeline(
        # "text-generation",
        tokenizer=tokenizer,
        model=model,
        device_map="auto",
        torch_dtype=torch.float16,
        pipeline_class=Text2TextGenerator,
    )

    output_all = []
    for batch in tqdm(dataloader):
        print_out = batch["print_out"]
        question = [p for p in print_out['question']]
        input_ids = batch["input_ids"]
        output_ids = batch["labels"]
        max_length = input_ids.size(1) + output_ids.size(1)
        pipe_out = generator(
            question,
            do_sample=True,
            top_p=0.9,
            max_length=max_length,
            num_return_sequences=1,
            return_full_text=False)
        print_out["gen_out"] = [out[0]["generated_text"].strip()
                                for out in pipe_out]
        output_all.append({"print_out": print_out})

    out_file_name = f"test_eval_out.json"
    metirc_file_name = f"test_metrics.json"

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
