import random
import torch
import wandb
import logging

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
import openai

from transformers import AutoTokenizer, AutoConfig, GPT2Tokenizer

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

from common_bench.dataset import CommonDataset
from common_bench.model import TranslationOutput
from common_bench.utils.py_io import *




util_logger = logging.getLogger(
    'common_bench.inference'
)

model_path_hf = {
    "flan-t5": ("google/flan-t5-xxl", "chenz16/flan-xxl-sharded-fp16"),
    "t0pp": ("bigscience/T0pp", "chenz16/T0pp-11b-sharded-fp16"),
    "unified-qa": ("allenai/unifiedqa-v2-t5-11b-1251000", "chenz16/unifiedqa-11b-sharded-fp16"),
    "gptj": ("EleutherAI/gpt-j-6B", "sharded-gpt-j-6B"),
    "macaw-11b": ("allenai/macaw-11b", "chenz16/macaw-11b-sharded-fp16"),
    "bloom-3b": ("bigscience/bloom-3b", "sharded-bloom-3b"),
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
    if args.do_icl:
        train_data = CommonDataset(
            util_logger,
            args,
            tokenizer,
            args.data_dir,
            data_type="train",
            is_training=False,
            ic_examples=[]
        )

        if args.search:
            example_data = read_jsonl(
                f"./data/{args.dataset}/train_{args.encoder}.jsonl")

            train_dict = {}
            for instance in train_data.data:
                key = instance["guid"]
                train_dict[key] = instance

            ic_examples = {}
            for instance in example_data:
                key = instance["guid"]
                ic_examples[key] = [
                    train_dict[id] for id in instance["examples"]]
                assert len(ic_examples[key]) == 16
                ic_examples[key] = ic_examples[key][:args.num_examples]
        else:
            ic_examples = random.choices(
                train_data.data, k=args.num_examples)
    else:
        ic_examples = []

    test_data = CommonDataset(
        util_logger,
        args,
        tokenizer,
        args.data_dir,
        data_type="test",
        is_training=False,
        ic_examples=ic_examples
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


class Text2Generator():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token

    def generate(self, print_out, **generate_kwargs):
        device = torch.cuda.current_device()

        output_length = 0
        answers = [data for data in print_out['answer']]
        for answer in answers:
            out_ids = self.tokenizer(answer, return_tensors="pt").input_ids
            output_length = max(output_length, out_ids.size(1))

        input_length = 0
        questions = [data for data in print_out['question']]
        for question in questions:
            input_ids = self.tokenizer(question, return_tensors="pt").input_ids
            input_length = max(input_length, input_ids.size(1))

        input_ids = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=input_length,
            return_tensors="pt"
        ).input_ids.to(device)

        greedy_outputs = self.model.generate(
            input_ids.to(device),
            max_new_tokens=output_length,
            **generate_kwargs
        )

        outputs = self.tokenizer.batch_decode(
            greedy_outputs,
            skip_special_tokens=True
        )

        clean_outputs = [gen.replace(q, "")
                         for q, gen in zip(questions, outputs)]

        return clean_outputs


def run_acclerate(args):
    torch.set_grad_enabled(False)

    model_name, local_name, model_class = parse_checkpoint_path(args)
    model, tokenizer, _ = load_model(model_name, local_name, model_class)

    if args.model_type != "t5":
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    dataloader = load_data(args, tokenizer)
    generator = Text2Generator(model, tokenizer)

    output_all = []
    for batch in tqdm(dataloader):
        print_out = batch["print_out"]
        pipe_out = generator.generate(
            print_out,
            num_beams=5,
            # top_k=20,
            num_return_sequences=1)
        print_out["gen_out"] = pipe_out
        output_all.append({"print_out": print_out})

    out_file_name = f"test_eval_out.json"
    metirc_file_name = f"test_metrics.json"

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

def run_gpt3(args):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    dataloader = load_data(args, tokenizer)
    openai.api_key = args.gpt3_key

    output_all = []
    for batch in tqdm(dataloader):
        print_out = batch["print_out"]
        # pipe_out = generator.generate(
        #     print_out,
        #     num_beams=5,
        #     # top_k=20,
        #     num_return_sequences=1)
        # print_out["gen_out"] = pipe_out
        response = openai.Completion.create(engine="text-davinci-002", 
                prompt=print_out['question'],
                temperature=0,
                max_tokens=1,
                top_p=0,
                frequency_penalty=0.0,
                presence_penalty=0.0
                )

        print_out["gen_out"] = response['choices'][0]['text']
        output_all.append({"print_out": print_out})

    out_file_name = f"test_eval_out.json"
    metirc_file_name = f"test_metrics.json"

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