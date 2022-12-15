import re
import torch
import string
import wandb
import itertools
import torch.nn as nn

from pathlib import Path
from typing import Dict
from dataclasses import dataclass

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

from common_bench.utils.py_io import *

model_class_registry = {
    "t5": AutoModelForSeq2SeqLM,
    "opt": AutoModelForCausalLM,
    "bloom": AutoModelForCausalLM,
    "gpt": AutoModelForCausalLM
}

model_path_hf = {
    "flan-t5": ("google/flan-t5-xxl", "chenz16/flan-xxl-sharded-fp16"),
    "t0pp": ("bigscience/T0pp", "chenz16/T0pp-11b-sharded-fp16"),
    "unified-qa": ("allenai/unifiedqa-v2-t5-11b-1251000", "chenz16/unifiedqa-11b-sharded-fp16"),
    "gptj": "EleutherAI/gpt-j-6B",
    "macaw-11b": ("allenai/macaw-11b", "chenz16/macaw-11b-sharded-fp16"),
    "bloom-3b": ("bigscience/bloom-3b", "sharded-bloom-3b"),
    "bloom-1b": ("bigscience/bloom-1b7", "sharded-bloom-1b7")
}


class TransformerModel(nn.Module):
    """Generic transformer-based pretrained encoder decoder (e.g., T5, BART, etc..)
    which has the added feature of doing on-the-fly generation during training and evaluation.
    """

    def __init__(self, model, tokenizer, model_config, global_config):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.global_config = global_config

        AutoModelForCausalLM.from_pretrained

    @classmethod
    def from_config(cls, config):
        """Loads a pretrained encoder decoder from configuration

        :param config: the global configuration
        """
        model_class = model_class_registry[config.model_type]
        hf_name = model_path_hf[config.model_name_or_path]
        if isinstance(hf_name, tuple):
            tokenizer = AutoTokenizer.from_pretrained(hf_name[0])
            model_config = AutoConfig.from_pretrained(hf_name[0])
            model = model_class.from_pretrained(
                hf_name[1],
                local_files_only=False,
                device_map="auto",
                offload_state_dict=True,
                offload_folder="offload",
                torch_dtype=torch.float16
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            model_config = AutoConfig.from_pretrained(hf_name)
            model = model_class.from_pretrained(
                hf_name, device_map="auto",
                offload_state_dict=True,
                offload_folder="offload",
                torch_dtype=torch.float16)

        if config.model_type == "gpt":
            model.config.pad_token_id = tokenizer.eos_token_id

        return cls(
            model,
            tokenizer,
            model_config,
            config,
        )

    def forward(self, features, print_out, evaluate=False):
        """A modified version of forward method for the underlying transformer model.
        :param features: the target inputs
        :param print_out: data to print out during evaluation
        """
        main_out = {}
        outputs = self.model(**features)
        if evaluate:
            main_out["print_out"] = print_out
            main_out["print_out"]["gen_out"] = self.generate(print_out)
        else:
            outputs = self.model(**features)
            main_out["loss"] = outputs.loss
        return main_out

    def generate(self, print_out):
        device = self.model.device
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
        max_length = input_length + output_length

        greedy_outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=1,
            use_cache=True
        )
        outputs = self.tokenizer.batch_decode(
            greedy_outputs,
            skip_special_tokens=True
        )
        return outputs

    def output_parser_metrics(self, raw_output):
        """Function responsible for parsing the raw_output and computing particular
        metrics from the model runner output.

        :param raw_output: the raw output created by the model runner
        :rtype: tuple
        """
        metrics = {}
        sout = TranslationOutput.from_output(
            self.global_config, raw_output)
        scores = sout.compute_metrics()
        metrics.update(scores)
        return (sout, metrics)

    def evaluate_output(self, output, out_file=None, metric_file=None, is_test=False):
        """Method for generating output produced during training and/or evaluation.

        :param output: the output generated by runner
        :raises: ValueError
        """
        sout, metrics = self.output_parser_metrics(output)
        if out_file:
            out_dir = Path(out_file).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = []
            for instance in sout:
                outputs.append(instance)
            write_json(outputs, out_file)

            if is_test:
                artifact = wandb.Artifact(f"test_eval_out", type='dataset')
                artifact.add_file(out_file)
                wandb.run.log_artifact(artifact)

        if metric_file:
            out_dir = Path(metric_file).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(metrics, metric_file)

            if is_test:
                artifact = wandb.Artifact(f"test_metrics", type='dataset')
                artifact.add_file(out_file)
                wandb.run.log_artifact(artifact)

        return metrics


@ dataclass
class TranslationOutput:
    """Helper class for translation output"""
    print_data: Dict

    @ classmethod
    def from_output(cls, output):
        """Loads from raw outputs

        :param outputs: the outputs produced by the model
        """

        print_data = cls.get_print_data(output, "print_out")
        return cls(print_data=print_data)

    @ classmethod
    def get_print_data(cls, output, print_key):
        print_data = {}
        print_out_keys = set(
            itertools.chain(*[list(i[print_key].keys()) for i in output])
        )

        for key_name in print_out_keys:
            raw_data = [t[print_key][key_name]
                        if key_name in t[print_key] else [] for t in output]
            print_data[key_name] = [t for t in itertools.chain(*raw_data)]

        return print_data

    @ property
    def questions(self):
        return self.print_data.get("question", [])

    @ property
    def targets(self):
        return self.print_data.get("answer", [])

    @ property
    def outputs(self):
        return self.print_data.get("gen_out", [])

    def compute_metrics(self):
        """Returns an exact match accuracy for generation

        :rtype: float or None
        """
        targets = self.targets
        outputs = self.outputs

        metrics = {}
        if targets and outputs and len(targets) == len(outputs):
            em_scores = [self.compute_exact_match(
                gen, label) for label, gen in zip(targets, outputs)]
            f1_scores = [self.compute_f1(gen, label)
                         for label, gen in zip(targets, outputs)]

            metrics["acc"] = sum(em_scores) / len(targets)
            metrics["f1"] = sum(f1_scores) / len(targets)

        return metrics

    def normalize_text(self, text):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(truth) in self.normalize_text(prediction))

    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        return 2 * (prec * rec) / (prec + rec)

    def enumerate_instances(self):
        """Enumerate through instances for printing
        """
        guids = self.print_data["guid"]
        questions = self.questions
        targets = self.targets
        outputs = self.outputs

        total_outputs = []
        for k, identifier in enumerate(guids):
            instance_dict = {}
            instance_dict["guid"] = identifier
            instance_dict["question"] = questions[k]
            instance_dict["gen_out"] = outputs[k]
            instance_dict["answer"] = targets[k]
            total_outputs.append(instance_dict)

        return total_outputs

    def __iter__(self):
        for item in self.enumerate_instances():
            yield item
