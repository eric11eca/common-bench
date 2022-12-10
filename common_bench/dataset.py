import uuid
import torch

from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from .utils.py_io import read_jsonl


class DataReader:
    """Custom dataset loader for QA problems."""

    @staticmethod
    def _read(instance, args):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param args: the configuration arguments
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        NotImplemented

    @classmethod
    def jsonl_file_reader(cls, path, config):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param path: the path to the target data file
        :param evaluation: indicator as to where this is an evaluation file (based on name)
        :param config: the configuration
        """
        total_data = read_jsonl(path)
        total_metadata = {}
        total_qa_data = []
        for instance in total_data:
            qa_data, metadata = cls._read(instance, config)
            total_qa_data.append(qa_data)
            total_metadata[qa_data["guid"]] = metadata

        for data in total_qa_data[:2]:
            pprint(data)

        return total_qa_data, total_metadata


class TomiDataReader(DataReader):
    @staticmethod
    def _read(instance, args):
        guid = instance["guid"]
        story = instance["story"]
        question = instance["question"]
        answer = instance["answer"]
        metadata = instance["metadata"]
        reformat = f"$answer$ ; $question$ = {question}; $context$ = {story}"

        data = {
            "guid": guid,
            "question": reformat,
            "answer": answer,
        }

        return data, metadata


class SocialIQADataReader(DataReader):
    @staticmethod
    def _read(instance, args):
        try:
            guid = instance["guid"]
        except KeyError:
            guid = str(uuid.uuid4())
        context = instance["context"]
        question = instance["question"]
        answer = instance["answer"]
        options = instance["options"]
        metadata = instance["metadata"]

        options = [f"({i}) {o}" for i, o in enumerate(options)]
        options = " ".join(options)
        reformat = f"$answer$ ; $mcoptions$ = {options}; $question$ = {question}; $context$ = {context}"

        data = {
            "guid": guid,
            "question": reformat,
            "answer": answer,
        }

        return data, metadata


class SocialChemDataReader(DataReader):
    @staticmethod
    def _read(instance, args):
        guid = instance["guid"]
        situation = instance["situation"]
        action = instance["action"]
        answer = instance["answer"]
        options = instance["options"]
        metadata = instance["metadata"]

        context = f"Situation: {situation}\nAction:{action}"
        question = f"Given the situation, what do you think about the action?"

        options = [f"({i}) {o}" for i, o in enumerate(options)]
        options = " ".join(options)
        reformat = f"$answer$ ; $mcoptions$ = {options}; $question$ = {question}; $context$ = {context}"

        data = {
            "guid": guid,
            "question": reformat,
            "answer": answer,
        }

        return data, metadata


class ScruplesAnecdoteDataReader(DataReader):
    @staticmethod
    def _read(instance, args):
        guid = instance["guid"]
        story = instance["story"]
        question = instance["question1"]
        answer = instance["answer1"]
        options = instance["options1"]
        metadata = instance["metadata"]

        options = [f"({i}) {o}" for i, o in enumerate(options)]
        options = " ".join(options)
        reformat = f"$answer$ ; $mcoptions$ = {options}; $question$ = {question}; $context$ = {story}"

        data = {
            "guid": guid,
            "question": reformat,
            "answer": answer,
        }

        return data, metadata


class ScruplesDilemmaDataReader(DataReader):
    @staticmethod
    def _read(instance, args):
        guid = instance["guid"]
        story = instance["story"]
        question = instance["question"]
        answer = instance["answer"]
        options = instance["options"]
        metadata = instance["metadata"]

        options = [f"({i}) {o}" for i, o in enumerate(options)]
        options = " ".join(options)
        reformat = f"$answer$ ; $mcoptions$ = {options}; $question$ = {question}; $context$ = {story}"

        data = {
            "guid": guid,
            "question": reformat,
            "answer": answer,
        }

        return data, metadata


class CommonDataset(object):
    def __init__(self, logger, args, tokenizer, data_path, data_type, is_training):
        self.data_path = data_path
        self.data_type = data_type
        self.task_name = args.dataset
        self.args = args

        reader_classes = {
            "tomi": TomiDataReader,
            "socialiqa": SocialIQADataReader,
            "socialchem": SocialChemDataReader,
        }
        self.reader = reader_classes[args.task]
        self.is_training = is_training
        self.logger = logger
        self.tokenizer = tokenizer
        self.dataloader = None
        self.load = False
        self.data, self.metadata = self.read_data_from_file()

    def __len__(self):
        return len(self.data)

    def read_data_from_file(self):
        file_path = f"{self.data_path}/{self.task_name}/{self.data_type}.jsonl"
        file_data = self.reader.jsonl_file_reader(file_path, self.args)
        return file_data

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataloader(self):
        meta_dataloader = CommonDataLoader(
            self.args,
            self.data,
            self.tokenizer,
            self.is_training
        )
        self.dataloader = meta_dataloader.dataloader
        return self.dataloader


class CommonDataLoader():
    def __init__(self, args, dataset, tokenizer, is_training):
        self.args = args
        self.task = args.task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.evaluate = not is_training

        if is_training:
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        collate_fn_dict = {
            "t5": self.text2text_collator,
        }

        collate_fn = collate_fn_dict[args.model_type]
        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

    def _tensorize(self, input_txt, output_txt, max_length=None):
        """Converts a list of strings into a tensor of ids

        :param input_txt: the input text
        :param output_txt: the output text
        :param max_length: the maximum length of the sequence
        :return: the tensorized input and output
        """
        pad = self.tokenizer.eos_token
        pad_id = self.tokenizer.encode(pad)

        ids1 = self.tokenizer(input_txt, return_tensors="pt")["input_ids"]
        ids2 = self.tokenizer(output_txt, return_tensors="pt")["input_ids"]

        max_length = ids1.size(
            1) + ids2.size(1) if max_length is None else max_length
        n_mask = max_length - ids1.size(1) - ids2.size(1)
        assert n_mask >= 0, (max_length, ids1.size(1), ids2.size(1))
        padding = torch.LongTensor(pad_id * n_mask).unsqueeze(0)

        input_ids = torch.cat((ids1, ids2, padding), dim=1)
        attention_mask = torch.LongTensor(
            [1] * (ids1.size(1) + ids2.size(1)) + [0] * n_mask).unsqueeze(0)
        token_type_ids = torch.LongTensor(
            [0] * ids1.size(1) + [1] * ids2.size(1) + [0] * n_mask).unsqueeze(0)

        assert input_ids.size(1) == attention_mask.size(
            1) == token_type_ids.size(1) == max_length

        return input_ids, attention_mask, token_type_ids

    def text2text_collator(self, batch):
        """Batch collator for this custom class
        :param batch: an incoming batch
        :param tokenizer: the model tokenizer
        :param args: the global configuration
        """
        questions = [data["question"] for data in batch]
        answers = [data["answer"] for data in batch]

        print_out = {
            "guid": [data['guid'] for data in batch],
            "question": questions,
            "answer": answers,
        }

        max_seq_length = max([len(self.tokenizer(q).input_ids)
                             for q in questions])
        max_out_length = max([len(self.tokenizer(a).input_ids)
                             for a in answers])

        tokenized_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_seq_length,
        )

        tokenized_outputs = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_out_length,
        )

        feature = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_outputs["input_ids"],
            "print_out": print_out,
        }

        return feature
