import torch
import logging
import pytorch_lightning as pl

from typing import Dict
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from common_bench.dataset import CommonDataset
from common_bench.model import TransformerModel
from common_bench.train import setup_trainer
from common_bench.utils.wandb_utils import setup_wandb

util_logger = logging.getLogger(
    'common_bench.runner'
)


class CommonBenchRunner(pl.LightningModule):
    def __init__(self, config):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()
        self.model_logger = util_logger
        self.hparams.update(vars(config))

        self.global_trainin_step = 0
        self.global_epoch_counter = 0

        self.model = TransformerModel.from_config(config)
        self.tokenizer = self.model.tokenizer

        self.load_dataset()
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

    def step(self, batch, is_train: bool) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param is_train: whether to run training or validation
        :rtype: dict
        :returns: dictionary that includes loss
        """
        print_out = batch["print_out"]
        evaluate = not is_train

        features = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }
        if "labels" in batch:
            features["labels"] = batch["labels"]
        else:
            features["labels"] = batch["input_ids"]

        output = self.model(features, print_out, evaluate)
        output_dict = {'loss': output["loss"]}

        if not is_train:
            output_dict["print_out"] = output["print_out"]

        return output_dict

    def training_step(self, batch, batch_idx) -> Dict:
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.step(batch, is_train=True)
        self.log(
            f'batch_train_loss',
            output_dict["loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        self.global_trainin_step += 1
        return output_dict

    def training_epoch_end(self, outputs):
        """Called at the end of the training epoch

        :param outputs: the outputs of the train step
        :rtype: None 
        """
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "avg_train_loss",
            avg_train_loss,
            on_step=False,
            on_epoch=True
        )
        self.global_epoch_counter += 1

    def validation_step(self, batch, batch_idx) -> Dict:
        """Runs a single validation step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        output_dict = self.step(batch, is_train=False)
        return output_dict

    def validation_epoch_end(self, outputs):
        """Called at the end of the validation epoch
        :param outputs: the outputs of the train step
        :rtype: None 
        """
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        epoch = self.global_epoch_counter
        step = self.global_trainin_step

        out_file_name = f"dev_out-epoch={epoch}_step={step}.json"
        metirc_file_name = f"val_metrics-epoch={epoch}_step={step}.json"
        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.run_dir}/{out_file_name}",
            f"{self.hparams.run_dir}/{metirc_file_name}"
        )

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )

    def test_step(self, batch, batch_idx) -> Dict:
        test_out = self.validation_step(batch, batch_idx)
        return test_out

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)

        out_file_name = f"test_eval_out.json"
        metirc_file_name = f"test_metrics.json"

        metrics_out = self.model.evaluate_output(
            outputs,
            f"{self.hparams.run_dir}/{out_file_name}",
            f"{self.hparams.run_dir}/{metirc_file_name}"
        )

        for metric_name, metric_value in metrics_out.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_epoch=True,
                prog_bar=True
            )

    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params": parameters_first,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": parameters_sec,
                "weight_decay": 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler

        """
        num_devices = self.hparams.n_gpu if torch.cuda.is_available() else 1
        effective_batch_size = self.hparams.train_batch_size * \
            self.hparams.gradient_accumulation_steps * num_devices

        total_steps = (len(self.train_dataloader().dataset) /
                       effective_batch_size) * self.hparams.num_train_epochs
        self.hparams.warmup_steps = (
            total_steps / effective_batch_size
        ) * self.hparams.warmup_proportion

        self.model_logger.info(
            'total_steps computed for scheduler: %s, warmup step: %s' % (
                total_steps, str(self.hparams.warmup_steps))
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return scheduler

    def load_dataset(self):
        """Loads the dataset

        """
        self.model_logger.info('Loading dataset')

        if self.hparams.do_train:
            self.train_data = CommonDataset(
                self.model_logger,
                self.hparams,
                self.tokenizer,
                self.hparams.data_dir,
                data_type="train",
                is_training=True
            )
            self.dev_data = CommonDataset(
                self.model_logger,
                self.hparams,
                self.tokenizer,
                self.hparams.data_dir,
                data_type="dev",
                is_training=False
            )
        if self.hparams.do_eval:
            self.test_data = CommonDataset(
                self.model_logger,
                self.hparams,
                self.tokenizer,
                self.hparams.data_dir,
                data_type="test",
                is_training=False
            )
        self.model_logger.info('Dataset loaded')

    def train_dataloader(self):
        """Loader to building training data.

        :rtype: DataLoader
        """
        dataloader = self.train_data.load_dataloader()
        self.model_logger.info(
            'Length of training data loader %d' % len(dataloader)
        )
        return dataloader

    def val_dataloader(self):
        """Loader to building validation data.

        :rtype: DataLoader
        """
        dataloader = self.dev_data.load_dataloader()
        self.model_logger.info(
            'Length of validation data loader %d' % len(dataloader)
        )
        return dataloader

    def test_dataloader(self):
        """Loader to building test data.

        :rtype: DataLoader
        """
        dataloader = self.test_data.load_dataloader()
        self.model_logger.info(
            'Length of test data loader %d' % len(dataloader)
        )
        return dataloader


def run(args):
    util_logger.info('Setting up configuration for model runner...')

    setup_wandb(args)
    model = CommonBenchRunner(args)
    trainer = setup_trainer(args)

    if args.do_train:
        if args.load_checkpoint is not None:
            trainer.fit(model, ckpt_path=args.load_checkpoint)
        else:
            trainer.fit(model)

    if args.do_eval:
        if args.load_checkpoint is not None:
            trainer.test(model, ckpt_path=args.load_checkpoint)
        else:
            trainer.test(model)
