# import os
# import torch
# import logging
# import deepspeed

# from tqdm import tqdm

# from transformers.models.t5.modeling_t5 import T5Block
# from transformers import AutoTokenizer, AutoConfig
# from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
# from transformers import pipeline

# from common_bench.dataset import CommonDataset
# from common_bench.utils.py_io import *

# from accelerate import infer_auto_device_map, init_empty_weights, Accelerator

# util_logger = logging.getLogger(
#     'common_bench.runner'
# )

# model_path_hf = {
#     "flan-t5": "google/flan-t5-xxl",
#     "t0pp": "bigscience/T0pp",
#     "unified-qa": "allenai/unifiedqa-v2-t5-11b-1251000",
#     "gptj": "EleutherAI/gpt-j-6B",
#     "macaw-11b": "allenai/macaw-11b",
#     "macaw-3b": "allenai/macaw-3b",
#     "macaw-large":  "sharded-macaw-large",  # "allenai/macaw-large",
#     "bloom": "bigscience/bloom",
#     "opt": "facebook/opt-66b"
# }

# model_class_registry = {
#     "t5": AutoModelForSeq2SeqLM,
#     "opt": AutoModelForCausalLM,
#     "bloom": AutoModelForCausalLM,
#     "gpt": AutoModelForCausalLM
# }

# model = None
# tokenizer = None
# model_config = None
# def load_model(args):
#     # model_class = model_class_registry[args.model_type]
#     # hf_name = model_path_hf[args.model_name_or_path]
#     # tokenizer = AutoTokenizer.from_pretrained(hf_name)
#     # model = model_class.from_pretrained(
#     #     hf_name, local_files_only=True,
#     #     device_map="auto", torch_dtype=torch.float16
#     # )
#     # model = model.eval()
#     # model_config = AutoConfig.from_pretrained(hf_name)

#     # return model, tokenizer, model_config
#     global model
#     global tokenizer
#     global model_config
#     if model is None:
#         print("Setting "+ f"{args.model_name_or_path}")

#         model = AutoModelForCausalLM.from_pretrained(
#             args.model_name_or_path, device_map="balanced_low_0", offload_folder="offload", offload_state_dict=True, torch_dtype=torch.float16
#         )

#         device_map = infer_auto_device_map(model)
#         device_map["model.decoder.layers.37"] = "disk"

#         tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#         tokenizer.padding_side = 'left'
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = model.config.eos_token_id

#         model_config = AutoConfig.from_pretrained(model)
#         print("Finished")

#         return model, tokenizer, model_config




# def load_data(args, tokenizer):
#     test_data = CommonDataset(
#         util_logger,
#         args,
#         tokenizer,
#         args.data_dir,
#         data_type="test",
#         is_training=False
#     )

#     dataloader = test_data.load_dataloader()
#     return dataloader


# def run_acclerate(args):
#     torch.set_grad_enabled(False)
#     local_rank = int(os.getenv("LOCAL_RANK", "0"))
#     world_size = 8

#     # deepspeed.init_distributed("nccl")
#     # rank = dist.get_rank()

#     # def print_rank0(*msg):
#     #     if rank != 0:
#     #         return
#     #     print(*msg)
#     # print_rank0(f"*** Loading the model {args.model_name_or_path}")

#     model, tokenizer, model_config = load_model(args)
#     dataloader = load_data(args, tokenizer)

#     dtype = torch.bfloat16 if args.model_name_or_path in ["bigscience/bloom", "bigscience/bigscience-small-testing"] else torch.float16
#     model_hidden_size = model_config.hidden_size
#     val_batch_size = 1 * world_size

#     ds_config = {
#         "fp16": {
#             "enabled": dtype == torch.float16,
#         },
#         "bf16": {
#             "enabled": dtype == torch.bfloat16,
#         },
#         "zero_optimization": {
#             "stage": 3,
#             "overlap_comm": True,
#             "contiguous_gradients": True,
#             "reduce_bucket_size": model_hidden_size * model_hidden_size,
#             "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
#             "stage3_param_persistence_threshold": 0,
#         },
#         "steps_per_print": 2000,
#         "train_batch_size": val_batch_size,
#         "train_micro_batch_size_per_gpu": 1,
#         "wall_clock_breakdown": False,
#     }

#     # if args.cpu_offload:
#     #     ds_config["zero_optimization"]["offload_param"] = dict(
#     #         device="cpu", pin_memory=True)
#     # dschf = HfDeepSpeedConfig(ds_config)

#     # print_rank0(ds_config)

#     pipe = pipeline(
#         "text2text-generation",
#         tokenizer=tokenizer,
#         model=model,
#         device=local_rank,
#     )

#     pipe.model = deepspeed.init_inference(
#         model=pipe.model, dtype=dtype,
#         mp_size=world_size)

#     output_all = []
#     for batch in tqdm(dataloader):
#         print_out = batch["print_out"]
#         questions = [data for data in print_out['question']]
#         pipe_out = pipe(questions)
#         output_all.extend(pipe_out)

#     out_file_name = f"test_eval_out.json"
#     metirc_file_name = f"test_metrics.json"

#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         print('dfdlfkl')
#         write_jsonl(output_all, f"{args.run_dir}/{out_file_name}")


import os
import deepspeed
import torch
torch.cuda.empty_cache()
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, init_empty_weights, Accelerator

# from accelerate import load_checkpoint_and_dispatch

# model = load_checkpoint_and_dispatch(
#     model, "/nlpdata1/share/models/opt-175b/parallel_consolidated_shards", device_map="auto"
# )

model_name = 'facebook/opt-66b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(model, max_memory = {0: "30GIB", 1: "30GIB", 2: "30GIB", 3: "30GIB", 4: "30GIB", 5: "30GIB", 6: "30GIB", 7: "30GIB"})
# device_map["model.decoder.layers.37"] = "disk"

model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir="/nlpdata1/home/sooh/opt66", device_map=device_map, offload_folder="offload", offload_state_dict = True, torch_dtype=torch.float16)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
config = AutoConfig.from_pretrained(model_name, cache_dir="/nlpdata1/home/sooh/opt66")

print("Finished")


# local_rank = int(os.getenv('LOCAL_RANK', '0'))
# world_size = int(os.getenv('WORLD_SIZE', '1'))
# generator = pipeline('text-generation', model=model,
#                      device=local_rank)



# generator.model = deepspeed.init_inference(generator.model,
#                                            mp_size=world_size,
#                                            dtype=torch.float,
#                                            replace_method='auto',
# 					   replace_with_kernel_inject=True)

# string = generator("DeepSpeed is", do_sample=True, min_length=50)
# if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#     print(string)