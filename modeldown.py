import os
import deepspeed
import torch
torch.cuda.empty_cache()
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, init_empty_weights, Accelerator


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
