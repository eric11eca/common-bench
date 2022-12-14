import torch
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import HfApi

# load model as float16
model = AutoModelForSeq2SeqLM.from_pretrained(
    "allenai/macaw-11b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
# shard model an push to hub
model.save_pretrained("sharded-macaw-11b", max_shard_size="2000MB")
