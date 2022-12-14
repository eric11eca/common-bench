import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM

HF_TOKEN = "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"

HF_NAME = "google/flan-t5-xxl"
# HF_NAME = "allenai/macaw-11b"
# HF_NAME = "bigscience/T0pp"
# HF_NAME = "allenai/unifiedqa-v2-t5-11b-1251000"
# HF_NAME = "facebook/opt-30b"

SHARD_NAME = "sharded-flan-xxl"
# SHARD_NAME = "sharded-macaw-11b"
# SHARD_NAME = "sharded-T0pp"
# SHARD_NAME = "sharded-unifiedqa-11b"
# SHARD_NAME = "sharded-opt-30b"

REPO_ID = "chenz16/flan-xxl-sharded-fp16"
# REPO_ID = "chenz16/macaw-11b-sharded-fp16"
# REPO_ID = "chenz16/T0pp-sharded-fp16"
# REPO_ID = "chenz16/unifiedqa-11b-sharded-fp16"
# REPO_ID = "chenz16/opt-30b-sharded-fp16"


# load model as float16
model = AutoModelForSeq2SeqLM.from_pretrained(
    HF_NAME, torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# shard model an push to hub
model.save_pretrained(SHARD_NAME, max_shard_size="2000MB")

api = HfApi()
api.upload_folder(
    folder_path=SHARD_NAME,
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
)
