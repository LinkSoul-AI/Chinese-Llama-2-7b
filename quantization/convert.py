# Modified from https://github.com/soulteary/docker-llama2-chat/blob/main/llama2-7b-cn-4bit/quantization_4bit.py

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = 'LinkSoul/Chinese-Llama-2-7b'
output_dir = "Chinese-Llama-2-7b-4bit"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map='auto'
)

model.save_pretrained(output_dir)
print("done")
