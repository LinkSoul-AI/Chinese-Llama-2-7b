# Chinese Llama 2 7B

[![](https://img.shields.io/badge/LLaMA2-Chinese-blue)](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b) [![](https://img.shields.io/badge/Commercial-Support-blue)](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b) [![](https://img.shields.io/badge/License-Apache_v2-blue)](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b/blob/main/LICENSE) [![](https://img.shields.io/badge/HuggingFace-Live_Demo-green)](https://huggingface.co/spaces/LinkSoul/Chinese-Llama-2-7b) [![](https://img.shields.io/badge/Datasets-instruction_merge_set-blue)](https://huggingface.co/datasets/LinkSoul/instruction_merge_set)

全部开源，完全可商用的**中文版 Llama2 模型及中英文 SFT 数据集**，输入格式严格遵循 *llama-2-chat* 格式，兼容适配所有针对原版 *llama-2-chat* 模型的优化。

![Chinese LLaMA2 7B](.github/preview.jpg)

## 基础演示

![Base Demo](.github/demo.gif)

## 在线试玩

> Talk is cheap, Show you the Demo.

- [Demo 地址 / HuggingFace Spaces](https://huggingface.co/spaces/LinkSoul/Chinese-Llama-2-7b)
- [Colab 一键启动](#) // 正在准备

## 资源下载

- 模型下载：[Chinese Llama2 Chat Model](https://huggingface.co/LinkSoul/Chinese-Llama-2-7b)

- 4bit量化：[Chinese Llama2 4bit Chat Model](https://huggingface.co/LinkSoul/Chinese-Llama-2-7b-4bit)

- GGML Q4 模型（社区版）：
  - [https://huggingface.co/rffx0/Chinese-Llama-2-7b-ggml-model-q4_0](https://huggingface.co/rffx0/Chinese-Llama-2-7b-ggml-model-q4_0)
  - [https://huggingface.co/soulteary/Chinese-Llama-2-7b-ggml-q4](https://huggingface.co/soulteary/Chinese-Llama-2-7b-ggml-q4)

> 我们使用了中英文 SFT 数据集，数据量 1000 万。

- 数据集：[https://huggingface.co/datasets/LinkSoul/instruction_merge_set](https://huggingface.co/datasets/LinkSoul/instruction_merge_set)

## 快速测试

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "LinkSoul/Chinese-Llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

prompt = instruction.format("用中文回答，When is the best time to visit Beijing, and do you have any suggestions for me?")
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
```

## Docker

你可以使用仓库中的 Dockerfile，来快速制作基于 Nvidia 最新版本的 `nvcr.io/nvidia/pytorch:23.06-py3` 基础镜像，在任何地方使用容器来运行中文的 LLaMA2 模型应用。

```bash
docker build -t linksoul/chinese-llama2-chat .
```

镜像构建完毕，使用命令运行镜像即可：

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v `pwd`/LinkSoul:/app/LinkSoul -p 7860:7860 linksoul/chinese-llama2-chat
```

## GGML / Llama.cpp

想要在 CPU 环境运行 LLaMA2 模型么？使用下面的方法吧。

- 使用 `ggml/convert_to_ggml.py` 进行转换操作，详见脚本支持的 CLI 参数。
- 或使用 `docker pull soulteary/llama2:converter` 下载模型格式转换工具镜像，在 Docker 容器中使用下面的两条命令完成操作（教程 [构建能够使用 CPU 运行的 MetaAI LLaMA2 中文大模型
](https://zhuanlan.zhihu.com/p/645426799)）：

```bash
python3 convert.py /app/LinkSoul/Chinese-Llama-2-7b/ --outfile /app/LinkSoul/Chinese-Llama-2-7b-ggml.bin
./quantize /app/LinkSoul/Chinese-Llama-2-7b-ggml.bin /app/LinkSoul/Chinese-Llama-2-7b-ggml-q4.bin q4_0
```

## 如何训练

```bash
DATASET="LinkSoul/instruction_merge_set"

DATA_CACHE_PATH="hf_datasets_cache"
MODEL_PATH="/PATH/TO/TRANSFORMERS/VERSION/LLAMA2"

output_dir="./checkpoints_llama2"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_port=25003 \
        train.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATASET} \
        --data_cache_path ${DATA_CACHE_PATH} \
        --bf16 True \
        --output_dir ${output_dir} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy 'no' \
        --save_strategy 'steps' \
        --save_steps 1200 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --fsdp 'full_shard auto_wrap' \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True
```

## 相关项目

- [Llama2](https://ai.meta.com/llama/)
- [soulteary/docker-llama2-chat](https://github.com/soulteary/docker-llama2-chat)


## 项目协议

[Apache-2.0 license](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b/blob/main/LICENSE)

## 微信交流群

<img src=".github/QRcode.jpg" alt="微信交流群" width="300"/>
