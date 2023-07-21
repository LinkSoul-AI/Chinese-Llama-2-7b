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

## 如何训练

```bash
python train.py --args ...
```

## 相关项目

- [Llama2](https://ai.meta.com/llama/)

## 项目协议

[Apache-2.0 license](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b/blob/main/LICENSE)
