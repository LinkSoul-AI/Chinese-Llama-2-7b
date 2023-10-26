r"""
Modified from https://github.com/THUDM/ChatGLM-6B/blob/main/api.py
Usage Example:
```python
import openai
if __name__ == "__main__":
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=[
            {"role": "user", "content": "你好"}
        ],
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)
```
"""

import time
import sys
from threading import Thread
from typing import Optional, List, Literal, Union

import torch
import uvicorn
from pydantic import BaseModel, Field
from loguru import logger as _logger
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import TextIteratorStreamer


def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    # _logger.add('logs/log.txt', level=logfile_level)
    return _logger


logger = define_log_level()


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


# def torch_gc():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


app = FastAPI()

B_INST, E_INST = "[INST]", "[/INST]"
SYSTEM = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


async def predict(streamer: TextIteratorStreamer):
    model_id = "llama2"

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    for new_text in streamer:
        if not new_text:
            continue
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "data: {}\n\n".format(chunk.json(exclude_unset=True, ensure_ascii=False))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


def get_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"<s>{B_INST} <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]

    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f"{user_input} {E_INST} {response.strip()} </s><s>{B_INST} ")

    message = message.strip() if do_strip else message
    texts.append(f"{message} {E_INST}")
    return "".join(texts)


@app.post("/v1/chat/completions")
async def create_item(request: ChatCompletionRequest):
    global model, tokenizer

    # 1. Build input_ids for LLM
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content
    prev_messages = request.messages[:-1]

    system = SYSTEM
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system = prev_messages.pop(0).content

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if (
                prev_messages[i].role == "user"
                and prev_messages[i + 1].role == "assistant"
            ):
                history.append((prev_messages[i].content, prev_messages[i + 1].content))

    prompt = get_prompt(query, history, system)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    logger.info(tokenizer.decode(input_ids))

    # 2. Build the options dict for prediction
    max_length = request.max_length or 4096
    top_p = request.top_p or 0.7
    temperature = request.temperature or 0.95
    is_stream = request.stream

    generation_kwargs = {
        "inputs": torch.tensor([input_ids]).cuda(),
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature,
    }

    if is_stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs['streamer'] = streamer

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generate = predict(streamer)
        return StreamingResponse(generate, media_type="text/event-stream")

    else:
        response = model.generate(**generation_kwargs)
        response = tokenizer.decode(response[0, len(input_ids):], skip_special_tokens=True)
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )

        return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


if __name__ == "__main__":
    model_path = "LinkSoul/Chinese-Llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=model_path.endswith("4bit"),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
