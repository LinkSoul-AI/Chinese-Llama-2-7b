import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Original version
# model_path = "LinkSoul/Chinese-Llama-2-7b"
# 4 bit version
model_path = "LinkSoul/Chinese-Llama-2-7b-4bit"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=model_path.endswith("4bit"),
        torch_dtype=torch.float16,
        device_map='auto'
    )
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

model_log = """\
   ___ _     _                              __    __    _              _       ____  
  / __\ |__ (_)_ __   ___  ___  ___        / /   / /   /_\    /\/\    /_\     |___ \ 
 / /  | '_ \| | '_ \ / _ \/ __|/ _ \      / /   / /   //_\\\  /    \  //_\\\      __) |
/ /___| | | | | | | |  __/\__ \  __/     / /___/ /___/  _  \/ /\/\ \/  _  \    / __/ 
\____/|_| |_|_|_| |_|\___||___/\___|     \____/\____/\_/ \_/\/    \/\_/ \_/   |_____|
                                                                                     """

logo = """\
                       __ _       _     __             _ 
                      / /(_)_ __ | | __/ _\ ___  _   _| |
                     / / | | '_ \| |/ /\ \ / _ \| | | | |
                    / /__| | | | |   < _\ \ (_) | |_| | |
                    \____/_|_| |_|_|\_\\\__/\___/ \__,_|_|
                                                         """

def get_prompt(message: str, chat_history: list[tuple[str, str]]) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)


print ("="*100)
print (model_log)
print (logo)
print ("-"*80)
print ("Have a try!")

s = ''
chat_history = []
while True:
    s = input("User: ")
    if s != '':
        prompt = get_prompt(s, chat_history)
        print ('Answer:')
        tokens = tokenizer(prompt, return_tensors='pt').input_ids
        generate_ids = model.generate(tokens.cuda(), max_new_tokens=4096, streamer=streamer)
        output = tokenizer.decode(generate_ids[0, len(tokens[0]):-1]).strip()
        chat_history.append([s, output])
        print ('-'*80)
