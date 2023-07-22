import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Original version
# model_path = "LinkSoul/Chinese-Llama-2-7b"
# 4 bit version
model_path = "LinkSoul/Chinese-Llama-2-7b-4bit"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
if model_path.endswith("4bit"):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
else:
    model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

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

print ("="*100)
print (model_log)
print (logo)
print ("-"*80)
print ("Have a try!")

s = ''
print ("User:")
while True:
    c = input()
    if c == '':
        prompt = instruction.format(s)
        print ('Answer:')
        generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
        print ('-'*80)
        s = ''
        print ("User:") 
    else:
        s += c
