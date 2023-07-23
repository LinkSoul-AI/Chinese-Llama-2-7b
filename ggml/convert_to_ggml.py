#!/usr/bin/env python3
# This script is slightly modified based on TheBlock's code.
# ("https://huggingface.co/junelee/wizard-vicuna-13b/discussions/2")

import argparse
import os
import subprocess

def main(llama_cpp_base, model, outbase, outdir):
    ggml_version = "v3"

    if not os.path.isdir(model):
        raise Exception(f"Could not find model dir at {model}")

    if not os.path.isfile(f"{model}/config.json"):
        raise Exception(f"Could not find config.json in {model}")

    os.makedirs(outdir, exist_ok=True)

    # print("Building llama.cpp")
    # subprocess.run(f"cd {llama_cpp_base} && git pull && make clean && LLAMA_CUBLAS=1 make", shell=True, check=True)

    fp16 = f"{outdir}/{outbase}.ggml{ggml_version}.fp16.bin"

    print(f"Making unquantised GGML at {fp16}")
    if not os.path.isfile(fp16):
        subprocess.run(f"python3 {llama_cpp_base}/convert.py {model} --outtype f16 --outfile {fp16}", shell=True, check=True)
    else:
        print(f"Unquantised GGML already exists at: {fp16}")

    print("Making quants")
    for type in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]:
        outfile = f"{outdir}/{outbase}.ggml{ggml_version}.{type}.bin"
        print(f"Making {type} : {outfile}")
        subprocess.run(f"{llama_cpp_base}/quantize {fp16} {outfile} {type}", shell=True, check=True)

    # Delete FP16 GGML when done making quantisations
    os.remove(fp16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Bash to Python.')
    parser.add_argument('--llama-cpp-base', help='llama.cpp directory')
    parser.add_argument('--model', help='Model directory')
    parser.add_argument('--outbase', help='Output base name')
    parser.add_argument('--outdir', help='Output directory')

    args = parser.parse_args()

    main(args.llama_cpp_base, args.model, args.outbase, args.outdir)
