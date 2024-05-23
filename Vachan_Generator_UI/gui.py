import streamlit as st
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

st.set_page_config(page_title="GPT-2 based Vachana Generator", page_icon=":book:")

st.title('GPT-2 based Vachana Generator')
st.caption("A vachana generator based on GPT-2, trained using nanoGPT on a huge list of basvanna's vachanas")

st.markdown("""
## What?
Basavanna was a medieval Indian poet and social reformer who wrote short "sonnets" of sorts, these were written in a Dravidian language called Kannada which is still widely spoken in the southern part of India. These sonnets or "vachanas" were often religious poems pertaining to faith or other social issues.

## Why?
- Kannada language project in college
- Curioisty to see how non-ascii characters perform in the GPT architecture

## How?
I didn't have the time to train an entire LLM from scratch or write my own architecture. There's no point in re-inventing the wheel. So I used [nanoGPT](https://github.com/karpathy/nanoGPT) to train a GPT-2 based model to generate these vachanas and see how coherent they are. It took me 5000 iterations and it's trained on CUDA machines (NVIDIA). Can't disclose more about the machine due to security issues.

## Misc info
Dataset: I used my own dataset called tinybasavanna, inspired by tinyshakespeare. It contains 1414 vachanas. Can be found in the github repository.

## Road map
- Test out meter in Halegannada (Old Kannada)
- See if output follows rules of meter in Old Kannada
""")

st.write('You can use this generator here to generate vachanas in Kannada')

# Default values
init_from = 'resume'
out_dir = 'out'
num_samples = 1
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# Get user input
start = st.text_input('Starter word', value="\\n")

if st.button('Generate'):
    # Setup
    exec(open('configurator.py').read())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load model
    if init_from == 'resume':
        ckpt_path = 'ckpt.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)

    # Load tokenizer
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = 'meta.pkl'
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # Generate
    start_ids = encode(start)
    if len(start_ids) > 0:
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        output = decode(y[0].tolist())
        st.write(output)
    else:
        st.write("Please enter a starter word.")