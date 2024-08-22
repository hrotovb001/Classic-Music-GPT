import torch
import os
import sys
from model import GPTConfig, GPT
from contextlib import nullcontext
import pickle
from music21 import *
import data.classical_music.tokens as tokens

sys.modules['tokens'] = tokens

seed = 1337
out_dir = "out-music-model"
device = 'cuda'
ptdtype = torch.float16
max_new_tokens = 1000 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

device_type = 'cuda' if 'cuda' in device else 'cpu'
torch.manual_seed(seed)
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
mtoi, itom = meta['mtoi'], meta['itom']
encode = lambda m: [mtoi[n] for n in m] # encoder: take melody, output a list of integers
decode = lambda l: [itom[i] for i in l] # decoder: take a list of integers, output a melody

start_ids = encode([None])
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        tk = tokens.detokenize_2(decode(y[0].tolist()))
        stream.Stream(tk).show("midi")
