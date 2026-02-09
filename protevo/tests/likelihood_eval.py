import os
import json

import numpy as np
import torch
import esm
from protevo.models import PeintTransformerVanilla, load_model

if __name__ == "__main__":

    transition_file = '/home/akoehl/Projects/peint/protevo/tests/example_transition.txt'

    with open(transition_file, 'r') as f:
        line = f.readline().strip()
        x, y, t = line.split(' ')
        t = float(t)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_checkpoint = '/home/akoehl/Projects/peint/model_checkpoints/epoch=2-step=40000.ckpt'

    #Use the Vanilla model for comparison
    model, vocab = load_model(model_checkpoint, use_cached_model= False, device=device, use_flash=False)

    x_tokens = [vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx]
    y_tokens = [vocab.cls_idx] + vocab.encode(y) #y_input doesn't have an eos
    y_targets = vocab.encode(y) + [vocab.eos_idx]

    x_toks = torch.tensor(x_tokens).unsqueeze(0).to(device)
    y_toks = torch.tensor(y_tokens).unsqueeze(0).to(device)
    y_targs = torch.tensor(y_targets).unsqueeze(0).to(device)
    ts = torch.tensor([t]).unsqueeze(0).to(device)

    x_attn_mask = x_toks.eq(vocab.padding_idx)
    y_attn_mask = y_toks.eq(vocab.padding_idx)

    with torch.no_grad():
        x_logits, y_logits, representations, self_attns, cross_attns = model(x_toks, y_toks, ts, x_attn_mask, y_attn_mask)

    old_logits = np.load('/home/akoehl/Projects/peint/protevo/tests/y_logits.npy')

    print('Max difference in logits:', np.max(np.abs(old_logits - y_logits.cpu().numpy())))
    print(f"Logits match: {np.allclose(old_logits, y_logits.cpu().numpy())}")
