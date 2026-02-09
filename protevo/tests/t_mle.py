import os
import json

import numpy as np
import torch
import esm
from protevo.models import PeintTransformerVanilla, load_model

def prepare_input(x, y, t=0.6):
    x_tokens = [vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx]
    y_tokens = [vocab.cls_idx] + vocab.encode(y)
    y_targets = vocab.encode(y) + [vocab.eos_idx]

    ts = torch.tensor([t], dtype=torch.float32).unsqueeze(0).to(device)

    x_toks = torch.tensor(x_tokens).unsqueeze(0).to(device)
    y_toks = torch.tensor(y_tokens).unsqueeze(0).to(device)
    y_targs = torch.tensor(y_targets).unsqueeze(0).to(device)

    x_attn_mask = x_toks.eq(vocab.padding_idx)
    y_attn_mask = y_toks.eq(vocab.padding_idx)

    return x_toks, y_toks, y_targs, ts, x_attn_mask, y_attn_mask

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_checkpoint = '/home/akoehl/Projects/peint/model_checkpoints/epoch=2-step=40000.ckpt'

    #Use the Vanilla model for comparison
    model, vocab = load_model(model_checkpoint, use_cached_model= False, device=device, use_flash=False)
    model = model.eval()

    transition_file = '/home/akoehl/Projects/peint/protevo/tests/example_transition.txt'

    with open(transition_file, 'r') as f:
        line = f.readline().strip()
        x, y, t = line.split(' ')
        t = float(t)

    batch_f = prepare_input(x, y)
    batch_r = prepare_input(y, x)

    t_guesses = torch.full_like(batch_f[3], 0.6, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([t_guesses], lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    eps = 5e-3

    print(f'Starting t_guess: {t_guesses.item()}')

    for step in range(100):

        optimizer.zero_grad()
        x, y, y_t, ts, x_mask, y_mask = batch_f
        
        x_logits, y_logits, representations, self_attns, cross_attns = model(x, y, t_guesses, x_mask, y_mask)

        forward_nll = torch.nn.functional.cross_entropy(
            y_logits.transpose(-1, -2), y_t, ignore_index=vocab.padding_idx, reduction = 'none'
        ).sum(dim=1)

        x, y, y_t, ts, x_mask, y_mask = batch_r

        x_logits, y_logits, representations, self_attns, cross_attns = model(x, y, t_guesses, x_mask, y_mask)
        reverse_nll = torch.nn.functional.cross_entropy(
            y_logits.transpose(-1, -2), y_t, ignore_index=vocab.padding_idx, reduction = 'none'
        ).sum(dim=1)

        loss = torch.mean(forward_nll + reverse_nll)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        t_guesses.data.clamp_(min=eps)

    t_mle = t_guesses.item()
    print(f"Absolute error in MLE t: {abs(t_mle - t):.3f}")

