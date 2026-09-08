"""Shared inference: per-mutant transition log-likelihood scoring.

Single implementation of the PEINT scoring loop used by both training-time DMS
evaluation (`_train_utils.DMSEvaluationCallback`) and offline scoring
(`compute_fitness`). Leaf module (depends only on torch/tqdm) so it can be imported
from anywhere in the VEP package without creating import cycles.
"""

from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def score_transition_pairs(
    model,
    vocab,
    device,
    pairs,
    t_value,
    batch_size: int = 32,
    progress: bool = False,
) -> List[float]:
    """Mean per-site conditional log-likelihood ``log p(y | x, t)`` for each pair.

    Args:
        model: a ``PeintTransformer`` (or its LightningModule wrapper); its
            ``forward(x, y, t, x_mask, y_mask)`` must return ``(x_logits, y_logits)``.
        vocab: the ESM vocab (provides cls/eos/padding idx and ``encode``).
        device: torch device.
        pairs: sequence of ``(x_seq, y_seq)`` string pairs (x = wild-type, y = mutant).
        t_value: evolutionary time (raw float).
        batch_size: inference batch size.
        progress: show a tqdm progress bar.

    Returns:
        List of floats (one log-likelihood per input pair, in input order).
    """
    model.eval()
    scores: List[float] = []
    iterator = range(0, len(pairs), batch_size)
    if progress:
        iterator = tqdm(iterator)
    for i in iterator:
        batch = pairs[i : i + batch_size]
        x_tokens, y_tokens, y_targets, t_tokens = [], [], [], []
        for x, y in batch:
            x_tokens.append(
                torch.tensor([vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx])
            )
            y_tokens.append(torch.tensor([vocab.cls_idx] + vocab.encode(y)))
            y_targets.append(torch.tensor(vocab.encode(y) + [vocab.eos_idx]))
            t_tokens.append(t_value)

        x_toks = torch.nn.utils.rnn.pad_sequence(
            x_tokens, batch_first=True, padding_value=vocab.padding_idx
        ).to(device)
        y_toks = torch.nn.utils.rnn.pad_sequence(
            y_tokens, batch_first=True, padding_value=vocab.padding_idx
        ).to(device)
        y_targs = torch.nn.utils.rnn.pad_sequence(
            y_targets, batch_first=True, padding_value=vocab.padding_idx
        ).to(device)
        ts = torch.tensor(t_tokens).unsqueeze(-1).to(device)

        x_attn_mask = x_toks.eq(vocab.padding_idx).to(device)
        y_attn_mask = y_toks.eq(vocab.padding_idx).to(device)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(x_toks, y_toks, ts, x_attn_mask, y_attn_mask)
        # Flash PeintTransformer returns (x_logits, y_logits); the vanilla model returns a
        # richer tuple (…, representations, attentions). Take y_logits, ignore any extras.
        y_logits = out[1] if isinstance(out, tuple) else out

        loss = -1 * F.cross_entropy(
            y_logits.transpose(-1, -2),
            y_targs,
            ignore_index=vocab.padding_idx,
            reduction="none",
        ).mean(dim=-1)

        # .float(): scores are bf16 for flash/ESM-C backbones, and numpy has no bfloat16.
        scores.extend(loss.float().cpu().numpy())

    return scores
