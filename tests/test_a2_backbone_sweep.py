"""A2 ablation tests: the ESM2 backbone size is a config switch that really works.

For each ESM2 size, builds the real backbone (downloaded once, cached) and a PEINT
model on it, and runs a forward pass, catching dim/state-dict/head-divisibility
issues. ESM2-150M is the published baseline (exercised elsewhere) and is skipped
here to avoid a 600 MB download.
"""

import os

import pytest
import torch

from protevo.models import build_esm_backbone
from protevo.models._transformer import PeintTransformerVanilla
from train_peint_model import parse_args_with_config

PEINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_HEADS = 20
# (registry name, expected hidden dim); 150M omitted to keep the test light.
SIZES = [("ESM2-8M", 320), ("ESM2-35M", 480)]


def _config(name):
    return os.path.join(PEINT_ROOT, "configs", "ablations", name)


@pytest.mark.slow
@pytest.mark.parametrize("name,dim", SIZES)
def test_backbone_size_builds_and_runs(name, dim):
    assert dim % NUM_HEADS == 0, f"num_heads={NUM_HEADS} must divide embed_dim={dim}"
    esm_model, vocab, embed_dim = build_esm_backbone(name, use_flash=False)
    assert embed_dim == dim

    model = PeintTransformerVanilla(
        esm_model=esm_model, esm_vocab=vocab, embed_dim=embed_dim,
        num_heads=NUM_HEADS, num_encoder_layers=5, num_decoder_layers=5,
        encoder_backbone=name,
    ).eval()

    x = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEFGHIK") + [vocab.eos_idx]).unsqueeze(0)
    y_in = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEYGHIK")).unsqueeze(0)
    t = torch.tensor([[0.1]], dtype=torch.float32)
    with torch.no_grad():
        _, y_logits, *_ = model(x, y_in, t, x.eq(vocab.padding_idx), y_in.eq(vocab.padding_idx))
    assert torch.isfinite(y_logits).all()
    assert y_logits.shape[-1] == len(vocab)
    assert model.config.encoder_backbone == name


def test_shipped_size_configs_valid_and_controlled():
    """esm2_8m / esm2_35m configs differ from baseline only in backbone+embed_dim."""
    base = parse_args_with_config(["--config", _config("baseline.yaml")])
    expected = {"esm2_8m.yaml": ("ESM2-8M", 320), "esm2_35m.yaml": ("ESM2-35M", 480)}
    ignore = {"esm_model", "embed_dim", "output_dir", "name_addon", "config"}
    for fname, (model_name, dim) in expected.items():
        args = parse_args_with_config(["--config", _config(fname)])
        assert args.esm_model == model_name
        assert args.embed_dim == dim
        assert dim % args.num_heads == 0
        a = {k: v for k, v in vars(args).items() if k not in ignore}
        b = {k: v for k, v in vars(base).items() if k not in ignore}
        assert a == b, f"{fname} differs from baseline beyond the backbone swap"
