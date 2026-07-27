"""A4 ablation tests: remove the extra encoder layers (num_encoder_layers=0).

The forward-path support for num_encoder_layers=0 (backbone output feeds the decoder
directly) is exercised in test_phase0_backbone.py; here we pin the A4 config to the
baseline-minus-one-axis contract and check the structural consequence (no encoder
layers, decoder intact).
"""

import os

import pytest

from train_peint_model import parse_args_with_config

PEINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(name):
    return os.path.join(PEINT_ROOT, "configs", "ablations", name)


def test_no_encoder_config_differs_from_baseline_only_in_encoder_layers():
    base = parse_args_with_config(["--config", _config("baseline.yaml")])
    a4 = parse_args_with_config(["--config", _config("no_encoder.yaml")])
    assert base.num_encoder_layers == 5
    assert a4.num_encoder_layers == 0
    assert a4.num_decoder_layers == base.num_decoder_layers == 5
    ignore = {"num_encoder_layers", "name_addon", "output_dir", "config"}
    a = {k: v for k, v in vars(a4).items() if k not in ignore}
    b = {k: v for k, v in vars(base).items() if k not in ignore}
    assert a == b, "no_encoder.yaml differs from baseline beyond num_encoder_layers"


@pytest.mark.slow
def test_zero_encoder_model_has_no_encoder_layers():
    """Structural check: E=0 builds 0 encoder blocks, decoder intact, forward runs."""
    import torch

    from protevo.models import build_esm_backbone
    from protevo.models._transformer import PeintTransformerVanilla

    esm, vocab, dim = build_esm_backbone("ESM2-8M", use_flash=False)
    model = PeintTransformerVanilla(
        esm_model=esm, esm_vocab=vocab, embed_dim=dim,
        num_heads=20, num_encoder_layers=0, num_decoder_layers=5,
    ).eval()
    assert len(model.enc_layers) == 0
    assert len(model.dec_layers) == 5

    x = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEFGHIK") + [vocab.eos_idx]).unsqueeze(0)
    y_in = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEYGHIK")).unsqueeze(0)
    t = torch.tensor([[0.1]], dtype=torch.float32)
    with torch.no_grad():
        _, y_logits, *_ = model(x, y_in, t, x.eq(vocab.padding_idx), y_in.eq(vocab.padding_idx))
    assert torch.isfinite(y_logits).all()
