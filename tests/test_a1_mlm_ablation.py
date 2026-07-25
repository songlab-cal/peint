"""A1 ablation tests: the auxiliary MLM objective is config-driven, not hard-coded.

Exercises PeintLightningModule._compute_losses directly with dummy logits/targets,
so no forward pass / GPU / Trainer is needed. Verifies:
  - mlm_weight defaults to 1.0 (published) and reproduces mlm_loss + tlm_loss;
  - mlm_weight=0.0 ablates MLM entirely (loss == tlm_loss, mlm term reported as 0);
  - fractional weights scale the MLM term linearly (fully general).
"""

import pytest
import torch

from protevo.models import build_esm_backbone
from protevo.models.training import PeintLightningModule

BACKBONE = "ESM2-8M"


@pytest.fixture(scope="module")
def backbone():
    esm_model, vocab, embed_dim = build_esm_backbone(BACKBONE, use_flash=False)
    return esm_model, vocab, embed_dim


def _module(backbone, **cfg):
    esm_model, vocab, embed_dim = backbone
    return PeintLightningModule(
        esm_model=esm_model,
        esm_vocab=vocab,
        max_seq_len=1022,
        num_heads=20,
        num_encoder_layers=5,
        num_decoder_layers=5,
        embed_dim=embed_dim,
        **cfg,
    ).eval()


def _dummy(vocab):
    torch.manual_seed(0)
    v = len(vocab)
    x_logits = torch.randn(2, 6, v)
    y_logits = torch.randn(2, 6, v)
    x_targets = torch.randint(0, v, (2, 6))
    y_targets = torch.randint(0, v, (2, 6))
    return x_logits, y_logits, x_targets, y_targets


@pytest.mark.slow
def test_default_weight_is_published(backbone):
    """No mlm_weight given -> 1.0 -> loss == mlm_loss + tlm_loss (published)."""
    _, vocab, _ = backbone
    m = _module(backbone)
    assert m.mlm_weight == 1.0
    loss, metrics = m._compute_losses(*_dummy(vocab))
    expected = metrics["mlm_loss"] + metrics["tlm_loss"]
    assert torch.allclose(loss, expected)


@pytest.mark.slow
def test_zero_weight_ablates_mlm(backbone):
    """mlm_weight=0.0 -> loss == tlm_loss, MLM term is exactly zero."""
    _, vocab, _ = backbone
    ref = _module(backbone)                       # weight 1.0, for the MLM value
    _, ref_metrics = ref._compute_losses(*_dummy(vocab))

    m0 = _module(backbone, mlm_weight=0.0)
    assert m0.mlm_weight == 0.0
    loss0, metrics0 = m0._compute_losses(*_dummy(vocab))

    assert torch.allclose(loss0, metrics0["tlm_loss"])
    assert float(metrics0["mlm_loss"]) == 0.0
    # tlm term is unaffected by the ablation
    assert torch.allclose(metrics0["tlm_loss"], ref_metrics["tlm_loss"])


@pytest.mark.slow
def test_fractional_weight_scales_mlm(backbone):
    """Fractional mlm_weight scales the MLM term linearly."""
    _, vocab, _ = backbone
    ref = _module(backbone)
    _, ref_metrics = ref._compute_losses(*_dummy(vocab))
    mlm, tlm = ref_metrics["mlm_loss"], ref_metrics["tlm_loss"]

    m = _module(backbone, mlm_weight=0.5)
    loss, _ = m._compute_losses(*_dummy(vocab))
    assert torch.allclose(loss, 0.5 * mlm + tlm, atol=1e-6)
