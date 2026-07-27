"""Phase-0 foundation tests: pluggable backbone + num_encoder_layers=0 + round-trip.

These build a real (tiny) ESM2-8M-backed model, so they are marked ``slow`` and
download the 8M backbone on first run (cached thereafter). They run on CPU using
the standard-attention Vanilla variant, so no Ampere GPU is required.

They guard the invariants every ablation depends on:
  - build_esm_backbone honors the requested size (registry is not hardcoded);
  - num_encoder_layers=0 runs and genuinely differs from the E=5 path;
  - hyper_parameters (incl. encoder_backbone + ablation fields) round-trip through
    a checkpoint and the reloaded model reproduces logits exactly.
"""

import os
import tempfile

import pytest
import torch

from protevo.models import build_esm_backbone
from protevo.models._loading import load_peint_model
from protevo.models._transformer import PeintTransformerVanilla

BACKBONE = "ESM2-8M"
EMBED_DIM = 320  # ESM2-8M hidden size


def _encode(vocab, seq):
    return [vocab.cls_idx] + vocab.encode(seq) + [vocab.eos_idx]


def _inputs(vocab):
    x = torch.tensor(_encode(vocab, "ACDEFGHIK")).unsqueeze(0)
    y_in = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEYGHIK")).unsqueeze(0)
    t = torch.tensor([[0.1]], dtype=torch.float32)
    return x, y_in, t, x.eq(vocab.padding_idx), y_in.eq(vocab.padding_idx)


@pytest.fixture(scope="module")
def backbone():
    esm_model, vocab, embed_dim = build_esm_backbone(BACKBONE, use_flash=False)
    assert embed_dim == EMBED_DIM
    return esm_model, vocab, embed_dim


def _build(backbone, n_enc, n_dec, **cfg):
    esm_model, vocab, embed_dim = backbone
    return PeintTransformerVanilla(
        esm_model=esm_model,
        esm_vocab=vocab,
        embed_dim=embed_dim,
        num_heads=20,
        num_encoder_layers=n_enc,
        num_decoder_layers=n_dec,
        **cfg,
    ).eval()


@pytest.mark.slow
def test_backbone_registry_not_hardcoded(backbone):
    """build_esm_backbone returns the requested (non-150M) size."""
    _, vocab, embed_dim = backbone
    assert embed_dim == EMBED_DIM
    assert len(vocab) == 33


@pytest.mark.slow
def test_zero_encoder_forward_differs_from_full(backbone):
    """E=0 (remove-encoder ablation) runs and is not identical to E=5."""
    _, vocab, _ = backbone
    x, y_in, t, xm, ym = _inputs(vocab)

    m5 = _build(backbone, 5, 5)
    m0 = _build(backbone, 0, 5)
    assert len(m0.enc_layers) == 0 and len(m0.dec_layers) == 5

    with torch.no_grad():
        x5, y5, *_ = m5(x, y_in, t, xm, ym)
        x0, y0, *_ = m0(x, y_in, t, xm, ym)

    assert torch.isfinite(y5).all() and torch.isfinite(y0).all()
    assert y0.shape == y5.shape
    # x_logits: E=0 skips the encoder layers, so it must differ from E=5.
    assert not torch.allclose(x5, x0)


@pytest.mark.slow
def test_checkpoint_roundtrip_reproduces_logits(backbone):
    """hyper_parameters (incl. ablation fields) round-trip; logits reproduce exactly."""
    _, vocab, embed_dim = backbone
    x, y_in, t, xm, ym = _inputs(vocab)

    ref = _build(
        backbone, 5, 5,
        encoder_backbone=BACKBONE, mlm_weight=0.0,
        use_attention_bias=True,
    )
    with torch.no_grad():
        _, ref_y, *_ = ref(x, y_in, t, xm, ym)

    hp = dict(
        max_seq_len=1022, num_heads=20, num_encoder_layers=5, num_decoder_layers=5,
        embed_dim=embed_dim, use_attention_bias=True, dropout_p=0.0,
        encoder_backbone=BACKBONE, mlm_weight=0.0,
        esm_finetune_mode="frozen", lora_rank=None, architecture="encoder_decoder",
    )
    prefixed = {f"model.{k}": v for k, v in ref.state_dict().items()}

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "tiny.ckpt")
        torch.save({"state_dict": prefixed, "hyper_parameters": hp}, ckpt_path)
        model, _ = load_peint_model(
            ckpt_path, device=torch.device("cpu"),
            model_type="standard", use_flash=False, strict_loading=False,
        )

    assert model.config.encoder_backbone == BACKBONE
    assert model.embed_dim == EMBED_DIM
    assert model.config.mlm_weight == 0.0
    with torch.no_grad():
        _, rt_y, *_ = model(x, y_in, t, xm, ym)
    assert torch.allclose(ref_y, rt_y, atol=1e-5)
