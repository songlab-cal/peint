"""A5 ablation tests: LoRA-finetuned backbone, with a strict fall-back guarantee.

Key properties verified:
  * FROZEN default is untouched — a frozen model builds without importing _lora and
    has no LoRA parameters (the "fall-back-able" requirement).
  * LoRA injects adapters on the target linears; the base stays frozen and only the
    adapters are trainable.
  * LoRA is init-identity: with zero-initialized B, the LoRA model's forward equals
    the frozen model's forward at initialization.
  * A LoRA checkpoint round-trips (rebuild from hparams applies LoRA, then loads).
"""

import os
import sys
import tempfile

import pytest
import torch

from protevo.models import build_esm_backbone
from protevo.models._loading import load_peint_model
from protevo.models._transformer import PeintTransformerVanilla
from train_peint_model import parse_args_with_config

BACKBONE = "ESM2-8M"
PEINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_lora_config_differs_from_baseline_only_in_finetune_axes():
    """Fast: lora.yaml differs from baseline only in the LoRA fine-tuning axes."""
    base = parse_args_with_config(
        ["--config", os.path.join(PEINT_ROOT, "configs/ablations/baseline.yaml")]
    )
    lora = parse_args_with_config(
        ["--config", os.path.join(PEINT_ROOT, "configs/ablations/lora.yaml")]
    )
    assert base.esm_finetune_mode == "frozen" and lora.esm_finetune_mode == "lora"
    assert lora.lora_rank == 16 and lora.lora_alpha == 16
    ignore = {"esm_finetune_mode", "lora_rank", "lora_alpha", "name_addon",
              "output_dir", "config"}
    a = {k: v for k, v in vars(lora).items() if k not in ignore}
    b = {k: v for k, v in vars(base).items() if k not in ignore}
    assert a == b, "lora.yaml differs from baseline beyond the LoRA axes"


def _build(**cfg):
    # Fresh backbone per model: apply_lora_to_backbone mutates the backbone in place,
    # so each model must own its backbone (as it does in production).
    esm, vocab, dim = build_esm_backbone(BACKBONE, use_flash=False)
    model = PeintTransformerVanilla(
        esm_model=esm, esm_vocab=vocab, embed_dim=dim,
        num_heads=20, num_encoder_layers=5, num_decoder_layers=5,
        encoder_backbone=BACKBONE, **cfg,
    ).eval()
    return model, vocab


def _inputs(vocab):
    x = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEFGHIK") + [vocab.eos_idx]).unsqueeze(0)
    y = torch.tensor([vocab.cls_idx] + vocab.encode("ACDEYGHIK")).unsqueeze(0)
    t = torch.tensor([[0.1]], dtype=torch.float32)
    return x, y, t, x.eq(vocab.padding_idx), y.eq(vocab.padding_idx)


@pytest.mark.slow
def test_frozen_default_does_not_touch_lora():
    """Fall-back: the default frozen model has no LoRA params and never imports _lora."""
    sys.modules.pop("protevo.models._lora", None)
    model, _ = _build()  # esm_finetune_mode defaults to "frozen"
    assert not any("lora" in n.lower() for n, _ in model.named_parameters())
    # building a frozen model must not import the LoRA module
    assert "protevo.models._lora" not in sys.modules
    # backbone fully frozen
    assert not any(p.requires_grad for p in model.esm.parameters())


@pytest.mark.slow
def test_lora_injects_trainable_adapters_over_frozen_base():
    model, _ = _build(esm_finetune_mode="lora", lora_rank=8)
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    assert lora_params, "no LoRA parameters were injected"
    # exactly the adapters are trainable inside the backbone
    trainable = [n for n, p in model.esm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    # base weights present and frozen
    base = [p for n, p in model.esm.named_parameters() if p.requires_grad is False]
    assert base


@pytest.mark.slow
def test_lora_is_identity_at_init():
    """Zero-init B => LoRA model reproduces the frozen model's forward at init."""
    torch.manual_seed(0)
    frozen, vocab = _build()
    torch.manual_seed(0)
    lora, _ = _build(esm_finetune_mode="lora", lora_rank=8)
    x, y, t, xm, ym = _inputs(vocab)

    # copy the non-LoRA (PEINT enc/dec/head) weights so only the adapter differs
    lora_sd = lora.state_dict()
    for k, v in frozen.state_dict().items():
        # frozen key q_proj.weight maps to lora key q_proj.base.weight in the backbone
        if k in lora_sd:
            lora_sd[k].copy_(v)
        else:
            kb = k.replace(".q_proj.", ".q_proj.base.").replace(".v_proj.", ".v_proj.base.")
            if kb in lora_sd:
                lora_sd[kb].copy_(v)
    lora.load_state_dict(lora_sd)

    with torch.no_grad():
        _, yf, *_ = frozen(x, y, t, xm, ym)
        _, yl, *_ = lora(x, y, t, xm, ym)
    assert torch.allclose(yf, yl, atol=1e-5)


@pytest.mark.slow
def test_lora_checkpoint_roundtrips():
    ref, vocab = _build(esm_finetune_mode="lora", lora_rank=8, lora_alpha=16)
    dim = ref.embed_dim
    x, y, t, xm, ym = _inputs(vocab)
    # perturb an adapter so it's not the zero-init trivial case
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith("lora_B"):
                p.add_(0.01)
        _, ref_y, *_ = ref(x, y, t, xm, ym)

    hp = dict(
        max_seq_len=1022, num_heads=20, num_encoder_layers=5, num_decoder_layers=5,
        embed_dim=dim, use_attention_bias=True, dropout_p=0.0, encoder_backbone=BACKBONE,
        mlm_weight=1.0, esm_finetune_mode="lora", lora_rank=8, lora_alpha=16,
        architecture="encoder_decoder",
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "lora.ckpt")
        torch.save({"state_dict": {f"model.{k}": v for k, v in ref.state_dict().items()},
                    "hyper_parameters": hp}, p)
        model, _ = load_peint_model(p, device=torch.device("cpu"),
                                    model_type="standard", use_flash=False)
    assert model.config.esm_finetune_mode == "lora" and model.config.lora_rank == 8
    with torch.no_grad():
        _, rt_y, *_ = model(x, y, t, xm, ym)
    assert torch.allclose(ref_y, rt_y, atol=1e-5)
