"""Unit tests for PeintConfig, including the ablation axis fields.

These tests are dependency-light (PeintConfig is pure-stdlib) and guard the
config plumbing that every ablation relies on: defaults must reproduce the
published PEINT behavior, and the config must round-trip through to_dict/from_dict.
"""

import pytest

from protevo.models._config import PeintConfig

BASE = dict(embed_dim=640, num_heads=20, num_encoder_layers=5, num_decoder_layers=5)


def test_defaults_match_published_behavior():
    """New ablation fields default to the published PEINT configuration."""
    cfg = PeintConfig(**BASE)
    assert cfg.mlm_weight == 1.0
    assert cfg.use_time_conditioning is True
    assert cfg.encoder_backbone == "ESM2-150M"
    assert cfg.esm_finetune_mode == "frozen"
    assert cfg.lora_rank is None
    assert cfg.architecture == "encoder_decoder"


def test_roundtrip_to_from_dict():
    """to_dict -> from_dict -> construct preserves every field, including axes."""
    cfg = PeintConfig(
        **BASE,
        mlm_weight=0.0,
        use_time_conditioning=False,
        encoder_backbone="ESM2-35M",
        esm_finetune_mode="lora",
        lora_rank=8,
        architecture="decoder_only",
    )
    cfg2 = PeintConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()


def test_from_dict_ignores_unknown_keys():
    """Old/newer checkpoints with extra keys still load."""
    d = dict(BASE)
    d["some_future_field"] = 123
    cfg = PeintConfig.from_dict(d)
    assert cfg.embed_dim == 640


def test_zero_encoder_layers_allowed():
    """num_encoder_layers=0 is now representable (A4: feed backbone to decoder)."""
    cfg = PeintConfig(**{**BASE, "num_encoder_layers": 0})
    assert cfg.num_encoder_layers == 0


def test_negative_encoder_layers_rejected():
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "num_encoder_layers": -1})


def test_zero_decoder_layers_rejected():
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "num_decoder_layers": 0})


def test_mlm_weight_negative_rejected():
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "mlm_weight": -0.5})


def test_invalid_finetune_mode_rejected():
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "esm_finetune_mode": "sometimes"})


def test_lora_requires_positive_rank():
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "esm_finetune_mode": "lora"})  # lora_rank=None
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "esm_finetune_mode": "lora", "lora_rank": 0})
    # Valid: positive rank
    cfg = PeintConfig(**{**BASE, "esm_finetune_mode": "lora", "lora_rank": 16})
    assert cfg.lora_rank == 16


def test_invalid_architecture_rejected():
    with pytest.raises(ValueError):
        PeintConfig(**{**BASE, "architecture": "seq2seq2seq"})
