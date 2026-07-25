"""Tests for the training launcher's YAML-config plumbing.

Fast (no model construction): they only exercise argument parsing / merging, which
every ablation depends on. Guards the precedence rule (CLI > YAML > defaults), the
unknown-key safety check, and that the shipped ablation configs are valid.
"""

import os

import pytest

from train_peint_model import build_parser, parse_args_with_config

PEINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(name):
    return os.path.join(PEINT_ROOT, "configs", "ablations", name)


def test_no_config_uses_published_defaults():
    args = parse_args_with_config([])
    assert args.mlm_weight == 1.0
    assert args.esm_finetune_mode == "frozen"
    assert args.architecture == "encoder_decoder"
    assert args.esm_model == "ESM2-150M"


def test_yaml_sets_defaults_and_cli_overrides(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("mlm_weight: 0.0\nseed: 0\nnum_encoder_layers: 5\n")
    # CLI passes --seed 7, which must override the YAML's seed: 0.
    args = parse_args_with_config(["--config", str(cfg), "--seed", "7"])
    assert args.mlm_weight == 0.0            # from YAML
    assert args.num_encoder_layers == 5      # YAML overrides argparse default (6)
    assert args.seed == 7                    # CLI overrides YAML


def test_yaml_unknown_key_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("mlm_weight: 0.0\ntypoo_field: 3\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        parse_args_with_config(["--config", str(cfg)])


def test_shipped_baseline_config_valid():
    args = parse_args_with_config(["--config", _config("baseline.yaml")])
    assert args.mlm_weight == 1.0
    assert args.num_encoder_layers == 5 and args.num_decoder_layers == 5
    assert args.embed_dim == 640 and args.num_heads == 20
    assert args.esm_model == "ESM2-150M"
    assert args.devices == [0, 1, 2, 3]
    assert args.lr == pytest.approx(3e-4)


def test_shipped_no_mlm_config_differs_only_in_mlm_weight():
    """no_mlm.yaml must be the baseline with mlm_weight flipped to 0."""
    base = parse_args_with_config(["--config", _config("baseline.yaml")])
    nomlm = parse_args_with_config(["--config", _config("no_mlm.yaml")])
    assert nomlm.mlm_weight == 0.0 and base.mlm_weight == 1.0
    ignore = {"mlm_weight", "name_addon", "output_dir", "config"}
    base_d = {k: v for k, v in vars(base).items() if k not in ignore}
    nomlm_d = {k: v for k, v in vars(nomlm).items() if k not in ignore}
    assert base_d == nomlm_d


def test_build_parser_has_ablation_flags():
    dests = {a.dest for a in build_parser()._actions}
    for flag in ("mlm_weight", "no_time_conditioning", "esm_finetune_mode",
                 "lora_rank", "architecture", "mask_prob", "config"):
        assert flag in dests
