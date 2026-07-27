"""Tests for the training launcher's YAML-config plumbing.

Fast (no model construction): they only exercise argument parsing / merging, which
every ablation depends on. Guards the precedence rule (CLI > YAML > defaults), the
unknown-key safety check, and that the shipped ablation configs are valid.
"""

import os

import pytest

from protevo.models._config import PeintConfig
from train_peint_model import build_parser, latest_checkpoint, parse_args_with_config

PEINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _config(name):
    return os.path.join(PEINT_ROOT, "configs", "ablations", name)


def test_baseline_is_the_published_default_config():
    """A0: baseline.yaml is the UNMODIFIED default config — every ablation axis
    matches PeintConfig's own default, so ablations differ from it in one axis only.
    """
    args = parse_args_with_config(["--config", _config("baseline.yaml")])
    defaults = PeintConfig.__dataclass_fields__
    # ablation axes at their published defaults
    assert args.mlm_weight == defaults["mlm_weight"].default == 1.0
    assert args.esm_finetune_mode == defaults["esm_finetune_mode"].default == "frozen"
    assert args.lora_rank == defaults["lora_rank"].default is None
    assert args.architecture == defaults["architecture"].default == "encoder_decoder"
    assert args.esm_model == defaults["encoder_backbone"].default == "ESM2-150M"
    # published architecture scalars
    assert (args.num_encoder_layers, args.num_decoder_layers) == (5, 5)
    assert args.embed_dim == 640 and args.num_heads == 20
    assert args.dropout_p == 0.0 and args.use_attention_bias is True
    assert args.mask_prob == 0.15


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
    assert args.devices == [0, 1]
    assert args.accumulate_grad_batches == 12  # 2-GPU protocol; 32*12*2 = 768 seqs/update
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


def test_latest_checkpoint_for_preemption_resume(tmp_path):
    """Auto-resume picks the newest checkpoint (recursively), for requeue safety."""
    assert latest_checkpoint(None) is None
    assert latest_checkpoint(str(tmp_path)) is None  # nothing yet

    # Simulate two run subdirs (e.g. resumed on a later day) with checkpoints.
    import os
    import time
    old = tmp_path / "run_a" / "step-4000.ckpt"
    new = tmp_path / "run_b" / "step-8000.ckpt"
    for p in (old, new):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    os.utime(old, (time.time() - 100, time.time() - 100))  # make `old` older
    assert latest_checkpoint(str(tmp_path)) == str(new)


def test_build_parser_has_ablation_flags():
    dests = {a.dest for a in build_parser()._actions}
    for flag in ("mlm_weight", "esm_finetune_mode",
                 "lora_rank", "architecture", "mask_prob", "config"):
        assert flag in dests
