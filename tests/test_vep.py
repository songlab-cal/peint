"""Tests for the VEP module (protevo.vep).

Unit tests (no downloads, no checkpoint) cover the two pieces of load-time logic that
are easy to get wrong: the HF->fair-esm vESM key converter and the ESM-architecture
inference used to rebuild the encoder from a full checkpoint. The integration test
exercises the real CPU scoring path on the bundled neuraminidase DMS data.
"""

import os

import numpy as np
import pytest
import torch

from protevo.vep.encoders import convert_hf_esm_state_dict_to_fair_esm
from protevo.vep._vep_utils import _infer_esm_arch_from_state_dict, load_model

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
VEP_CKPT = os.path.join(REPO_ROOT, "model_checkpoints", "vep.ckpt")
NRAM_CSV = os.path.join(REPO_ROOT, "NRAM_I33A0_Jiang_2016.csv")


class TestVESMConverter:
    """convert_hf_esm_state_dict_to_fair_esm is a pure key-remapper (unit-testable)."""

    def test_key_remapping(self):
        hf_sd = {
            "esm.embeddings.word_embeddings.weight": torch.zeros(3, 4),
            "esm.encoder.layer.0.attention.self.query.weight": torch.zeros(4, 4),
            "esm.encoder.layer.0.attention.self.query.bias": torch.zeros(4),
            "esm.encoder.layer.0.intermediate.dense.weight": torch.zeros(8, 4),
            "esm.encoder.layer.2.LayerNorm.weight": torch.zeros(4),
            "lm_head.decoder.weight": torch.zeros(3, 4),
        }
        out = convert_hf_esm_state_dict_to_fair_esm(hf_sd)
        assert out.keys() == {
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.q_proj.bias",
            "layers.0.fc1.weight",
            "layers.2.final_layer_norm.weight",
            "lm_head.weight",
        }

    def test_drops_position_and_rotary_keys(self):
        hf_sd = {
            "esm.embeddings.position_embeddings.weight": torch.zeros(2, 4),
            "esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq": torch.zeros(2),
        }
        assert convert_hf_esm_state_dict_to_fair_esm(hf_sd) == {}


class TestArchInference:
    """_infer_esm_arch_from_state_dict reads (num_layers, embed_dim, heads) from a ckpt."""

    def test_infers_150m_geometry(self):
        sd = {f"model.esm.layers.{i}.self_attn.q_proj.weight": torch.zeros(1) for i in range(30)}
        sd["model.esm.embed_tokens.weight"] = torch.zeros(33, 640)
        assert _infer_esm_arch_from_state_dict(sd) == (30, 640, 20)

    def test_infers_650m_geometry(self):
        sd = {f"model.esm.layers.{i}.self_attn.q_proj.weight": torch.zeros(1) for i in range(33)}
        sd["model.esm.embed_tokens.weight"] = torch.zeros(33, 1280)
        assert _infer_esm_arch_from_state_dict(sd) == (33, 1280, 20)

    def test_raises_on_stripped_checkpoint(self):
        # No model.esm.* keys -> cannot rebuild the encoder from the checkpoint.
        with pytest.raises(ValueError):
            _infer_esm_arch_from_state_dict({"model.enc_layers.0.self_attn.q_proj.weight": torch.zeros(1)})


@pytest.mark.integration
@pytest.mark.slow
class TestCPUScoring:
    """End-to-end CPU scoring of the (stripped) distributed vep.ckpt on real DMS data."""

    @pytest.fixture(scope="class")
    def nram_pairs(self):
        if not os.path.exists(NRAM_CSV):
            pytest.skip("NRAM_I33A0_Jiang_2016.csv not available")
        import pandas as pd

        df = pd.read_csv(NRAM_CSV).iloc[:8].reset_index(drop=True)

        def wild_type_from_mutant(mutant, seq):
            wt_aa, mut_aa = mutant[0], mutant[-1]
            idx = int(mutant[1:-1]) - 1
            assert seq[idx] == mut_aa
            return seq[:idx] + wt_aa + seq[idx + 1:]

        wt = wild_type_from_mutant(df["mutant"][0], df["mutated_sequence"][0])
        return [(wt, s) for s in df["mutated_sequence"]]

    def test_load_and_score_cpu(self, nram_pairs):
        if not os.path.exists(VEP_CKPT):
            pytest.skip("model_checkpoints/vep.ckpt not available")
        from protevo.models._transformer import PeintTransformerVanilla
        from protevo.vep._scoring import score_transition_pairs

        device = torch.device("cpu")
        model, vocab = load_model(VEP_CKPT, device, use_flash=False)
        assert isinstance(model, PeintTransformerVanilla)

        scores = score_transition_pairs(model, vocab, device, nram_pairs, t_value=1.0, batch_size=8)
        assert len(scores) == len(nram_pairs)
        assert all(np.isfinite(float(s)) for s in scores)
        # Scores must vary across distinct mutants (sanity: model is actually reading input).
        assert len(set(round(float(s), 6) for s in scores)) > 1
