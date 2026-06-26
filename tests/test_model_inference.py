"""Tests for PEINT model inference and reproducibility."""

import pytest
import torch
import numpy as np

from tests.conftest import prepare_model_input


class TestModelReproducibility:
    """Test that model produces consistent outputs."""

    @pytest.mark.integration
    def test_logits_match_reference(self, loaded_model, example_transition, reference_logits, device):
        """Verify model output matches saved reference logits."""
        model, vocab = loaded_model
        x, y, t = example_transition

        x_toks, y_toks, y_targs, ts, x_attn_mask, y_attn_mask = prepare_model_input(
            x, y, t, vocab, device
        )

        with torch.no_grad():
            x_logits, y_logits, representations, self_attns, cross_attns = model(
                x_toks, y_toks, ts, x_attn_mask, y_attn_mask
            )

        y_logits_np = y_logits.cpu().numpy()

        # Check shape matches
        assert y_logits_np.shape == reference_logits.shape, \
            f"Shape mismatch: {y_logits_np.shape} vs {reference_logits.shape}"

        # Check values are close
        max_diff = np.max(np.abs(reference_logits - y_logits_np))
        assert np.allclose(reference_logits, y_logits_np, rtol=1e-4, atol=1e-4), \
            f"Logits differ from reference. Max difference: {max_diff}"

    @pytest.mark.integration
    def test_model_returns_expected_outputs(self, loaded_model, example_transition, device):
        """Verify model returns all expected output tensors."""
        model, vocab = loaded_model
        x, y, t = example_transition

        x_toks, y_toks, y_targs, ts, x_attn_mask, y_attn_mask = prepare_model_input(
            x, y, t, vocab, device
        )

        with torch.no_grad():
            outputs = model(x_toks, y_toks, ts, x_attn_mask, y_attn_mask)

        # PeintTransformerVanilla returns 5 values
        assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"

        x_logits, y_logits, representations, self_attns, cross_attns = outputs

        # Check logit shapes
        batch_size = x_toks.size(0)
        x_seq_len = x_toks.size(1)
        y_seq_len = y_toks.size(1)
        vocab_size = len(vocab)

        assert x_logits.shape == (batch_size, x_seq_len, vocab_size)
        assert y_logits.shape == (batch_size, y_seq_len, vocab_size)

        # Check representations dict is populated
        assert isinstance(representations, dict)
        assert len(representations) > 0

        # Check attention dicts are populated
        assert isinstance(self_attns, dict)
        assert isinstance(cross_attns, dict)


class TestModelGeneration:
    """Test model sequence generation."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_generate_produces_valid_sequences(self, loaded_model, example_transition, device):
        """Test that generate() produces valid amino acid sequences."""
        model, vocab = loaded_model
        x, y, t = example_transition

        x_tokens = [vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx]
        x_toks = torch.tensor(x_tokens).unsqueeze(0).to(device)
        ts = torch.tensor([t], dtype=torch.float32).unsqueeze(0).to(device)

        # Generate with deterministic sampling (p=0 means argmax)
        generated = model.generate(
            x=x_toks,
            t=ts,
            max_decode_steps=50,
            device=device,
            temperature=1.0,
            p=0.0
        )

        assert len(generated) == 1, "Should return one sequence per batch item"
        assert isinstance(generated[0], str), "Generated output should be string"
        assert len(generated[0]) > 0, "Generated sequence should not be empty"

        # Check all characters are valid amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        for char in generated[0]:
            assert char in valid_aas, f"Invalid amino acid: {char}"


class TestLikelihoodEvaluation:
    """Test evaluate_likelihood, which must work with or without Flash Attention.

    The loaded_model fixture uses the standard-attention (Vanilla) model, so these
    tests exercise the non-Flash likelihood path.
    """

    @pytest.mark.integration
    def test_evaluate_likelihood_shape_and_finite(self, loaded_model, example_transition, device):
        """evaluate_likelihood returns one finite NLL per target sequence."""
        model, vocab = loaded_model
        x, y, t = example_transition

        nlls = np.atleast_1d(model.evaluate_likelihood(
            x=x, y=[y, y, y], t=[t, t, t], device=device, batch_size=8
        ))

        assert nlls.shape == (3,)
        assert np.isfinite(nlls).all()
        # Identical targets must receive identical scores (deterministic).
        assert np.allclose(nlls, nlls[0])

    @pytest.mark.integration
    def test_evaluate_likelihood_batching_consistent(self, loaded_model, example_transition, device):
        """Scores are independent of batch size (single vs multiple batches)."""
        model, vocab = loaded_model
        x, y, t = example_transition

        targets = [x, y, y, x, y]
        times = [t] * len(targets)

        one_batch = np.atleast_1d(model.evaluate_likelihood(
            x=x, y=targets, t=times, device=device, batch_size=len(targets)
        ))
        many_batches = np.atleast_1d(model.evaluate_likelihood(
            x=x, y=targets, t=times, device=device, batch_size=2
        ))

        assert np.allclose(one_batch, many_batches, atol=1e-4)
