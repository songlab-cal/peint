"""Tests for PEINT time MLE estimation."""

import pytest
import torch

from tests.conftest import prepare_model_input


class TestTimeMLE:
    """Test maximum likelihood estimation of evolutionary time."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_time_mle_converges(self, loaded_model, example_transition, device):
        """Test that time MLE optimization converges near true value."""
        model, vocab = loaded_model
        x, y, true_t = example_transition

        # Prepare forward and reverse batches
        batch_f = prepare_model_input(x, y, true_t, vocab, device)
        batch_r = prepare_model_input(y, x, true_t, vocab, device)

        # Initialize time guess away from true value
        initial_guess = 0.6
        t_guesses = torch.full((1, 1), initial_guess, requires_grad=True, device=device)

        optimizer = torch.optim.Adam([t_guesses], lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        eps = 5e-3

        # Run optimization
        for step in range(100):
            optimizer.zero_grad()

            # Forward direction
            x_toks, y_toks, y_targs, _, x_mask, y_mask = batch_f
            x_logits, y_logits, _, _, _ = model(x_toks, y_toks, t_guesses, x_mask, y_mask)
            forward_nll = torch.nn.functional.cross_entropy(
                y_logits.transpose(-1, -2), y_targs,
                ignore_index=vocab.padding_idx, reduction='none'
            ).sum(dim=1)

            # Reverse direction
            x_toks, y_toks, y_targs, _, x_mask, y_mask = batch_r
            x_logits, y_logits, _, _, _ = model(x_toks, y_toks, t_guesses, x_mask, y_mask)
            reverse_nll = torch.nn.functional.cross_entropy(
                y_logits.transpose(-1, -2), y_targs,
                ignore_index=vocab.padding_idx, reduction='none'
            ).sum(dim=1)

            loss = torch.mean(forward_nll + reverse_nll)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            t_guesses.data.clamp_(min=eps)

        t_mle = t_guesses.item()
        absolute_error = abs(t_mle - true_t)

        # The MLE should converge reasonably close to true value
        # Using a tolerance of 0.1 since this is a single sequence
        assert absolute_error < 0.1, \
            f"Time MLE did not converge. True: {true_t:.4f}, Estimated: {t_mle:.4f}, Error: {absolute_error:.4f}"

    @pytest.mark.integration
    def test_time_gradient_flows(self, loaded_model, example_transition, device):
        """Test that gradients flow through time parameter."""
        model, vocab = loaded_model
        x, y, t = example_transition

        x_toks, y_toks, y_targs, _, x_mask, y_mask = prepare_model_input(x, y, t, vocab, device)

        t_param = torch.tensor([[0.5]], requires_grad=True, device=device)

        x_logits, y_logits, _, _, _ = model(x_toks, y_toks, t_param, x_mask, y_mask)
        loss = torch.nn.functional.cross_entropy(
            y_logits.transpose(-1, -2), y_targs,
            ignore_index=vocab.padding_idx
        )
        loss.backward()

        assert t_param.grad is not None, "Gradient should flow to time parameter"
        assert not torch.isnan(t_param.grad).any(), "Gradient should not be NaN"
        assert not torch.isinf(t_param.grad).any(), "Gradient should not be infinite"
