"""Pytest configuration and fixtures for PEINT tests."""

import os
import pytest
import torch
import numpy as np

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'protevo', 'tests')
MODEL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_checkpoints')


def get_checkpoint_path():
    """Locate the PEINT base checkpoint used by the reference-logits fixtures.

    Prefers ``peint.ckpt`` (the model that ``y_logits.npy`` / ``example_transition``
    were generated from); otherwise falls back to the first checkpoint in sorted
    order for determinism.
    """
    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        return None
    checkpoints = sorted(f for f in os.listdir(MODEL_CHECKPOINT_DIR) if f.endswith('.ckpt'))
    if not checkpoints:
        return None
    preferred = 'peint.ckpt' if 'peint.ckpt' in checkpoints else checkpoints[0]
    return os.path.join(MODEL_CHECKPOINT_DIR, preferred)


@pytest.fixture(scope="session")
def device():
    """Get available torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def checkpoint_path():
    """Get checkpoint path, skip test if not available."""
    path = get_checkpoint_path()
    if path is None:
        pytest.skip("No model checkpoint available")
    return path


@pytest.fixture(scope="session")
def loaded_model(checkpoint_path, device):
    """Load model for testing (uses Vanilla/non-Flash for compatibility)."""
    from protevo.models import load_model
    model, vocab = load_model(
        checkpoint_path,
        use_cached_model=False,
        device=device,
        use_flash=False
    )
    model = model.eval()
    return model, vocab


@pytest.fixture
def example_transition():
    """Load example transition data."""
    transition_file = os.path.join(TEST_DATA_DIR, 'example_transition.txt')
    with open(transition_file, 'r') as f:
        line = f.readline().strip()
        x, y, t = line.split(' ')
        t = float(t)
    return x, y, t


@pytest.fixture
def reference_logits():
    """Load reference logits for reproducibility check."""
    logits_file = os.path.join(TEST_DATA_DIR, 'y_logits.npy')
    return np.load(logits_file)


def prepare_model_input(x, y, t, vocab, device):
    """Prepare input tensors for model forward pass."""
    x_tokens = [vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx]
    y_tokens = [vocab.cls_idx] + vocab.encode(y)
    y_targets = vocab.encode(y) + [vocab.eos_idx]

    x_toks = torch.tensor(x_tokens).unsqueeze(0).to(device)
    y_toks = torch.tensor(y_tokens).unsqueeze(0).to(device)
    y_targs = torch.tensor(y_targets).unsqueeze(0).to(device)
    ts = torch.tensor([t], dtype=torch.float32).unsqueeze(0).to(device)

    x_attn_mask = x_toks.eq(vocab.padding_idx)
    y_attn_mask = y_toks.eq(vocab.padding_idx)

    return x_toks, y_toks, y_targs, ts, x_attn_mask, y_attn_mask
