"""The flash-attn-free path: peint must import without flash-attn, and the pure-torch rotary
stand-ins must agree numerically with flash-attn's own implementation.

Level-1 replotting imports peint (several figure producers do) but never runs a model, so an
import that hard-requires flash-attn blocks data-only panels on CPU machines.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

from peint.models._rotary_fallback import (
    RotaryEmbedding as FallbackRotary,
    apply_rotary_emb_torch as fallback_apply,
    rotate_half as fallback_rotate_half,
)

flash_rotary = pytest.importorskip(
    "flash_attn.layers.rotary",
    reason="equivalence can only be checked where flash-attn is installed",
)

HEAD_DIM, SEQLENS, NHEADS, BATCH = 32, (1, 7, 64), 4, 2


def test_import_peint_without_flash_attn():
    """`import peint` must succeed when flash_attn is unimportable."""
    script = textwrap.dedent(
        """
        import sys, importlib.abc

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "flash_attn" or fullname.startswith("flash_attn."):
                    raise ImportError(fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import peint
        from peint.models import _transformer_modules as tm
        assert tm.FLASH_AVAILABLE is False
        # the non-flash attention stack must be constructible
        tm.VanillaMHA(embed_dim=64, num_heads=4)
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK" in r.stdout


@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("interleaved", [False, True])
def test_apply_rotary_matches_flash(seqlen, interleaved):
    torch.manual_seed(0)
    x = torch.randn(BATCH, seqlen, NHEADS, HEAD_DIM, dtype=torch.float32)
    cos = torch.randn(seqlen, HEAD_DIM // 2, dtype=torch.float32)
    sin = torch.randn(seqlen, HEAD_DIM // 2, dtype=torch.float32)
    ours = fallback_apply(x, cos, sin, interleaved=interleaved)
    theirs = flash_rotary.apply_rotary_emb_torch(x, cos, sin, interleaved=interleaved)
    assert torch.equal(ours, theirs), (ours - theirs).abs().max().item()


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotate_half_matches_flash(interleaved):
    torch.manual_seed(1)
    x = torch.randn(BATCH, 5, NHEADS, HEAD_DIM, dtype=torch.float32)
    assert torch.equal(
        fallback_rotate_half(x, interleaved), flash_rotary.rotate_half(x, interleaved)
    )


@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cos_sin_cache_matches_flash(seqlen, dtype):
    """The cache PEINT reads (_cos_cached/_sin_cached) must be bit-identical."""
    ours = FallbackRotary(HEAD_DIM, base=10000.0, interleaved=False, scale_base=None,
                          pos_idx_in_fp32=True)
    theirs = flash_rotary.RotaryEmbedding(HEAD_DIM, base=10000.0, interleaved=False,
                                          scale_base=None, pos_idx_in_fp32=True)
    ours._update_cos_sin_cache(seqlen, device=torch.device("cpu"), dtype=dtype)
    theirs._update_cos_sin_cache(seqlen, device=torch.device("cpu"), dtype=dtype)
    assert torch.equal(ours.inv_freq, theirs.inv_freq)
    assert torch.equal(ours._cos_cached, theirs._cos_cached)
    assert torch.equal(ours._sin_cached, theirs._sin_cached)


def test_scale_base_is_rejected_rather_than_silently_wrong():
    with pytest.raises(NotImplementedError):
        FallbackRotary(HEAD_DIM, scale_base=512)
