"""Pure-torch stand-ins for the two rotary helpers PEINT uses from flash-attn.

``_transformer_modules`` imports these only when ``flash_attn`` is unavailable, so that
``import peint`` works without it -- the plotting code imports peint but never runs a model.
Semantics mirror ``flash_attn.layers.rotary``; ``tests/test_rotary_fallback.py`` asserts
numerical equivalence against the real implementation whenever flash-attn is installed.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """x: (batch, seqlen, nheads, headdim); cos/sin: (seqlen, rotary_dim / 2)."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    pattern = "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    cos = repeat(cos, pattern)
    sin = repeat(sin, pattern)
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )

class RotaryEmbedding(nn.Module):
    """The part of flash-attn's RotaryEmbedding that PEINT uses: the inverse-frequency
    buffer and the cos/sin cache. PEINT always constructs it with ``scale_base=None``,
    which is the only case implemented here."""

    def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None,
                 pos_idx_in_fp32=True, device=None):
        super().__init__()
        if scale_base is not None:
            raise NotImplementedError(
                "scale_base needs flash-attn's RotaryEmbedding; install flash-attn."
            )
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scale_base = None
        self.scale = None
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                inv_freq = (
                    self._compute_inv_freq(device=device)
                    if self.inv_freq.dtype != torch.float32
                    else self.inv_freq
                )
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
