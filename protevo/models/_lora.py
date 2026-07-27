"""Self-contained LoRA adapters for the frozen backbone (A5 ablation).

This module is imported ONLY when ``esm_finetune_mode == "lora"``. The default
``frozen`` path never touches it, so the published behavior is bit-identical and
there is no new dependency (no ``peft``) on the default code path.

LoRA (Hu et al., 2021) freezes a linear layer's weight W and learns a low-rank
update: y = Wx + (alpha/r) * B(A x), with A in R^{r x in}, B in R^{out x r}. B is
initialized to zero so the adapter output is exactly zero at init — i.e. the LoRA
model reproduces the frozen model until training moves the adapters.
"""

import math

import torch
import torch.nn as nn

# Which backbone linear layers get adapters. Standard LoRA adapts the attention
# query/value projections; the ESM2 flash and non-flash backbones both name these
# ``q_proj``/``v_proj`` (see RopeFlashMHA / ESM2 MultiheadAttention).
DEFAULT_TARGET_SUFFIXES = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.base.requires_grad_(False)  # base weight stays frozen
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B left at zero -> zero adapter output at init (identity to frozen).
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out.to(base_out.dtype)


def apply_lora_to_backbone(
    module: nn.Module,
    rank: int,
    alpha=None,
    target_suffixes=DEFAULT_TARGET_SUFFIXES,
    dropout: float = 0.0,
) -> int:
    """Replace target ``nn.Linear`` submodules of ``module`` in place with LoRALinear.

    The base weights stay frozen; only the LoRA A/B params are trainable, so the
    standard ``requires_grad`` optimizer filter picks up exactly the adapters (no
    change to the training module needed).

    Args:
        module: backbone to adapt (mutated in place).
        rank: LoRA rank (> 0).
        alpha: LoRA scaling numerator; defaults to ``rank`` (scaling = 1.0).
        target_suffixes: adapt linears whose attribute name ends with one of these.
        dropout: dropout on the LoRA input.

    Returns:
        Number of linear layers adapted (raises if none matched, to catch a typo).
    """
    if alpha is None:
        alpha = rank
    # Snapshot the module list first: we mutate the tree while iterating.
    targets = [
        name
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and name.rsplit(".", 1)[-1] in target_suffixes
    ]
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        base = getattr(parent, leaf)
        setattr(parent, leaf, LoRALinear(base, rank, alpha, dropout))
    if not targets:
        raise ValueError(
            f"LoRA matched no layers for suffixes {target_suffixes}; "
            "check the target names against the backbone."
        )
    return len(targets)
