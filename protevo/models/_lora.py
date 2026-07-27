"""LoRA adapters for the frozen backbone (A5 ablation), backed by `peft`.

This module is imported ONLY when ``esm_finetune_mode == "lora"``. The default
``frozen`` path never touches it, so the published behavior is bit-identical and
there is no ``peft`` dependency on the default code path.

We use the standard `peft` LoRA implementation (Hu et al., 2021) rather than a
hand-rolled one: it is the community standard, and — since PEINT is intended to
ship through the HuggingFace ecosystem — LoRA adapters land in peft's standard
format, which the ecosystem loads natively. peft's ``inject_adapter_in_model``
adapts arbitrary ``nn.Module``s in place (it does not require a HuggingFace
``PreTrainedModel``), so the custom ``ESM2Flash`` backbone is used unchanged.

LoRA freezes a linear's weight W and learns a low-rank update:
y = Wx + (alpha/r) * B(A x). peft zero-initializes B, so the adapter output is
exactly zero at init — the LoRA model reproduces the frozen model until training
moves the adapters (verified against the ESM2 backbone).
"""

# Which backbone linear layers get adapters. Standard LoRA adapts the attention
# query/value projections; the ESM2 flash and non-flash backbones both name these
# ``q_proj``/``v_proj`` (see RopeFlashMHA / ESM2 MultiheadAttention).
DEFAULT_TARGET_SUFFIXES = ("q_proj", "v_proj")


def apply_lora_to_backbone(
    module,
    rank: int,
    alpha=None,
    target_suffixes=DEFAULT_TARGET_SUFFIXES,
    dropout: float = 0.0,
) -> int:
    """Inject peft LoRA adapters into ``module`` in place; return #layers adapted.

    The caller is expected to have frozen the base weights first; peft creates the
    LoRA A/B params as trainable, so the standard ``requires_grad`` optimizer filter
    picks up exactly the adapters (no change to the training module needed).

    Args:
        module: backbone to adapt (mutated in place).
        rank: LoRA rank (> 0).
        alpha: LoRA scaling numerator; defaults to ``rank`` (scaling = 1.0).
        target_suffixes: adapt linears whose attribute name matches one of these.
        dropout: LoRA dropout.

    Raises:
        ImportError: if peft is not installed (only hit on the LoRA path).
        ValueError: if rank is not positive, or no target layers matched.
    """
    if not (isinstance(rank, int) and rank > 0):
        raise ValueError(f"LoRA rank must be a positive int, got {rank!r}")
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as e:  # pragma: no cover - only when peft is absent
        raise ImportError(
            "esm_finetune_mode='lora' requires the 'peft' package "
            "(pip install peft, or install the package's [train] extra)."
        ) from e

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank,
        target_modules=list(target_suffixes),
        lora_dropout=dropout,
        bias="none",
    )
    inject_adapter_in_model(config, module)

    n_adapted = sum(
        1 for name, _ in module.named_parameters() if name.endswith("lora_A.default.weight")
    )
    if n_adapted == 0:
        raise ValueError(
            f"LoRA matched no layers for target_modules={list(target_suffixes)}; "
            "check the target names against the backbone."
        )
    return n_adapted
