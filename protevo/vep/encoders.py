"""Frozen pLM encoders for PEINT.

This is the single integration point for swapping the encoder that PEINT sits on top
of. `load_esm_model(which_esm)` returns `(ESM2Flash, esm_vocab)` ready to be passed to
`PeintLightningModule`. Supported keys live in `_config.ESM_REGISTRY`:

  - stock ESM2:  "150M", "650M", "3B", "15B"  (loaded from fair-esm)
  - vESM:        "vesm_650M"                   (HuggingFace EsmForMaskedLM weights,
                                                converted into the fair-esm layout)

Why a converter is needed: this codebase runs a FlashAttention re-implementation of
fair-esm's ESM2 (`ESM2Flash`), whereas vESM is distributed as a HuggingFace
`transformers` `EsmForMaskedLM` checkpoint. vESM_650M is architecturally identical to
ESM2-650M, so only the *weights* differ. `convert_hf_esm_state_dict_to_fair_esm`
remaps the HF state-dict keys to the fair-esm layout; we then load them into the
`ESM2Flash` scaffold with `strict=False` (exactly as stock ESM2 is loaded today —
rotary buffers legitimately differ).

NOTE: scoring/eval need no encoder awareness — frozen encoder weights are saved inside
each PEINT checkpoint, so `_vep_utils.load_model` reconstructs them from the ckpt.
"""

from typing import Tuple

import torch

from protevo.models._flash_esm import ESM2Flash, ESM2Model
from protevo.vep._config import ESM_REGISTRY


# ---------------------------------------------------------------------------
# HuggingFace EsmForMaskedLM  ->  fair-esm ESM2  state-dict conversion
# ---------------------------------------------------------------------------
# Verified against transformers EsmForMaskedLM and esm.model.esm2.ESM2 key layouts.
# Per-layer (HF suffix -> fair-esm suffix), with i the layer index:
_HF_LAYER_SUFFIX_MAP = {
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "attention.LayerNorm": "self_attn_layer_norm",
    "intermediate.dense": "fc1",
    "output.dense": "fc2",
    "LayerNorm": "final_layer_norm",
}
# Top-level (HF key -> fair-esm key)
_HF_TOPLEVEL_MAP = {
    "esm.embeddings.word_embeddings.weight": "embed_tokens.weight",
    "esm.encoder.emb_layer_norm_after.weight": "emb_layer_norm_after.weight",
    "esm.encoder.emb_layer_norm_after.bias": "emb_layer_norm_after.bias",
    "esm.contact_head.regression.weight": "contact_head.regression.weight",
    "esm.contact_head.regression.bias": "contact_head.regression.bias",
    "lm_head.bias": "lm_head.bias",
    "lm_head.dense.weight": "lm_head.dense.weight",
    "lm_head.dense.bias": "lm_head.dense.bias",
    "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
    "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
    "lm_head.decoder.weight": "lm_head.weight",
}
# HF keys we intentionally drop (fair-esm computes these as buffers / has no analogue):
#   *.position_embeddings.weight  (ESM2 uses rotary, no learned positions)
#   *.rotary_embeddings.inv_freq  (recomputed buffer; ESM2Flash keeps its own)


def convert_hf_esm_state_dict_to_fair_esm(hf_sd: dict) -> dict:
    """Remap a HuggingFace EsmForMaskedLM state dict to the fair-esm ESM2 layout.

    Pure function (no I/O), so it is unit-testable without downloading weights. Keys
    that have no fair-esm analogue (learned position embeddings, rotary inv_freq) are
    dropped; the resulting dict is meant to be loaded with `strict=False`.
    """
    out = {}
    unmapped = []
    for k, v in hf_sd.items():
        if k in _HF_TOPLEVEL_MAP:
            out[_HF_TOPLEVEL_MAP[k]] = v
            continue
        if k.endswith("position_embeddings.weight") or k.endswith("rotary_embeddings.inv_freq"):
            continue  # intentionally dropped
        if k.startswith("esm.encoder.layer."):
            # esm.encoder.layer.{i}.{suffix}.{param}
            rest = k[len("esm.encoder.layer.") :]
            idx, suffix_param = rest.split(".", 1)
            matched = False
            for hf_suffix, fe_suffix in _HF_LAYER_SUFFIX_MAP.items():
                if suffix_param.startswith(hf_suffix + "."):
                    param = suffix_param[len(hf_suffix) + 1 :]
                    out[f"layers.{idx}.{fe_suffix}.{param}"] = v
                    matched = True
                    break
            if not matched:
                unmapped.append(k)
            continue
        unmapped.append(k)
    if unmapped:
        print(f"[encoders] WARNING: {len(unmapped)} unmapped vESM keys (ignored): {unmapped[:6]}{' ...' if len(unmapped) > 6 else ''}")
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _build_esm_from_fair_esm(esm_model, use_flash: bool = True):
    """Build an ESM2 scaffold with the architecture of `esm_model` and copy its weights.

    `use_flash` selects the FlashAttention encoder (`ESM2Flash`, needs a GPU) or the
    standard PyTorch encoder (`ESM2Model`, CPU-capable). The two share weights; only the
    attention/rotary implementation differs.
    """
    cls = ESM2Flash if use_flash else ESM2Model
    built = cls(
        num_layers=esm_model.num_layers,
        embed_dim=esm_model.embed_dim,
        attention_heads=esm_model.attention_heads,
        alphabet="ESM-1b",
        token_dropout=True,
        dropout_p=0.0,  # ESM2 does not use dropout
    )
    # rot emb differs -> strict=False (mismatched rotary keys are expected)
    built.load_state_dict(esm_model.state_dict(), strict=False)
    return built


def _load_stock_esm(which_esm: str, use_flash: bool = True):
    """Load a stock ESM2 model from fair-esm and wrap it as ESM2Flash/ESM2Model."""
    import esm

    loaders = {
        "150M": esm.pretrained.esm2_t30_150M_UR50D,
        "650M": esm.pretrained.esm2_t33_650M_UR50D,
        "3B": esm.pretrained.esm2_t36_3B_UR50D,
        "15B": esm.pretrained.esm2_t48_15B_UR50D,
    }
    if which_esm not in loaders:
        raise ValueError(f"Invalid stock ESM model: {which_esm}")
    print(f"Loading stock ESM2 ({which_esm})...")
    esm_model, esm_vocab = loaders[which_esm]()
    built = _build_esm_from_fair_esm(esm_model, use_flash=use_flash)
    del esm_model  # required for the Lightning trainer to behave
    return built, esm_vocab


def _load_vesm(which_esm: str, use_flash: bool = True):
    """Load vESM weights into an ESM2Flash scaffold of the matching base size.

    vESM is a HuggingFace EsmForMaskedLM `.pth` state dict. We build the fair-esm
    scaffold for the base size (for architecture + the ESM-1b vocab), convert the vESM
    keys, and load them with strict=False.
    """
    from protevo.vep._config import ENCODER_STAGING_DIR

    spec = ESM_REGISTRY[which_esm]
    base = spec["base"]
    print(f"Loading vESM ({which_esm}; base {base})...")

    # Scaffold (architecture + vocab) from the stock base.
    flash, esm_vocab = _load_stock_esm(base, use_flash=use_flash)

    # vESM weights (HF EsmForMaskedLM state dict): prefer a pre-staged shared copy
    # (so SLURM compute nodes need no internet), else download from the Hub.
    staged = ENCODER_STAGING_DIR / spec["hf_file"]
    if staged.exists():
        print(f"  using staged weights: {staged}")
        pth = str(staged)
    else:
        from huggingface_hub import hf_hub_download

        pth = hf_hub_download(repo_id=spec["hf_repo"], filename=spec["hf_file"])
    hf_sd = torch.load(pth, map_location="cpu")
    if isinstance(hf_sd, dict) and "state_dict" in hf_sd:
        hf_sd = hf_sd["state_dict"]

    fair_sd = convert_hf_esm_state_dict_to_fair_esm(hf_sd)
    missing, unexpected = flash.load_state_dict(fair_sd, strict=False)
    # vESM may ship a FULL fine-tuned model (e.g. VESM_150M) or only a PARTIAL set of
    # fine-tuned weights (e.g. VESM_650M overlays only the top encoder layers + word
    # embeddings + lm_head). The scaffold already holds the full stock base, so keys absent
    # from the vESM file correctly retain their base weights — `missing` is expected/benign.
    n_overlaid = len(fair_sd) - len(unexpected)
    print(f"  overlaid {n_overlaid}/{len(fair_sd)} vESM tensors onto the stock {base} encoder")
    if unexpected:
        print(f"[encoders] WARNING: {len(unexpected)} vESM keys not found in ESM2Flash: {list(unexpected)[:6]}")
    return flash, esm_vocab


def load_esm_model(which_esm: str = "150M", use_flash: bool = True) -> Tuple[object, object]:
    """Return `(encoder, esm_vocab)` for the requested encoder key.

    Keys: "150M" | "650M" | "3B" | "15B" | "vesm_650M" (see _config.ESM_REGISTRY).
    `use_flash=True` builds the FlashAttention encoder (`ESM2Flash`, GPU); `use_flash=False`
    builds the standard PyTorch encoder (`ESM2Model`), which also runs on CPU.
    """
    if which_esm not in ESM_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{which_esm}'. Options: {sorted(ESM_REGISTRY)}"
        )
    if "hf_repo" in ESM_REGISTRY[which_esm]:
        return _load_vesm(which_esm, use_flash=use_flash)
    return _load_stock_esm(which_esm, use_flash=use_flash)


# ---------------------------------------------------------------------------
# Verification (run before launching a vESM training run)
# ---------------------------------------------------------------------------
@torch.no_grad()
def verify_vesm_conversion(which_esm: str = "vesm_650M", seq: str = None, atol: float = 1e-2):
    """Sanity check: HF vESM logits vs converted ESM2Flash logits on one sequence.

    Downloads vESM and the HF base model; intended to be run once (ideally on GPU)
    before training. Returns the max absolute logit difference.
    """
    import esm as _esm
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, EsmForMaskedLM

    spec = ESM_REGISTRY[which_esm]
    base = spec["base"]
    hf_base_id = {
        "150M": "facebook/esm2_t30_150M_UR50D",
        "650M": "facebook/esm2_t33_650M_UR50D",
    }[base]
    if seq is None:
        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"

    # HF vESM reference
    hf = EsmForMaskedLM.from_pretrained(hf_base_id)
    pth = hf_hub_download(repo_id=spec["hf_repo"], filename=spec["hf_file"])
    hf.load_state_dict(torch.load(pth, map_location="cpu"), strict=False)
    hf.eval()
    tok = AutoTokenizer.from_pretrained(hf_base_id)
    hf_ids = tok(seq, return_tensors="pt")["input_ids"]
    hf_logits = hf(hf_ids).logits

    # Converted ESM2Flash
    flash, vocab = load_esm_model(which_esm)
    flash.eval()
    fe_ids = torch.tensor([[vocab.cls_idx] + vocab.encode(seq) + [vocab.eos_idx]])
    fe_logits = flash(fe_ids)["logits"]

    max_diff = (hf_logits - fe_logits).abs().max().item()
    print(f"[verify_vesm_conversion] max|Δlogits| = {max_diff:.4g} (atol={atol})")
    assert max_diff < atol, "vESM conversion mismatch — check the key mapping"
    return max_diff
