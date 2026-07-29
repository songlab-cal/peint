"""A3 hardening: verify the ESM-C backbone integration is correct.

The ESM-C model code was ported from research work that was never carefully
checked; for a Nature revision it must be proven correct before any training run.

Two tiers:
  * vocab tests — fast, CPU, only need the `esm3` package (skipped if absent).
  * model tests — build the real ESM-C-300M backbone and a PEINT model on it, so
    they need `esm3`, a GPU (ESM-C uses flash-attention), and a ~1.2 GB download.
    Marked ``slow`` and skipped when esm3/GPU are unavailable.

They guard: vocab correctness (len, special tokens, encode/decode, non-AA guard),
that the embedding and sequence_head weights actually load from the backbone, that
the wrapped representation equals the raw ESM-C output, and end-to-end forward
shapes/finiteness.
"""

import pytest
import torch

esm3 = pytest.importorskip("esm3", reason="ESM-C tests require the esm3 package")

from protevo.datasets._vocab import get_esmc_vocab  # noqa: E402
from protevo.models._esm_registry import ESMC_REGISTRY, build_esm_backbone  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
gpu_only = pytest.mark.skipif(not HAS_CUDA, reason="ESM-C (flash-attn) needs a GPU")

AA = "ACDEFGHIKLMNPQRSTVWY"


# ----------------------------- vocab (fast, CPU) -----------------------------

def test_esmc_vocab_structure():
    v = get_esmc_vocab()
    # ESM-C's sequence vocab is 33 tokens (the model head is wider, 64).
    assert len(v) == 33
    for name in ("cls_idx", "eos_idx", "pad_idx", "mask_idx", "unk_idx"):
        idx = getattr(v, name)
        assert 0 <= idx < len(v), f"{name}={idx} out of range"
    assert v.padding_idx == v.pad_idx


def test_esmc_vocab_encode_decode_roundtrip():
    v = get_esmc_vocab()
    ids = v.encode(list(AA))
    assert all(0 <= i < len(v) for i in ids), "encoded ids out of vocab range"
    assert "".join(v.decode(ids)) == AA  # standard AAs round-trip exactly


def test_esmc_vocab_amino_acids_are_distinct_tokens():
    """All 20 AAs map to distinct, non-special token ids (needed so generation can
    mask non-AA tokens correctly)."""
    v = get_esmc_vocab()
    ids = v.encode(list(AA))
    specials = {v.cls_idx, v.eos_idx, v.pad_idx, v.mask_idx, v.unk_idx}
    assert len(set(ids)) == 20
    assert specials.isdisjoint(ids)


# --------------------------- model (slow, GPU) --------------------------------

@pytest.fixture(scope="module")
def esmc_backbone():
    module, vocab, embed_dim = build_esm_backbone("esmc")
    return module, vocab, embed_dim


@pytest.mark.slow
@gpu_only
def test_esmc_backbone_builds_with_expected_dims(esmc_backbone):
    module, vocab, embed_dim = esmc_backbone
    assert embed_dim == 960                       # esmc_300m hidden size
    assert type(module).__name__ == "ESMC"
    assert len(vocab) == 33
    assert ESMC_REGISTRY["esmc"] == ("esmc_300m", 960)


@pytest.mark.slow
@gpu_only
def test_esmc_embedding_and_lm_head_load_from_backbone(esmc_backbone):
    """The PEINT embedding/lm_head must be the backbone's actual pretrained weights."""
    from protevo.models._transformer import PeintTransformer
    module, vocab, embed_dim = esmc_backbone
    model = PeintTransformer(
        esm_model=module, esm_vocab=vocab, embed_dim=embed_dim,
        num_heads=20, num_encoder_layers=5, num_decoder_layers=5,
        encoder_backbone="esmc",
    ).eval()
    assert model._is_esmc
    # The PEINT embedding/lm_head are loaded FROM the ESM-C backbone's pretrained
    # weights, but held in float32 modules (the backbone is bf16 for its flash attn;
    # the PEINT side is float32 with bf16 supplied by autocast at train/eval time).
    # So compare VALUES, not dtype — the bf16 weights are exactly representable in
    # float32, so an exact equality holds once dtypes are matched.
    # embedding initialized from ESM-C's embed table (compare on cpu: the backbone may
    # sit on cuda while this un-.cuda()'d model's copies are on cpu).
    assert torch.equal(model.embedding.weight.detach().float().cpu(), module.embed.weight.detach().float().cpu())
    # lm_head is ESM-C's sequence_head (64-wide)
    ref = module.sequence_head.state_dict()
    got = model.lm_head.state_dict()
    assert got.keys() == ref.keys()
    for k in ref:
        assert torch.equal(got[k].float().cpu(), ref[k].float().cpu()), f"lm_head param {k} not loaded from sequence_head"


@pytest.mark.slow
@gpu_only
def test_esmc_representation_matches_raw_backbone(esmc_backbone):
    """model._compute_language_model_representations(x) == raw ESM-C embeddings."""
    from protevo.models._transformer import PeintTransformer
    module, vocab, embed_dim = esmc_backbone
    model = PeintTransformer(
        esm_model=module, esm_vocab=vocab, embed_dim=embed_dim,
        num_heads=20, num_encoder_layers=5, num_decoder_layers=5,
        encoder_backbone="esmc",
    ).eval().cuda()
    x = torch.tensor([vocab.cls_idx] + vocab.encode(list("ACDEFGHIK")) + [vocab.eos_idx]).unsqueeze(0).cuda()
    with torch.no_grad():
        rep = model._compute_language_model_representations(x)
        raw = model.esm(x).embeddings
    assert rep.shape[-1] == 960
    assert torch.equal(rep, raw)


@pytest.mark.slow
@gpu_only
def test_esmc_end_to_end_forward(esmc_backbone):
    """Full PEINT-ESM-C forward: finite logits over the 64-wide ESM-C head."""
    from protevo.models._transformer import PeintTransformer
    module, vocab, embed_dim = esmc_backbone
    model = PeintTransformer(
        esm_model=module, esm_vocab=vocab, embed_dim=embed_dim,
        num_heads=20, num_encoder_layers=5, num_decoder_layers=5,
        encoder_backbone="esmc",
    ).eval().cuda()
    x = torch.tensor([vocab.cls_idx] + vocab.encode(list("ACDEFGHIK")) + [vocab.eos_idx]).unsqueeze(0).cuda()
    y_in = torch.tensor([vocab.cls_idx] + vocab.encode(list("ACDEYGHIK"))).unsqueeze(0).cuda()
    t = torch.tensor([[0.1]], dtype=torch.float32).cuda()
    xm = x.eq(vocab.padding_idx)
    ym = y_in.eq(vocab.padding_idx)
    # PeintTransformer is the flash variant; flash attention requires bf16, so it runs
    # under bf16 autocast (exactly as training with precision='bf16' and the flash eval
    # do). A plain float32 forward is not a supported mode for a flash model — the golden
    # ESM2 reference test uses the Vanilla (non-flash) variant for its float32 forward.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x_logits, y_logits = model(x, y_in, t, xm, ym)
    assert x_logits.shape[-1] == 64 and y_logits.shape[-1] == 64
    assert torch.isfinite(x_logits).all() and torch.isfinite(y_logits).all()
