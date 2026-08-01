"""Inference-time utilities: fast tokenization and multi-GPU work sharding.

Nothing here changes what PEINT computes — these are throughput tools for the
shipped checkpoints. See ``benchmarks/inference/parity.py`` for the gate that
holds the optimized paths bit-exact against the pristine release tree.
"""

from protevo.inference._batching import (
    DEFAULT_MAX_TOKENS,
    fixed_size_batches,
    largest_batch_first,
    padding_waste,
    token_budget_batches,
)
from protevo.inference._shard import (
    available_gpus,
    item_seed,
    run_sharded,
    shard_indices,
)
from protevo.inference._tokenize import (
    build_token_lut,
    encode_all,
    encode_batch,
    encode_one,
    pad_encoded,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "available_gpus",
    "build_token_lut",
    "encode_all",
    "encode_batch",
    "encode_one",
    "fixed_size_batches",
    "item_seed",
    "largest_batch_first",
    "pad_encoded",
    "padding_waste",
    "run_sharded",
    "shard_indices",
    "token_budget_batches",
]
