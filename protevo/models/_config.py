"""Configuration dataclass for PEINT models.

This module provides typed configuration to replace scattered kwargs.get() calls,
ensuring typos are caught at instantiation time rather than silently using defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PeintConfig:
    """Configuration for PEINT transformer models.

    All architecture and training hyperparameters are defined here with explicit
    types and defaults. Using a dataclass ensures:
    - Typos in parameter names raise errors immediately
    - IDE autocompletion works correctly
    - Default values are documented in one place

    Attributes:
        embed_dim: Embedding dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder transformer layers
        num_decoder_layers: Number of decoder transformer layers
        max_seq_len: Maximum sequence length (default: 1022 for ESM2)
        dropout_p: Dropout probability (default: 0.0)
        use_attention_bias: Whether to use bias in attention layers (default: True)
        label_smoothing: Label smoothing for cross-entropy loss (default: 0.0)
        max_encoder_seq_len: Max encoder sequence length for cached decoders (default: 1024)
        max_decoder_seq_len: Max decoder sequence length for cached decoders (default: 1024)
        weight_decay: Weight decay for optimizer (default: 0.0)
    """

    embed_dim: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    max_seq_len: int = 1022
    dropout_p: float = 0.0
    use_attention_bias: bool = True
    label_smoothing: float = 0.0
    max_encoder_seq_len: int = 1024
    max_decoder_seq_len: int = 1024
    weight_decay: float = 0.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be positive")
        if self.num_decoder_layers <= 0:
            raise ValueError("num_decoder_layers must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.dropout_p <= 1.0:
            raise ValueError("dropout_p must be between 0 and 1")
        if not 0.0 <= self.label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be between 0 and 1")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PeintConfig":
        """Create config from dictionary, ignoring unknown keys.

        This allows loading from checkpoints that may have extra keys.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "max_seq_len": self.max_seq_len,
            "dropout_p": self.dropout_p,
            "use_attention_bias": self.use_attention_bias,
            "label_smoothing": self.label_smoothing,
            "max_encoder_seq_len": self.max_encoder_seq_len,
            "max_decoder_seq_len": self.max_decoder_seq_len,
            "weight_decay": self.weight_decay,
        }
