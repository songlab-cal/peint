"""Unified homology detection framework.

This module provides interchangeable PEINT and DIAMOND methods for
protein homology detection with a common interface.

Example:
    >>> from protevo.homology_detection import (
    ...     PeintHomologySearcher, PeintSearchConfig,
    ...     load_fasta
    ... )
    >>> config = PeintSearchConfig(checkpoint_path="model.ckpt")
    >>> searcher = PeintHomologySearcher(config)
    >>> database = load_fasta("database.fasta")
    >>> queries = load_fasta("queries.fasta")
    >>> hits = searcher.search(database, queries)
"""

from protevo.homology_detection._base import (
    DistanceMatrix,
    HomologyHit,
    HomologySearcher,
    Proteome,
    get_distance,
    load_distance_matrix,
    load_fasta,
    load_proteomes,
)
from protevo.homology_detection._diamond import (
    DiamondHomologySearcher,
    DiamondSearchConfig,
)
from protevo.homology_detection._peint import (
    PeintHomologySearcher,
    PeintSearchConfig,
)

__all__ = [
    # Base classes and types
    "HomologyHit",
    "HomologySearcher",
    "Proteome",
    "DistanceMatrix",
    # PEINT
    "PeintHomologySearcher",
    "PeintSearchConfig",
    # DIAMOND
    "DiamondHomologySearcher",
    "DiamondSearchConfig",
    # Utilities
    "load_fasta",
    "load_proteomes",
    "load_distance_matrix",
    "get_distance",
]
