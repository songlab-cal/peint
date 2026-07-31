"""Base classes and utilities for homology detection.

This module provides the abstract base class for homology searchers and shared
utilities for loading sequences and writing results.
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type aliases
Proteome = Dict[str, List[Tuple[str, str]]]  # proteome_name -> [(seq_id, sequence), ...]
DistanceMatrix = Dict[Tuple[str, str], float]  # (proteome1, proteome2) -> distance


@dataclass
class HomologyHit:
    """A single homology search hit.

    Attributes:
        query_id: Identifier of the query sequence
        db_id: Identifier of the database/reference sequence
        score: Similarity score (interpretation depends on method)
        extra: Method-specific additional fields
    """

    query_id: str
    db_id: str
    score: float
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to flat dictionary for CSV output."""
        result = {"query_id": self.query_id, "db_id": self.db_id, "score": self.score}
        result.update(self.extra)
        return result


class HomologySearcher(ABC):
    """Abstract base class for homology search methods."""

    @abstractmethod
    def search(
        self,
        database: List[Tuple[str, str]],
        queries: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[HomologyHit]:
        """Search query sequences against a database.

        Args:
            database: List of (id, sequence) tuples for reference database
            queries: List of (id, sequence) tuples for query sequences
            top_k: If set, return only top K matches per query

        Returns:
            List of HomologyHit objects
        """
        pass

    @abstractmethod
    def all_vs_all(self, sequences: List[Tuple[str, str]]) -> List[HomologyHit]:
        """Compare all sequences against each other.

        Args:
            sequences: List of (id, sequence) tuples

        Returns:
            List of HomologyHit objects for all pairwise comparisons
        """
        pass

    @abstractmethod
    def all_vs_all_proteomes(
        self,
        proteomes: Proteome,
        distances: Optional[DistanceMatrix] = None,
    ) -> List[HomologyHit]:
        """Compare all sequences across proteomes.

        Args:
            proteomes: Dict mapping proteome names to list of (id, sequence) tuples
            distances: Optional distance matrix for evolutionary times (method-specific)

        Returns:
            List of HomologyHit objects
        """
        pass

    @staticmethod
    def write_results(
        hits: List[HomologyHit], path: str, format: str = "csv"
    ) -> None:
        """Write search results to file.

        Pure serialization - it never touched ``self``, and making that explicit
        lets the multi-GPU CLI path write results without constructing a searcher
        (and so without loading a model onto GPU 0 for nothing). Existing
        ``searcher.write_results(hits, path)`` calls are unaffected.

        Args:
            hits: List of HomologyHit objects
            path: Output file path
            format: 'csv' or 'json'
        """
        if not hits:
            logger.warning("No results to write")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w") as f:
                json.dump([h.to_dict() for h in hits], f, indent=2)
        else:
            rows = [h.to_dict() for h in hits]
            fieldnames = list(rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Wrote {len(hits)} results to {path}")


def load_fasta(path: str) -> List[Tuple[str, str]]:
    """Load sequences from a FASTA file.

    Args:
        path: Path to FASTA file

    Returns:
        List of (sequence_id, sequence) tuples
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:]  # Remove '>'
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_id is not None:
            sequences.append((current_id, "".join(current_seq)))

    return sequences


def load_proteomes(proteome_dir: str) -> Proteome:
    """Load all FASTA files from a directory as proteomes.

    Args:
        proteome_dir: Directory containing .fasta files

    Returns:
        Dict mapping proteome name (filename without extension) to sequences
    """
    proteome_dir = Path(proteome_dir)
    proteomes = {}

    for fasta_path in proteome_dir.glob("*.fasta"):
        proteome_name = fasta_path.stem
        proteomes[proteome_name] = load_fasta(str(fasta_path))
        logger.info(f"Loaded {len(proteomes[proteome_name])} sequences from {proteome_name}")

    return proteomes


def load_distance_matrix(path: str, default: float = 1.0) -> DistanceMatrix:
    """Load distance matrix from CSV/TSV file.

    Expected format (auto-detects delimiter):
        ,human,mouse,yeast
        human,0.0,0.5,1.2
        mouse,0.5,0.0,1.1
        yeast,1.2,1.1,0.0

    Args:
        path: Path to distance matrix file
        default: Default distance for missing pairs

    Returns:
        Dict mapping (proteome1, proteome2) to distance
    """
    distances = {}

    with open(path) as f:
        content = f.read()

    # Auto-detect delimiter
    delimiter = "\t" if "\t" in content.split("\n")[0] else ","

    lines = content.strip().split("\n")
    header = lines[0].split(delimiter)[1:]  # Skip first empty cell

    for line in lines[1:]:
        parts = line.split(delimiter)
        row_name = parts[0]
        values = parts[1:]

        for col_name, value in zip(header, values):
            try:
                dist = float(value)
            except ValueError:
                dist = default
            distances[(row_name, col_name)] = dist
            if (col_name, row_name) not in distances:
                distances[(col_name, row_name)] = dist

    return distances


def get_distance(
    proteome1: str,
    proteome2: str,
    distances: Optional[DistanceMatrix],
    default: float = 1.0,
) -> float:
    """Get distance between two proteomes."""
    if distances is None:
        return default
    return distances.get((proteome1, proteome2), default)
