"""DIAMOND-based homology detection.

This module provides homology detection using DIAMOND, a fast sequence
aligner for protein and translated DNA searches.
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from peint.homology_detection._base import (
    DistanceMatrix,
    HomologyHit,
    HomologySearcher,
    Proteome,
)

logger = logging.getLogger(__name__)


def _check_diamond_available() -> bool:
    """Check if DIAMOND is installed and available."""
    return shutil.which("diamond") is not None


@dataclass
class DiamondSearchConfig:
    """Configuration for DIAMOND homology search.

    Attributes:
        evalue: E-value threshold for hits
        threads: Number of threads to use
        sensitivity: DIAMOND sensitivity mode (fast, mid-sensitive, sensitive, more-sensitive, very-sensitive, ultra-sensitive)
        max_target_seqs: Maximum number of target sequences per query (0 for unlimited)
    """

    evalue: float = 0.001
    threads: int = 4
    sensitivity: str = "more-sensitive"
    max_target_seqs: int = 0


class DiamondHomologySearcher(HomologySearcher):
    """Homology searcher using DIAMOND blastp.

    DIAMOND provides fast sequence alignment using E-values as similarity scores.
    Lower E-values indicate more significant matches.
    """

    def __init__(self, config: DiamondSearchConfig):
        """Initialize DIAMOND homology searcher.

        Args:
            config: Search configuration

        Raises:
            RuntimeError: If DIAMOND is not installed
        """
        if not _check_diamond_available():
            raise RuntimeError(
                "DIAMOND is not installed or not in PATH. "
                "Install with: conda install -c bioconda diamond"
            )
        self.config = config

    @staticmethod
    def _clean_id(seq_id: str) -> str:
        """Clean sequence ID for FASTA compatibility."""
        return seq_id.replace(" ", "_")

    def _write_fasta(self, sequences: List[Tuple[str, str]], path: Path) -> None:
        """Write sequences to a FASTA file."""
        with open(path, "w") as f:
            for seq_id, seq in sequences:
                clean_id = self._clean_id(seq_id)
                f.write(f">{clean_id}\n{seq}\n")

    def _make_db(self, fasta_path: Path, db_path: Path) -> None:
        """Create DIAMOND database from FASTA file."""
        cmd = [
            "diamond",
            "makedb",
            "--in",
            str(fasta_path),
            "-d",
            str(db_path),
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create DIAMOND database: {result.stderr}")

    def _run_blastp(
        self, query_path: Path, db_path: Path, output_path: Path
    ) -> None:
        """Run DIAMOND blastp search."""
        cmd = [
            "diamond",
            "blastp",
            "-d",
            str(db_path),
            "-q",
            str(query_path),
            "-o",
            str(output_path),
            f"--{self.config.sensitivity}",
            "-p",
            str(self.config.threads),
            "-e",
            str(self.config.evalue),
            "-f",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "evalue",
            "bitscore",
            "--quiet",
        ]
        if self.config.max_target_seqs > 0:
            cmd.extend(["-k", str(self.config.max_target_seqs)])
        else:
            cmd.append("-k0")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"DIAMOND blastp failed: {result.stderr}")

    def _parse_results(self, output_path: Path) -> List[HomologyHit]:
        """Parse DIAMOND tabular output to HomologyHit objects."""
        hits = []
        if not output_path.exists():
            return hits

        with open(output_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 6:
                    continue
                qseqid, sseqid, pident, length, evalue, bitscore = parts[:6]
                hits.append(
                    HomologyHit(
                        query_id=qseqid,
                        db_id=sseqid,
                        score=float(evalue),
                        extra={
                            "pident": float(pident),
                            "length": int(length),
                            "bitscore": float(bitscore),
                        },
                    )
                )
        return hits

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
            List of HomologyHit objects sorted by score (best first per query)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write database and create DIAMOND db
            db_fasta = tmpdir / "database.fasta"
            db_dmnd = tmpdir / "database"
            self._write_fasta(database, db_fasta)
            self._make_db(db_fasta, db_dmnd)

            # Write queries
            query_fasta = tmpdir / "queries.fasta"
            self._write_fasta(queries, query_fasta)

            # Run search
            output_file = tmpdir / "results.tsv"
            self._run_blastp(query_fasta, db_dmnd, output_file)

            # Parse results
            hits = self._parse_results(output_file)

        # Sort by query_id then score, apply top_k per query
        if top_k is not None:
            hits_by_query: Dict[str, List[HomologyHit]] = {}
            for hit in hits:
                hits_by_query.setdefault(hit.query_id, []).append(hit)

            filtered_hits = []
            for query_id, query_hits in hits_by_query.items():
                query_hits.sort(key=lambda x: x.score)
                filtered_hits.extend(query_hits[:top_k])
            hits = filtered_hits

        return hits

    def all_vs_all(self, sequences: List[Tuple[str, str]]) -> List[HomologyHit]:
        """Compare all sequences against each other.

        Args:
            sequences: List of (id, sequence) tuples

        Returns:
            List of HomologyHit objects for all pairwise comparisons
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write sequences and create database
            fasta_path = tmpdir / "sequences.fasta"
            db_path = tmpdir / "sequences"
            self._write_fasta(sequences, fasta_path)
            self._make_db(fasta_path, db_path)

            # Run all-vs-all search
            output_file = tmpdir / "results.tsv"
            self._run_blastp(fasta_path, db_path, output_file)

            # Parse results, excluding self-hits
            all_hits = self._parse_results(output_file)
            hits = [h for h in all_hits if h.query_id != h.db_id]

        return hits

    def all_vs_all_proteomes(
        self,
        proteomes: Proteome,
        distances: Optional[DistanceMatrix] = None,
    ) -> List[HomologyHit]:
        """Compare all sequences across proteomes.

        Note: DIAMOND ignores the distances parameter as it uses E-values
        rather than evolutionary time-based scoring.

        Args:
            proteomes: Dict mapping proteome names to list of (id, sequence) tuples
            distances: Ignored (included for interface compatibility)

        Returns:
            List of HomologyHit objects with proteome information in extra fields
        """
        if distances is not None:
            logger.info("DIAMOND ignores distance matrix (uses E-values instead)")

        # Collect all sequences with proteome info
        # Use cleaned IDs for mapping since DIAMOND output uses cleaned IDs
        all_sequences = []
        seq_to_proteome = {}
        for proteome_name, seqs in proteomes.items():
            for seq_id, seq in seqs:
                all_sequences.append((seq_id, seq))
                clean_id = self._clean_id(seq_id)
                seq_to_proteome[clean_id] = proteome_name

        # Run all-vs-all
        hits = self.all_vs_all(all_sequences)

        # Add proteome information to hits (IDs are already cleaned by DIAMOND)
        for hit in hits:
            hit.extra["ref_proteome"] = seq_to_proteome.get(hit.db_id, "unknown")
            hit.extra["query_proteome"] = seq_to_proteome.get(hit.query_id, "unknown")

        # Filter to only cross-proteome hits
        hits = [
            h for h in hits if h.extra["ref_proteome"] != h.extra["query_proteome"]
        ]

        return hits
