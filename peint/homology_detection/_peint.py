"""PEINT-based homology detection.

This module provides homology detection using the PEINT model's likelihood
evaluation. The encoder output is cached for efficiency when comparing
many sequences against a reference.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from peint.homology_detection._base import (
    DistanceMatrix,
    HomologyHit,
    HomologySearcher,
    Proteome,
    get_distance,
)
from peint.models._loading import load_peint_model
from peint.models._transformer_modules import FLASH_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class PeintSearchConfig:
    """Configuration for PEINT homology search.

    Attributes:
        checkpoint_path: Path to PEINT model checkpoint
        device: Torch device to use (e.g., 'cuda', 'cpu')
        time: Default evolutionary time for comparisons
        batch_size: Batch size for decoder evaluation
        use_flash: Whether to use Flash Attention
    """

    checkpoint_path: str
    device: str = "cuda"
    time: float = 1.0
    batch_size: int = 32
    use_flash: bool = True


class PeintHomologySearcher(HomologySearcher):
    """Homology searcher using PEINT model likelihood evaluation.

    The PEINT model evaluates P(query | reference, time), where lower negative
    log-likelihood indicates higher similarity. The encoder (~175M params with ESM2)
    is much more expensive than the decoder (~25M params), so encoder outputs are
    cached for efficiency.
    """

    def __init__(self, config: PeintSearchConfig):
        """Initialize PEINT homology searcher.

        Args:
            config: Search configuration

        Raises:
            RuntimeError: If Flash Attention is not available (required for evaluator)
        """
        if not FLASH_AVAILABLE:
            raise RuntimeError(
                "PEINT homology search requires Flash Attention (GPU compute capability >= 8.0). "
                "Use DIAMOND for homology search on this hardware."
            )

        self.config = config
        self.device = torch.device(config.device)

        if self.device.type != "cuda":
            logger.warning("CUDA not available - performance will be significantly reduced")

        logger.info(f"Loading model from {config.checkpoint_path}")
        self.model, self.vocab = load_peint_model(
            checkpoint_path=config.checkpoint_path,
            device=self.device,
            model_type="evaluator",
            use_flash=config.use_flash,
        )

    def _compare_sequences(
        self,
        reference_seq: str,
        query_seqs: List[str],
        times: List[float],
    ) -> List[float]:
        """Compute negative log-likelihoods for query sequences against a reference.

        Args:
            reference_seq: Single reference sequence string
            query_seqs: List of query sequence strings
            times: List of evolutionary times (one per query)

        Returns:
            List of negative log-likelihoods (lower = more similar)
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                nlls = self.model.evaluate_likelihood(
                    x=reference_seq,
                    y=query_seqs,
                    t=times,
                    device=self.device,
                    batch_size=self.config.batch_size,
                )
        return nlls.tolist() if hasattr(nlls, "tolist") else list(nlls)

    def search(
        self,
        database: List[Tuple[str, str]],
        queries: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[HomologyHit]:
        """Search query sequences against a database.

        For each database sequence as reference, evaluates likelihoods of all queries.

        Args:
            database: List of (id, sequence) tuples for reference database
            queries: List of (id, sequence) tuples for query sequences
            top_k: If set, return only top K matches per query

        Returns:
            List of HomologyHit objects sorted by score (best first per query)
        """
        # Collect results per query
        query_results: Dict[str, List[HomologyHit]] = {qid: [] for qid, _ in queries}

        query_seqs = [seq for _, seq in queries]
        query_ids = [qid for qid, _ in queries]
        times = [self.config.time] * len(queries)

        for db_id, db_seq in tqdm(database, desc="Searching database"):
            nlls = self._compare_sequences(db_seq, query_seqs, times)

            for query_id, nll in zip(query_ids, nlls):
                query_results[query_id].append(
                    HomologyHit(
                        query_id=query_id,
                        db_id=db_id,
                        score=nll,
                        extra={"time": self.config.time},
                    )
                )

        # Flatten, sort by score per query, and optionally truncate
        all_hits = []
        for query_id in query_ids:
            hits = sorted(query_results[query_id], key=lambda x: x.score)
            if top_k is not None:
                hits = hits[:top_k]
            all_hits.extend(hits)

        return all_hits

    def all_vs_all(self, sequences: List[Tuple[str, str]]) -> List[HomologyHit]:
        """Compare all sequences against each other.

        Args:
            sequences: List of (id, sequence) tuples

        Returns:
            List of HomologyHit objects for all pairwise comparisons
        """
        hits = []
        seq_ids = [sid for sid, _ in sequences]
        seqs = [seq for _, seq in sequences]
        times = [self.config.time] * len(sequences)

        for i, (ref_id, ref_seq) in enumerate(tqdm(sequences, desc="All-vs-all")):
            # Compare against all other sequences
            other_indices = [j for j in range(len(sequences)) if j != i]
            other_seqs = [seqs[j] for j in other_indices]
            other_ids = [seq_ids[j] for j in other_indices]
            other_times = [times[j] for j in other_indices]

            if not other_seqs:
                continue

            nlls = self._compare_sequences(ref_seq, other_seqs, other_times)

            for query_id, nll in zip(other_ids, nlls):
                hits.append(
                    HomologyHit(
                        query_id=query_id,
                        db_id=ref_id,
                        score=nll,
                        extra={"time": self.config.time},
                    )
                )

        return hits

    def all_vs_all_proteomes(
        self,
        proteomes: Proteome,
        distances: Optional[DistanceMatrix] = None,
        skip_same_proteome: bool = True,
    ) -> List[HomologyHit]:
        """Compare all sequences across proteomes.

        For each reference sequence, evaluates likelihoods of all sequences in other
        proteomes (and optionally the same proteome). Uses distance matrix for
        evolutionary times if provided.

        Args:
            proteomes: Dict mapping proteome names to list of (id, sequence) tuples
            distances: Optional distance matrix for evolutionary times
            skip_same_proteome: If True, skip comparisons within same proteome

        Returns:
            List of HomologyHit objects
        """
        hits = []
        proteome_names = list(proteomes.keys())

        total_refs = sum(len(seqs) for seqs in proteomes.values())
        pbar = tqdm(total=total_refs, desc="Processing references")

        for ref_proteome in proteome_names:
            for ref_id, ref_seq in proteomes[ref_proteome]:
                # Collect all query sequences for this reference
                query_seqs = []
                query_times = []
                query_info = []  # (query_id, query_proteome)

                for query_proteome in proteome_names:
                    if skip_same_proteome and query_proteome == ref_proteome:
                        continue

                    time_val = get_distance(
                        ref_proteome, query_proteome, distances, self.config.time
                    )

                    for query_id, query_seq in proteomes[query_proteome]:
                        query_seqs.append(query_seq)
                        query_times.append(time_val)
                        query_info.append((query_id, query_proteome))

                if not query_seqs:
                    pbar.update(1)
                    continue

                # Compare all queries against this reference
                nlls = self._compare_sequences(ref_seq, query_seqs, query_times)

                for nll, time_val, (query_id, query_proteome) in zip(
                    nlls, query_times, query_info
                ):
                    hits.append(
                        HomologyHit(
                            query_id=query_id,
                            db_id=ref_id,
                            score=nll,
                            extra={
                                "time": time_val,
                                "ref_proteome": ref_proteome,
                                "query_proteome": query_proteome,
                            },
                        )
                    )

                pbar.update(1)

        pbar.close()
        return hits
