"""Per-site log-likelihood of transitions under PEINT.

PEINT reads an unaligned source sequence ``x`` and autoregressively scores an
unaligned target ``y`` at evolutionary time ``t``. To compare it against
site-independent models such as LG, every per-residue log-likelihood is projected
back onto the alignment the transition was drawn from:

1. Forward pass on ``(x, y, t)`` gives ``log P(y_i | x, t)`` per residue of ``y``.
2. The alignment mask drops the residues that are insertions relative to the
   query, which the a3m alignment does not represent.
3. The gap pattern of the aligned target places what remains in its column.

The resulting arrays have the same shape as the classical models' output, so the
two can be compared column by column.
"""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from protevo import caching as protevo_caching
from protevo import io
from protevo.datasets import PeintCollator
from protevo.models import load_peint_model
from protevo.utils import amino_acids

from ._aligned_transitions import DEFAULT_MAX_LENGTH, AlignedTransitionsDataset

logger = logging.getLogger(__name__)

# Gap columns are scored as 0 so that summing over sites ignores them. Averaging
# over sites has to mask them out instead.
GAP_LOG_LIKELIHOOD = 0.0


@torch.no_grad()
def evaluate_transitions_log_likelihood_per_site(
    model,
    vocab,
    dataset: AlignedTransitionsDataset,
    device: torch.device,
    batch_size: int = 8,
    restrict_to_amino_acids: bool = True,
    progress: bool = False,
) -> np.ndarray:
    """Score one family's transitions and project them onto its alignment.

    Args:
        model: A PEINT model. Not ``PeintEvaluator``: its encoder cache assumes
            one source sequence for many targets, and here every transition has
            its own source.
        vocab: ESM alphabet the model was loaded with.
        dataset: Transitions of a single family.
        device: Device to run the forward passes on.
        batch_size: Transitions per forward pass.
        restrict_to_amino_acids: Renormalize each position over the 20 amino
            acids, discarding the gap and special tokens PEINT can also emit.
            Keep this on to compare against models defined on 20 states.
        progress: Show a progress bar over batches.

    Returns:
        A ``[num_transitions, alignment_width]`` array of log-likelihoods. Gap
        columns hold :data:`GAP_LOG_LIKELIHOOD`. Columns that were not scored are
        ``NaN``: whole rows for transitions too long to run, single columns for
        residues that carry no mass under ``restrict_to_amino_acids``.
    """
    collator = PeintCollator(vocab=vocab, mask_prob=0.0)
    non_amino_acid_tokens = torch.tensor(
        [vocab.get_idx(tok) for tok in vocab.all_toks if tok not in amino_acids],
        device=device,
    )

    per_site = np.full((dataset.num_transitions, dataset.alignment_width), np.nan)

    starts = range(0, len(dataset), batch_size)
    for start in tqdm(starts, desc=dataset.family, disable=not progress):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        x, _, y, y_targets, t, x_pad_mask, y_pad_mask = [
            tensor.to(device)
            for tensor in collator([transition.model_input for transition in batch])
        ]

        logits = model.evaluate_transition_logits(x, y, t, x_pad_mask, y_pad_mask).float()
        if restrict_to_amino_acids:
            logits[:, :, non_amino_acid_tokens] = -torch.inf

        log_likelihood = -F.cross_entropy(
            logits.transpose(1, 2),
            y_targets,
            ignore_index=vocab.padding_idx,
            reduction="none",
        )
        is_target = y_targets.ne(vocab.padding_idx)

        for transition, row, row_is_target in zip(
            batch, log_likelihood.cpu().numpy(), is_target.cpu().numpy()
        ):
            residues = row[row_is_target][:-1]  # the last target is EOS
            per_site[transition.row] = GAP_LOG_LIKELIHOOD
            per_site[transition.row, transition.columns] = residues[transition.keep]

    unscorable = np.isinf(per_site)
    if unscorable.any():
        # Ambiguous residues (X and friends) carry no mass once the distribution
        # is restricted to the 20 amino acids. Report them as unscored, not -inf.
        logger.warning(
            "%s: %d residues have no probability mass under the amino acid "
            "restriction; leaving them unscored.",
            dataset.family,
            int(unscorable.sum()),
        )
        per_site[unscorable] = np.nan

    return per_site


def sum_over_sites(log_likelihood_per_site: np.ndarray) -> np.ndarray:
    """Total log-likelihood of each transition, ignoring unscored columns.

    Transitions with no scored column at all stay ``NaN`` rather than summing
    to zero.
    """
    unscored = np.isnan(log_likelihood_per_site)
    return np.where(
        unscored.all(axis=1), np.nan, np.nansum(log_likelihood_per_site, axis=1)
    )


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_transitions_log_likelihood_dir",
        "output_transitions_log_likelihood_per_site_dir",
    ],
    exclude_args=["device", "batch_size", "use_flash"],
    exclude_args_if_default=["restrict_to_amino_acids", "max_length"],
    write_extra_log_files=True,
)
def evaluate_peint_model_transitions_log_likelihood__cached(
    transitions_dir: str,
    aligned_transitions_dir: str,
    alignment_mask_dir: str,
    model_checkpoint_path: str,
    families: List[str],
    device: str = "cpu",
    batch_size: int = 8,
    use_flash: bool = True,
    restrict_to_amino_acids: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
    output_transitions_log_likelihood_dir: Optional[str] = None,
    output_transitions_log_likelihood_per_site_dir: Optional[str] = None,
    _version: str = "2026_07_25_v1",
) -> None:
    """Compute transitions log-likelihood under PEINT, indexed by alignment column.

    Mirrors ``evaluate_lg_model_transitions_log_likelihood__cached``: same output
    directories, same file layout, one row per transition.

    Args:
        transitions_dir: The unaligned transitions to score. The transitions for
            family 'family' should be in the file '{family}.txt'.
        aligned_transitions_dir: The same transitions, aligned.
        alignment_mask_dir: Per-residue masks marking which residues of the
            unaligned sequences occupy an alignment column.
        model_checkpoint_path: PEINT checkpoint to evaluate.
        families: List of families for which to compute the log-likelihood.
        device: Device to run the forward passes on.
        batch_size: Transitions per forward pass.
        use_flash: Use Flash Attention when it is available. The Flash path runs
            in bfloat16, which costs roughly 0.1 nats of noise on an individual
            site; pass False to score in fp32 instead.
        restrict_to_amino_acids: Renormalize each position over the 20 amino
            acids, so the likelihoods are comparable to the classical models'.
        max_length: Longest unaligned sequence to score; longer transitions are
            written out unscored.
        output_transitions_log_likelihood_dir: Where the log-likelihoods will get
            written. The log-likelihoods for family 'family' will be in the file
            '{family}.txt', with one line per transition.
        output_transitions_log_likelihood_per_site_dir: Where the per-site
            log-likelihoods will get written. The log-likelihoods for family
            'family' will be in the file '{family}.txt', with one row per
            transition and one column per alignment column.
    """
    device = torch.device(device)
    model, vocab = load_peint_model(
        model_checkpoint_path,
        device=device,
        model_type="standard",
        use_flash=use_flash,
    )

    logger.info("Going to score %d families on %s", len(families), device)

    for family in tqdm(families, desc="families"):
        dataset = AlignedTransitionsDataset(
            transitions_dir=transitions_dir,
            aligned_transitions_dir=aligned_transitions_dir,
            alignment_mask_dir=alignment_mask_dir,
            family=family,
            vocab=vocab,
            max_length=max_length,
        )
        if len(dataset) < dataset.num_transitions:
            logger.warning(
                "%s: %d of %d transitions exceed max_length=%d and are left unscored.",
                family,
                dataset.num_transitions - len(dataset),
                dataset.num_transitions,
                max_length,
            )

        per_site = evaluate_transitions_log_likelihood_per_site(
            model=model,
            vocab=vocab,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            restrict_to_amino_acids=restrict_to_amino_acids,
        )

        io.write_transitions_log_likelihood_per_site(
            transitions_log_likelihood_per_site=per_site.tolist(),
            transitions_log_likelihood_per_site_path=os.path.join(
                output_transitions_log_likelihood_per_site_dir, family + ".txt"
            ),
        )
        protevo_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_per_site_dir,
            parallel_arg=family,
        )

        io.write_transitions_log_likelihood(
            transitions_log_likelihood=sum_over_sites(per_site).tolist(),
            transitions_log_likelihood_path=os.path.join(
                output_transitions_log_likelihood_dir, family + ".txt"
            ),
        )
        protevo_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_dir,
            parallel_arg=family,
        )
