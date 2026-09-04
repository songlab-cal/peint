"""Transitions paired with the alignment columns they came from.

PEINT scores unaligned sequences, so its per-residue log-likelihoods are indexed
by residue. Classical site-independent models (LG, WAG) score alignment columns.
This module holds the bookkeeping needed to move from one indexing to the other.
"""

import os
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from peint import io
from peint.datasets._torch_datasets import MIN_TIME_THRESHOLD
from peint.utils import gap_character

# ESM's vocabulary covers every ambiguous amino acid code except J.
_TO_ESM_VOCAB = str.maketrans({"J": "I"})

# Longest unaligned sequence PEINT scores: ESM adds CLS and EOS on top of this.
DEFAULT_MAX_LENGTH = 1022


class AlignedTransition(NamedTuple):
    """One transition, ready to score and to project back onto the alignment.

    Attributes:
        model_input: ``(x, y, t, length)`` in the layout ``PeintCollator`` expects.
        keep: Bool mask over the residues of the unaligned ``y``; True where the
            residue occupies an alignment column.
        columns: Alignment column of each kept residue, in order.
        row: Index of this transition in the family's files.
    """

    model_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    keep: np.ndarray
    columns: np.ndarray
    row: int


class AlignedTransitionsDataset(Dataset):
    """Unaligned transitions for one family, indexed back to alignment columns.

    Three files describe the same transitions, one row each:

    ``transitions_dir/{family}.txt``
        The unaligned pair ``(x, y, t)``. Insertions relative to the query are
        retained, so these are the sequences PEINT consumes.
    ``alignment_mask_dir/{family}.txt``
        A ``0``/``1`` string per sequence, as long as the unaligned sequence.
        ``1`` marks a residue that occupies an alignment column, ``0`` an
        insertion that the a3m alignment drops.
    ``aligned_transitions_dir/{family}.txt``
        The same transitions as alignment rows. Only the gap pattern of ``y`` is
        used, to place each scored residue in its column.

    Transitions longer than ``max_length`` are dropped from the dataset but still
    counted in :attr:`num_transitions`, so their rows can be left unscored
    without losing correspondence with the classical models.

    Args:
        transitions_dir: Directory of unaligned transitions.
        aligned_transitions_dir: Directory of the same transitions, aligned.
        alignment_mask_dir: Directory of per-residue alignment masks.
        family: Family name, i.e. the file stem shared by all three directories.
        vocab: ESM alphabet used to tokenize the sequences.
        max_length: Longest unaligned sequence to score.
    """

    def __init__(
        self,
        transitions_dir: str,
        aligned_transitions_dir: str,
        alignment_mask_dir: str,
        family: str,
        vocab,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        transitions = io.read_transitions(os.path.join(transitions_dir, family + ".txt"))
        aligned = io.read_transitions(os.path.join(aligned_transitions_dir, family + ".txt"))
        # The mask files share the transitions format: a 0/1 string per sequence.
        masks = io.read_transitions(os.path.join(alignment_mask_dir, family + ".txt"))
        _validate_family(transitions, aligned, masks, family)

        self.family = family
        self.num_transitions = len(transitions)
        self.alignment_width = len(aligned[0][0])

        self.transitions = []
        for row, ((x, y, t), (_, aligned_y, _), (_, y_mask, _)) in enumerate(
            zip(transitions, aligned, masks)
        ):
            if max(len(x), len(y)) > max_length:
                continue
            self.transitions.append(
                AlignedTransition(
                    model_input=(
                        torch.tensor(vocab.encode(x.translate(_TO_ESM_VOCAB))),
                        torch.tensor(vocab.encode(y.translate(_TO_ESM_VOCAB))),
                        max(t, MIN_TIME_THRESHOLD) * torch.ones(len(x)),
                        len(x),
                    ),
                    keep=np.array([m == "1" for m in y_mask]),
                    columns=np.flatnonzero([c != gap_character for c in aligned_y]),
                    row=row,
                )
            )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> AlignedTransition:
        return self.transitions[idx]

    def scored_columns_mask(self) -> np.ndarray:
        """``[num_transitions, alignment_width]`` mask of the columns a model scores.

        True wherever the aligned target has a residue and the transition is
        short enough to run. Gap columns are False, so this is what to average
        over: the log-likelihood arrays themselves store 0 in gap columns, which
        is not distinguishable from a genuine score.
        """
        mask = np.zeros((self.num_transitions, self.alignment_width), dtype=bool)
        for transition in self.transitions:
            mask[transition.row, transition.columns] = True
        return mask


def _validate_family(transitions, aligned, masks, family: str) -> None:
    """Check that the three files describe the same transitions of one family."""
    if not len(transitions) == len(aligned) == len(masks):
        raise ValueError(
            f"Family '{family}' has {len(transitions)} unaligned transitions, "
            f"{len(aligned)} aligned transitions and {len(masks)} alignment masks."
        )
    widths = {len(seq) for x, y, _ in aligned for seq in (x, y)}
    if len(widths) != 1:
        raise ValueError(f"Family '{family}' has ragged alignment rows: widths {sorted(widths)}.")

    for row, ((_, y, _), (_, aligned_y, _), (_, y_mask, _)) in enumerate(
        zip(transitions, aligned, masks)
    ):
        if len(y_mask) != len(y):
            raise ValueError(
                f"Family '{family}' transition {row}: alignment mask covers "
                f"{len(y_mask)} residues but the unaligned target has {len(y)}."
            )
        kept = "".join(residue for residue, m in zip(y, y_mask) if m == "1")
        if kept != aligned_y.replace(gap_character, ""):
            raise ValueError(
                f"Family '{family}' transition {row}: the masked unaligned target "
                f"does not match the aligned target. Are the three directories "
                f"from the same dataset?"
            )


def list_families(
    transitions_dir: str,
    aligned_transitions_dir: str,
    alignment_mask_dir: str,
) -> List[str]:
    """Families that have a transitions file in all three directories."""

    def stems(directory: str) -> set:
        return {f[: -len(".txt")] for f in os.listdir(directory) if f.endswith(".txt")}

    return sorted(
        stems(transitions_dir) & stems(aligned_transitions_dir) & stems(alignment_mask_dir)
    )
