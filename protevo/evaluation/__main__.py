"""CLI for scoring transitions with PEINT.

Usage:
    python -m protevo.evaluation \\
        --transitions-dir local_data/unaligned/test_transitions_dir/output_transitions_dir \\
        --aligned-transitions-dir local_data/aligned/test_transitions_dir \\
        --alignment-mask-dir local_data/unaligned/test_alignment_mask_dir \\
        --checkpoint model_checkpoints/peint.ckpt \\
        --output-dir likelihoods --num-families 5 --device cuda
"""

import argparse
import logging
import os

import torch

from ._aligned_transitions import DEFAULT_MAX_LENGTH, list_families
from ._likelihood import evaluate_peint_model_transitions_log_likelihood__cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score transitions with PEINT, indexed by alignment column."
    )
    parser.add_argument("--transitions-dir", required=True, help="Unaligned transitions")
    parser.add_argument(
        "--aligned-transitions-dir", required=True, help="The same transitions, aligned"
    )
    parser.add_argument(
        "--alignment-mask-dir", required=True, help="Per-residue alignment masks"
    )
    parser.add_argument("--checkpoint", required=True, help="PEINT checkpoint to evaluate")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Written to as {output-dir}/transitions_log_likelihood[_per_site]/{family}.txt",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Families to score (default: every family present in all three directories)",
    )
    parser.add_argument(
        "--num-families", type=int, default=None, help="Score only the first N families"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--no-flash", action="store_true", help="Do not use Flash Attention"
    )
    parser.add_argument(
        "--no-amino-acid-restriction",
        action="store_true",
        help="Score over the full vocabulary instead of renormalizing over the 20 amino acids",
    )
    args = parser.parse_args()

    families = args.families or list_families(
        args.transitions_dir, args.aligned_transitions_dir, args.alignment_mask_dir
    )
    if args.num_families is not None:
        families = families[: args.num_families]
    if not families:
        raise SystemExit("No families found in all three directories.")

    logger.info("Scoring %d families with %s", len(families), args.checkpoint)

    evaluate_peint_model_transitions_log_likelihood__cached(
        transitions_dir=args.transitions_dir,
        aligned_transitions_dir=args.aligned_transitions_dir,
        alignment_mask_dir=args.alignment_mask_dir,
        model_checkpoint_path=args.checkpoint,
        families=families,
        device=args.device,
        batch_size=args.batch_size,
        use_flash=not args.no_flash,
        restrict_to_amino_acids=not args.no_amino_acid_restriction,
        max_length=args.max_length,
        output_transitions_log_likelihood_dir=os.path.join(
            args.output_dir, "transitions_log_likelihood"
        ),
        output_transitions_log_likelihood_per_site_dir=os.path.join(
            args.output_dir, "transitions_log_likelihood_per_site"
        ),
    )

    logger.info("Wrote likelihoods to %s", args.output_dir)


if __name__ == "__main__":
    main()
