"""Unified CLI for homology detection.

This module provides a command-line interface for running homology searches
using either PEINT or DIAMOND methods.

Usage:
    # Search mode (queries against database)
    python -m peint.homology_detection search \\
        --method peint --checkpoint model.ckpt --time 1.0 \\
        --database db.fasta --queries queries.fasta --output results.csv

    python -m peint.homology_detection search \\
        --method diamond --evalue 0.001 --threads 8 \\
        --database db.fasta --queries queries.fasta --output results.csv

    # All-vs-all (single FASTA)
    python -m peint.homology_detection all-vs-all \\
        --method peint --checkpoint model.ckpt --default-time 1.0 \\
        --fasta sequences.fasta --output results.csv

    # All-vs-all proteomes (directory of FASTAs)
    python -m peint.homology_detection all-vs-all \\
        --method peint --checkpoint model.ckpt \\
        --proteome-dir /path/to/proteomes --distance-matrix distances.csv \\
        --output results.csv
"""

import argparse
import logging
import sys

from peint.homology_detection._base import (
    load_distance_matrix,
    load_fasta,
    load_proteomes,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--method",
        choices=["peint", "diamond"],
        required=True,
        help="Homology search method",
    )
    parser.add_argument("--output", required=True, help="Output file path (.csv or .json)")
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default=None,
        help="Output format (default: inferred from extension)",
    )


def add_peint_args(parser: argparse.ArgumentParser) -> None:
    """Add PEINT-specific arguments."""
    parser.add_argument(
        "--checkpoint", help="Path to PEINT model checkpoint (required for PEINT)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Evolutionary time for comparisons (default: 1.0)",
    )
    parser.add_argument(
        "--default-time",
        type=float,
        default=1.0,
        help="Default evolutionary time when distance matrix entry missing (default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for decoder evaluation (default: 32)",
    )
    parser.add_argument(
        "--no-flash",
        action="store_true",
        help="Disable Flash Attention",
    )


def add_diamond_args(parser: argparse.ArgumentParser) -> None:
    """Add DIAMOND-specific arguments."""
    parser.add_argument(
        "--evalue",
        type=float,
        default=0.001,
        help="E-value threshold (default: 0.001)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads (default: 4)",
    )
    parser.add_argument(
        "--sensitivity",
        choices=[
            "fast",
            "mid-sensitive",
            "sensitive",
            "more-sensitive",
            "very-sensitive",
            "ultra-sensitive",
        ],
        default="more-sensitive",
        help="DIAMOND sensitivity mode (default: more-sensitive)",
    )


def create_searcher(args):
    """Create the appropriate homology searcher based on args."""
    if args.method == "peint":
        if not args.checkpoint:
            logger.error("--checkpoint is required for PEINT method")
            sys.exit(1)

        from peint.homology_detection._peint import (
            PeintHomologySearcher,
            PeintSearchConfig,
        )

        config = PeintSearchConfig(
            checkpoint_path=args.checkpoint,
            time=getattr(args, "time", None) or getattr(args, "default_time", 1.0),
            batch_size=args.batch_size,
            use_flash=not args.no_flash,
        )
        return PeintHomologySearcher(config)
    else:
        from peint.homology_detection._diamond import (
            DiamondHomologySearcher,
            DiamondSearchConfig,
        )

        config = DiamondSearchConfig(
            evalue=args.evalue,
            threads=args.threads,
            sensitivity=args.sensitivity,
        )
        return DiamondHomologySearcher(config)


def get_output_format(args) -> str:
    """Determine output format from args or filename."""
    if args.output_format:
        return args.output_format
    return "json" if args.output.endswith(".json") else "csv"


def cmd_search(args):
    """Handle the search subcommand."""
    searcher = create_searcher(args)

    database = load_fasta(args.database)
    queries = load_fasta(args.queries)
    logger.info(f"Loaded {len(database)} database and {len(queries)} query sequences")

    hits = searcher.search(database=database, queries=queries, top_k=args.top_k)
    logger.info(f"Found {len(hits)} hits")

    searcher.write_results(hits, args.output, format=get_output_format(args))


def cmd_all_vs_all(args):
    """Handle the all-vs-all subcommand."""
    searcher = create_searcher(args)

    if args.fasta:
        # Single FASTA mode
        sequences = load_fasta(args.fasta)
        logger.info(f"Loaded {len(sequences)} sequences")

        hits = searcher.all_vs_all(sequences)
        logger.info(f"Found {len(hits)} hits")

    elif args.proteome_dir:
        # Proteome directory mode
        proteomes = load_proteomes(args.proteome_dir)
        logger.info(f"Loaded {len(proteomes)} proteomes")

        distances = None
        if args.distance_matrix:
            default_time = getattr(args, "default_time", 1.0)
            distances = load_distance_matrix(args.distance_matrix, default=default_time)
            logger.info(f"Loaded distance matrix with {len(distances)} pairs")

        # PEINT supports additional options
        if args.method == "peint":
            hits = searcher.all_vs_all_proteomes(
                proteomes=proteomes,
                distances=distances,
                skip_same_proteome=not args.include_same_proteome,
            )
        else:
            hits = searcher.all_vs_all_proteomes(proteomes=proteomes, distances=distances)

        logger.info(f"Found {len(hits)} hits")

    else:
        logger.error("Either --fasta or --proteome-dir is required")
        sys.exit(1)

    searcher.write_results(hits, args.output, format=get_output_format(args))


def main():
    parser = argparse.ArgumentParser(
        description="Unified homology detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search subcommand
    search_parser = subparsers.add_parser(
        "search", help="Search query sequences against a database"
    )
    add_common_args(search_parser)
    add_peint_args(search_parser)
    add_diamond_args(search_parser)
    search_parser.add_argument(
        "--database", required=True, help="FASTA file with database sequences"
    )
    search_parser.add_argument(
        "--queries", required=True, help="FASTA file with query sequences"
    )
    search_parser.add_argument(
        "--top-k", type=int, help="Return only top K matches per query"
    )

    # All-vs-all subcommand
    all_vs_all_parser = subparsers.add_parser(
        "all-vs-all", help="Compare all sequences against each other"
    )
    add_common_args(all_vs_all_parser)
    add_peint_args(all_vs_all_parser)
    add_diamond_args(all_vs_all_parser)

    input_group = all_vs_all_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fasta", help="Single FASTA file with all sequences")
    input_group.add_argument(
        "--proteome-dir", help="Directory containing .fasta files (one per proteome)"
    )

    all_vs_all_parser.add_argument(
        "--distance-matrix", help="CSV/TSV file with pairwise proteome distances"
    )
    all_vs_all_parser.add_argument(
        "--include-same-proteome",
        action="store_true",
        help="Include comparisons within same proteome (PEINT only)",
    )

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "all-vs-all":
        cmd_all_vs_all(args)


if __name__ == "__main__":
    main()
