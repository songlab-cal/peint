import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from peint.models._wag import evaluate_wag_model_transitions_log_likelihood_per_site

# The WAG rate matrix ships with this repository; resolve it relative to this file so the
# default works from any working directory.
DEFAULT_RATE_MATRIX = Path(__file__).resolve().parents[2] / "data" / "rate_matrices" / "wag.txt"


def load_wag_rate_matrix(rate_matrix_path: str) -> pd.DataFrame:
    """Load WAG rate matrix from file."""
    return pd.read_csv(rate_matrix_path, sep="\t", index_col=0)


def evaluate_wag_for_family(
    family: str,
    data_path: str,
    out_path: str,
    rate_matrix: pd.DataFrame,
    t_wag: float = 1.0,
    force_recompute: bool = False,
):
    """Evaluate WAG model for a specific protein family."""
    output_path = os.path.join(out_path, f"{family}_wag_preds.txt")
    if os.path.exists(output_path) and not force_recompute:
        print(f"{family} WAG output exists, skipping")
        return 0

    family_data_file = os.path.join(data_path, f"{family}.txt")
    if not os.path.exists(family_data_file):
        print(f"Transitions for {family} does not exist, skipping")
        return 1

    pairs = []
    with open(family_data_file, "r") as fin:
        for l in fin:
            pairs.append(l.strip().split())
    if len(pairs[0][0]) > 1022:
        print(f"{family} is too long, skipping")
        return 1

    # Scale transitions by t_wag
    transitions = [(x, y, t_wag) for (x, y) in pairs]

    # Compute WAG mean log-likelihoods per site
    wag_scores_per_site = evaluate_wag_model_transitions_log_likelihood_per_site(
        transitions=transitions,
        rate_matrix=rate_matrix,
    )
    wag_scores = [sum(x) / len(x) for x in wag_scores_per_site]

    # Save scores
    with open(output_path, "w") as fout:
        for score in wag_scores:
            fout.write(f"{score}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compute WAG scores for protein families"
    )

    parser.add_argument("--data_dir", type=str, help="Path to the transitions data")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to save WAG predictions"
    )
    parser.add_argument(
        "--t_wag", type=float, default=1.0, help="Time parameter for WAG model"
    )
    parser.add_argument(
        "--family_subset",
        type=str,
        default=None,
        help="Path to optional set of families",
    )
    parser.add_argument(
        "--rate_matrix_path",
        type=str,
        default=str(DEFAULT_RATE_MATRIX),
        help="Path to WAG rate matrix (defaults to the copy shipped in this repository)",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recompute WAG scores",
    )

    args = parser.parse_args()

    # Convert to Path objects
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # Create WAG predictions directory structure
    wag_predictions_dir = output_dir / f"t_{str(args.t_wag).replace('.', '_')}"
    os.makedirs(wag_predictions_dir, exist_ok=True)

    # Load WAG rate matrix
    print(f"Loading WAG rate matrix from {args.rate_matrix_path}")
    rate_matrix = load_wag_rate_matrix(args.rate_matrix_path)

    # Load family names
    if args.family_subset:
        families = []
        with open(args.family_subset, "r") as fin:
            for line in fin:
                families.append(line.strip())
    else:
        families = [f.split(".")[0] for f in os.listdir(data_dir) if f.endswith(".txt")]

    print(f"Processing {len(families)} families with t_wag={args.t_wag}")
    print(f"WAG predictions will be saved to: {wag_predictions_dir}")

    # Process each family
    for i, family in enumerate(tqdm(families, desc="Computing WAG scores")):
        print(f"Processing family: {family} ({i + 1} / {len(families)})")
        retcode = evaluate_wag_for_family(
            family=family,
            data_path=str(data_dir),
            out_path=str(wag_predictions_dir),
            rate_matrix=rate_matrix,
            t_wag=args.t_wag,
            force_recompute=args.force_recompute,
        )

        if retcode != 0:
            print(f"Failed to process family {family}")

    print(f"WAG score computation completed. Results saved to: {wag_predictions_dir}")


if __name__ == "__main__":
    main()
