import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from peint.vep._vep_utils import PROTEINGYM_DIR, VEP_DATA_DIR, load_model
from peint.vep._scoring import score_transition_pairs


def _dms_subs_folder():
    """Folder of per-assay DMS substitution CSVs.

    The vendored ProteinGym3 copy can be permission-restricted (its inner
    ``DMS_ProteinGym_substitutions`` symlinks into another user's private dir);
    set ``VEP_DMS_SUBS_DIR`` to a readable mirror to override.
    """
    env = os.environ.get("VEP_DMS_SUBS_DIR")
    if env:
        return Path(env)
    return (
        PROTEINGYM_DIR
        / "input_data"
        / "ProteinGym_v1.3"
        / "DMS_ProteinGym_substitutions"
        / "DMS_ProteinGym_substitutions"
    )


def evaluate_vep_for_family(
    family,
    data_path,
    out_path,
    model,
    device,
    vocab,
    batch_size=32,
    t=1.0,
):
    """Evaluate VEP for a specific protein family."""
    output_path = os.path.join(out_path, f"{family}_preds.txt")
    if os.path.exists(output_path):
        print(f"{family} output exists, skipping")
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

    t_value = t

    scores = score_transition_pairs(
        model, vocab, device, pairs, t_value, batch_size=batch_size, progress=True
    )

    with open(output_path, "w") as fout:
        for p in scores:
            fout.write(f"{p}\n")

    return 0


def create_wag_corrected_score_files(
    lls_dir: str,
    wag_predictions_dir: str,
    output_scores_folder: str,
    t_wag: float,
    DMS_indices: list = None,
    clinical: bool = False,
):
    """
    Create WAG-corrected score files by subtracting WAG log likelihoods from Peint scores.

    Args:
        lls_dir: Directory containing the Peint log likelihood files.
        wag_predictions_dir: Directory containing the WAG predictions.
        output_scores_folder: Directory to save the output WAG-corrected score files.
        t_wag: Time parameter used for WAG predictions.
        DMS_indices: List of DMS indices to process.
        clinical: Whether to use clinical data instead of DMS data.
    """
    DMS_reference_file_path = (
        PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"
    )
    DMS_data_folder = _dms_subs_folder()

    # Make clinical replacements if needed
    if clinical:
        DMS_reference_file_path = os.path.abspath(DMS_reference_file_path)
        DMS_data_folder = os.path.abspath(DMS_data_folder)
    else:
        DMS_reference_file_path = os.path.abspath(DMS_reference_file_path)
        DMS_data_folder = os.path.abspath(DMS_data_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_scores_folder, exist_ok=True)

    # Set default DMS_indices if not provided
    DMS_indices = list(range(217))

    # Process each DMS index
    for DMS_index in tqdm(DMS_indices):
        # Load the mapping file
        mapping_protein_seq_DMS = pd.read_csv(DMS_reference_file_path)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id = list_DMS[DMS_index]
        print(
            f"Creating WAG-corrected score file for: {DMS_id} ({DMS_index + 1} / {len(DMS_indices)})"
        )

        # Get file names and target sequence
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][
            mapping_protein_seq_DMS["DMS_id"] == DMS_id
        ].values[0]
        target_seq = (
            mapping_protein_seq_DMS["target_seq"][
                mapping_protein_seq_DMS["DMS_id"] == DMS_id
            ]
            .values[0]
            .upper()
        )

        # Load DMS data
        DMS_data_path = os.path.join(DMS_data_folder, DMS_file_name)
        DMS_data = pd.read_csv(DMS_data_path, low_memory=False)

        # Read Peint log likelihoods
        family = DMS_id
        peint_score_fpath = os.path.join(lls_dir, family + "_preds.txt")
        if not os.path.exists(peint_score_fpath):
            print(f"Peint score does not exist for {DMS_id}, skipping")
            continue
        with open(peint_score_fpath, "r") as fin:
            peint_scores = [float(l.rstrip("\n")) for l in fin]

        # Read WAG log likelihoods
        t_wag_str = str(t_wag).replace(".", "_")
        wag_score_fpath = os.path.join(
            wag_predictions_dir, "t_" + t_wag_str, family + "_wag_preds.txt"
        )
        if not os.path.exists(wag_score_fpath):
            print(f"WAG score does not exist for {DMS_id}, skipping")
            continue
        with open(wag_score_fpath, "r") as fin:
            wag_scores = [float(l.rstrip("\n")) for l in fin]

        # Check that scores have same length
        if len(peint_scores) != len(wag_scores):
            print(
                f"Score length mismatch for {DMS_id}: Peint={len(peint_scores)}, WAG={len(wag_scores)}, skipping"
            )
            continue

        # Compute WAG-corrected scores
        wag_corrected_scores = [
            peint - wag for peint, wag in zip(peint_scores, wag_scores)
        ]
        DMS_data["wag_corrected_score"] = wag_corrected_scores

        # Save to file
        scoring_filename = os.path.join(output_scores_folder, DMS_id + ".csv")
        if clinical:
            DMS_data[["mutant", "wag_corrected_score", "DMS_bin_score"]].to_csv(
                scoring_filename, index=False
            )
        else:
            DMS_data[["mutant", "wag_corrected_score", "DMS_score"]].to_csv(
                scoring_filename, index=False
            )


def create_peint_score_files(
    lls_dir: str,
    output_scores_folder: str,
    DMS_indices: list = None,
    clinical: bool = False,
    indels: bool = False,
    overwrite: bool = False,
):
    """
    Create score files for the peint model using log likelihoods.

    Args:
        lls_dir: Directory containing the log likelihood files.
        output_scores_folder: Directory to save the output score files.
        DMS_indices: List of DMS indices to process.
        clinical: Whether to use clinical data instead of DMS data.
        indels: Whether to use indels data instead of substitutions data.
        overwrite: Whether to overwrite existing score files.
    """
    if indels:
        # For indels, we don't need a reference file and use a different data folder
        DMS_reference_file_path = None
        DMS_data_folder = (
            VEP_DATA_DIR / "proteingym_indels_data" / "DMS_ProteinGym_indels"
        )
        DMS_data_folder = os.path.abspath(DMS_data_folder)
    else:
        DMS_reference_file_path = (
            PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"
        )
        DMS_data_folder = _dms_subs_folder()

        # Make clinical replacements if needed
        if clinical:
            DMS_reference_file_path = os.path.abspath(DMS_reference_file_path)
            DMS_data_folder = os.path.abspath(DMS_data_folder)
        else:
            DMS_reference_file_path = os.path.abspath(DMS_reference_file_path)
            DMS_data_folder = os.path.abspath(DMS_data_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_scores_folder, exist_ok=True)

    # Set default DMS_indices if not provided
    if indels:
        # For indels, get the number of indels files available
        indels_files = [
            f for f in os.listdir(DMS_data_folder) if f.endswith("_indels.csv")
        ]
        DMS_indices = list(range(len(indels_files)))
    else:
        # Default for substitutions
        DMS_indices = list(range(217))

    # Process each DMS index
    for DMS_index in tqdm(DMS_indices):
        if indels:
            # For indels, get family names directly from the files in the directory
            if DMS_index >= len(indels_files):
                continue
            DMS_file_name = indels_files[DMS_index]
            DMS_id = DMS_file_name.replace("_indels.csv", "")
            print(
                f"Creating score file for: {DMS_id} ({DMS_index + 1} / {len(indels_files)}) with peint model"
            )
        else:
            # Load the mapping file
            mapping_protein_seq_DMS = pd.read_csv(DMS_reference_file_path)
            list_DMS = mapping_protein_seq_DMS["DMS_id"]
            DMS_id = list_DMS[DMS_index]
            print(
                f"Creating score file for: {DMS_id} ({DMS_index + 1} / {len(DMS_indices)}) with peint model"
            )

            # Get file names and target sequence
            DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][
                mapping_protein_seq_DMS["DMS_id"] == DMS_id
            ].values[0]
            target_seq = (
                mapping_protein_seq_DMS["target_seq"][
                    mapping_protein_seq_DMS["DMS_id"] == DMS_id
                ]
                .values[0]
                .upper()
            )

        # Get family name
        family = DMS_id
        scoring_filename = os.path.join(output_scores_folder, DMS_id + ".csv")
        if os.path.exists(scoring_filename) and not overwrite:
            print(f"Score file for {DMS_id} already exists, skipping")
            continue

        # Load DMS data
        DMS_data_path = os.path.join(DMS_data_folder, DMS_file_name)
        DMS_data = pd.read_csv(DMS_data_path, low_memory=False)

        # Read log likelihoods
        score_fpath = os.path.join(lls_dir, family + "_preds.txt")
        if not os.path.exists(score_fpath):
            print(f"Score does not exists for {DMS_id}, skipping")
            continue
        with open(score_fpath, "r") as fin:
            lls = [float(l.rstrip("\n")) for l in fin]

        # Assign scores
        model_scores = lls
        DMS_data["peint_score"] = model_scores

        if indels:
            # For indels, use mutated_sequence column instead of mutant
            DMS_data[["mutated_sequence", "peint_score", "DMS_score"]].to_csv(
                scoring_filename, index=False
            )
        elif clinical:
            DMS_data[["mutant", "peint_score", "DMS_bin_score"]].to_csv(
                scoring_filename, index=False
            )
        else:
            DMS_data[["mutant", "peint_score", "DMS_score"]].to_csv(
                scoring_filename, index=False
            )


def compute_wag_corrected_spearman(output_scores_folder, df_dms=None):
    """
    Compute Spearman correlation for each assay using WAG-corrected scores.

    Args:
        output_scores_folder: Directory containing WAG-corrected score files.
        df_dms: DataFrame containing DMS information.

    Returns:
        DataFrame with family, assay_type, and spearman correlation.
    """
    df_results = []
    for file in tqdm(os.listdir(output_scores_folder)):
        family = file.split(".")[0]
        df_scores = pd.read_csv(os.path.join(output_scores_folder, file))
        spearman = scipy.stats.spearmanr(
            df_scores["DMS_score"], df_scores["wag_corrected_score"]
        )[0]

        assay_type = df_dms[(df_dms["DMS_id"] == family)].iloc[0][
            "coarse_selection_type"
        ]
        df_results.append((family, assay_type, spearman))
    return pd.DataFrame(df_results, columns=["family", "assay_type", "spearman"])


def compute_per_assay_spearman(output_scores_folder, df_dms=None, indels=False):
    """
    Compute Spearman correlation for each assay.

    Args:
        output_scores_folder: Directory containing score files.
        df_dms: DataFrame containing DMS information (not used for indels).
        indels: Whether to process indels data (no assay types available).

    Returns:
        DataFrame with family, assay_type (if available), and spearman correlation.
    """
    df_results = []
    for file in tqdm(os.listdir(output_scores_folder)):
        family = file.split(".")[0]
        df_scores = pd.read_csv(os.path.join(output_scores_folder, file))
        spearman = scipy.stats.spearmanr(
            df_scores["DMS_score"], df_scores["peint_score"]
        )[0]

        if indels:
            # For indels, no assay type information is available
            df_results.append((family, "N/A", spearman))
        else:
            assay_type = df_dms[(df_dms["DMS_id"] == family)].iloc[0][
                "coarse_selection_type"
            ]
            df_results.append((family, assay_type, spearman))
    return pd.DataFrame(df_results, columns=["family", "assay_type", "spearman"])


def compute_spearman_by_mutation_depth(output_scores_folder, df_dms):
    """
    Compute Spearman correlation stratified by mutational depth for each family.
    Mutations with depth >= 5 are grouped together as "5+".

    Args:
        output_scores_folder: Directory containing score files.
        df_dms: DataFrame containing DMS information.

    Returns:
        DataFrame with columns: [mutational_depth, family, assay_type, count, spearman]
    """
    if df_dms is None:
        raise ValueError("df_dms is required")

    def get_mutation_depth(mutant):
        """Parse mutant string to get number of mutations, capping at 5+."""
        if pd.isna(mutant):
            return 0
        # Mutations are separated by colons, e.g., "S35Y:H37E" has 2 mutations
        depth = len(mutant.split(":"))
        # Group mutations with depth >= 5 into "5+"
        return "5+" if depth >= 5 else str(depth)

    # Compute per-family correlation by mutation depth
    results = []
    for file in tqdm(os.listdir(output_scores_folder)):
        family = file.split(".")[0]
        df_scores = pd.read_csv(os.path.join(output_scores_folder, file))

        # Get assay type
        assay_type = df_dms[(df_dms["DMS_id"] == family)].iloc[0][
            "coarse_selection_type"
        ]

        # Add mutation depth column
        df_scores["mutation_depth"] = df_scores["mutant"].apply(get_mutation_depth)

        # Group by mutation depth and compute correlation
        for depth, group in df_scores.groupby("mutation_depth"):
            if len(group) < 2:
                # Skip if not enough samples to compute correlation
                continue
            spearman = scipy.stats.spearmanr(group["DMS_score"], group["peint_score"])[
                0
            ]
            results.append((depth, family, assay_type, len(group), spearman))

    return pd.DataFrame(
        results,
        columns=["mutational_depth", "family", "assay_type", "count", "spearman"],
    )


def compute_esm_spearman(esm_models=["ESM2_150M", "ESM2_650M", "ESM2_3B", "ESM2_15B"]):
    output_scores_folder = (
        PROTEINGYM_DIR / "input_data/ProteinGym_v1.3/zero_shot_substitutions_scores"
    )
    df_dms = pd.read_csv(PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv")
    transition_dir = (
        PROTEINGYM_DIR
        / "_cache_cherryml/create_test_transition_pairs/3a13efc22507796bfec09150d959917b6e034c3c29639e25c0cadf0f02921d4c/output_transition_pairs_dir/"
    )
    families = [
        f.split(".")[0] for f in os.listdir(transition_dir) if f.endswith(".txt")
    ]
    df_results_esm = defaultdict(list)
    df_depth_esm = defaultdict(list)

    def get_mutation_depth(mutant):
        """Parse mutant string to get number of mutations, capping at 5+."""
        if pd.isna(mutant):
            return 0
        # Mutations are separated by colons, e.g., "S35Y:H37E" has 2 mutations
        depth = len(mutant.split(":"))
        # Group mutations with depth >= 5 into "5+"
        return "5+" if depth >= 5 else str(depth)

    for family in tqdm(families):
        score_fpath = output_scores_folder / f"{family}.csv"
        if not os.path.exists(score_fpath):
            continue
        df_scores = pd.read_csv(score_fpath)

        # Get assay type
        assay_type = df_dms[(df_dms["DMS_id"] == family)].iloc[0][
            "coarse_selection_type"
        ]

        # Add mutation depth column
        df_scores["mutation_depth"] = df_scores["mutant"].apply(get_mutation_depth)

        for model in esm_models:
            # Overall spearman
            spearman = scipy.stats.spearmanr(df_scores["DMS_score"], df_scores[model])[
                0
            ]
            df_results_esm[model].append((family, assay_type, spearman))

            # Per mutation depth spearman
            for depth, group in df_scores.groupby("mutation_depth"):
                if len(group) < 2:
                    # Skip if not enough samples to compute correlation
                    continue
                spearman_depth = scipy.stats.spearmanr(
                    group["DMS_score"], group[model]
                )[0]
                df_depth_esm[model].append(
                    (depth, family, assay_type, len(group), spearman_depth)
                )

    for model, df in df_results_esm.items():
        df = pd.DataFrame(df, columns=["family", "assay_type", "spearman"])
        output_dir = Path(VEP_DATA_DIR / f"test_lls/{model}/")
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "spearman_results.csv", index=False)

        # Save mutation depth results
        df_depth = pd.DataFrame(
            df_depth_esm[model],
            columns=["mutational_depth", "family", "assay_type", "count", "spearman"],
        )
        df_depth.to_csv(output_dir / "spearman_by_mutation_depth.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Running VEP with the Transformer"
    )

    parser.add_argument("--checkpoints", type=str, nargs="+", help="checkpoint names")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to save results to"
    )
    parser.add_argument("--times", type=str, nargs="+", help="times to evaluate")
    parser.add_argument("--data_dir", type=str, help="Path to the DMS transitions")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--family_subset",
        type=str,
        nargs="?",
        default=None,
        help="Path to an optional set of families",
    )
    parser.add_argument(
        "--indels",
        action="store_true",
        help="Evaluate on indels data instead of substitutions",
    )
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="Skip model loading and evaluation; rebuild score files and "
        "per-assay Spearman from existing *_preds.txt (recovery / rescore).",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Suffix to add to the output directory",
    )
    parser.add_argument(
        "--wag_predictions_dir",
        type=str,
        default=None,
        help="Path to WAG predictions directory (optional, enables WAG correction)",
    )
    parser.add_argument(
        "--t_wag",
        type=float,
        default=None,
        help="Time parameter for WAG model (defaults to same as t)",
    )

    args = parser.parse_args()

    # Convert to Path objects for easier path manipulation
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # Load the DMS information for per-assay Spearman calculation (not needed for indels)
    df_dms = None
    if not args.indels:
        df_dms = pd.read_csv(
            PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"
        )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load family names
    if args.indels:
        # For indels, get family names from the indels CSV files
        indels_data_folder = (
            VEP_DATA_DIR / "proteingym_indels_data" / "DMS_ProteinGym_indels"
        )
        families = []
        for f in os.listdir(indels_data_folder):
            if f.endswith("_indels.csv"):
                family_name = f.replace("_indels.csv", "")
                families.append(family_name)
        print(f"Found {len(families)} indels families")
    elif args.family_subset:
        families = []
        with open(args.family_subset, "r") as fin:
            for l in fin:
                families.append(l.rstrip("\n"))
    else:
        families = [f.split(".")[0] for f in os.listdir(data_dir) if f.endswith(".txt")]

    # Process each checkpoint
    checkpoints = args.checkpoints
    # Sweep over time
    if args.times is None:
        times = [1.0]
    else:
        times = sorted([float(t) for t in args.times])

    # Set t_wag default
    if args.t_wag is None:
        t_wag = times[0]  # Default to same as t
    else:
        t_wag = args.t_wag

    for i, ckpt_path in enumerate(checkpoints):
        # Extract run name and checkpoint name from path
        ckpt_path = Path(ckpt_path)
        run_name = ckpt_path.parent.name
        ckpt_name = ckpt_path.name.replace(".ckpt", "")

        # Set t value
        for t in times:
            if t < 0.05:
                t = 0.05
            t_str = f"t_{str(t).replace('.', '_')}"

            # Create flattened output directory structure
            output_subdir = output_dir / f"{run_name}-{ckpt_name}-{t_str}"
            if args.output_suffix:
                output_subdir = (
                    output_dir / f"{run_name}-{ckpt_name}-{t_str}-{args.output_suffix}"
                )
            os.makedirs(output_subdir, exist_ok=True)

            print(f"Processing checkpoint: {ckpt_path}")
            print(f"Output directory: {output_subdir}")

            if args.score_only:
                print(
                    "score_only: skipping model load + evaluation; rebuilding "
                    "score files and Spearman from existing *_preds.txt"
                )
            else:
                # Load model (FlashAttention on GPU, standard PyTorch encoder on CPU)
                model, vocab = load_model(
                    model_checkpoint_path=str(ckpt_path),
                    device=device,
                    use_flash=(device.type == "cuda"),
                )

                # Evaluate each family
                for j, family in enumerate(families):
                    print(
                        f"Evaluating model on: {family} ({j + 1} / {len(families)}) with peint model"
                    )
                    retcode = evaluate_vep_for_family(
                        family,
                        str(data_dir),
                        str(output_subdir),
                        model,
                        device,
                        vocab,
                        batch_size=args.batch_size,
                        t=t,
                    )

            # Create scores directory and generate score files
            output_scores_folder = output_subdir / "scores"
            os.makedirs(output_scores_folder, exist_ok=True)

            create_peint_score_files(
                lls_dir=str(output_subdir),
                output_scores_folder=str(output_scores_folder),
                indels=args.indels,
            )

            # Compute per-assay Spearman correlations
            df_results = compute_per_assay_spearman(
                output_scores_folder, df_dms, indels=args.indels
            )

            # Save results
            df_results.to_csv(output_subdir / "spearman_results.csv", index=False)

            # Compute Spearman correlations by mutation depth (only for substitutions)
            if not args.indels:
                print("\nComputing Spearman correlation by mutation depth...")
                df_depth = compute_spearman_by_mutation_depth(
                    output_scores_folder, df_dms=df_dms
                )
                df_depth.to_csv(
                    output_subdir / "spearman_by_mutation_depth.csv",
                    index=False,
                )

            # Print summary statistics
            print(f"\nResults for {run_name}_{ckpt_name}_{t_str}:")
            mean_spearman = df_results["spearman"].mean()
            print(f"Mean Spearman correlation: {mean_spearman:.4f}")
            if not args.indels:
                print("\nSpearman correlation by assay type:")
                print(df_results.groupby("assay_type")["spearman"].mean())
                print("\nSpearman correlation by mutation depth:")
                depth_summary = (
                    df_depth.groupby("mutational_depth")
                    .agg({"count": "sum", "spearman": "mean"})
                    .sort_index()
                )
                print(depth_summary)
            else:
                print("Indels evaluation completed (no assay type breakdown available)")

            # WAG correction if requested
            if args.wag_predictions_dir:
                if args.indels:
                    raise ValueError(
                        "WAG correction is not supported for indels data. "
                        "Please run without --indels flag or without --wag_predictions_dir."
                    )

                print(f"\nComputing WAG-corrected scores with t_wag={t_wag}")
                t_wag_str = str(t_wag).replace(".", "_")

                # Create WAG-corrected scores directory
                wag_corrected_scores_folder = (
                    output_subdir / f"wag_corrected_scores_t_{t_wag_str}"
                )
                os.makedirs(wag_corrected_scores_folder, exist_ok=True)

                create_wag_corrected_score_files(
                    lls_dir=str(output_subdir),
                    wag_predictions_dir=args.wag_predictions_dir,
                    output_scores_folder=str(wag_corrected_scores_folder),
                    t_wag=t_wag,
                )

                # Compute WAG-corrected Spearman correlations
                df_wag_results = compute_wag_corrected_spearman(
                    str(wag_corrected_scores_folder), df_dms
                )

                # Save WAG-corrected results
                wag_results_path = (
                    output_subdir / f"wag_corrected_spearman_results_t_{t_wag_str}.csv"
                )
                df_wag_results.to_csv(wag_results_path, index=False)

                # Print WAG-corrected summary statistics
                print(
                    f"\nWAG-corrected results for {run_name}_{ckpt_name}_{t_str} (t_wag={t_wag}):"
                )
                mean_wag_spearman = df_wag_results["spearman"].mean()
                print(
                    f"Mean WAG-corrected Spearman correlation: {mean_wag_spearman:.4f}"
                )
                print("WAG-corrected Spearman correlation by assay type:")
                print(df_wag_results.groupby("assay_type")["spearman"].mean())
