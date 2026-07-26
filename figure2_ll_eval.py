"""Figure 2: per-site likelihood of held-out transitions, PEINT vs classical models.

Every model is scored on the same transitions and reported as a mean per-site
likelihood in bins of evolutionary time. The baselines (uniform random guess,
WAG, LG) read aligned transitions; PEINT reads the unaligned ones and its
per-residue likelihoods are projected back onto the alignment, so the two are
comparable column by column.

The dataset directories were produced by ``pfam_15k__treewise_train_test_split``
elsewhere and copied into ``local_data``; this script reads them directly rather
than rebuilding them, which would need the raw a3m files. Everything this script
computes itself is cached under ``_cache_peint``, so a re-run only does the work
that is missing.

Usage:
    python figure2_ll_eval.py           # the full evaluation
    python figure2_ll_eval.py --num-test-families 20 --num-train-families 200
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cherryml import caching as cherryml_caching
from tqdm import tqdm

from protevo import caching as protevo_caching
from protevo import models
from protevo.evaluation import evaluate_peint_model_transitions_log_likelihood__cached
from protevo.io import read_transitions, read_transitions_log_likelihood_per_site
from protevo.utils import (
    get_quantile_idx,
    get_quantization_points_from_geometric_grid,
)

# Dataset directories, copied over from where the split was originally built.
ALIGNED_TRAIN_TRANSITIONS_DIR = "local_data/aligned/train_transitions_dir"
ALIGNED_TRAIN_SITE_RATES_4CAT_DIR = (
    "local_data/aligned/train_site_rates_4cat_dir/output_site_rates_dir"
)
ALIGNED_TEST_TRANSITIONS_DIR = "local_data/aligned/test_transitions_dir"
UNALIGNED_TEST_TRANSITIONS_DIR = (
    "local_data/unaligned/test_transitions_dir/output_transitions_dir"
)
UNALIGNED_TEST_ALIGNMENT_MASK_DIR = "local_data/unaligned/test_alignment_mask_dir"

PEINT_CHECKPOINT_FILE = "model_checkpoints/peint.ckpt"

# Families held out of training regardless of where the shuffle put them.
HELD_OUT_CAS = ["5e2r_1_A", "1ekj_1_C"]
NUM_TRAIN_FAMILIES = 14500
SPLIT_SEED = 42


def default_num_processes() -> int:
    """Physical cores this process may use, which is what Open MPI will allocate.

    CherryML's C++ transition counter runs under Open MPI, which counts slots in
    cores rather than hardware threads and aborts outright when asked for more
    than it has. ``os.sched_getaffinity`` reports threads, so halve it on an SMT
    machine.
    """
    threads = len(os.sched_getaffinity(0))
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as f:
            threads_per_core = len(f.read().strip().replace("-", ",").split(","))
    except OSError:
        threads_per_core = 1
    return max(1, threads // max(1, threads_per_core))


def build_family_split(transitions_dir: str):
    """Reproduce the train/test split the model was trained under.

    ``pfam_15k.get_families`` returns the sorted family names of the a3m
    directory, which are also the file stems here, so shuffling that list under
    the same seed recovers the original split without the raw a3m files.

    Returns:
        ``(families_train, families_test, train_held_out_subset)``, where the
        last is a random sample of training families the same size as the test
        set, for reading off how much of the gap is generalization.
    """
    all_families = sorted(
        f[: -len(".txt")] for f in os.listdir(transitions_dir) if f.endswith(".txt")
    )
    random.Random(SPLIT_SEED).shuffle(all_families)
    families_train = sorted(all_families[:NUM_TRAIN_FAMILIES])
    families_test = sorted(all_families[NUM_TRAIN_FAMILIES:])

    families_test = sorted(families_test + HELD_OUT_CAS)
    families_train = [f for f in families_train if f not in HELD_OUT_CAS]

    subset_idx = random.Random(SPLIT_SEED).sample(
        range(len(families_train)), len(families_test)
    )
    train_held_out_subset = [families_train[i] for i in subset_idx]

    # The trained checkpoint is named for its 14498 training families; if these
    # counts drift, the split no longer matches what the model was trained on.
    assert len(families_train) == 14498, len(families_train)
    assert len(families_test) == 553, len(families_test)
    assert not set(families_train) & set(families_test)

    return families_train, families_test, train_held_out_subset


def accumulate_by_time_bin(families, transitions_dir, per_site_dirs, quantization_points):
    """Total log-likelihood and site count per model, binned by evolutionary time.

    Transitions that any model left unscored are dropped for every model, so the
    comparison always runs over the same transitions.

    Args:
        families: Families to read.
        transitions_dir: Aligned transitions, read for their times.
        per_site_dirs: Model name -> directory of per-site log-likelihoods.
        quantization_points: Bin edges, as floats.

    Returns:
        ``(totals, counts)``, each a dict of model name -> array over bins.
    """
    totals = {name: np.zeros(len(quantization_points)) for name in per_site_dirs}
    counts = {name: np.zeros(len(quantization_points), dtype=int) for name in per_site_dirs}

    for family in tqdm(families):
        transitions = read_transitions(os.path.join(transitions_dir, family + ".txt"))
        per_site = {
            name: np.array(
                read_transitions_log_likelihood_per_site(
                    os.path.join(directory, family + ".txt")
                )
            )
            for name, directory in per_site_dirs.items()
        }

        # PEINT leaves transitions it could not score (too long) as NaN.
        scored = ~np.isnan(np.stack(list(per_site.values()))).any(axis=(0, 2))

        for i, (_, _, t) in enumerate(transitions):
            if not scored[i]:
                continue
            bin_idx = get_quantile_idx(quantization_points, t)
            for name, log_likelihoods in per_site.items():
                totals[name][bin_idx] += log_likelihoods[i].sum()
                counts[name][bin_idx] += log_likelihoods.shape[1]

    return totals, counts


def plot_mean_likelihood(totals, counts, quantization_points, output_path):
    """Mean per-site likelihood against evolutionary time, one line per model."""
    sns.set_theme(style="white")
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.minor.left"] = True
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams.update(
        {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    )

    fig, ax = plt.subplots(figsize=(3, 2))
    max_x = 0
    for name in totals:
        populated = counts[name] > 0
        xs = np.array(quantization_points)[populated]
        ys = np.exp(totals[name][populated] / counts[name][populated])
        if len(xs) > 0:
            max_x = max(max_x, xs.max())

        style = {"linewidth": 0.75, "legend": False}
        if name == "Random guess":
            style.update({"linestyle": "--", "color": "gray"})
        sns.lineplot(x=xs, y=ys, label=name, ax=ax, **style)

    ticks_base = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels_base = [f"{t:.1f}" if i in [0, 4] else "" for i, t in enumerate(ticks_base)]
    extra_ticks = list(range(2, int(max_x) + 1))

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.set_xlabel("Evolutionary time")
    ax.set_ylabel("Mean per-site likelihood")
    ax.set_xscale("log")
    ax.set_xlim(1e-1, max(max_x, 1.0))
    ax.set_xticks(
        ticks_base + extra_ticks,
        labels=labels_base + [str(t) for t in extra_ticks],
    )
    ax.legend(fontsize=7, loc="upper right", frameon=True, ncol=1)
    ax.grid(which="major", axis="y", linestyle=":", linewidth=0.5)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=PEINT_CHECKPOINT_FILE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=default_num_processes(),
        help="CherryML counts transitions with an Open MPI binary, which allocates "
             "one slot per physical core and refuses to start if asked for more, "
             "so this defaults to cores rather than hardware threads",
    )
    parser.add_argument(
        "--num-test-families",
        type=int,
        default=None,
        help="Evaluate only the first N test families, for a quick smoke test",
    )
    parser.add_argument(
        "--num-train-families",
        type=int,
        default=None,
        help="Train the classical models on only the first N families. Caches "
             "under its own key, so this never overwrites a full-split model.",
    )
    args = parser.parse_args()

    protevo_caching.set_cache_dir("_cache_peint")
    protevo_caching.set_read_only(False)
    protevo_caching.set_log_level(9)

    # CherryML keeps its own cache, and it is a separate caching system: giving
    # it its own directory keeps the two from writing over each other.
    cherryml_caching.set_cache_dir("_cache_cherryml")
    cherryml_caching.set_read_only(False)
    cherryml_caching.set_log_level(9)

    families_train, families_test, train_held_out_subset = build_family_split(
        ALIGNED_TEST_TRANSITIONS_DIR
    )
    if args.num_test_families is not None:
        families_test = families_test[: args.num_test_families]
        train_held_out_subset = train_held_out_subset[: args.num_test_families]
    if args.num_train_families is not None:
        families_train = families_train[: args.num_train_families]

    print(f"Families train: {len(families_train)}")
    print(f"Families test: {len(families_test)}")
    print(f"Families train held-out subset: {len(train_held_out_subset)}")

    ###### MODEL TRAINING ######
    # The uniform random guess model has nothing to train.
    print("Training WAG model ...")
    wag_model_output_dir = models.train_wag_model__cached(
        train_transitions_dir=ALIGNED_TRAIN_TRANSITIONS_DIR,
        families=families_train,
        num_processes=args.num_processes,
    )["output_model_dir"]

    print("Training LG (4RC) model ...")
    lg_4rc_model_output_dir = models.train_lg_model__cached(
        train_transitions_dir=ALIGNED_TRAIN_TRANSITIONS_DIR,
        train_site_rates_dir=ALIGNED_TRAIN_SITE_RATES_4CAT_DIR,
        families=families_train,
        num_processes=args.num_processes,
    )["output_model_dir"]

    # PEINT is already trained on the full-length sequences; injected below.

    ########### GET LIKELIHOODS ###########
    def per_site_dirs_for(families):
        """Score every model on the given families, returning their output dirs."""
        random_guess = (
            models.evaluate_uniform_random_guess_model_transitions_log_likelihood__cached(
                transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
                families=families,
            )
        )
        wag = models.evaluate_wag_model_transitions_log_likelihood__cached(
            transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
            families=families,
            model_dir=wag_model_output_dir,
            num_processes=args.num_processes,
            condition_on_non_gap=True,
        )
        lg_4rc = models.evaluate_lg_model_transitions_log_likelihood__cached(
            transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
            site_rates_dir=ALIGNED_TRAIN_SITE_RATES_4CAT_DIR,
            families=families,
            model_dir=lg_4rc_model_output_dir,
            num_processes=args.num_processes,
            condition_on_non_gap=True,
        )
        peint = evaluate_peint_model_transitions_log_likelihood__cached(
            transitions_dir=UNALIGNED_TEST_TRANSITIONS_DIR,
            aligned_transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
            alignment_mask_dir=UNALIGNED_TEST_ALIGNMENT_MASK_DIR,
            model_checkpoint_path=args.checkpoint,
            families=families,
            device=args.device,
            batch_size=args.batch_size,
        )
        return {
            name: result["output_transitions_log_likelihood_per_site_dir"]
            for name, result in [
                ("Random guess", random_guess),
                ("WAG", wag),
                ("LG (4 rate categories)", lg_4rc),
                ("PEINT", peint),
            ]
        }

    print("Getting test predictions ...")
    test_per_site_dirs = per_site_dirs_for(families_test)
    print("Getting train held-out subset predictions ...")
    train_held_out_per_site_dirs = per_site_dirs_for(train_held_out_subset)

    ########### ANALYZE AND PLOT ###########
    quantization_points = [
        float(q) for q in get_quantization_points_from_geometric_grid()
    ]

    print("Reading test transitions log likelihoods ...")
    test_totals, test_counts = accumulate_by_time_bin(
        families_test, ALIGNED_TEST_TRANSITIONS_DIR, test_per_site_dirs, quantization_points
    )
    print("Reading train held-out subset transitions log likelihoods ...")
    train_totals, train_counts = accumulate_by_time_bin(
        train_held_out_subset,
        ALIGNED_TEST_TRANSITIONS_DIR,
        train_held_out_per_site_dirs,
        quantization_points,
    )

    for name in test_totals:
        scored_sites = test_counts[name].sum()
        mean_ll = test_totals[name].sum() / scored_sites
        print(f"  {name:<24} mean per-site LL = {mean_ll:+.4f} over {scored_sites} sites")

    plot_mean_likelihood(
        test_totals, test_counts, quantization_points, "figure2_likelihood_eval_test.pdf"
    )
    plot_mean_likelihood(
        train_totals,
        train_counts,
        quantization_points,
        "figure2_likelihood_eval_train_held_out.pdf",
    )


if __name__ == "__main__":
    main()
