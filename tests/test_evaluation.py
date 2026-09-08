"""Tests for per-site likelihood evaluation against the alignment.

The fixture family ``4djg_1_B`` ships with the reference per-site
log-likelihoods produced by the previous version of this repository, so the
integration tests check the current interface against known-good numbers.
"""

import os

import esm
import numpy as np
import pytest

from peint.evaluation import (
    AlignedTransitionsDataset,
    evaluate_transitions_log_likelihood_per_site,
    list_families,
    sum_over_sites,
)
from peint.io import read_transitions, read_transitions_log_likelihood_per_site

EVAL_TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'peint', 'tests', 'eval_test_dir')
FAMILY = '4djg_1_B'


@pytest.fixture(scope="module")
def eval_dirs():
    """The three input directories the evaluation reads."""
    return {
        'transitions_dir': os.path.join(EVAL_TEST_DIR, 'transitions_dir'),
        'aligned_transitions_dir': os.path.join(EVAL_TEST_DIR, 'aligned_transitions_dir'),
        'alignment_mask_dir': os.path.join(EVAL_TEST_DIR, 'alignment_mask_dir'),
    }


@pytest.fixture(scope="module")
def vocab():
    return esm.data.Alphabet.from_architecture("ESM-1b")


@pytest.fixture(scope="module")
def dataset(eval_dirs, vocab):
    return AlignedTransitionsDataset(family=FAMILY, vocab=vocab, **eval_dirs)


@pytest.fixture(scope="module")
def reference_log_likelihood():
    """Per-site log-likelihoods from the previous version of the repository."""
    return np.array(read_transitions_log_likelihood_per_site(
        os.path.join(EVAL_TEST_DIR, 'reference_log_likelihood_per_site_dir', FAMILY + '.txt')
    ))


@pytest.fixture(scope="module")
def per_site_log_likelihood(dataset, checkpoint_path, device):
    """Score the fixture family in fp32, so the values are checkpoint-exact."""
    from peint.models import load_peint_model

    model, model_vocab = load_peint_model(
        checkpoint_path, device=device, model_type='standard', use_flash=False
    )
    return evaluate_transitions_log_likelihood_per_site(
        model=model, vocab=model_vocab, dataset=dataset, device=device, batch_size=8
    )


def test_list_families_finds_fixture(eval_dirs):
    assert list_families(**eval_dirs) == [FAMILY]


def test_dataset_indexes_residues_back_to_alignment_columns(dataset):
    """Every residue kept by the alignment mask gets exactly one column."""
    assert len(dataset) == dataset.num_transitions

    for transition in dataset:
        _, y, _, _ = transition.model_input
        assert len(transition.keep) == len(y), "mask must cover the unaligned target"
        assert transition.keep.sum() == len(transition.columns)
        assert transition.columns.max() < dataset.alignment_width


def test_scored_columns_mask_counts_non_gap_residues(dataset, eval_dirs):
    """The scored columns are exactly the non-gap columns of the aligned targets."""
    aligned = read_transitions(
        os.path.join(eval_dirs['aligned_transitions_dir'], FAMILY + '.txt')
    )
    expected = np.array([[residue != '-' for residue in y] for _, y, _ in aligned])

    assert np.array_equal(dataset.scored_columns_mask(), expected)


def test_over_length_transitions_are_excluded_but_still_counted(eval_dirs, vocab):
    """Transitions too long to score keep their row, so rows stay aligned."""
    dataset = AlignedTransitionsDataset(family=FAMILY, vocab=vocab, max_length=40, **eval_dirs)

    assert dataset.num_transitions == 44
    assert 0 < len(dataset) < dataset.num_transitions

    unscored_rows = (~dataset.scored_columns_mask()).all(axis=1).sum()
    assert unscored_rows == dataset.num_transitions - len(dataset)


def test_mismatched_directories_are_rejected(eval_dirs, vocab, tmp_path):
    """A mask that does not match the alignment is a corrupt dataset, not a warning."""
    masks = read_transitions(os.path.join(eval_dirs['alignment_mask_dir'], FAMILY + '.txt'))
    x_mask, y_mask, t = masks[0]
    corrupted = tmp_path / 'alignment_mask_dir'
    corrupted.mkdir()
    (corrupted / (FAMILY + '.txt')).write_text(
        f"{len(masks)} transitions\n"
        + "\n".join(
            [f"{x_mask} {'0' + y_mask[1:]} {t}"]  # drop the first kept residue
            + [f"{x} {y} {t}" for x, y, t in masks[1:]]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="does not match the aligned target"):
        AlignedTransitionsDataset(
            transitions_dir=eval_dirs['transitions_dir'],
            aligned_transitions_dir=eval_dirs['aligned_transitions_dir'],
            alignment_mask_dir=str(corrupted),
            family=FAMILY,
            vocab=vocab,
        )


def test_sum_over_sites_ignores_unscored_columns():
    per_site = np.array([
        [-1.0, 0.0, -2.0],       # a gap column contributes nothing
        [-1.0, np.nan, -2.0],    # an unscored column is skipped
        [np.nan, np.nan, np.nan],  # a transition that was never scored
    ])

    totals = sum_over_sites(per_site)

    assert totals[0] == pytest.approx(-3.0)
    assert totals[1] == pytest.approx(-3.0)
    assert np.isnan(totals[2])


def test_uniform_random_guess_follows_the_same_gap_convention(eval_dirs):
    """The baseline has to score gaps the way PEINT does, or the two can't be compared."""
    from peint.models._uniform_random_guess import (
        evaluate_uniform_random_guess_model_transitions_log_likelihood_per_site,
    )

    aligned = read_transitions(
        os.path.join(eval_dirs['aligned_transitions_dir'], FAMILY + '.txt')
    )
    per_site = np.array(
        evaluate_uniform_random_guess_model_transitions_log_likelihood_per_site(aligned)
    )
    is_residue = np.array([[y_i != '-' for y_i in y] for _, y, _ in aligned])

    assert (per_site[~is_residue] == 0.0).all()
    assert per_site[is_residue] == pytest.approx(-np.log(20))


@pytest.mark.integration
def test_matches_previous_repository_likelihoods(
    per_site_log_likelihood, reference_log_likelihood
):
    """The refactored interface reproduces the likelihoods it replaces."""
    assert per_site_log_likelihood.shape == reference_log_likelihood.shape
    np.testing.assert_allclose(
        per_site_log_likelihood, reference_log_likelihood, atol=1e-4
    )


@pytest.mark.integration
def test_gap_columns_carry_no_likelihood(per_site_log_likelihood, dataset):
    """Gap columns are exactly zero, so summing over sites ignores them."""
    scored = dataset.scored_columns_mask()

    assert (per_site_log_likelihood[~scored] == 0.0).all()
    assert np.isfinite(per_site_log_likelihood[scored]).all()
    assert (per_site_log_likelihood[scored] < 0).all()


@pytest.mark.integration
def test_amino_acid_restriction_normalizes_over_twenty_states(
    dataset, checkpoint_path, device
):
    """Restricting to amino acids moves mass onto them, raising every score."""
    from peint.models import load_peint_model

    model, model_vocab = load_peint_model(
        checkpoint_path, device=device, model_type='standard', use_flash=False
    )
    scored = dataset.scored_columns_mask()

    restricted = evaluate_transitions_log_likelihood_per_site(
        model=model, vocab=model_vocab, dataset=dataset, device=device,
        restrict_to_amino_acids=True,
    )
    unrestricted = evaluate_transitions_log_likelihood_per_site(
        model=model, vocab=model_vocab, dataset=dataset, device=device,
        restrict_to_amino_acids=False,
    )

    assert (restricted[scored] > unrestricted[scored]).all()


@pytest.mark.integration
def test_batching_does_not_change_scores(dataset, checkpoint_path, device):
    """Padding a batch must not leak into the per-residue likelihoods."""
    from peint.models import load_peint_model

    model, model_vocab = load_peint_model(
        checkpoint_path, device=device, model_type='standard', use_flash=False
    )
    scored = dataset.scored_columns_mask()

    one_at_a_time = evaluate_transitions_log_likelihood_per_site(
        model=model, vocab=model_vocab, dataset=dataset, device=device, batch_size=1
    )
    batched = evaluate_transitions_log_likelihood_per_site(
        model=model, vocab=model_vocab, dataset=dataset, device=device, batch_size=16
    )

    np.testing.assert_allclose(batched[scored], one_at_a_time[scored], atol=1e-4)
