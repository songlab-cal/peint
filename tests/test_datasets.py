"""Tests for PEINT dataset classes."""

import os
import pytest
import torch
import tempfile

from peint.datasets import (
    PeintDataset,
    PeintCollator,
    get_a3m_families,
    a3m_dataset__cached,
)


# Path to test a3m directory with 2 files: O13297.a3m and Q9T0N8.a3m
A3M_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "peint", "tests", "a3m_test_dir")
# Path to rate matrices
RATE_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rate_matrices", "wag.txt")


class TestGetA3mFamilies:
    """Test get_a3m_families functionality."""

    def test_returns_all_families_by_default(self):
        """Test that num_families=-1 returns all families."""
        families = get_a3m_families(A3M_TEST_DIR)
        assert families == ["O13297", "Q9T0N8"]

    def test_returns_all_families_explicitly(self):
        """Test that num_families=-1 explicitly returns all families."""
        families = get_a3m_families(A3M_TEST_DIR, num_families=-1)
        assert families == ["O13297", "Q9T0N8"]

    def test_returns_subset_of_families(self):
        """Test that num_families=1 returns exactly 1 family."""
        families = get_a3m_families(A3M_TEST_DIR, num_families=1)
        assert len(families) == 1
        assert families[0] in ["O13297", "Q9T0N8"]

    def test_warns_when_requesting_more_than_available(self):
        """Test that requesting more families than available issues a warning."""
        with pytest.warns(UserWarning, match="Requested 10 families but only 2 available"):
            families = get_a3m_families(A3M_TEST_DIR, num_families=10)
        assert families == ["O13297", "Q9T0N8"]

    def test_raises_on_invalid_directory(self):
        """Test that invalid directory raises ValueError."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            get_a3m_families("/nonexistent/path")

    def test_raises_on_empty_directory(self):
        """Test that directory with no a3m files raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No .a3m files found"):
                get_a3m_families(tmpdir)

    def test_deterministic_shuffling(self):
        """Test that same seed produces same subset."""
        families1 = get_a3m_families(A3M_TEST_DIR, num_families=1, random_seed=123)
        families2 = get_a3m_families(A3M_TEST_DIR, num_families=1, random_seed=123)
        assert families1 == families2

    def test_different_seeds_can_produce_different_results(self):
        """Test that different seeds can produce different subsets."""
        # With only 2 families, we need to try a few seeds to find ones that differ
        results = set()
        for seed in range(10):
            families = get_a3m_families(A3M_TEST_DIR, num_families=1, random_seed=seed)
            results.add(families[0])
        # Should have found both families with different seeds
        assert len(results) == 2


class TestA3mDataset:
    """Test a3m_dataset__cached functionality."""

    @pytest.fixture
    def cache_dir(self):
        """Set up temporary cache directory for cherryml and peint."""
        from cherryml import caching as cherryml_caching
        from peint import caching as peint_caching
        with tempfile.TemporaryDirectory() as tmpdir:
            cherryml_caching.set_cache_dir(tmpdir)
            peint_caching.set_cache_dir(tmpdir)
            yield tmpdir

    @pytest.mark.slow
    def test_no_train_test_split(self, cache_dir):
        """Test dataset creation without train/test split."""
        data_dirs = a3m_dataset__cached(
            a3m_dir=A3M_TEST_DIR,
            num_families=1,
            num_sequences_per_family=32,
            num_processes=1,
            do_train_test_split=False,
            rate_matrix_path=RATE_MATRIX_PATH,
        )

        # Check expected keys for no-split mode
        assert "families" in data_dirs
        assert "transitions_dir" in data_dirs
        assert "msa_dir" in data_dirs
        assert "tree_dir" in data_dirs
        assert "site_rates_4cat_dir" in data_dirs

        # Should NOT have train/test specific keys
        assert "train_transitions_dir" not in data_dirs
        assert "test_transitions_dir" not in data_dirs

        # Verify directories exist
        assert os.path.isdir(data_dirs["transitions_dir"])
        assert os.path.isdir(data_dirs["msa_dir"])
        assert os.path.isdir(data_dirs["tree_dir"])

        # Verify family files exist
        family = data_dirs["families"][0]
        assert os.path.isfile(os.path.join(data_dirs["transitions_dir"], f"{family}.txt"))

    @pytest.mark.slow
    def test_with_train_test_split(self, cache_dir):
        """Test dataset creation with train/test split."""
        data_dirs = a3m_dataset__cached(
            a3m_dir=A3M_TEST_DIR,
            num_families=1,
            num_sequences_per_family=32,
            num_processes=1,
            do_train_test_split=True,
            rate_matrix_path=RATE_MATRIX_PATH,
        )

        # Check expected keys for split mode
        assert "families" in data_dirs
        assert "train_transitions_dir" in data_dirs
        assert "test_transitions_dir" in data_dirs
        assert "train_msa_dir" in data_dirs
        assert "test_msa_dir" in data_dirs
        assert "full_tree_dir" in data_dirs
        assert "train_tree_dir" in data_dirs
        assert "test_tree_dir" in data_dirs

        # Should NOT have no-split keys
        assert "transitions_dir" not in data_dirs

        # Verify directories exist
        assert os.path.isdir(data_dirs["train_transitions_dir"])
        assert os.path.isdir(data_dirs["test_transitions_dir"])


class TestPeintDataset:
    """Test PeintDataset functionality."""

    @pytest.fixture
    def sample_data_dir(self):
        """Create a temporary directory with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample family file
            data_file = os.path.join(tmpdir, 'test_family.txt')
            with open(data_file, 'w') as f:
                f.write("header\n")  # Dataset skips first line
                f.write("ACDEFGHIK ACDEYGHIK 0.1\n")
                f.write("MNPQRSTVW MNPQRSTVW 0.05\n")
                f.write("ACDEFGHIKLMNPQRSTVWY ACDEFGHIKLMNPQRSTVWY 0.2\n")
            yield tmpdir

    @pytest.fixture
    def mock_vocab(self):
        """Create a minimal mock vocabulary for testing."""
        class MockVocab:
            def __init__(self):
                self.padding_idx = 0
                self.cls_idx = 1
                self.eos_idx = 2
                self.mask_idx = 3
                # Standard amino acids start at index 4
                self._tok_to_idx = {aa: i + 4 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

            def encode(self, seq):
                return [self._tok_to_idx[aa] for aa in seq]

            def __len__(self):
                return 24  # pad, cls, eos, mask + 20 AAs

        return MockVocab()

    def test_dataset_loads_data(self, sample_data_dir, mock_vocab):
        """Test that dataset loads sequences from files."""
        dataset = PeintDataset(
            data_path=sample_data_dir,
            vocab=mock_vocab,
            families=['test_family'],
            max_len=100
        )

        assert len(dataset) == 3, "Should load 3 sequences"

    def test_dataset_filters_long_sequences(self, sample_data_dir, mock_vocab):
        """Test that sequences exceeding max_len are filtered."""
        dataset = PeintDataset(
            data_path=sample_data_dir,
            vocab=mock_vocab,
            families=['test_family'],
            max_len=10  # Only first two sequences are <= 10 chars
        )

        assert len(dataset) == 2, "Should filter out long sequence"

    def test_dataset_getitem_returns_expected_types(self, sample_data_dir, mock_vocab):
        """Test that __getitem__ returns correct types."""
        dataset = PeintDataset(
            data_path=sample_data_dir,
            vocab=mock_vocab,
            families=['test_family']
        )

        x, y, t, length = dataset[0]

        assert isinstance(x, torch.Tensor), "x should be tensor"
        assert isinstance(y, torch.Tensor), "y should be tensor"
        assert isinstance(t, float), "t should be a scalar float (memory-efficient store)"
        assert isinstance(length, int), "length should be int"

    def test_dataset_time_minimum(self, sample_data_dir, mock_vocab):
        """Test that times below minimum are clamped."""
        dataset = PeintDataset(
            data_path=sample_data_dir,
            vocab=mock_vocab,
            families=['test_family']
        )

        # Second sequence has t=0.05, first has t=0.1
        _, _, t, _ = dataset[1]

        # Time (a scalar now) should be clamped to >= 5e-3
        assert t >= 5e-3, "Time value should be >= 5e-3"


class TestPeintCollator:
    """Test PeintCollator functionality."""

    @pytest.fixture
    def mock_vocab(self):
        """Create a minimal mock vocabulary."""
        class MockVocab:
            def __init__(self):
                self.padding_idx = 0
                self.cls_idx = 1
                self.eos_idx = 2
                self.mask_idx = 3

            def __len__(self):
                return 24

        return MockVocab()

    def test_collator_pads_sequences(self, mock_vocab):
        """Test that collator properly pads variable length sequences."""
        collator = PeintCollator(vocab=mock_vocab, mask_prob=0.0)

        # Create batch with different lengths
        batch = [
            (torch.tensor([4, 5, 6]), torch.tensor([4, 5, 6]), 0.1, 3),
            (torch.tensor([7, 8]), torch.tensor([7, 8]), 0.2, 2),
        ]

        x_inputs, x_targets, y_inputs, y_targets, ts, x_pad_mask, y_pad_mask = collator(batch)

        # All tensors should have same sequence length dimension
        assert x_inputs.shape[1] == x_targets.shape[1]
        assert y_inputs.shape[1] == y_targets.shape[1]

        # Shorter sequence should have padding
        assert x_pad_mask[1].any(), "Shorter sequence should have padding"

    def test_collator_adds_special_tokens(self, mock_vocab):
        """Test that collator adds CLS and EOS tokens."""
        collator = PeintCollator(vocab=mock_vocab, mask_prob=0.0)

        batch = [
            (torch.tensor([4, 5]), torch.tensor([4, 5]), 0.1, 2),
        ]

        x_inputs, x_targets, y_inputs, y_targets, ts, _, _ = collator(batch)

        # x should have CLS at start and EOS at end
        assert x_inputs[0, 0].item() == mock_vocab.cls_idx
        assert x_inputs[0, -1].item() == mock_vocab.eos_idx

        # y_inputs should have CLS at start
        assert y_inputs[0, 0].item() == mock_vocab.cls_idx

    def test_collator_mlm_masking(self, mock_vocab):
        """Test that MLM masking is applied when mask_prob > 0."""
        collator = PeintCollator(vocab=mock_vocab, mask_prob=1.0)  # 100% masking

        batch = [
            (torch.tensor([4, 5, 6, 7, 8]), torch.tensor([4, 5, 6, 7, 8]), 0.1, 5),
        ]

        x_inputs, x_targets, _, _, _, _, _ = collator(batch)

        # With 100% mask prob, all non-special tokens should be masked
        # (except CLS and EOS which are added after masking)
        mask_count = (x_inputs == mock_vocab.mask_idx).sum().item()
        assert mask_count > 0, "Some tokens should be masked"
