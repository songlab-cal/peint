"""Tests for unified homology detection module.

I/O tests run without GPU. Model tests are marked as integration tests
and require both GPU and model checkpoint.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from peint.homology_detection import (
    DiamondHomologySearcher,
    DiamondSearchConfig,
    HomologyHit,
    get_distance,
    load_distance_matrix,
    load_fasta,
    load_proteomes,
)

# Test data directory
HOMOLOGY_TEST_DIR = os.path.join(
    os.path.dirname(__file__), "..", "peint", "tests", "homology_test_dir"
)


class TestHomologyHit:
    def test_to_dict_basic(self):
        hit = HomologyHit(query_id="q1", db_id="db1", score=0.5)
        result = hit.to_dict()

        assert result["query_id"] == "q1"
        assert result["db_id"] == "db1"
        assert result["score"] == 0.5

    def test_to_dict_with_extra(self):
        hit = HomologyHit(
            query_id="q1",
            db_id="db1",
            score=0.5,
            extra={"time": 1.0, "proteome": "human"},
        )
        result = hit.to_dict()

        assert result["time"] == 1.0
        assert result["proteome"] == "human"


class TestLoadFasta:
    def test_load_fasta_basic(self):
        fasta_path = os.path.join(HOMOLOGY_TEST_DIR, "human.fasta")
        sequences = load_fasta(fasta_path)

        assert len(sequences) == 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in sequences)

        # Check first sequence
        seq_id, seq = sequences[0]
        assert "ADRB2_HUMAN" in seq_id
        assert seq.startswith("MGQPGN")
        assert len(seq) > 100

    def test_load_fasta_all_files(self):
        for name in ["human", "mouse", "bovine"]:
            fasta_path = os.path.join(HOMOLOGY_TEST_DIR, f"{name}.fasta")
            sequences = load_fasta(fasta_path)
            assert len(sequences) == 3


class TestLoadProteomes:
    def test_load_proteomes(self):
        proteomes = load_proteomes(HOMOLOGY_TEST_DIR)

        assert len(proteomes) == 3
        assert set(proteomes.keys()) == {"human", "mouse", "bovine"}

        for name, seqs in proteomes.items():
            assert len(seqs) == 3
            for seq_id, seq in seqs:
                assert isinstance(seq_id, str)
                assert isinstance(seq, str)
                assert len(seq) > 0


class TestLoadDistanceMatrix:
    def test_load_csv_matrix(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(",human,mouse,bovine\n")
            f.write("human,0.0,0.5,0.8\n")
            f.write("mouse,0.5,0.0,0.6\n")
            f.write("bovine,0.8,0.6,0.0\n")
            f.flush()

            distances = load_distance_matrix(f.name)

        os.unlink(f.name)

        assert distances[("human", "mouse")] == 0.5
        assert distances[("mouse", "human")] == 0.5
        assert distances[("human", "bovine")] == 0.8
        assert distances[("human", "human")] == 0.0

    def test_load_tsv_matrix(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("species\thuman\tmouse\n")
            f.write("human\t0.0\t1.2\n")
            f.write("mouse\t1.2\t0.0\n")
            f.flush()

            distances = load_distance_matrix(f.name)

        os.unlink(f.name)

        assert distances[("human", "mouse")] == 1.2

    def test_default_value_for_invalid(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(",a,b\n")
            f.write("a,0.0,NA\n")
            f.write("b,NA,0.0\n")
            f.flush()

            distances = load_distance_matrix(f.name, default=999.0)

        os.unlink(f.name)

        assert distances[("a", "b")] == 999.0


class TestGetDistance:
    def test_with_matrix(self):
        distances = {("a", "b"): 1.5, ("b", "a"): 1.5}
        assert get_distance("a", "b", distances, default=0.0) == 1.5

    def test_missing_from_matrix(self):
        distances = {("a", "b"): 1.5}
        assert get_distance("a", "c", distances, default=2.0) == 2.0

    def test_none_matrix(self):
        assert get_distance("a", "b", None, default=1.0) == 1.0


class TestWriteResults:
    def test_write_csv(self):
        from peint.homology_detection._base import HomologySearcher

        # Create a concrete implementation for testing
        class DummySearcher(HomologySearcher):
            def search(self, database, queries, top_k=None):
                return []

            def all_vs_all(self, sequences):
                return []

            def all_vs_all_proteomes(self, proteomes, distances=None):
                return []

        searcher = DummySearcher()
        hits = [
            HomologyHit(query_id="A", db_id="B", score=1.5, extra={"time": 1.0}),
            HomologyHit(query_id="A", db_id="C", score=2.0, extra={"time": 1.0}),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        searcher.write_results(hits, path, format="csv")

        with open(path) as f:
            content = f.read()

        os.unlink(path)

        assert "query_id" in content
        assert "db_id" in content
        assert "score" in content
        assert "1.5" in content

    def test_write_json(self):
        from peint.homology_detection._base import HomologySearcher

        class DummySearcher(HomologySearcher):
            def search(self, database, queries, top_k=None):
                return []

            def all_vs_all(self, sequences):
                return []

            def all_vs_all_proteomes(self, proteomes, distances=None):
                return []

        searcher = DummySearcher()
        hits = [HomologyHit(query_id="A", db_id="B", score=1.5)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        searcher.write_results(hits, path, format="json")

        with open(path) as f:
            import json

            data = json.load(f)

        os.unlink(path)

        assert len(data) == 1
        assert data[0]["query_id"] == "A"


class TestDiamondHomologySearcher:
    """Unit tests for DIAMOND searcher (mock subprocess)."""

    def test_init_raises_without_diamond(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="DIAMOND is not installed"):
                DiamondHomologySearcher(DiamondSearchConfig())

    def test_write_fasta(self):
        with patch("shutil.which", return_value="/usr/bin/diamond"):
            searcher = DiamondHomologySearcher(DiamondSearchConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            fasta_path = Path(tmpdir) / "test.fasta"
            sequences = [("seq1", "ACDEF"), ("seq2", "GHIKL")]

            searcher._write_fasta(sequences, fasta_path)

            with open(fasta_path) as f:
                content = f.read()

            assert ">seq1" in content
            assert "ACDEF" in content
            assert ">seq2" in content
            assert "GHIKL" in content

    def test_parse_results(self):
        with patch("shutil.which", return_value="/usr/bin/diamond"):
            searcher = DiamondHomologySearcher(DiamondSearchConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            output_path = Path(tmpdir) / "results.tsv"
            with open(output_path, "w") as f:
                f.write("query1\tdb1\t95.5\t100\t1e-50\t200.5\n")
                f.write("query1\tdb2\t80.0\t90\t1e-20\t150.0\n")

            hits = searcher._parse_results(output_path)

            assert len(hits) == 2
            assert hits[0].query_id == "query1"
            assert hits[0].db_id == "db1"
            assert hits[0].score == 1e-50
            assert hits[0].extra["pident"] == 95.5
            assert hits[0].extra["bitscore"] == 200.5

    def test_search_with_mock(self):
        with patch("shutil.which", return_value="/usr/bin/diamond"):
            searcher = DiamondHomologySearcher(DiamondSearchConfig())

        # Mock the subprocess calls
        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""

            # If this is a blastp call, create mock output
            if "blastp" in cmd:
                output_path = cmd[cmd.index("-o") + 1]
                with open(output_path, "w") as f:
                    f.write("q1\tdb1\t90.0\t100\t1e-30\t180.0\n")

            return result

        with patch("subprocess.run", side_effect=mock_run):
            database = [("db1", "ACDEFGHIKLMNPQRSTVWY")]
            queries = [("q1", "ACDEFGHIKLMNPQRSTVWY")]

            hits = searcher.search(database, queries)

            assert len(hits) == 1
            assert hits[0].query_id == "q1"
            assert hits[0].db_id == "db1"


@pytest.mark.skipif(
    shutil.which("diamond") is None, reason="DIAMOND not installed"
)
class TestDiamondIntegration:
    """Integration tests for DIAMOND (requires diamond binary)."""

    def test_all_vs_all(self):
        config = DiamondSearchConfig(evalue=10.0, threads=1)
        searcher = DiamondHomologySearcher(config)

        sequences = load_fasta(os.path.join(HOMOLOGY_TEST_DIR, "human.fasta"))
        hits = searcher.all_vs_all(sequences)

        # Should have hits (excluding self-hits)
        assert len(hits) > 0
        # No self-hits
        for hit in hits:
            assert hit.query_id != hit.db_id

    def test_search(self):
        config = DiamondSearchConfig(evalue=10.0, threads=1)
        searcher = DiamondHomologySearcher(config)

        database = load_fasta(os.path.join(HOMOLOGY_TEST_DIR, "human.fasta"))
        queries = load_fasta(os.path.join(HOMOLOGY_TEST_DIR, "mouse.fasta"))[:1]

        hits = searcher.search(database, queries, top_k=2)

        # Should have hits
        assert len(hits) > 0
        # At most 2 per query
        assert len(hits) <= 2

    def test_all_vs_all_proteomes(self):
        config = DiamondSearchConfig(evalue=10.0, threads=1)
        searcher = DiamondHomologySearcher(config)

        proteomes = load_proteomes(HOMOLOGY_TEST_DIR)
        hits = searcher.all_vs_all_proteomes(proteomes)

        # Should have cross-proteome hits
        assert len(hits) > 0
        for hit in hits:
            assert hit.extra["ref_proteome"] != hit.extra["query_proteome"]


# Integration tests requiring GPU and model checkpoint
@pytest.mark.integration
class TestPeintHomologyIntegration:
    """Integration tests that require GPU, model checkpoint, and Flash Attention."""

    @pytest.fixture
    def peint_searcher(self, checkpoint_path, device):
        """Create PeintHomologySearcher for testing."""
        from peint.homology_detection import PeintHomologySearcher, PeintSearchConfig
        from peint.models._transformer_modules import FLASH_AVAILABLE

        if not FLASH_AVAILABLE:
            pytest.skip("PEINT homology search requires Flash Attention (GPU compute >= 8.0)")

        config = PeintSearchConfig(
            checkpoint_path=checkpoint_path,
            device=str(device),
            time=1.0,
            batch_size=4,
            use_flash=True,
        )
        return PeintHomologySearcher(config)

    def test_search(self, peint_searcher):
        database = load_fasta(os.path.join(HOMOLOGY_TEST_DIR, "human.fasta"))
        queries = load_fasta(os.path.join(HOMOLOGY_TEST_DIR, "mouse.fasta"))[:2]

        hits = peint_searcher.search(database=database, queries=queries, top_k=2)

        # 2 queries, top_k=2 each
        assert len(hits) <= 4

        for hit in hits:
            assert isinstance(hit.score, float)
            assert hit.score > 0
            assert "time" in hit.extra

    def test_all_vs_all(self, peint_searcher):
        sequences = load_fasta(os.path.join(HOMOLOGY_TEST_DIR, "human.fasta"))[:3]
        hits = peint_searcher.all_vs_all(sequences)

        # 3 sequences, each compared to 2 others = 6 comparisons
        assert len(hits) == 6

        for hit in hits:
            assert hit.query_id != hit.db_id
            assert hit.score > 0

    def test_all_vs_all_proteomes(self, peint_searcher):
        proteomes = load_proteomes(HOMOLOGY_TEST_DIR)

        hits = peint_searcher.all_vs_all_proteomes(
            proteomes=proteomes, skip_same_proteome=True
        )

        # 3 proteomes, 3 seqs each, comparing across proteomes only
        # Each of 9 refs compares to 6 queries (other 2 proteomes * 3 seqs)
        expected_count = 9 * 6
        assert len(hits) == expected_count

        for hit in hits:
            assert hit.extra["ref_proteome"] != hit.extra["query_proteome"]
            assert hit.score > 0

    def test_all_vs_all_proteomes_with_distances(self, peint_searcher):
        proteomes = load_proteomes(HOMOLOGY_TEST_DIR)

        # Create a distance matrix
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(",human,mouse,bovine\n")
            f.write("human,0.0,0.5,0.8\n")
            f.write("mouse,0.5,0.0,0.6\n")
            f.write("bovine,0.8,0.6,0.0\n")
            f.flush()
            distances = load_distance_matrix(f.name)

        os.unlink(f.name)

        hits = peint_searcher.all_vs_all_proteomes(
            proteomes=proteomes, distances=distances, skip_same_proteome=True
        )

        # Verify times match distance matrix
        for hit in hits:
            ref_prot = hit.extra["ref_proteome"]
            query_prot = hit.extra["query_proteome"]
            expected_time = distances.get((ref_prot, query_prot))
            assert hit.extra["time"] == expected_time
