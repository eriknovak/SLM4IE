"""Tests for slm4ie/tokenizers/corpus.py."""

import gzip
import json
from pathlib import Path
from typing import Dict, List

import pytest

from slm4ie.tokenizers.corpus import (
    SampleBudget,
    check_sample_side,
    corpus_fingerprint,
    count_sample_characters,
    fit_corpus_split,
    iter_dedup_documents,
    iter_sample,
    load_corpus_split,
    materialize_sample,
    sample_corpus,
    save_corpus_split,
    write_sample,
)


def _write_corpus(root: Path, docs_by_source: Dict[str, List[str]]) -> Path:
    """Write a fake `06_sentence_dedup` tree with one shard per source.

    Args:
        root (Path): Corpus root to create.
        docs_by_source (Dict[str, List[str]]): Texts keyed by dataset name.

    Returns:
        Path: The corpus root, for chaining.
    """
    for source, texts in docs_by_source.items():
        sub = root / source
        sub.mkdir(parents=True, exist_ok=True)
        with gzip.open(sub / "00000.jsonl.gz", "wt", encoding="utf-8") as handle:
            for i, text in enumerate(texts):
                record = {"text": text, "id": f"{source}-{i}", "metadata": {"dataset": source}}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return root


class TestIterDedupDocuments:
    """Tests for iter_dedup_documents."""

    def test_reads_all_sources(self, tmp_path: Path):
        """Every shard across subdirs is read."""
        root = _write_corpus(tmp_path / "dd", {"a": ["x", "y"], "b": ["z"]})
        records = list(iter_dedup_documents(root))
        assert {r["text"] for r in records} == {"x", "y", "z"}

    def test_skips_state_dirs(self, tmp_path: Path):
        """Underscore-prefixed subdirs (dedup state) are ignored."""
        root = _write_corpus(tmp_path / "dd", {"a": ["x"]})
        (root / "_dedup_state").mkdir()
        (root / "_dedup_state" / "00000.jsonl.gz").write_bytes(b"garbage")
        records = list(iter_dedup_documents(root))
        assert [r["text"] for r in records] == ["x"]

    def test_dataset_filter(self, tmp_path: Path):
        """The datasets filter restricts which subdirs are read."""
        root = _write_corpus(tmp_path / "dd", {"a": ["x"], "b": ["z"]})
        records = list(iter_dedup_documents(root, datasets=["b"]))
        assert [r["text"] for r in records] == ["z"]


class TestSampleCorpus:
    """Tests for sample_corpus budget and weighting."""

    def test_doc_budget_caps_output(self, tmp_path: Path):
        """A document budget limits how many texts are yielded."""
        root = _write_corpus(tmp_path / "dd", {"a": [f"doc{i}" for i in range(20)]})
        texts = list(sample_corpus(root, SampleBudget(max_docs=5)))
        assert len(texts) <= 5

    def test_source_weight_zero_excludes(self, tmp_path: Path):
        """A zero weight excludes a source entirely."""
        root = _write_corpus(tmp_path / "dd", {"a": ["aa"], "b": ["bb"]})
        budget = SampleBudget(source_weights={"a": 1.0, "b": 0.0})
        texts = set(sample_corpus(root, budget))
        assert texts == {"aa"}

    def test_deterministic_under_seed(self, tmp_path: Path):
        """Two runs with the same seed yield identical samples."""
        root = _write_corpus(tmp_path / "dd", {"a": [f"d{i}" for i in range(30)]})
        first = list(sample_corpus(root, SampleBudget(max_docs=10, seed=7)))
        second = list(sample_corpus(root, SampleBudget(max_docs=10, seed=7)))
        assert first == second


class TestSampleCache:
    """Tests for the cache writer/reader round-trip."""

    def test_round_trip_flattens_newlines(self, tmp_path: Path):
        """Cached docs round-trip with internal newlines collapsed."""
        cache = tmp_path / "sample.txt.gz"
        write_sample(iter(["a\nb", "c d"]), cache)
        assert list(iter_sample(cache)) == ["a b", "c d"]

    def test_plain_text_cache(self, tmp_path: Path):
        """A non-gz cache path produces a plain text file."""
        cache = tmp_path / "sample.txt"
        write_sample(iter(["hello"]), cache)
        assert cache.read_text(encoding="utf-8").strip() == "hello"
        assert list(iter_sample(cache)) == ["hello"]


class TestCorpusSplit:
    """Tests for fitting, persisting and sampling a corpus split."""

    def _corpus(self, tmp_path: Path) -> Path:
        """Write a two-source corpus of uneven size.

        Args:
            tmp_path (Path): Pytest temporary directory.

        Returns:
            Path: The corpus root.
        """
        return _write_corpus(
            tmp_path / "dd",
            {"big": [f"big doc {i}" for i in range(50)], "small": [f"small {i}" for i in range(10)]},
        )

    def test_each_source_split_in_exact_proportion(self, tmp_path: Path):
        """Every source puts round(80%) of its documents on the training side."""
        split = fit_corpus_split(self._corpus(tmp_path), 0.8, max_workers=1)
        assert split.counts["big"]["train"]["documents"] == 40
        assert split.counts["small"]["train"]["documents"] == 8
        assert split.counts["small"]["eval"]["documents"] == 2

    def test_side_of_agrees_with_counts(self, tmp_path: Path):
        """Classifying every document reproduces the fitted side counts."""
        split = fit_corpus_split(self._corpus(tmp_path), 0.8, max_workers=1)
        sides = [split.side_of("big", f"big-{i}") for i in range(50)]
        assert sides.count("train") == 40

    def test_split_counts_words(self, tmp_path: Path):
        """Word counts per side add up to the source's whitespace words."""
        split = fit_corpus_split(self._corpus(tmp_path), 0.8, max_workers=1)
        sides = split.counts["big"]
        assert sides["train"]["words"] + sides["eval"]["words"] == 150

    def test_split_round_trips_through_json(self, tmp_path: Path):
        """A saved split loads back identical."""
        split = fit_corpus_split(self._corpus(tmp_path), 0.8, max_workers=1)
        assert load_corpus_split(save_corpus_split(split, tmp_path / "split.json")) == split

    def test_sample_draws_only_from_its_side(self, tmp_path: Path):
        """An evaluation sample holds only evaluation-side documents."""
        root = self._corpus(tmp_path)
        split = fit_corpus_split(root, 0.8, max_workers=1)
        eval_texts = {f"big doc {i}" for i in range(50) if split.side_of("big", f"big-{i}") == "eval"}
        drawn = set(sample_corpus(root, SampleBudget(max_docs=1000), split, "eval"))
        assert {t for t in drawn if t.startswith("big")} == eval_texts

    def test_sample_without_side_rejected(self, tmp_path: Path):
        """A split without a side is a caller error."""
        root = self._corpus(tmp_path)
        split = fit_corpus_split(root, 0.8, max_workers=1)
        with pytest.raises(ValueError, match="side"):
            list(sample_corpus(root, SampleBudget(max_docs=5), split))

    def test_materialized_sample_records_its_sources(self, tmp_path: Path):
        """The sidecar lists the documents drawn per source."""
        root = self._corpus(tmp_path)
        split = fit_corpus_split(root, 0.8, max_workers=1)
        cache = materialize_sample(root, SampleBudget(max_docs=1000), tmp_path / "s.txt", split, "train")
        sources = json.loads((tmp_path / "s.txt.sources.json").read_text())["sources"]
        assert sources["small"]["documents"] == 8
        assert len(list(iter_sample(cache))) == 48


class TestSplitGuards:
    """Tests for the checks that refuse stale splits and samples."""

    def test_fingerprint_changes_when_a_shard_is_rewritten(self, tmp_path: Path):
        """Rewriting a shard with other content changes the corpus fingerprint."""
        root = _write_corpus(tmp_path / "dd", {"a": ["x", "y"]})
        before = corpus_fingerprint(root)
        _write_corpus(root, {"a": ["x", "y", "a longer third document"]})
        assert corpus_fingerprint(root) != before

    def test_fitted_split_carries_the_fingerprint(self, tmp_path: Path):
        """A fitted split records the fingerprint of the corpus it was fitted on."""
        root = _write_corpus(tmp_path / "dd", {"a": ["x", "y"]})
        assert fit_corpus_split(root, 0.5, max_workers=1).corpus_fingerprint == corpus_fingerprint(root)

    def test_sample_from_the_right_side_passes(self, tmp_path: Path):
        """A sample drawn from the training side passes the training-side check."""
        root = _write_corpus(tmp_path / "dd", {"a": [f"doc {i}" for i in range(10)]})
        split = fit_corpus_split(root, 0.8, max_workers=1)
        cache = materialize_sample(root, SampleBudget(max_docs=100), tmp_path / "s.txt", split, "train")
        check_sample_side(cache, "train")

    def test_sample_from_the_other_side_refused(self, tmp_path: Path):
        """A training sample is refused where an evaluation sample is needed."""
        root = _write_corpus(tmp_path / "dd", {"a": [f"doc {i}" for i in range(10)]})
        split = fit_corpus_split(root, 0.8, max_workers=1)
        cache = materialize_sample(root, SampleBudget(max_docs=100), tmp_path / "s.txt", split, "train")
        with pytest.raises(ValueError, match="side"):
            check_sample_side(cache, "eval")

    def test_sample_without_sidecar_refused(self, tmp_path: Path):
        """A sample drawn before splits existed has no sidecar and is refused."""
        cache = write_sample(iter(["old"]), tmp_path / "old.txt")
        with pytest.raises(ValueError, match="None"):
            check_sample_side(cache, "train")


def test_character_counts_skip_spaces_and_compose_accents(tmp_path: Path):
    """Spaces are not counted, and a decomposed "č" counts as the one character "č"."""
    cache = write_sample(iter(["mac\u030cka  spi"]), tmp_path / "sample.txt")
    counts = count_sample_characters(cache)
    assert counts["č"] == 1
    assert " " not in counts and "\u030c" not in counts
