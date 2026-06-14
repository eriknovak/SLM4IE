"""Tests for slm4ie/tokenizers/corpus.py."""

import gzip
import json
from pathlib import Path
from typing import Dict, List

from slm4ie.tokenizers.corpus import (
    SampleBudget,
    iter_dedup_documents,
    iter_sample_cache,
    sample_corpus,
    write_sample_cache,
)


def _write_corpus(root: Path, docs_by_source: Dict[str, List[str]]) -> Path:
    """Write a fake `05_2_dedup` tree with one shard per source.

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
        write_sample_cache(iter(["a\nb", "c d"]), cache)
        assert list(iter_sample_cache(cache)) == ["a b", "c d"]

    def test_plain_text_cache(self, tmp_path: Path):
        """A non-gz cache path produces a plain text file."""
        cache = tmp_path / "sample.txt"
        write_sample_cache(iter(["hello"]), cache)
        assert cache.read_text(encoding="utf-8").strip() == "hello"
        assert list(iter_sample_cache(cache)) == ["hello"]
