"""Tests for the no-judge corpus profile statistics."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import pytest

from slm4ie.data.curate.inspect.profile import (
    cached_source_counts,
    count_source_documents,
    iter_stage_sentinels,
    language_confidence,
    oov_rate,
    percentiles,
    sample_documents,
    sloleks_forms,
    tokens,
    type_token_ratio,
)


def _write_shard(path: Path, count: int, start: int = 0) -> None:
    """Write numbered documents as one gzipped JSONL shard.

    Args:
        path: Shard to write.
        count: Documents to write.
        start: Number of the first document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(start, start + count):
            handle.write(json.dumps({"id": f"doc:{index}", "text": f"text {index}"}) + "\n")


class _Detector:
    """Stand-in for a lingua detector returning fixed confidences per text."""

    def __init__(self, answers: dict) -> None:
        """Store the answer for each text.

        Args:
            answers: Text to a list of (ISO code, confidence) pairs, best first.
        """
        self.answers = answers

    def compute_language_confidence_values(self, text: str) -> List[SimpleNamespace]:
        """Return the stored confidences shaped like lingua's.

        Args:
            text: The text to score.

        Returns:
            Objects carrying `language.iso_code_639_1.name` and `value`.
        """
        pairs: Sequence[Tuple[str, float]] = self.answers[text]
        return [
            SimpleNamespace(language=SimpleNamespace(iso_code_639_1=SimpleNamespace(name=code.upper())), value=value)
            for code, value in pairs
        ]


class TestLexicalStatistics:
    """Tokens, percentiles, type-token ratio and out-of-vocabulary rate."""

    def test_tokens_are_lowercased_word_runs(self) -> None:
        """Punctuation splits tokens and case is folded."""
        assert tokens("Dober dan, Svet!") == ["dober", "dan", "svet"]

    def test_percentiles_use_the_nearest_rank(self) -> None:
        """Percentiles pick a sample value, never an interpolated one."""
        result = percentiles(list(range(1, 101)), points=(5, 50, 95))
        assert result == {"p5": 6.0, "p50": 51.0, "p95": 95.0}
        assert percentiles([], points=(50,)) == {"p50": 0.0}

    def test_type_token_ratio_is_measured_at_the_budget(self) -> None:
        """Only the first `budget` tokens count, so a longer sample is comparable."""
        ratio, used = type_token_ratio(["a", "b", "a", "b", "c", "d"], budget=4)
        assert (ratio, used) == (0.5, 4)

    def test_oov_rate_ignores_numbers(self) -> None:
        """Digits are neither known nor unknown words."""
        rate, checked = oov_rate(["hiša", "xyzzy", "2024", "a1"], {"hiša"})
        assert (rate, checked) == (0.5, 2)

    def test_sloleks_forms_holds_lemmas_and_forms(self, tmp_path: Path) -> None:
        """Both the lemma and every inflected form are loaded, lowercased."""
        path = tmp_path / "sloleks.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"lemma": "Hiša", "forms": [{"form": "hiše"}, {"form": "Hiši"}]}) + "\n")
        assert sloleks_forms(path) == {"hiša", "hiše", "hiši"}


class TestLanguageConfidence:
    """The share called Slovene and the confidence given to Slovene."""

    def test_counts_the_target_even_when_it_is_not_first(self) -> None:
        """A document called Croatian still contributes its Slovene confidence."""
        detector = _Detector({"a": [("sl", 0.9), ("hr", 0.1)], "b": [("hr", 0.6), ("sl", 0.4)]})
        result = language_confidence(["a", "b"], detector)
        assert result["in_language_share"] == 0.5
        assert result["predicted"] == {"sl": 1, "hr": 1}
        assert (result["confidence"]["p5"], result["confidence"]["p95"]) == (0.4, 0.9)


class TestSampleDocuments:
    """Sampling follows file order at a stride, with a fallback for small sources."""

    def test_stride_spreads_the_sample(self, tmp_path: Path) -> None:
        """Every tenth document is taken from a source large enough."""
        _write_shard(tmp_path / "00000.jsonl.gz", 100)
        sampled = sample_documents(tmp_path, per_source=5, stride=10)
        assert [row["id"] for row in sampled] == [f"doc:{index}" for index in (0, 10, 20, 30, 40)]

    def test_small_source_falls_back_to_every_document(self, tmp_path: Path) -> None:
        """A source too small to fill half the sample at the stride is read whole."""
        _write_shard(tmp_path / "00000.jsonl.gz", 12)
        sampled = sample_documents(tmp_path, per_source=10, stride=10)
        assert len(sampled) == 10


class TestStageSentinels:
    """The survival funnel reads its counts from stage sentinels."""

    def test_yields_counts_per_stage_and_source(self, tmp_path: Path) -> None:
        """Every `.complete` sentinel yields its stage, source and counts."""
        for stage, records_out in (("01_language", 90), ("02_spam", 80)):
            sentinel = tmp_path / stage / "demo" / ".complete"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text(json.dumps({"records_in": 100, "records_out": records_out}), encoding="utf-8")
        assert list(iter_stage_sentinels(tmp_path, ("01_language", "02_spam"))) == [
            ("01_language", "demo", 100, 90),
            ("02_spam", "demo", 100, 80),
        ]


class TestCountSourceDocuments:
    """The dedup stages' per-source counts are read off their output."""

    def test_counts_every_shard_of_every_source(self, tmp_path: Path) -> None:
        """Shards of one source add up, and each source is counted apart."""
        _write_shard(tmp_path / "a" / "00000.jsonl.gz", 3)
        _write_shard(tmp_path / "a" / "00001.jsonl.gz", 2, start=3)
        _write_shard(tmp_path / "b" / "00000.jsonl.gz", 4)
        assert count_source_documents(tmp_path, workers=1) == {"a": 5, "b": 4}


class TestCachedSourceCounts:
    """A corpus-wide stage's per-source counts are counted once per state of the stage."""

    @staticmethod
    def _stage(root: Path, digest: str) -> Path:
        """Write a stage with one two-document source and a completed sentinel.

        Args:
            root: The stage folder.
            digest: The document digest the sentinel records.

        Returns:
            The stage folder.
        """
        _write_shard(root / "a" / "00000.jsonl.gz", 2)
        payload = {"completed_at": "", "config_hash": "", "config_slice": {}, "records_in": 2, "records_out": 2}
        (root / ".complete").write_text(json.dumps({**payload, "document_digest": digest}), encoding="utf-8")
        return root

    def test_reuses_the_cache_while_the_digest_holds(self, tmp_path: Path) -> None:
        """A cache taken at the stage's current digest is returned without reading shards."""
        stage = self._stage(tmp_path / "stage", "sum256:abc")
        cache = tmp_path / "counts.json"
        cache.write_text(json.dumps({"document_digest": "sum256:abc", "counts": {"a": 99}}), encoding="utf-8")
        assert cached_source_counts(stage, cache, workers=1) == {"a": 99}

    def test_recounts_when_the_digest_changed(self, tmp_path: Path) -> None:
        """A cache from an earlier state of the stage is replaced by a fresh count."""
        stage = self._stage(tmp_path / "stage", "sum256:new")
        cache = tmp_path / "counts.json"
        cache.write_text(json.dumps({"document_digest": "sum256:old", "counts": {"a": 99}}), encoding="utf-8")
        assert cached_source_counts(stage, cache, workers=1) == {"a": 2}
        assert json.loads(cache.read_text(encoding="utf-8"))["document_digest"] == "sum256:new"

    def test_rejects_a_stage_without_a_sentinel(self, tmp_path: Path) -> None:
        """An unfinished stage is not counted."""
        with pytest.raises(FileNotFoundError, match="no completed sentinel"):
            cached_source_counts(tmp_path, tmp_path / "counts.json", workers=1)
