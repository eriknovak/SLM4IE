"""Tests for slm4ie.data.curate.stats.CorpusStats."""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
import json
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("datatrove")

from datatrove.data import Document  # noqa: E402

from slm4ie.data.curate.stats import CorpusStats  # noqa: E402


def _doc(text: str, *, dataset: str, domain: str, doc_id: str) -> Document:
    """Build a Document carrying the metadata CorpusStats reads."""
    return Document(text=text, id=doc_id, metadata={"dataset": dataset, "domain": domain})


def _run(stats: CorpusStats, docs: List[Document]) -> None:
    """Drive `stats.run` to exhaustion so the bundle is written to disk."""
    for _ in stats.run(iter(docs)):
        pass


class TestCorpusStats:
    """Aggregation behavior of the corpus-stats block."""

    def test_per_domain_and_per_dataset_word_counts(self, tmp_path: Path) -> None:
        """Grouped word counts match a hand-summed expectation."""
        out = tmp_path / "agg.json"
        stats = CorpusStats(output_path=out, stopwords=set(), top_k_words=100)
        docs = [
            _doc(
                "Slovenščina je uradni jezik Republike Slovenije.",
                dataset="kzb", domain="scientific", doc_id="a1",
            ),
            _doc(
                "Pravna pravila sledijo zakonu in ustavnim načelom.",
                dataset="coleslaw", domain="legal", doc_id="b1",
            ),
            _doc(
                "Akademske raziskave preučujejo družbene vzorce in jezikovne pojave.",
                dataset="kzb", domain="scientific", doc_id="a2",
            ),
        ]
        _run(stats, docs)
        bundle = json.loads(out.read_text(encoding="utf-8"))

        assert bundle["total_docs"] == 3
        assert bundle["total_words"] > 0

        scientific = bundle["by_domain"]["scientific"]
        legal = bundle["by_domain"]["legal"]
        assert scientific["doc_count"] == 2
        assert legal["doc_count"] == 1
        assert scientific["word_count"] + legal["word_count"] == bundle["total_words"]

        kzb = bundle["by_dataset"]["kzb"]
        coleslaw = bundle["by_dataset"]["coleslaw"]
        assert kzb["doc_count"] == 2
        assert coleslaw["doc_count"] == 1
        # Per-domain and per-dataset words must agree on the total.
        assert kzb["word_count"] + coleslaw["word_count"] == bundle["total_words"]

    def test_word_freq_table_excludes_stopwords_and_short_tokens(self, tmp_path: Path) -> None:
        """Stopwords and sub-`_MIN_WORD_LEN` tokens are dropped from the table."""
        out = tmp_path / "agg.json"
        stats = CorpusStats(output_path=out, stopwords={"in"}, top_k_words=50)
        _run(
            stats,
            [
                _doc(
                    "jezik in jezik a in jezik besedilo",
                    dataset="kzb", domain="scientific", doc_id="x1",
                ),
            ],
        )
        bundle = json.loads(out.read_text(encoding="utf-8"))
        table = dict(bundle["word_freq_top_50"])
        assert table["jezik"] == 3
        assert "besedilo" in table
        assert "in" not in table  # stopword
        assert "a" not in table  # shorter than _MIN_WORD_LEN

    def test_share_of_total_words_sums_to_one(self, tmp_path: Path) -> None:
        """Per-domain shares sum to 1 (within float tolerance)."""
        out = tmp_path / "agg.json"
        stats = CorpusStats(output_path=out, stopwords=set())
        _run(
            stats,
            [
                _doc("ena dva tri", dataset="d1", domain="A", doc_id="1"),
                _doc("štiri pet šest sedem", dataset="d2", domain="B", doc_id="2"),
                _doc("osem devet deset", dataset="d3", domain="A", doc_id="3"),
            ],
        )
        bundle = json.loads(out.read_text(encoding="utf-8"))
        total_share = sum(b["share_of_total_words"] for b in bundle["by_domain"].values())
        assert abs(total_share - 1.0) < 1e-9
