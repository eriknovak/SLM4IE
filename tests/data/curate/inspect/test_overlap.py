"""Tests for the containment audit's overlap measurement over the extracted tier."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import slm4ie.data.curate.inspect.overlap as overlap
from slm4ie.data.curate.inspect.overlap import measure_overlap, normalize_url, sentence_units


def _write(folder: Path, key: str, documents: List[Dict[str, Any]]) -> None:
    """Write *documents* as `extracted/<key>.jsonl`."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"uid": f"{key}:{i}", **doc}) for i, doc in enumerate(documents)]
    (folder / f"{key}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sentence(i: int) -> str:
    """Return a distinct sentence long enough to count as a unit."""
    return f"Ta dolgi stavek številka {i} je dovolj dolg, da se upošteva pri merjenju."


def _rows(tmp_path: Path, **kwargs: Any) -> Dict[tuple, Dict[str, Any]]:
    """Run the measurement over `tmp_path/extracted` and key its rows by pair."""
    rows = measure_overlap(tmp_path / "extracted", out_dir=tmp_path / "out", keep_one_in=1, **kwargs)
    return {(row["a"], row["b"]): row for row in rows}


def test_normalize_url_drops_scheme_www_slash_and_tracking() -> None:
    """Variants of one page normalise to the same string."""
    a = normalize_url("https://WWW.Example.si/Pot/?utm_source=x&id=3#top")
    b = normalize_url("http://example.si/Pot?id=3&fbclid=abc")
    assert a == b == "example.si/Pot?id=3"


def test_sentence_units_ignore_tokenisation_and_short_sentences() -> None:
    """Spacing around punctuation and case do not matter; short sentences are skipped."""
    tokenised = sentence_units("Hvala . Ta stavek je zapisan nekoliko drugače , a je isti .")
    plain = sentence_units("Hvala. TA STAVEK je zapisan nekoliko drugače, a je isti.")
    assert tokenised == plain
    assert len(tokenised) == 1


def test_url_overlap_counts_shared_pages(tmp_path: Path) -> None:
    """URL overlap is the share of each side's distinct URLs found in the other."""
    extracted = tmp_path / "extracted"
    pages = [f"https://site.si/{i}" for i in range(4)]
    _write(extracted, "a", [{"text": "x", "metadata": {"url": url}} for url in pages])
    _write(extracted, "b", [{"text": "x", "metadata": {"u": url}} for url in pages[:1] + ["https://other.si/"]])
    rows = _rows(tmp_path, keys=["a", "b"])
    assert rows[("a", "b")]["shared_urls"] == 1
    assert rows[("a", "b")]["url_share"] == 0.25
    assert rows[("b", "a")]["url_share"] == 0.5


def test_text_overlap_measures_contained_corpus(tmp_path: Path) -> None:
    """A corpus copied into a larger one is fully found there; the larger only partly."""
    extracted = tmp_path / "extracted"
    small = [{"text": f"{_sentence(i)} {_sentence(i + 100)}"} for i in range(5)]
    big = small + [{"text": _sentence(i + 200)} for i in range(10)]
    _write(extracted, "small", small)
    _write(extracted, "big", big)
    rows = _rows(tmp_path, keys=["small", "big"])
    assert rows[("small", "big")]["text_share"] == 1.0
    assert rows[("big", "small")]["text_share"] == 0.5
    assert rows[("small", "big")]["url_share"] is None


def test_document_sample_is_seeded(tmp_path: Path) -> None:
    """The measured side is a seeded document sample of the requested size."""
    extracted = tmp_path / "extracted"
    _write(extracted, "a", [{"text": _sentence(i)} for i in range(200)])
    _write(extracted, "b", [{"text": _sentence(i)} for i in range(200)])
    first = _rows(tmp_path, keys=["a", "b"], sample_docs=50, seed=7)
    second = _rows(tmp_path, keys=["a", "b"], sample_docs=50, seed=7)
    assert first == second
    assert 25 <= first[("a", "b")]["sample_docs"] <= 75
    assert first[("a", "b")]["text_share"] == 1.0


@pytest.mark.parametrize("chunk_bytes", [1, 37, 1 << 20])
def test_byte_ranges_cover_every_line_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_bytes: int) -> None:
    """Splitting a corpus into byte ranges reads each document exactly once."""
    extracted = tmp_path / "extracted"
    _write(extracted, "a", [{"text": _sentence(i), "metadata": {"url": f"https://a.si/{i}"}} for i in range(30)])
    _write(extracted, "b", [{"text": _sentence(i), "metadata": {"url": f"https://a.si/{i}"}} for i in range(10)])
    monkeypatch.setattr(overlap, "CHUNK_BYTES", chunk_bytes)
    rows = _rows(tmp_path, keys=["a", "b"])
    assert rows[("a", "b")]["docs_a"] == 30
    assert rows[("a", "b")]["sample_docs"] == 30
    assert rows[("a", "b")]["urls_a"] == 30
    assert rows[("b", "a")]["text_share"] == 1.0
    assert rows[("a", "b")]["text_share"] == round(10 / 30, 4)
