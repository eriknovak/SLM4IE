"""Tests for the curation-thresholds experiment's quality re-score in its analysis.py."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import yaml

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "data" / "curation-thresholds-slovenian" / "analysis.py"


def _load() -> ModuleType:
    """Import the experiment's analysis.py, whose folder name is no package name.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("curation_thresholds_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered so a process pool can pickle the script's worker functions by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load()


def _verdict(coherence: int = 5, text_type: str = "prose") -> Dict[str, Any]:
    """Build a judge verdict on the rubric.

    Args:
        coherence: Coherence on the 1-5 scale.
        text_type: The text shape.

    Returns:
        A verdict dict.
    """
    return {"coherence": coherence, "text_type": text_type, "adult_or_spam": False}


class _KeepIfMarked:
    """Stand-in quality filter that keeps documents containing `keep`."""

    def filter(self, doc: Any) -> Any:
        """Keep a document whose text contains `keep`.

        Args:
            doc: The document.

        Returns:
            True to keep, else a drop reason.
        """
        return True if "keep" in doc.text else (False, "gopher_below_alpha_threshold")


def _write_shard(stage: Path, source: str, documents: List[Dict[str, Any]], compact: bool) -> None:
    """Write one source's shard and sentinel, compact as datatrove writes or spaced as convert does.

    Args:
        stage: The stage folder.
        source: The source subfolder.
        documents: The documents, in order.
        compact: Whether to drop the spaces after separators.
    """
    (stage / source).mkdir(parents=True)
    (stage / source / ".complete").write_text(json.dumps({"document_digest": "d1"}), encoding="utf-8")
    separators = (",", ":") if compact else (", ", ": ")
    with gzip.open(stage / source / "00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(doc, separators=separators) + "\n" for doc in documents)


def test_refilter_splits_rescued_from_still_dropped() -> None:
    """A kept drop is rescued and a dropped one stays dropped."""
    verdicts = {"a": _verdict(text_type="garbage"), "b": _verdict(coherence=1)}

    rows = analysis.refilter_rows(
        {"c4": {"a": "", "b": ""}}, {"a": "keep me", "b": "drop me"}, set(), verdicts, {"c4": _KeepIfMarked()}, {}
    )

    source = next(row for row in rows if row["source"] == "c4")
    assert (source["rescued"], source["still_dropped"]) == (1, 1)


def test_refilter_counts_changed_text_apart_from_absent_documents() -> None:
    """A document whose text changed and an absent one never reach the filter, counted apart."""
    verdicts = {"a": _verdict(), "b": _verdict(), "c": _verdict()}

    rows = analysis.refilter_rows(
        {"c4": {"a": "", "b": "", "c": ""}}, {"a": "keep"}, {"b"}, verdicts, {"c4": _KeepIfMarked()}, {}
    )

    source = next(row for row in rows if row["source"] == "c4")
    assert (source["reach_quality"], source["text_changed"], source["absent"]) == (1, 1, 1)


def test_refilter_pools_each_route_and_all_sources() -> None:
    """The route row pools its sources and the overall row pools every source."""
    openings = {"c4": {"a": ""}, "hplt": {"b": ""}, "kas": {"c": ""}}
    texts = {"a": "drop", "b": "drop", "c": "drop"}
    routes = {"c4": "raw_crawl", "hplt": "raw_crawl", "kas": "medical_academic"}
    filters = dict.fromkeys(openings, _KeepIfMarked())

    rows = analysis.refilter_rows(openings, texts, set(), {key: _verdict() for key in "abc"}, filters, routes)

    pooled = {row["route"]: row["still_dropped"] for row in rows if row["source"] == "pooled"}
    assert pooled == {"all": 3, "medical_academic": 1, "raw_crawl": 2}


def test_refilter_rescues_everything_where_the_filter_is_off() -> None:
    """A source whose route switches the quality filter off keeps every drop."""
    rows = analysis.refilter_rows({"kzb": {"a": ""}}, {"a": "drop"}, set(), {"a": _verdict()}, {"kzb": None}, {})

    assert next(row for row in rows if row["source"] == "kzb")["rescued"] == 1


def test_fetch_full_texts_reads_compact_and_spaced_shards(tmp_path: Path) -> None:
    """Ids are found in datatrove's compact lines and in convert's spaced, escaped ones."""
    stage = tmp_path / "02_spam"
    _write_shard(stage, "c4", [{"text": 'quoted ","id":"x" inside', "id": "a", "metadata": {}}], compact=True)
    _write_shard(stage, "solar", [{"text": "Kakšna je, več", "id": "solar:š1", "dataset": "solar"}], compact=False)

    texts, _ = analysis.fetch_full_texts(
        stage, {"c4": {"a": "quoted"}, "solar": {"solar:š1": "Kakšna"}}, tmp_path / "c.json", 1
    )

    assert texts == {"a": 'quoted ","id":"x" inside', "solar:š1": "Kakšna je, več"}


def test_fetch_full_texts_keeps_the_judged_text_and_flags_a_changed_one(tmp_path: Path) -> None:
    """A repeated id keeps the text the judge read; an id seen only with other text is flagged."""
    stage = tmp_path / "02_spam"
    documents = [
        {"text": "other opening", "id": "b", "metadata": {}},
        {"text": "judged opening, then more", "id": "b", "metadata": {}},
        {"text": "rewritten text", "id": "c", "metadata": {}},
    ]
    _write_shard(stage, "c4", documents, compact=True)

    texts, changed = analysis.fetch_full_texts(
        stage, {"c4": {"b": "judged opening", "c": "original"}}, tmp_path / "c.json", 1
    )

    assert (texts, changed) == ({"b": "judged opening, then more"}, {"c"})


def test_fetch_full_texts_rereads_when_the_wanted_documents_change(tmp_path: Path) -> None:
    """A cache taken for other documents is not reused."""
    stage = tmp_path / "02_spam"
    _write_shard(stage, "c4", [{"text": "one", "id": "a"}, {"text": "two", "id": "b"}], compact=True)
    analysis.fetch_full_texts(stage, {"c4": {"a": "one"}}, tmp_path / "c.json", 1)

    texts, _ = analysis.fetch_full_texts(stage, {"c4": {"a": "one", "b": "two"}}, tmp_path / "c.json", 1)

    assert texts == {"a": "one", "b": "two"}


def test_quality_filters_apply_the_route_settings(tmp_path: Path) -> None:
    """A source's filter carries its route's quality settings, the letter-rule repair included."""
    loose = {"max_non_alpha_words_ratio": 0.5, "alpha_words_skip_punctuation": True}
    config = tmp_path / "routes.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "stopwords": "sl",
                "quality": {"max_non_alpha_words_ratio": 0.8},
                "profiles": {"loose": {"quality": loose}},
                "overrides": {"oss": {"profile": "loose"}},
            }
        ),
        encoding="utf-8",
    )

    filters = analysis.quality_filters(config, ["oss", "c4"])

    settings = [(f.max_non_alpha_words_ratio, f.alpha_words_skip_punctuation) for f in (filters["oss"], filters["c4"])]
    assert settings == [(0.5, True), (0.8, False)]
