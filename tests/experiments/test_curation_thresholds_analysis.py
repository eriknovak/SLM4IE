"""Tests for the curation-thresholds experiment's quality re-score in its analysis.py."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest
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


def _strict_filter() -> Any:
    """Build a quality filter needing 5 words and the stopword `in`.

    Returns:
        The filter, its other rules at datatrove's defaults.
    """
    return analysis.LetterShareGopherQualityFilter(
        language=analysis.Languages.slovenian, stop_words=["in"], min_doc_words=5, min_stop_words=1
    )


def test_failing_rules_lists_every_rule_past_the_first() -> None:
    """A short text without stopwords fails the length rule, then the stopword rule."""
    doc = analysis.Document(text="mačka pes", id="a")

    assert analysis.failing_rules(_strict_filter(), doc) == ["min_doc_words", "min_stop_words"]


def test_failing_rules_leaves_the_filter_unchanged() -> None:
    """Switching rules off works on copies, so the filter keeps dropping."""
    quality_filter = _strict_filter()
    analysis.failing_rules(quality_filter, analysis.Document(text="mačka pes", id="a"))

    assert quality_filter.min_doc_words == 5


def test_drop_rule_rows_groups_still_dropped_documents_by_their_failed_rules() -> None:
    """Kept documents are left out; dropped ones are counted per failed set with their target hits."""
    texts = {"a": "mačka pes", "b": "mačka in lisica in medved in", "c": "lisica volk"}
    verdicts = {"a": _verdict(coherence=1), "b": _verdict(), "c": _verdict()}

    rows = analysis.drop_rule_rows(
        {"siparl": dict.fromkeys(texts, "")}, texts, verdicts, {"siparl": _strict_filter()}, {"siparl": "curated_prose"}
    )

    assert [(row["failed_rules"], row["still_dropped"], row["target_hits"]) for row in rows] == [
        ("min_doc_words+min_stop_words", 2, 1)
    ]


def _leaflet(doc_id: str, text_id: str, text: str) -> Dict[str, Any]:
    """Build a medical-source document as a stage shard holds it.

    Args:
        doc_id: The document id.
        text_id: The release's own text id.
        text: The text.

    Returns:
        A document dict.
    """
    return {"id": doc_id, "text": text, "metadata": {"text_id": text_id}}


def test_copy_origin_rows_splits_exact_dedup_drops_by_text_id() -> None:
    """A drop sharing its copy's text id counts apart from one filed under another text id."""
    entering = [
        _leaflet("a", "t1", "navodilo"),
        _leaflet("b", "t1", "navodilo"),
        _leaflet("c", "t2", "navodilo"),
        _leaflet("d", "t3", "drugo"),
    ]

    rows = analysis.copy_origin_rows(entering, {"a", "d"}, {"a", "d"})

    assert [(row["origin"], row["documents"]) for row in rows] == [
        ("dropped: same text under the same text id", 1),
        ("dropped: same text under another text id", 1),
        ("dropped: no copy kept in the source", 0),
        ("kept: one text per record", 2),
        ("kept: a second text of a kept record", 0),
    ]


def test_copy_origin_rows_counts_a_repeated_record_once_under_another_text_id() -> None:
    """Two copies under a text id other than the kept one's are one repeat and one other-id drop."""
    entering = [_leaflet("a", "t1", "navodilo"), _leaflet("b", "t2", "navodilo"), _leaflet("c", "t2", "navodilo")]

    rows = analysis.copy_origin_rows(entering, {"a"}, {"a"})

    assert [row["documents"] for row in rows[:2]] == [1, 1]


def test_copy_origin_rows_counts_a_text_with_no_kept_copy_as_another_sources() -> None:
    """A text exact dedup keeps no copy of in the source counts wholly as kept from another source."""
    entering = [_leaflet("a", "t1", "navodilo"), _leaflet("b", "t2", "navodilo")]

    rows = analysis.copy_origin_rows(entering, set(), set())

    assert [row["documents"] for row in rows[:3]] == [0, 0, 2]


def test_copy_origin_rows_counts_a_second_text_of_one_record_and_whether_both_finish() -> None:
    """Two kept texts under one text id are a second text; it counts as finished only beside its twin."""
    entering = [_leaflet("a", "t1", "'navodilo'"), _leaflet("b", "t1", "‘navodilo’"), _leaflet("c", "t2", "drugo")]

    rows = analysis.copy_origin_rows(entering, {"a", "b", "c"}, {"b", "c"})

    assert [(row["documents"], row["finished_corpus"]) for row in rows[3:]] == [(2, 2), (1, 0)]


def test_judge_input_rows_spreads_calibration_documents_evenly() -> None:
    """Two calibration documents among six fresh ones land after the second and the fourth, ids unchanged."""
    sample = [{"id": f"f{index}", "text": "x"} for index in range(6)]
    calibration = [{"id": "c1", "text": "y"}, {"id": "c2", "text": "z"}]

    rows = analysis.judge_input_rows(sample, calibration)

    assert [row["id"] for row in rows] == ["f0", "f1", "c1", "f2", "f3", "c2", "f4", "f5"]


def test_judge_input_rows_judges_a_document_the_draw_holds_once() -> None:
    """A calibration document already in the draw with the same text is not added again."""
    rows = analysis.judge_input_rows(
        [{"id": "f0", "text": "x"}, {"id": "f1", "text": "y"}], [{"id": "f1", "text": "y"}]
    )

    assert [row["id"] for row in rows] == ["f0", "f1"]


def test_judge_input_rows_rejects_a_shared_id_with_other_text() -> None:
    """A calibration id the draw holds with other text cannot share one verdict."""
    with pytest.raises(ValueError, match="f0"):
        analysis.judge_input_rows([{"id": "f0", "text": "x"}], [{"id": "f0", "text": "old"}])


def test_judge_input_rows_appends_what_a_short_draw_cannot_hold() -> None:
    """A draw shorter than the calibration set still carries every calibration document."""
    rows = analysis.judge_input_rows([{"id": "f0", "text": "x"}], [{"id": "a", "text": "y"}, {"id": "b", "text": "z"}])

    assert [row["id"] for row in rows] == ["f0", "a", "b"]


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


def _label(doc_id: str, coherence: int, language: str = "sl") -> Dict[str, Any]:
    """Build a calibration label or verdict on the rubric for one document.

    Args:
        doc_id: The document id.
        coherence: Coherence on the 1-5 scale.
        language: The language code.

    Returns:
        A label dict.
    """
    return {"id": doc_id, **_verdict(coherence), "language": language}


def test_agreement_rows_gate_passes_when_judge_and_person_agree() -> None:
    """Matching calls at the lenient bar give kappa 1 and pass the gate."""
    labels = [_label("a", 1), _label("b", 5)]
    rows = analysis.agreement_rows(labels, {label["id"]: label for label in labels})
    assert (rows[0]["test"], rows[0]["kappa"], rows[0]["passes_gate"]) == ("bad text", 1.0, True)


def test_agreement_rows_gate_fails_below_the_bar() -> None:
    """A judge that calls bad what the person keeps falls below the gate."""
    labels = [_label("a", 1), _label("b", 5)]
    rows = analysis.agreement_rows(labels, {"a": _label("a", 5), "b": _label("b", 1)})
    assert rows[0]["passes_gate"] is False


def test_agreement_rows_rejects_a_labelled_document_without_a_verdict() -> None:
    """A calibration document the run skipped stops the gate reading."""
    with pytest.raises(KeyError):
        analysis.agreement_rows([_label("a", 1)], {})


def _drops(stage: str, source: str, bad: int, good: int) -> Dict[Any, List[Dict[str, Any]]]:
    """Build one filter's judged drops on one source.

    Args:
        stage: The filter.
        source: The source.
        bad: How many drops are garbage at coherence 1.
        good: How many are coherent prose.

    Returns:
        Verdicts keyed by stage and source, as `judged_drops` gives them.
    """
    return {(stage, source): [_verdict(1, "garbage")] * bad + [_verdict(5)] * good}


def test_clause_rows_confirms_when_every_read_rate_holds() -> None:
    """A clause whose pooled and per-source rates all reach 0.6 is confirmed."""
    precision = analysis.precision_rows(_drops("quality", "a", 20, 5))
    h3 = next(row for row in analysis.clause_rows(precision, True) if row["clause"] == "H3")
    assert (h3["call"], h3["rates_read"]) == ("confirmed", 2)


def test_clause_rows_refutes_when_one_source_fails() -> None:
    """One read source below 0.5 refutes the clause even when the pool holds."""
    precision = analysis.precision_rows({**_drops("quality", "a", 60, 0), **_drops("quality", "b", 5, 20)})
    h3 = next(row for row in analysis.clause_rows(precision, True) if row["clause"] == "H3")
    assert (h3["call"], h3["lowest_at"]) == ("refuted", "quality / b")


def test_clause_rows_leaves_a_filter_without_drops_inconclusive() -> None:
    """A filter with no judged drops has nothing to confirm."""
    h1 = next(row for row in analysis.clause_rows(analysis.precision_rows({}), True) if row["clause"] == "H1")
    assert (h1["call"], h1["rates_read"]) == ("inconclusive", 0)


def _funnel(source: str, dropped: int, stage: str = "quality") -> Dict[str, Any]:
    """Build one source's funnel row in which only one filter drops documents.

    Args:
        source: The source.
        dropped: How many documents the filter drops out of 1,000.
        stage: The filter that drops them.

    Returns:
        A row shaped as `funnel_rows` gives it.
    """
    order = ["convert", "language", "spam", "quality", "repetition"]
    after = order.index(stage)
    return {"source": source, **{name: 1000 - dropped * (i >= after) for i, name in enumerate(order)}}


def _trade(rows: List[Dict[str, Any]], stage: str, source: str) -> Dict[str, Any]:
    """Pick one row of the trade-off table.

    Args:
        rows: Rows from `trade_off_rows`.
        stage: The filter.
        source: The source, or `pooled`.

    Returns:
        The matching row.
    """
    return next(row for row in rows if (row["stage"], row["source"]) == (stage, source))


def test_trade_off_rows_scales_the_judged_share_to_the_source_drops() -> None:
    """Three in four judged drops bad turns 400 drops into 300 bad removed and 100 good lost."""
    rows = analysis.trade_off_rows([_funnel("a", 400)], _drops("quality", "a", 15, 5), True)
    row = _trade(rows, "quality", "a")
    assert (row["dropped"], row["bad_removed"], row["good_lost"], row["call"]) == (400, 300, 100, "keep")


def test_trade_off_rows_switches_off_where_less_than_half_is_bad() -> None:
    """A filter on a source dropping more good than bad text is called off at the 1:1 rate."""
    rows = analysis.trade_off_rows([_funnel("a", 400)], _drops("quality", "a", 9, 11), True)
    assert _trade(rows, "quality", "a")["call"] == "switch off"


def test_trade_off_rows_leaves_a_source_with_few_judged_drops_uncalled() -> None:
    """Below 20 judged drops the estimate is listed without a call."""
    rows = analysis.trade_off_rows([_funnel("a", 400)], _drops("quality", "a", 15, 4), True)
    assert _trade(rows, "quality", "a")["call"] == ""


def test_trade_off_rows_pools_by_volume_not_by_judged_drops() -> None:
    """The pooled share weighs each source by what it drops, not by how many drops were judged."""
    drops = {**_drops("quality", "a", 20, 0), **_drops("quality", "b", 0, 20)}
    rows = analysis.trade_off_rows([_funnel("a", 100), _funnel("b", 300)], drops, True)
    assert _trade(rows, "quality", "pooled")["bad_share"] == 0.25


def test_trade_off_rows_gives_the_pooled_row_no_call() -> None:
    """The 1:1 call is made per source only (D12)."""
    rows = analysis.trade_off_rows([_funnel("a", 400)], _drops("quality", "a", 20, 0), True)
    assert _trade(rows, "quality", "pooled")["call"] == ""


def test_trade_off_rows_counts_a_language_drop_as_bad_only_when_not_slovene() -> None:
    """Slovene garbage dropped by the language filter is good text lost, by the filter's own target (D1)."""
    verdicts = [{**_verdict(1, "garbage"), "language": "sl"}] * 10 + [{**_verdict(), "language": "hr"}] * 30
    rows = analysis.trade_off_rows([_funnel("a", 400, "language")], {("language", "a"): verdicts}, True)
    assert _trade(rows, "language", "a")["bad_removed"] == 300
