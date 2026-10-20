"""Tests for the curation-thresholds experiment's quality re-score in its analysis.py."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

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


def test_candidate_filters_apply_the_route_quality_settings(tmp_path: Path) -> None:
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

    filters = analysis.candidate_filters(config, ["oss", "c4"], "quality")

    settings = [(f.max_non_alpha_words_ratio, f.alpha_words_skip_punctuation) for f in (filters["oss"], filters["c4"])]
    assert settings == [(0.5, True), (0.8, False)]


def test_candidate_filters_apply_the_route_repetition_settings(tmp_path: Path) -> None:
    """A source's repetition filter carries its route's limits, and another source keeps the defaults."""
    config = tmp_path / "routes.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "stopwords": "sl",
                "repetition": {},
                "profiles": {"loose": {"repetition": {"dup_line_frac": 0.6, "top_n_grams": [[2, 0.3]]}}},
                "overrides": {"oss": {"profile": "loose"}},
            }
        ),
        encoding="utf-8",
    )

    filters = analysis.candidate_filters(config, ["oss", "c4"], "repetition")

    settings = [(f.dup_line_frac, f.top_n_grams[0]) for f in (filters["oss"], filters["c4"])]
    assert settings == [(0.6, (2, 0.3)), (0.3, (2, 0.2))]


def _line_filter() -> Any:
    """Build a repetition filter checking only duplicate lines and the top 2-gram.

    Returns:
        The filter.
    """
    return analysis.GopherRepetitionFilter(
        dup_line_frac=0.3,
        dup_para_frac=None,
        dup_line_char_frac=None,
        dup_para_char_frac=None,
        top_n_grams=((2, 0.2),),
        dup_n_grams=(),
        language=analysis.Languages.slovenian,
    )


def test_failing_rules_switches_off_repetition_rules_one_by_one() -> None:
    """A text repeating one line fails the duplicate-line rule, then the top n-gram family."""
    doc = analysis.Document(text="mačka pes\nmačka pes\nmačka pes\nlisica", id="a")

    assert analysis.failing_rules(_line_filter(), doc) == ["dup_line_frac", "top_n_grams"]


def test_rule_trade_off_rows_scales_each_rule_set_to_the_source_drops() -> None:
    """Two of four judged drops failing only the length rule stand for half of the 400 the source drops."""
    texts = {"a": "mačka pes", "b": "lisica volk", "c": "mačka", "d": "pes"}
    verdicts = {"a": _verdict(coherence=1), "b": _verdict(), "c": _verdict(), "d": _verdict()}
    quality_filter = analysis.LetterShareGopherQualityFilter(
        language=analysis.Languages.slovenian, stop_words=["in"], min_doc_words=5, min_stop_words=0
    )

    rows = analysis.rule_trade_off_rows(
        "quality",
        {"a": dict.fromkeys(texts, "")},
        texts,
        verdicts,
        {"a"},
        {"a": quality_filter},
        [_funnel("a", 400)],
        True,
    )

    row = rows[0]
    assert (row["failed_rules"], row["judged"], row["dropped"], row["bad_removed"]) == ("min_doc_words", 4, 400, 100)


def test_rule_trade_off_rows_leaves_out_what_the_candidate_filter_keeps() -> None:
    """An earlier audit's drop the candidate filter keeps is not one of its drops."""
    texts = {"a": "mačka pes", "b": "mačka in lisica in medved"}
    rows = analysis.rule_trade_off_rows(
        "quality",
        {"a": dict.fromkeys(texts, "")},
        texts,
        {"a": _verdict(), "b": _verdict()},
        set(),
        {"a": _strict_filter()},
        [_funnel("a", 400)],
        True,
    )

    assert [(row["judged"], row["dropped"]) for row in rows] == [(1, 400)]


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


def test_merge_drops_lets_the_later_sample_win_a_shared_id() -> None:
    """A document in both samples keeps the later sample's opening; others are pooled per source."""
    merged = analysis.merge_drops({"c4": {"a": "old", "b": "b"}}, {"c4": {"a": "new"}, "hplt": {"c": "c"}})
    assert merged == {"c4": {"a": "new", "b": "b"}, "hplt": {"c": "c"}}


def test_read_setting_changes_adds_every_change_of_a_route_together(tmp_path: Path) -> None:
    """Each change stands alone, and the route's changes merge per filter under one more variant."""
    path = tmp_path / "changes.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "changes": {
                    "raw": {
                        "short": {"stage": "quality", "settings": {"min_doc_words": 10}},
                        "stop": {"stage": "quality", "settings": {"min_stop_words": 0}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    variants = analysis.read_setting_changes(path)

    assert variants["raw"][analysis.ALL_CHANGES] == {"quality": {"min_doc_words": 10, "min_stop_words": 0}}


def test_read_setting_changes_rejects_a_filter_without_rules(tmp_path: Path) -> None:
    """Only the quality and repetition filters have settings a change can loosen."""
    path = tmp_path / "changes.yaml"
    path.write_text(
        yaml.safe_dump({"changes": {"raw": {"off": {"stage": "spam", "settings": {"enabled": False}}}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'spam'"):
        analysis.read_setting_changes(path)


def test_rescue_outcomes_keeps_a_quality_drop_only_if_repetition_passes_it(tmp_path: Path) -> None:
    """Lifting the length floor rescues a short text, but not one the repetition filter then drops."""
    config = tmp_path / "routes.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "stopwords": "sl",
                "quality": {"min_doc_words": 5, "min_stop_words": 0},
                "repetition": {"top_n_grams": [], "dup_n_grams": []},
                "profiles": {"raw": {}},
                "overrides": {"c4": {"profile": "raw"}},
            }
        ),
        encoding="utf-8",
    )
    texts = {"a": "mačka in pes", "b": "pes\npes\npes"}
    variants = {"raw": {"short": {"quality": {"min_doc_words": 0}}}}

    outcomes = analysis.rescue_outcomes(
        config, variants, {"c4": "raw"}, {"quality": {"c4": dict.fromkeys(texts, "")}}, {"quality": texts}, 1
    )

    assert outcomes == {("quality", "c4", "a"): {"short": True}, ("quality", "c4", "b"): {"short": False}}


def _outcomes(kept_bad: int, kept_good: int, dropped: int) -> Dict[Any, Dict[str, bool]]:
    """Build one source's quality drops and whether a change keeps each.

    Args:
        kept_bad: Bad drops the change keeps.
        kept_good: Good drops the change keeps.
        dropped: Good drops it still drops.

    Returns:
        Outcomes as `rescue_outcomes` gives them, for source `a`, ids prefixed by their group.
    """
    groups = {"bad": (kept_bad, True), "good": (kept_good, True), "still": (dropped, False)}
    return {("quality", "a", f"{name}{i}"): {"x": kept} for name, (count, kept) in groups.items() for i in range(count)}


def _change_verdicts(outcomes: Dict[Any, Dict[str, bool]]) -> Dict[str, Dict[str, Any]]:
    """Give each id from `_outcomes` a verdict, bad for the `bad` group.

    Args:
        outcomes: From `_outcomes`.

    Returns:
        Verdicts by document id.
    """
    return {doc_id: _verdict(1, "garbage") if doc_id.startswith("bad") else _verdict() for _, _, doc_id in outcomes}


def test_setting_change_rows_scales_what_a_change_keeps_to_the_source_drops() -> None:
    """One bad and one good kept of four judged drops stand for 100 bad and 100 good of 400."""
    outcomes = _outcomes(1, 1, 2)
    rows = analysis.setting_change_rows(outcomes, {"a": "raw"}, _change_verdicts(outcomes), [_funnel("a", 400)], True)

    row = next(row for row in rows if row["source"] == "a")
    assert (row["rescued"], row["bad_gained"], row["good_gained"]) == (200, 100, 100)


def test_setting_change_rows_adopts_a_route_keeping_more_good_than_bad() -> None:
    """A route's pooled change keeping 20 judged drops, 5 of them bad, is adopted at the 1:1 rate."""
    outcomes = _outcomes(5, 15, 20)
    rows = analysis.setting_change_rows(outcomes, {"a": "raw"}, _change_verdicts(outcomes), [_funnel("a", 400)], True)

    assert [(row["source"], row["call"]) for row in rows] == [("pooled", "adopt"), ("a", "")]


def test_setting_change_rows_rejects_a_route_keeping_as_much_bad_as_good() -> None:
    """At half bad a change gains nothing at the 1:1 rate, so the filter stays as built."""
    outcomes = _outcomes(10, 10, 20)
    rows = analysis.setting_change_rows(outcomes, {"a": "raw"}, _change_verdicts(outcomes), [_funnel("a", 400)], True)

    assert rows[0]["call"] == "reject"


def test_read_setting_changes_rejects_a_setting_the_filter_lacks(tmp_path: Path) -> None:
    """A misspelt knob would silently change nothing, so it stops the run."""
    path = tmp_path / "changes.yaml"
    path.write_text(
        yaml.safe_dump({"changes": {"raw": {"short": {"stage": "quality", "settings": {"min_words": 10}}}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_words"):
        analysis.read_setting_changes(path)


def test_read_setting_changes_tries_a_lone_change_only_once(tmp_path: Path) -> None:
    """A route with one change gets no second variant repeating it."""
    path = tmp_path / "changes.yaml"
    path.write_text(
        yaml.safe_dump({"changes": {"med": {"off": {"stage": "repetition", "settings": {"enabled": False}}}}}),
        encoding="utf-8",
    )

    assert list(analysis.read_setting_changes(path)["med"]) == ["off"]


def _predicted(route: str, change: str, source: str, good: int, bad: int) -> Dict[str, Any]:
    """Build one row of the step-10 estimate, as `setting_change_rows` gives it.

    Args:
        route: The route.
        change: The setting change, or `ALL_CHANGES`.
        source: The source, or `pooled`.
        good: Good text the change was predicted to keep.
        bad: Bad text it was predicted to keep.

    Returns:
        A row carrying the columns `gain_rows` reads.
    """
    return {"route": route, "change": change, "source": source, "good_gained": good, "bad_gained": bad}


#: One route, `raw`, whose one change loosens the quality filter.
_QUALITY_CHANGED = {"raw": {"looser": {"quality": {"min_doc_words": 10}}}}


def _gains(
    frozen: List[Dict[str, Any]],
    loosened: List[Dict[str, Any]],
    frozen_drops: Dict[Any, List[Dict[str, Any]]],
    loosened_drops: Dict[Any, List[Dict[str, Any]]],
    variants: Dict[str, Any] = _QUALITY_CHANGED,
    predicted: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run `gain_rows` with every source on route `raw`.

    Args:
        frozen: Funnel rows of the frozen build.
        loosened: Funnel rows of the looser build.
        frozen_drops: Judged drops of the frozen build.
        loosened_drops: Judged drops of the looser build.
        variants: The setting changes per route.
        predicted: Step-10 rows; none by default.

    Returns:
        The rows `gain_rows` gives.
    """
    routes = {row["source"]: "raw" for row in frozen}
    return analysis.gain_rows(frozen, loosened, frozen_drops, loosened_drops, routes, variants, predicted or [], True)


def test_gain_rows_counts_what_the_frozen_build_drops_and_the_looser_one_keeps() -> None:
    """400 frozen drops at 3/4 bad minus 100 looser drops all bad gains 300 documents, 200 of them bad."""
    rows = _gains(
        [_funnel("a", 400)], [_funnel("a", 100)], _drops("quality", "a", 15, 5), _drops("quality", "a", 20, 0)
    )
    source_row = next(row for row in rows if row["source"] == "a")

    assert (source_row["gained"], source_row["bad_gained"], source_row["good_gained"]) == (300, 200, 100)


def test_gain_rows_reads_the_repetition_filter_against_the_quality_output() -> None:
    """A repetition filter dropping 400 of the quality filter's output gains those 400 when switched off."""
    variants = {"raw": {"off": {"repetition": {"enabled": False}}}}
    rows = _gains([_funnel("a", 400, "repetition")], [_funnel("a", 0)], _drops("repetition", "a", 0, 20), {}, variants)

    assert (rows[0]["gained"], rows[0]["good_gained"]) == (400, 400)


def test_gain_rows_gives_bad_text_an_interval_around_the_estimate() -> None:
    """All 20 judged drops good still leaves room for some bad text, from the adjusted share's variance."""
    rows = _gains([_funnel("a", 400)], [_funnel("a", 0)], _drops("quality", "a", 0, 20), {})

    assert rows[0]["bad_gained_low"] < rows[0]["bad_gained"] == 0 < rows[0]["bad_gained_high"]


def test_gain_rows_lets_a_source_estimate_fall_below_zero_by_chance() -> None:
    """A looser draw with a higher bad share than the frozen one gives negative bad text, left uncut."""
    rows = _gains(
        [_funnel("a", 200)], [_funnel("a", 100)], _drops("quality", "a", 0, 20), _drops("quality", "a", 20, 0)
    )
    source_row = next(row for row in rows if row["source"] == "a")

    assert source_row["bad_gained"] == -100 and source_row["bad_gained_low"] < -100 < source_row["bad_gained_high"]


def test_gain_rows_pools_each_route_by_volume() -> None:
    """A route's pooled row sums its sources' documents and bad text gained."""
    drops = {**_drops("quality", "a", 20, 0), **_drops("quality", "b", 0, 20)}
    rows = _gains([_funnel("a", 100), _funnel("b", 300)], [_funnel("a", 0), _funnel("b", 0)], drops, {})

    assert (rows[0]["source"], rows[0]["gained"], rows[0]["bad_gained"]) == ("pooled", 400, 100)


def test_gain_rows_skips_a_filter_its_route_leaves_unchanged() -> None:
    """Quality drops alike in both builds add nothing, even with no judged drop to score them."""
    variants = {"raw": {"off": {"repetition": {"enabled": False}}}}
    rows = _gains([_funnel("a", 100)], [_funnel("a", 100)], {}, {}, variants)

    assert (rows[0]["gained"], rows[0]["bad_gained"]) == (0, 0)


def test_gain_rows_rejects_an_unchanged_filter_dropping_differently() -> None:
    """A filter whose settings did not change must drop the same documents in both builds."""
    variants = {"raw": {"off": {"repetition": {"enabled": False}}}}

    with pytest.raises(ValueError, match="unchanged"):
        _gains([_funnel("a", 100)], [_funnel("a", 50)], {}, {}, variants)


def test_gain_rows_sets_the_estimate_with_every_change_of_the_route_beside_it() -> None:
    """Where a route tried several changes, the prediction is the one applying them all."""
    variants = {
        "raw": {
            "one": {"quality": {"min_doc_words": 10}},
            "two": {"quality": {"min_stop_words": 0}},
            analysis.ALL_CHANGES: {"quality": {"min_doc_words": 10, "min_stop_words": 0}},
        }
    }
    predicted = [
        _predicted("raw", "one", "pooled", 5, 1),
        _predicted("raw", analysis.ALL_CHANGES, "pooled", 50, 10),
    ]
    rows = _gains([_funnel("a", 100)], [_funnel("a", 0)], _drops("quality", "a", 20, 0), {}, variants, predicted)

    assert (rows[0]["predicted_good_gained"], rows[0]["predicted_bad_gained"]) == (50, 10)


def test_gain_rows_rejects_several_changes_without_one_applying_them_all() -> None:
    """With no variant applying every change, no single estimate stands for the looser build."""
    variants = {"raw": {"one": {"quality": {"min_doc_words": 10}}, "two": {"quality": {"min_stop_words": 0}}}}

    with pytest.raises(ValueError, match="no 'all changes together' variant"):
        _gains([_funnel("a", 100)], [_funnel("a", 0)], _drops("quality", "a", 20, 0), {}, variants)


def test_gain_rows_rejects_builds_feeding_different_documents_to_the_quality_filter() -> None:
    """Gains read as drop differences only when both builds start the filters from the same documents."""
    looser = {**_funnel("a", 0), "spam": 999}

    with pytest.raises(ValueError, match="different count"):
        _gains([_funnel("a", 100)], [looser], _drops("quality", "a", 20, 0), {})


def test_gain_rows_rejects_drops_with_no_judged_document() -> None:
    """A changed filter that drops documents nobody judged leaves its bad share unknown, so the run stops."""
    with pytest.raises(ValueError, match="none of them was judged"):
        _gains([_funnel("a", 100)], [_funnel("a", 0)], {}, {})


def _kept(bad: int, good: int) -> List[Dict[str, Any]]:
    """Build one source's judged kept documents.

    Args:
        bad: How many are garbage at coherence 1.
        good: How many are coherent prose.

    Returns:
        Verdicts as `judged_kept` lists them for one source.
    """
    return [_verdict(1, "garbage")] * bad + [_verdict(5)] * good


def test_judged_kept_reads_only_documents_kept_at_the_last_filter() -> None:
    """A document drawn as kept by quality or dropped by repetition is no part of the kept corpus."""
    sample = [
        {"id": "k", "dataset": "a", "cells": [{"stage": "repetition", "decision": "kept"}]},
        {"id": "q", "dataset": "a", "cells": [{"stage": "quality", "decision": "kept"}]},
        {"id": "d", "dataset": "a", "cells": [{"stage": "repetition", "decision": "dropped"}]},
    ]
    verdicts = {doc_id: _verdict() for doc_id in ("k", "q", "d")}

    assert analysis.judged_kept(sample, verdicts, {"a"}) == {"a": [_verdict()]}


def test_judged_kept_rejects_a_kept_document_without_a_verdict() -> None:
    """An unjudged kept document would bias the source's share, so the run stops."""
    sample = [{"id": "k", "dataset": "a", "cells": [{"stage": "repetition", "decision": "kept"}]}]

    with pytest.raises(KeyError, match="no verdict"):
        analysis.judged_kept(sample, {}, {"a"})


def test_residual_rows_weighs_each_source_by_what_it_keeps() -> None:
    """A clean source keeping 900 beside one keeping 100 at half bad gives a pooled rate of 0.05."""
    builds = {"baseline": ([_funnel("a", 100), _funnel("b", 900)], {"a": _kept(0, 40), "b": _kept(20, 20)})}
    pooled = analysis.residual_rows(builds)[0]

    assert (pooled["source"], pooled["kept"], pooled["bad_rate"], pooled["good_kept"]) == ("pooled", 1000, 0.05, 950)


def test_residual_rows_rejects_a_source_keeping_documents_nobody_judged() -> None:
    """A kept corpus share with no judged document has no rate, so the run stops."""
    with pytest.raises(ValueError, match="none was judged"):
        analysis.residual_rows({"baseline": ([_funnel("a", 0)], {})})


def test_residual_rise_rows_meets_the_bar_with_more_good_text_and_no_rise() -> None:
    """Keeping twice the documents at the baseline's rate answers the amended question yes (D17)."""
    builds = {
        "baseline": ([_funnel("a", 500)], {"a": _kept(4, 36)}),
        "loosened": ([_funnel("a", 0)], {"a": _kept(4, 36)}),
    }
    row = analysis.residual_rise_rows(builds, {"loosened": False})[0]

    assert (row["bad_rate_rise"], row["good_kept_gained"], row["meets_bar"]) == (0.0, 450, True)
    assert row["bad_rate_rise_low"] < 0 < row["bad_rate_rise_high"]


def test_residual_rise_rows_meets_the_bar_while_the_interval_reaches_below_two_points() -> None:
    """A rise of four points on a small draw still meets the bar, since its interval reaches under two (D17)."""
    builds = {
        "baseline": ([_funnel("a", 500)], {"a": _kept(4, 36)}),
        "loosened": ([_funnel("a", 0)], {"a": _kept(14, 86)}),
    }
    row = analysis.residual_rise_rows(builds, {"loosened": False})[0]

    assert (row["bad_rate_rise"], row["meets_bar"]) == (0.04, True)
    assert row["bad_rate_rise_low"] <= analysis.MAX_BAD_RATE_RISE


def test_residual_rise_rows_misses_the_bar_when_the_whole_interval_lies_above_two_points() -> None:
    """A rise whose interval sits wholly above two points misses the bar however much good text is kept."""
    builds = {
        "baseline": ([_funnel("a", 500)], {"a": _kept(40, 960)}),
        "loosened": ([_funnel("a", 0)], {"a": _kept(200, 800)}),
    }
    row = analysis.residual_rise_rows(builds, {"loosened": False})[0]

    assert row["bad_rate_rise_low"] > analysis.MAX_BAD_RATE_RISE and not row["meets_bar"]
