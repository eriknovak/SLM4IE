"""Tests for the KPI-coverage experiment's per-stage funnel in its analysis.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "data" / "kpi-coverage-slovenian" / "analysis.py"


def _load() -> ModuleType:
    """Import the experiment's analysis.py, whose folder name is no package name.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("kpi_coverage_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load()

#: Documents each fixture source writes at the five per-source stages, then after both dedups.
_COUNTS: Dict[str, Tuple[int, ...]] = {
    "web_a": (100, 90, 80, 70, 60, 50, 40),
    "web_b": (10, 10, 9, 8, 8, 6, 5),
    "med": (20, 20, 20, 18, 18, 9, 3),
}


def _sentinel(folder: Path, records_in: int, records_out: int) -> None:
    """Write a stage sentinel with its document counts.

    Args:
        folder: The unit folder.
        records_in: Documents the unit read.
        records_out: Documents the unit wrote.
    """
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"records_in": records_in, "records_out": records_out, "document_digest": "sum256:abc"}
    (folder / ".complete").write_text(json.dumps(payload), encoding="utf-8")


def _build(root: Path) -> Path:
    """Write a fixture build: per-source sentinels, dedup sentinels and final statistics.

    Args:
        root: Folder to build in.

    Returns:
        The build folder.
    """
    for source, counts in _COUNTS.items():
        previous = counts[0]
        for stage, written in zip(analysis.SCOPED_STAGES, counts[:5]):
            _sentinel(root / stage / source, previous, written)
            previous = written
        per_dataset = root / "07_statistics" / "per_dataset"
        per_dataset.mkdir(parents=True, exist_ok=True)
        final = {
            "dataset": source,
            "domain": _SOURCES[source]["domain"],
            "doc_count": counts[6],
            "word_count": 10 * counts[6],
        }
        (per_dataset / f"{source}.json").write_text(json.dumps(final), encoding="utf-8")
    exact_in, exact_out, final_out = (sum(counts[index] for counts in _COUNTS.values()) for index in (4, 5, 6))
    _sentinel(root / analysis.COUNTED_STAGE, exact_in, exact_out)
    _sentinel(root / "06_sentence_dedup", exact_out, final_out)
    return root


_EXACT = {source: counts[5] for source, counts in _COUNTS.items()}
_SOURCES = {
    "web_a": {"domain": "web", "access": "open"},
    "web_b": {"domain": "web", "access": "gated"},
    "med": {"domain": "medical", "access": "open"},
}


def test_funnel_follows_each_source_through_every_stage(tmp_path: Path) -> None:
    """Each source's row carries its documents after every stage and its final tokens."""
    rows = analysis.funnel_source_rows("loosened", _build(tmp_path), _EXACT, _SOURCES)
    med = next(row for row in rows if row["source"] == "med")
    stages = [med[column] for column in ("convert", "language", "spam", "quality", "repetition")]
    assert stages + [med["exact_dedup"], med["sentence_dedup"]] == list(_COUNTS["med"])
    assert (med["tokens_estimated"], med["retained"]) == (60, 0.15)


def test_funnel_rejects_a_stage_that_read_other_than_the_stage_before_wrote(tmp_path: Path) -> None:
    """A stage reused from another build shows as a break in the chain."""
    build = _build(tmp_path)
    _sentinel(build / "03_quality" / "web_b", 7, 8)
    with pytest.raises(ValueError, match="web_b: 03_quality read 7"):
        analysis.funnel_source_rows("loosened", build, _EXACT, _SOURCES)


def test_funnel_rejects_exact_dedup_counts_that_miss_the_stage_sentinel(tmp_path: Path) -> None:
    """Stale exact-dedup counts fail against the corpus-wide sentinel."""
    with pytest.raises(ValueError, match="exact_dedup documents sum to 66"):
        analysis.funnel_source_rows("loosened", _build(tmp_path), {**_EXACT, "web_a": 51}, _SOURCES)


def test_domain_rows_sum_the_sources_sharing_a_label(tmp_path: Path) -> None:
    """Two web sources fold into one web row, ranked above the smaller medical row."""
    rows = analysis.funnel_domain_rows(analysis.funnel_source_rows("loosened", _build(tmp_path), _EXACT, _SOURCES))
    assert [(row["source_domain"], row["sources"], row["convert"], row["sentence_dedup"]) for row in rows] == [
        ("web", 2, 110, 45),
        ("medical", 1, 20, 3),
    ]


def test_funnel_keeps_a_source_that_lost_its_last_document_before_the_end(tmp_path: Path) -> None:
    """A source with no statistics file ends at zero instead of vanishing from the funnel."""
    build = _build(tmp_path)
    (build / "07_statistics" / "per_dataset" / "med.json").unlink()
    _sentinel(build / analysis.COUNTED_STAGE, 86, 56)
    _sentinel(build / "06_sentence_dedup", 56, 45)
    rows = analysis.funnel_source_rows("loosened", build, {**_EXACT, "med": 0}, _SOURCES)
    med = next(row for row in rows if row["source"] == "med")
    assert (med["repetition"], med["exact_dedup"], med["sentence_dedup"], med["words_final"]) == (18, 0, 0, 0)


def test_funnel_rejects_statistics_for_a_source_no_stage_converted(tmp_path: Path) -> None:
    """A statistics file with no sentinels behind it names a build mixed from two source sets."""
    build = _build(tmp_path)
    stray = build / "07_statistics" / "per_dataset" / "stray.json"
    stray.write_text(
        json.dumps({"dataset": "stray", "domain": "web", "doc_count": 1, "word_count": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"no stage converted: \['stray'\]"):
        analysis.funnel_source_rows("loosened", build, _EXACT, _SOURCES)


_LABELS = ("medical", "news", "web")
#: Judge domain, assigned label and fallback label per document: medical is 2 of 3 right both ways.
_LABELLED = [
    analysis.Labelled("medical", "medical", "medical"),
    analysis.Labelled("medical", "medical", "web"),
    analysis.Labelled("medical", "news", "web"),
    analysis.Labelled("news", "medical", "web"),
    analysis.Labelled("news", "news", "news"),
    analysis.Labelled("web", "web", "web"),
]
#: One document per judge label, every one assigned right.
_PERFECT = [analysis.Labelled(label, label, "web") for label in analysis.LABELS]
_LEXICON_CONFIG = "weights: {strong: 3}\nlexicons:\n  medical:\n    strong: [zdravil*]\nfallback: {}\nrule: {threshold: 50, margin: 15, min_words: 50}\n"
_RULE = {"threshold": 20.0, "margin": 25.0, "min_words": 1}


def test_label_metrics_reads_precision_and_recall_per_label() -> None:
    """Two of three medical documents assigned medical, and one other assigned it, is 2/3 both ways."""
    metrics = analysis.label_metrics(_LABELLED, _LABELS)
    assert metrics["medical"] == {"documents": 3, "assigned": 3, "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}


def test_label_metrics_scores_zero_for_a_label_never_assigned() -> None:
    """A label with no assigned document has zero precision, recall and F1 instead of a division error."""
    metrics = analysis.label_metrics([analysis.Labelled("news", "web", "web")], ("news", "web"))
    assert (metrics["news"]["precision"], metrics["news"]["recall"], metrics["news"]["f1"]) == (0.0, 0.0, 0.0)


def test_macro_f1_counts_every_label_once() -> None:
    """Macro-F1 is the plain mean over labels, so the one-document web label weighs as much as medical."""
    assert analysis.macro_f1(analysis.label_metrics(_LABELLED, _LABELS)) == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)


def test_off_source_read_leaves_out_the_documents_the_source_already_labels() -> None:
    """Off its own source, medical loses its one leaflet and reads 1 of 2 both ways."""
    metrics = analysis.label_metrics(analysis.off_source(_LABELLED, "medical"), _LABELS)
    assert metrics["medical"] == {"documents": 2, "assigned": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_rule_grid_picks_the_best_fit_then_the_rule_that_assigns_more() -> None:
    """Three pairs fit equally; the pick is the lowest threshold among them, then the lowest margin."""
    scored = [
        {"judge_domain": "medical", "fallback": "web", "scores": {"medical": 60.0, "web": 0.0}, "words": 100},
        {"judge_domain": "web", "fallback": "web", "scores": {"medical": 30.0, "web": 10.0}, "words": 100},
    ]
    grid = analysis.rule_grid_rows(scored, 1, (20.0, 50.0), (0.0, 25.0))
    assert [(row["threshold"], row["margin"]) for row in grid if row["picked"]] == [(20.0, 25.0)]


def test_check_rule_tuned_names_the_config_rule_and_the_pick() -> None:
    """A config rule that is not the tune-side pick stops the test-side read."""
    grid = [{"threshold": 20.0, "margin": 25.0, "picked": True}]
    with pytest.raises(ValueError, match="threshold=60 margin=10 is not the tune-side pick threshold=20 margin=25"):
        analysis.check_rule_tuned(analysis.DecisionRule(60.0, 10.0, 50), grid, Path("classifier-lexicon.yaml"))


def test_lexicons_are_frozen_at_the_first_test_read(tmp_path: Path) -> None:
    """The first read writes the hash of the lexicons; the same lexicons pass again."""
    config = tmp_path / "classifier-lexicon.yaml"
    config.write_text(_LEXICON_CONFIG, encoding="utf-8")
    analysis.check_lexicons_frozen(config, tmp_path / "frozen.json")
    analysis.check_lexicons_frozen(config, tmp_path / "frozen.json")
    assert (tmp_path / "frozen.json").exists()


def test_changed_lexicons_stop_a_second_test_read(tmp_path: Path) -> None:
    """A lexicon edited after the test side was read is refused, so the test side is read once per lexicon."""
    config = tmp_path / "classifier-lexicon.yaml"
    config.write_text(_LEXICON_CONFIG, encoding="utf-8")
    analysis.check_lexicons_frozen(config, tmp_path / "frozen.json")
    config.write_text(_LEXICON_CONFIG.replace("zdravil*", "bolnik*"), encoding="utf-8")
    with pytest.raises(ValueError, match="lexicons hash to .* frozen as"):
        analysis.check_lexicons_frozen(config, tmp_path / "frozen.json")


def test_gate_passes_when_both_floors_hold() -> None:
    """A perfect labelling of every label clears both floors."""
    assert analysis.gate_rows({"test": _PERFECT}, _RULE)[0]["passes_gate"] is True


def test_gate_fails_on_off_source_medical_precision_alone() -> None:
    """One web-crawl news document assigned medical halves off-source medical precision, under its floor."""
    rows = analysis.gate_rows({"test": [*_PERFECT, analysis.Labelled("news", "medical", "web")]}, _RULE)
    assert (rows[0]["medical_precision_off_source"], rows[0]["passes_gate"]) == (0.5, False)


def test_gate_ignores_a_medical_source_document_the_source_already_labels() -> None:
    """A document from the medical source assigned medical does not count towards the off-source floor."""
    rows = analysis.gate_rows({"test": [*_PERFECT, analysis.Labelled("news", "medical", "medical")]}, _RULE)
    assert (rows[0]["medical_precision"], rows[0]["passes_gate"]) == (0.5, True)


def test_gate_domain_rows_carry_the_off_source_read() -> None:
    """The medical row reads 2 of 3 over all documents and 1 of 2 off the medical source."""
    row = next(row for row in analysis.gate_domain_rows({"test": _LABELLED}) if row["judge_domain"] == "medical")
    assert (row["precision"], row["off_source_documents"], row["off_source_precision"]) == (0.6667, 2, 0.5)


def test_confusion_rows_count_every_pair_of_labels() -> None:
    """The confusion table holds one row per judge and assigned label, zeros included."""
    rows = analysis.confusion_rows({"test": _LABELLED})
    counted = {(row["judge_domain"], row["assigned_domain"]): row["documents"] for row in rows if row["documents"]}
    assert (len(rows), counted["medical", "news"], counted["news", "medical"]) == (len(analysis.LABELS) ** 2, 1, 1)


def test_lexicon_read_folds_the_judge_and_the_assigned_label() -> None:
    """An academic-judged document from an academic source whose lexicon leader is scientific is a correct science."""
    rule = analysis.DecisionRule(threshold=50.0, margin=15.0, min_words=1)
    scored = [
        {"judge_domain": "science", "fallback": "science", "scores": {"scientific": 80.0, "web": 0.0}, "words": 100}
    ]
    assert analysis.label_scored(scored, rule) == [analysis.Labelled("science", "science", "science")]


def test_scoring_folds_the_judge_label_and_the_source_fallback() -> None:
    """An academic-judged document from an academic source is scored as science with a science fallback."""
    lexicons = analysis.load_lexicons(
        {"weights": {"strong": 3}, "lexicons": {"scientific": {"strong": ["x"]}}, "fallback": {}, "rule": _RULE}
    )
    gold = [{"id": "d", "judge_domain": "academic", "source_domain": "academic"}]
    row = analysis.score_gold({"d": "besedilo"}, gold, lexicons)[0]
    assert (row["judge_domain"], row["fallback"]) == ("science", "science")


def test_scientific_and_academic_are_scored_as_one_label() -> None:
    """Both judge labels fold into science, which takes scientific's place among the twelve scored labels."""
    merged = (analysis.merge_label("scientific"), analysis.merge_label("academic"), analysis.merge_label("legal"))
    assert (merged, len(analysis.LABELS), analysis.LABELS[1]) == (("science", "science", "legal"), 12, "science")


_SUPERVISED_CONFIG = (
    "max_chars: 30\nfolds: 2\nworkers: 1\nseed: 1\nregularisation: [1]\nmax_iterations: 100\ncandidates:\n  tfidf:\n"
    "    features: tfidf\n    word_ngrams: [1, 1]\n    char_ngrams: [2, 3]\n    min_df: 1\n    max_features: 50\n"
)
#: Three documents per label, each its own copy group.
_TUNE = [
    {"id": str(i), "text": f"{word} besedilo {i}", "judge_domain": label, "group": str(i)}
    for i, (word, label) in enumerate([("zdravilo", "medical")] * 3 + [("sodišče", "legal")] * 3)
]


def test_candidate_fit_is_saved_and_reused(tmp_path: Path) -> None:
    """The second call loads the saved fit instead of fitting again, with the same grid."""
    config = tmp_path / "classifier-supervised.yaml"
    config.write_text(_SUPERVISED_CONFIG, encoding="utf-8")
    _, grid = analysis.fit_or_load_candidate("tfidf", config, _TUNE, "abc", tmp_path / "models")
    _, reused = analysis.fit_or_load_candidate("tfidf", config, _TUNE, "abc", tmp_path / "models")
    assert reused == grid


def test_changed_tune_side_stops_a_saved_fit_from_being_reused(tmp_path: Path) -> None:
    """A saved fit from other documents is refused, so the test side is read once per fit."""
    config = tmp_path / "classifier-supervised.yaml"
    config.write_text(_SUPERVISED_CONFIG, encoding="utf-8")
    analysis.fit_or_load_candidate("tfidf", config, _TUNE, "abc", tmp_path / "models")
    with pytest.raises(ValueError, match="was fitted from inputs hashing to .* delete .*tfidf.pkl to refit"):
        analysis.fit_or_load_candidate("tfidf", config, _TUNE[1:], "abc", tmp_path / "models")


#: A corpus pass over the fixture build: web_a holds ten documents the classifier reads as medical.
_PASS = {
    "documents": {"web_a": {"web": 30, "medical": 10}, "web_b": {"web": 5}, "med": {"medical": 3}},
    "words": {"web_a": {"web": 300, "medical": 100}, "web_b": {"web": 50}, "med": {"medical": 30}},
    "gold_ids": [],
}


def _kpi(
    build: str, rows: List[Dict[str, Any]], domain: str, measure: str, labels: str, sources: str = "all"
) -> Dict[str, Any]:
    """Pick one row of `kpi_rows` by what it counts."""
    return next(
        row
        for row in rows
        if (row["build"], row["domain"], row["measure"], row["labels"], row["sources"])
        == (build, domain, measure, labels, sources)
    )


@pytest.fixture
def funnel(tmp_path: Path) -> List[Dict[str, Any]]:
    """The fixture build's funnel rows."""
    return analysis.funnel_source_rows("loosened", _build(tmp_path), _EXACT, _SOURCES)


def test_corpus_pass_matching_the_statistics_stage_is_read(tmp_path: Path, funnel: List[Dict[str, Any]]) -> None:
    """A pass that counts each source as the statistics stage did comes back whole."""
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(_PASS), encoding="utf-8")
    assert analysis.read_corpus_labels(path, "loosened", funnel) == _PASS


def test_corpus_pass_counting_a_source_otherwise_is_rejected(tmp_path: Path, funnel: List[Dict[str, Any]]) -> None:
    """A source whose words differ from the statistics stage names both counts."""
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({**_PASS, "words": {**_PASS["words"], "med": {"medical": 31}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="loosened med: the corpus pass counts 3 documents and 31 words"):
        analysis.read_corpus_labels(path, "loosened", funnel)


def test_medical_documents_counted_under_each_labelling_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source labels see the medical source alone; document labels add the web documents read as medical."""
    monkeypatch.setattr(analysis, "DOMAIN_DOCUMENT_TARGET", 5)
    rows = analysis.kpi_rows("loosened", _PASS, _SOURCES, True)
    source, document = (_kpi("loosened", rows, "medical", "documents", labels) for labels in ("source", "document"))
    assert (source["count"], source["passes"], document["count"], document["passes"]) == (3, False, 13, True)


def test_document_labels_are_not_read_when_the_classifier_fails_its_gate() -> None:
    """A failed gate leaves every count under document labels unread, and the source-label ones read."""
    rows = analysis.kpi_rows("loosened", _PASS, _SOURCES, False)
    source, document = (_kpi("loosened", rows, "medical", "documents", labels) for labels in ("source", "document"))
    assert (source["passes"], document["passes"]) == (False, None)


def test_count_equal_to_its_target_misses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass is a total above the target, so a total at it fails."""
    monkeypatch.setattr(analysis, "DOMAIN_DOCUMENT_TARGET", 3)
    assert not _kpi("loosened", analysis.kpi_rows("loosened", _PASS, _SOURCES, True), "medical", "documents", "source")[
        "passes"
    ]


def test_open_sources_leave_out_the_gated_ones() -> None:
    """The corpus read on open sources drops the gated web_b's words."""
    rows = analysis.kpi_rows("loosened", _PASS, _SOURCES, True)
    every, public = (_kpi("loosened", rows, "corpus", "words", "either", sources) for sources in ("all", "open"))
    assert (every["count"], public["count"]) == (480, 430)


def test_scientific_and_academic_sources_count_as_science() -> None:
    """Under source labels, science sums the scientific and the academic sources."""
    sources = {"oss": {"domain": "scientific", "access": "open"}, "kas": {"domain": "academic", "access": "gated"}}
    labels = {
        "documents": {"oss": {"science": 4}, "kas": {"student": 6}},
        "words": {"oss": {"science": 40}, "kas": {"student": 60}},
    }
    rows = analysis.kpi_rows("loosened", labels, sources, True)
    assert _kpi("loosened", rows, "science", "documents", "source")["count"] == 10


def _medical_gap(funnel: List[Dict[str, Any]], labels: str) -> Dict[str, Any]:
    """Attribute the fixture pass's missed targets and pick the medical documents one under a labelling rule."""
    gaps = analysis.gap_rows(analysis.kpi_rows("loosened", _PASS, _SOURCES, True), funnel)
    return next(
        gap
        for gap in gaps
        if (gap["domain"], gap["measure"], gap["labels"], gap["sources"]) == ("medical", "documents", labels, "all")
    )


def test_curation_loss_names_the_first_stage_at_or_below_the_target(
    monkeypatch: pytest.MonkeyPatch, funnel: List[Dict[str, Any]]
) -> None:
    """The medical source converts 20 documents and first holds 10 or fewer after exact dedup."""
    monkeypatch.setattr(analysis, "DOMAIN_DOCUMENT_TARGET", 10)
    gap = _medical_gap(funnel, "source")
    assert (gap["converted"], gap["final"], gap["cause"], gap["stage"]) == (20, 3, "curation loss", "exact_dedup")


def test_domain_that_never_held_the_target_is_a_source_gap(
    monkeypatch: pytest.MonkeyPatch, funnel: List[Dict[str, Any]]
) -> None:
    """A target above what the medical source converted is a source gap, naming no stage."""
    monkeypatch.setattr(analysis, "DOMAIN_DOCUMENT_TARGET", 25)
    gap = _medical_gap(funnel, "source")
    assert (gap["cause"], gap["stage"]) == ("source gap", "")


def test_shortfall_under_document_labels_is_not_attributable(
    monkeypatch: pytest.MonkeyPatch, funnel: List[Dict[str, Any]]
) -> None:
    """Document labels exist only at the final stage, so their shortfall names no cause."""
    monkeypatch.setattr(analysis, "DOMAIN_DOCUMENT_TARGET", 20)
    assert _medical_gap(funnel, "document")["cause"] == "not attributable"


def _dataset(name: str, domains: str, **fields: str) -> Dict[str, Any]:
    """Write one catalogue row, open, outside the registry and unannotated unless told otherwise."""
    row = {
        "name": name,
        "domains": domains,
        "access": "open",
        "in_registry": "no",
        "provenance_class": "native",
        "annotation": "none",
        "documents": "",
        "kpi2_fit": "unknown",
        "url": "",
    }
    return row | fields


def test_gap_catalogue_lists_open_datasets_the_corpus_lacks() -> None:
    """Datasets already in the registry or not openly downloadable are left out, the largest first."""
    catalogue = [
        _dataset("leaflets", "medical", in_registry="yes", documents="900"),
        _dataset("guidelines", "medical", access="not_downloadable", documents="800"),
        _dataset("notes", "medical", documents="20"),
        _dataset("ner", "medical", documents="300"),
    ]
    rows = analysis.gap_catalogue_rows([{"domain": "medical"}], catalogue)
    assert [row["dataset"] for row in rows] == ["ner", "notes"]


def test_ie_examples_keep_span_annotated_datasets_under_folded_domains() -> None:
    """Only extraction-span datasets count, and an academic one lists under science."""
    catalogue = [
        _dataset("terms", "academic", annotation="IE spans", documents="5"),
        _dataset("treebank", "academic", annotation="linguistic", documents="9"),
    ]
    assert [(row["domain"], row["dataset"]) for row in analysis.ie_example_rows(catalogue)] == [("science", "terms")]
