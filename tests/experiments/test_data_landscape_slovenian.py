"""Tests for the data-landscape-slovenian analysis script.

Two things the experiment's issue asked for: the per-domain aggregation on a
small in-tree fixture, and a schema check over the committed catalogue. The
script imports datachart at module level, so the whole module skips when the
`analysis` dependency group is not installed.
"""

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pytest
import yaml

pytest.importorskip("datachart")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "data" / "data-landscape-slovenian"


@pytest.fixture(scope="module")
def analysis() -> ModuleType:
    """Loads `analysis.py` as a module, since the experiment folder is not a package.

    Returns:
        The imported analysis module.
    """
    spec = importlib.util.spec_from_file_location("landscape_analysis", EXPERIMENT / "analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(**fields: object) -> Dict[str, str]:
    """Builds one survey row on the D4 schema with every field present.

    Args:
        **fields: Values to set; the rest are blank.

    Returns:
        The row, as the raw JSON files carry it.
    """
    base = {
        "name": "",
        "url": "",
        "licence": "",
        "access": "open",
        "languages": "sl",
        "domains": "other",
        "documents": "",
        "words": "",
        "size_verified": "no",
        "annotation": "none",
        "length_class": "document",
        "format": "",
        "notes": "",
    }
    return {**base, **{k: str(v) for k, v in fields.items()}}


@pytest.fixture
def fixture_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, analysis: ModuleType) -> Dict[str, Path]:
    """Points the script at a two-family survey, a two-entry registry and a corpus count.

    Returns:
        The fixture files by role, for tests that read them back.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    publisher = [
        row(name="Medical corpus", url="https://hdl.handle.net/11356/1983", domains="medical", documents=100),
        row(
            name="Science corpus",
            url="https://hdl.handle.net/11356/9001",
            domains="scientific",
            documents=20000,
            words=400000,
        ),
        row(
            name="Translated science",
            url="https://hdl.handle.net/11356/9002",
            domains="scientific",
            documents=500,
            annotation="parallel",
            notes="machine translation of an English set",
        ),
        row(
            name="Locked corpus",
            url="https://hdl.handle.net/11356/9003",
            domains="medical",
            access="login",
            documents=50000,
            words=1000000,
        ),
    ]
    mirror = [
        row(name="Medical corpus", url="https://elg.example/medical", domains="medical", licence="CC BY 4.0"),
    ]
    (raw / "publisher.json").write_text(json.dumps(publisher), encoding="utf-8")
    (raw / "mirror.json").write_text(json.dumps(mirror), encoding="utf-8")

    registry = tmp_path / "download.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "povejmo_vemo_med": {"urls": ["https://www.clarin.si/repository/xmlui/handle/11356/1983"]},
                    "unrelated": {"repo_id": "someone/else"},
                }
            }
        ),
        encoding="utf-8",
    )
    stats = tmp_path / "aggregate.json"
    stats.write_text(json.dumps({"by_dataset": {"povejmo_vemo_med": {"doc_count": 90, "word_count": 300000}}}))

    monkeypatch.setattr(analysis, "RAW_DIR", raw)
    monkeypatch.setattr(analysis, "FAMILY_FILES", [("publisher", "publisher.json"), ("mirror", "mirror.json")])
    monkeypatch.setattr(analysis, "REGISTRY", registry)
    monkeypatch.setattr(analysis, "CORPUS_STATISTICS", stats)
    monkeypatch.setattr(analysis, "SIZES_TABLE", tmp_path / "no-sizes.csv")
    return {"raw": raw, "registry": registry, "stats": stats}


def annotated(analysis: ModuleType) -> List[Dict[str, str]]:
    """Runs the script's row pipeline on whatever the fixtures point it at.

    Args:
        analysis: The analysis module.

    Returns:
        The annotated catalogue rows.
    """
    return analysis.annotate(analysis.merge_duplicates(analysis.load_rows()))


def test_merge_keeps_publisher_and_backfills_from_mirror(analysis: ModuleType, fixture_paths: Dict[str, Path]) -> None:
    """The publisher's row wins a merge and the mirror only fills its blanks (D10)."""
    rows = {r["name"]: r for r in annotated(analysis)}
    assert len(rows) == 4
    medical = rows["Medical corpus"]
    assert medical["family"] == "publisher"
    assert medical["url"] == "https://hdl.handle.net/11356/1983"
    assert medical["licence"] == "CC BY 4.0"
    assert medical["mirrors"] == "https://elg.example/medical"


def test_registry_match_and_corpus_count_size_an_unsized_row(
    analysis: ModuleType, fixture_paths: Dict[str, Path]
) -> None:
    """A registry row with no reported size takes the corpus build's word count (D6)."""
    rows = {r["name"]: r for r in annotated(analysis)}
    medical = rows["Medical corpus"]
    assert medical["in_registry"] == "yes"
    assert medical["registry_key"] == "povejmo_vemo_med"
    assert medical["words"] == "300000"
    assert medical["words_basis"] == "corpus"
    assert medical["documents_corpus"] == "90"
    assert medical["tokens_estimated"] == "600000"
    assert medical["kpi4_fit"] == "yes"
    science = rows["Science corpus"]
    assert science["in_registry"] == "no"
    assert science["words_basis"] == "reported"


def test_survey_size_beats_the_corpus_count_and_fills_documents(
    analysis: ModuleType, fixture_paths: Dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This experiment's own count wins over the corpus count and sets documents (M6)."""
    sizes = tmp_path / "sizes.csv"
    sizes.write_text(
        "source,catalogue_name,items_total,items_counted,words_counted,words_estimated,basis,method,sized_on\n"
        "case-reports,Medical corpus,80,80,5000,5000,full,counted every row,2026-09-21\n"
        "journal,Translated science,2000,40,8000,400000,sampled,forty articles,2026-09-21\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis, "SIZES_TABLE", sizes)
    rows = {r["name"]: r for r in annotated(analysis)}
    medical = rows["Medical corpus"]
    assert medical["words"] == "5000" and medical["words_basis"] == "counted"
    assert medical["words_corpus"] == "300000"
    assert medical["documents"] == "80" and medical["documents_counted"] == "80"
    journal = rows["Translated science"]
    assert journal["words"] == "400000" and journal["words_basis"] == "sampled"
    assert journal["kpi4_fit"] == "yes"


def test_provenance_reads_the_row_text(analysis: ModuleType, fixture_paths: Dict[str, Path]) -> None:
    """Provenance is read from the row's own name and notes (D11)."""
    rows = {r["name"]: r for r in annotated(analysis)}
    assert rows["Science corpus"]["provenance"] == "native"
    assert rows["Translated science"]["provenance"] == "translated"
    generated = analysis.provenance(row(name="Synthetic notes", notes="LLM-generated clinical notes"))
    assert generated == "generated"


def test_kpi_fit_reads_missing_counts_as_unknown(analysis: ModuleType) -> None:
    """A row with no reported count reads `unknown` for both KPIs (D9)."""
    assert analysis.kpi_fit(row(domains="medical", documents=20000, words=300000)) == ("yes", "yes")
    assert analysis.kpi_fit(row(domains="medical", documents=500, words=1000)) == ("no", "no")
    assert analysis.kpi_fit(row(domains="medical")) == ("unknown", "unknown")
    assert analysis.kpi_fit(row(domains="news", documents=20000)) == ("n/a", "unknown")


def test_verdict_never_reads_a_missing_size_as_a_miss(analysis: ModuleType) -> None:
    """A short total with unsized datasets is `unknown`, never `no` (D9)."""
    assert analysis.verdict(600000, 500000, uncounted=5) == "yes"
    assert analysis.verdict(100, 500000, uncounted=0) == "no"
    assert analysis.verdict(100, 500000, uncounted=1) == "unknown"


def test_summary_totals_per_domain_under_each_access_filter(
    analysis: ModuleType, fixture_paths: Dict[str, Path]
) -> None:
    """Per-domain totals widen as the access filter loosens (D8)."""
    summary = {(r["domain"], r["access_filter"]): r for r in analysis.summarise(annotated(analysis))}
    medical_open = summary[("medical", "open")]
    assert medical_open["datasets"] == 1
    assert medical_open["words"] == 300000
    assert medical_open["datasets_sized_by_corpus"] == 1
    assert medical_open["documents"] == 100
    assert medical_open["meets_kpi2_documents"] == "no"
    assert medical_open["meets_kpi4_tokens"] == "yes"
    medical_login = summary[("medical", "open+login")]
    assert medical_login["datasets"] == 2
    assert medical_login["words"] == 1300000
    assert medical_login["meets_kpi2_documents"] == "yes"
    science_open = summary[("scientific", "open")]
    assert science_open["documents"] == 20500
    assert science_open["documents_native"] == 20000
    assert science_open["datasets_without_word_count"] == 1
    assert science_open["meets_kpi2_native"] == "yes"
    assert science_open["meets_kpi4_tokens"] == "yes"
    assert summary[("finance", "all")]["datasets"] == 0
    assert summary[("finance", "all")]["meets_kpi4_tokens"] == "no"


def test_committed_catalogue_matches_the_schema(analysis: ModuleType) -> None:
    """The committed catalogue carries exactly the schema's columns (D4)."""
    with (EXPERIMENT / "tables" / "catalogue.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analysis.CATALOGUE_COLUMNS
        rows = list(reader)
    assert rows
    access = {"open", "login", "gated", "not_downloadable"}
    annotation = {"none", "linguistic", "labels", "parallel", "IE spans", "IE spans,labels"}
    for r in rows:
        assert r["name"] and r["url"], r
        assert r["access"] in access, r["name"]
        assert r["annotation"] in annotation, r["name"]
        assert {d.strip() for d in r["domains"].split(",")} <= set(analysis.DOMAINS), r["name"]
        assert r["provenance"] in {"native", "translated", "generated"}, r["name"]
        assert r["words_basis"] in {"reported", "corpus", "", *analysis.SIZE_BASIS.values()}, r["name"]
        assert r["in_registry"] in {"yes", "no"} and bool(r["registry_key"]) == (r["in_registry"] == "yes"), r["name"]
        assert r["kpi2_fit"] in {"yes", "no", "unknown", "n/a"} and r["kpi4_fit"] in {"yes", "no", "unknown"}, r["name"]
        for count in ("documents", "documents_counted", "words", "tokens_estimated"):
            assert r[count] == "" or r[count].isdigit(), (r["name"], count)
