"""Tests for the MkDocs hook that feeds the publications template."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

HOOK_PATH = Path(__file__).resolve().parents[2] / "website" / "hooks" / "publications.py"


def _load_hook():
    """Import the hook module from its path, as MkDocs does."""
    spec = importlib.util.spec_from_file_location("publications_hook", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()

LABELS = {
    "journal": "Journal",
    "conference": "Conference",
    "preprint": "Preprint",
    "dataset": "Dataset",
    "model": "Model",
    "poster": "Poster",
    "talk": "Talk",
}

JOURNAL_2027 = {
    "kind": "journal",
    "title": "Newer article",
    "authors": "E. Novak",
    "venue": "Applied Artificial Intelligence",
    "details": "40(1)",
    "year": 2027,
    "url": "https://example.org/article",
    "doi": "10.1000/xyz",
    "pdf": "https://example.org/article.pdf",
    "bibtex": "@article{novak2027,\n  title={Newer article}\n}\n",
    "thumbnail": "newer.png",
    "summary": "One line on what it shows.",
}
DATASET_2026_MARCH = {"kind": "dataset", "title": "March dataset", "venue": "Hugging Face", "year": 2026, "month": 3}
TALK_2026_OCTOBER = {"kind": "talk", "title": "October talk", "venue": "SiKDD", "year": 2026, "month": 10}
POSTER_2026 = {"kind": "poster", "title": "Undated poster", "venue": "EMNLP", "year": 2026}


def _docs_dir(tmp_path: Path, *thumbnails: str) -> Path:
    """Build a docs tree holding the given thumbnail files."""
    folder = tmp_path / hook.THUMBNAIL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for name in thumbnails:
        (folder / name).write_bytes(b"png")
    return tmp_path


def test_no_entries_yields_no_entries_and_no_filters(tmp_path: Path) -> None:
    """With nothing published the template gets empty lists and shows the empty state."""
    entries = hook.prepare_entries([], tmp_path, LABELS)
    assert entries == []
    assert hook.build_filters(entries) == []


def test_entries_sorted_newest_first_by_year_then_month(tmp_path: Path) -> None:
    """The most recent entry leads; an undated entry sorts after dated ones of its year."""
    entries = hook.prepare_entries([POSTER_2026, DATASET_2026_MARCH, TALK_2026_OCTOBER], tmp_path, LABELS)
    assert [e["title"] for e in entries] == ["October talk", "March dataset", "Undated poster"]


def test_every_kind_carries_its_label_and_colour_family(tmp_path: Path) -> None:
    """Journal and conference share a family, as do dataset and model, and poster and talk."""
    assert hook.KINDS == {
        "journal": "paper",
        "conference": "paper",
        "preprint": "preprint",
        "dataset": "resource",
        "model": "resource",
        "poster": "presentation",
        "talk": "presentation",
    }
    entry = hook.prepare_entries([DATASET_2026_MARCH], tmp_path, LABELS)[0]
    assert (entry["label"], entry["family"]) == ("Dataset", "resource")


def test_labels_come_from_the_page_not_the_hook(tmp_path: Path) -> None:
    """A translated page's labels reach both the entry and its filter chip."""
    entries = hook.prepare_entries([DATASET_2026_MARCH], tmp_path, {**LABELS, "dataset": "Podatkovna zbirka"})
    assert entries[0]["label"] == "Podatkovna zbirka"
    assert hook.build_filters(entries)[0]["label"] == "Podatkovna zbirka"


def test_front_matter_missing_a_label_is_rejected() -> None:
    """A page that forgets to label a kind fails the build and names the kind."""
    assert hook.check_labels(LABELS) == LABELS
    with pytest.raises(ValueError, match="talk"):
        hook.check_labels({k: v for k, v in LABELS.items() if k != "talk"})
    with pytest.raises(ValueError, match="kind_labels"):
        hook.check_labels(None)


def test_full_entry_resolves_doi_thumbnail_and_bibtex(tmp_path: Path) -> None:
    """A bare DOI becomes a resolver link, the thumbnail a site path, the BibTeX is trimmed."""
    entry = hook.prepare_entries([JOURNAL_2027], _docs_dir(tmp_path, "newer.png"), LABELS)[0]
    assert entry["doi_url"] == "https://doi.org/10.1000/xyz"
    assert entry["thumbnail"] == f"{hook.THUMBNAIL_DIR.as_posix()}/newer.png"
    assert entry["bibtex"] == "@article{novak2027,\n  title={Newer article}\n}"
    assert (entry["details"], entry["pdf"], entry["summary"]) == (
        "40(1)",
        "https://example.org/article.pdf",
        "One line on what it shows.",
    )


def test_optional_fields_survive_as_none(tmp_path: Path) -> None:
    """A bare entry still renders; the template guards every optional field."""
    entry = hook.prepare_entries([POSTER_2026], tmp_path, LABELS)[0]
    for key in ("authors", "details", "url", "doi_url", "pdf", "bibtex", "thumbnail", "summary"):
        assert entry[key] is None


def test_filters_count_each_present_kind_in_declared_order(tmp_path: Path) -> None:
    """Chips appear only for kinds that have entries, in the order `KINDS` declares."""
    entries = hook.prepare_entries([TALK_2026_OCTOBER, DATASET_2026_MARCH, POSTER_2026], tmp_path, LABELS)
    assert hook.build_filters(entries) == [
        {"kind": "dataset", "label": "Dataset", "family": "resource", "count": 1},
        {"kind": "poster", "label": "Poster", "family": "presentation", "count": 1},
        {"kind": "talk", "label": "Talk", "family": "presentation", "count": 1},
    ]


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    """A typo in `kind` fails the build instead of silently dropping the entry."""
    with pytest.raises(ValueError, match="kind"):
        hook.prepare_entries([{**POSTER_2026, "kind": "papr"}], tmp_path, LABELS)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    """A misspelt field fails the build rather than vanishing from the page."""
    with pytest.raises(ValueError, match="bibtx"):
        hook.prepare_entries([{**POSTER_2026, "bibtx": "@misc{}"}], tmp_path, LABELS)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    """An entry without a title names the missing field in the error."""
    entry = {k: v for k, v in POSTER_2026.items() if k != "title"}
    with pytest.raises(ValueError, match="title"):
        hook.prepare_entries([entry], tmp_path, LABELS)


def test_month_out_of_range_is_rejected(tmp_path: Path) -> None:
    """A month outside 1-12 fails the build."""
    with pytest.raises(ValueError, match="month"):
        hook.prepare_entries([{**POSTER_2026, "month": 13}], tmp_path, LABELS)


def test_missing_thumbnail_file_is_rejected(tmp_path: Path) -> None:
    """A thumbnail that is not on disk fails the build instead of rendering a broken image."""
    with pytest.raises(ValueError, match="newer.png"):
        hook.prepare_entries([JOURNAL_2027], _docs_dir(tmp_path), LABELS)


def test_load_data_reads_entries(tmp_path: Path) -> None:
    """An empty publications list is valid; entries come back as written."""
    data = tmp_path / "publications.yml"
    data.write_text("publications: []\n", encoding="utf-8")
    assert hook.load_data(data) == []
    data.write_text("publications:\n  - kind: journal\n    title: T\n    venue: V\n    year: 2026\n", encoding="utf-8")
    assert hook.load_data(data)[0]["title"] == "T"


def test_load_data_rejects_a_file_without_the_list(tmp_path: Path) -> None:
    """A data file without the `publications` key fails the build."""
    data = tmp_path / "publications.yml"
    data.write_text("entries: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="publications"):
        hook.load_data(data)


def _context_for(tmp_path: Path, meta: dict) -> dict:
    """Run the hook against a one-file docs tree and return the resulting context."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "publications.yml").write_text("publications: []\n", encoding="utf-8")
    page = SimpleNamespace(meta=meta)
    return hook.on_page_context({}, page, {"docs_dir": str(tmp_path)}, None)


def test_context_is_filled_on_every_publications_page(tmp_path: Path) -> None:
    """Any page using the template, in any language, gets its entries and filters."""
    context = _context_for(tmp_path, {"template": "publications.html", "kind_labels": LABELS})
    assert context[hook.ENTRIES_KEY] == []
    assert context[hook.FILTERS_KEY] == []


def test_context_is_untouched_on_every_other_page(tmp_path: Path) -> None:
    """No other page pays for the data file being read."""
    assert _context_for(tmp_path, {"template": "about.html"}) == {}
