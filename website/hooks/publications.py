"""MkDocs hook that feeds the publications data file to the publications template.

`website/data/publications.yml` holds one entry per output: journal and
conference papers, preprints, datasets, models, posters and talks. On build this
hook validates every entry, sorts them newest first, counts them per kind for
the filter chips, and puts the result in the Jinja context that
`website/overrides/publications.html` renders. Every visible string, the type
labels included, comes from the page's front matter, so each language's page
supplies its own. Wired through `hooks:` in `mkdocs.yml`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Pages rendered with this template get the data, in whatever language.
TEMPLATE = "publications.html"
DATA_FILE = Path("data") / "publications.yml"
# First-page images, relative to the docs directory.
THUMBNAIL_DIR = Path("assets") / "imgs" / "publications"

# Context keys the publications template reads.
ENTRIES_KEY = "publication_entries"
FILTERS_KEY = "publication_filters"

# Colour family per `kind`, in filter-chip order; labels come from front matter.
KINDS: Dict[str, str] = {
    "journal": "paper",
    "conference": "paper",
    "preprint": "preprint",
    "dataset": "resource",
    "model": "resource",
    "poster": "presentation",
    "talk": "presentation",
}
REQUIRED = ("kind", "title", "venue", "year")
OPTIONAL = ("authors", "details", "month", "url", "doi", "pdf", "bibtex", "thumbnail", "summary")


def load_data(path: Path) -> List[Dict[str, Any]]:
    """Read the publications data file.

    Args:
        path: The YAML file, a mapping with a `publications` list.

    Returns:
        The entry mappings, empty when no publication exists yet.

    Raises:
        ValueError: If the `publications` key is missing.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "publications" not in data:
        raise ValueError(f"{path}: expected a mapping with `publications`")
    return list(data["publications"] or [])


def check_labels(labels: Any) -> Dict[str, str]:
    """Check that a page's front matter labels every kind.

    Args:
        labels: The page's `kind_labels` front matter value.

    Returns:
        The labels, keyed by kind.

    Raises:
        ValueError: If `labels` is not a mapping or leaves a kind unlabelled.
    """
    missing = [kind for kind in KINDS if not isinstance(labels, dict) or kind not in labels]
    if missing:
        raise ValueError(f"publications page front matter `kind_labels` is missing {', '.join(missing)}")
    return labels


def _validate(entry: Dict[str, Any], docs_dir: Path) -> None:
    """Fail the build on a malformed entry rather than dropping it silently."""
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise ValueError(f"publication entry {entry!r} is missing {', '.join(missing)}")
    title = entry["title"]
    if entry["kind"] not in KINDS:
        raise ValueError(f"publication {title!r}: unknown kind {entry['kind']!r}, expected one of {list(KINDS)}")
    unknown = set(entry) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"publication {title!r}: unknown fields {sorted(unknown)}")
    if "month" in entry and not 1 <= int(entry["month"]) <= 12:
        raise ValueError(f"publication {title!r}: month {entry['month']!r} is not between 1 and 12")
    if "thumbnail" in entry and not (docs_dir / THUMBNAIL_DIR / entry["thumbnail"]).is_file():
        raise ValueError(f"publication {title!r}: thumbnail {entry['thumbnail']!r} not found in {THUMBNAIL_DIR}")


def _prepare(entry: Dict[str, Any], labels: Dict[str, str]) -> Dict[str, Any]:
    """Shape one entry for the template, resolving its label, DOI and thumbnail paths."""
    thumbnail: Optional[str] = entry.get("thumbnail")
    bibtex: Optional[str] = entry.get("bibtex")
    return {
        "kind": entry["kind"],
        "label": labels[entry["kind"]],
        "family": KINDS[entry["kind"]],
        "title": entry["title"],
        "authors": entry.get("authors"),
        "venue": entry["venue"],
        "details": entry.get("details"),
        "year": entry["year"],
        "url": entry.get("url"),
        "doi_url": f"https://doi.org/{entry['doi']}" if entry.get("doi") else None,
        "pdf": entry.get("pdf"),
        "bibtex": bibtex.strip() if bibtex else None,
        "thumbnail": (THUMBNAIL_DIR / thumbnail).as_posix() if thumbnail else None,
        "summary": entry.get("summary"),
    }


def prepare_entries(entries: List[Dict[str, Any]], docs_dir: Path, labels: Dict[str, str]) -> List[Dict[str, Any]]:
    """Validate the entries and sort them newest first.

    Args:
        entries: Entry mappings with `kind`, `title`, `venue`, `year` and any of
            the `OPTIONAL` fields.
        docs_dir: The docs directory, against which thumbnails are checked.
        labels: The display label per kind, as returned by `check_labels`.

    Returns:
        The prepared entries, ordered by year and then month, newest first; an
        entry without a month follows the dated entries of its year, and ties
        keep the data file's order.

    Raises:
        ValueError: If an entry lacks a required field, names an unknown kind or
            field, has a month outside 1-12, or points at a missing thumbnail.
    """
    for entry in entries:
        _validate(entry, docs_dir)
    ordered = sorted(entries, key=lambda e: (int(e["year"]), int(e.get("month", 0))), reverse=True)
    return [_prepare(e, labels) for e in ordered]


def build_filters(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Count the prepared entries per kind for the filter chips.

    Args:
        entries: Entries as returned by `prepare_entries`.

    Returns:
        One mapping per kind that has entries, in the order `KINDS` declares,
        with its `kind`, `label`, colour `family` and `count`.
    """
    filters = []
    for kind, family in KINDS.items():
        matching = [e for e in entries if e["kind"] == kind]
        if matching:
            filters.append({"kind": kind, "label": matching[0]["label"], "family": family, "count": len(matching)})
    return filters


def on_page_context(context: Dict[str, Any], page: Any, config: Dict[str, Any], nav: Any) -> Dict[str, Any]:
    """Put the sorted publications and their filters in a publications page context.

    Args:
        context: The Jinja context the page's template renders with.
        page: The MkDocs page; only pages using `TEMPLATE` are touched, and
            their `kind_labels` front matter names each kind.
        config: The MkDocs config; `docs_dir` locates the data file and thumbnails.
        nav: The MkDocs navigation, unused.

    Returns:
        The context, carrying the entries and the filters on a publications page
        and unchanged on every other page.

    Raises:
        ValueError: If the page's front matter leaves a kind unlabelled.
    """
    if page.meta.get("template") != TEMPLATE:
        return context
    docs_dir = Path(config["docs_dir"])
    labels = check_labels(page.meta.get("kind_labels"))
    entries = prepare_entries(load_data(docs_dir / DATA_FILE), docs_dir, labels)
    context[ENTRIES_KEY] = entries
    context[FILTERS_KEY] = build_filters(entries)
    return context
