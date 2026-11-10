"""Checks the Figure rows the record, programme and entry specs declare.

A Figure row cites one figure as an image link, `- **<label>**:
![<caption>](<path>)`: the record Verdict's Decision figure, a programme
belief's Figure and an entry's Algorithm and Worked example Figure
(figures.md, Figure rows). A belief's Figure repeats once per belief, so
these checks read every row of the label in its section, not
`Section.rows`, which keeps the last. A row still holding `<…>` template
guidance is left to the section's guidance rule. The link is read from the
document's folder, as the page reads it.

An entry's figure is a datachart chart `save_figure` stamped, or a diagram
SVG with its source file beside it, the same name with a diagram suffix.

The record's own figures, each finding's Result figure and the Verdict's
Decision figure, are held to the figure rules of figures.md (FIG-020..025):
a stamp at the current check version, text kept as text, the file name, a
caption, the cited clause against the stamp's claim and the claim's arms
against the figure's table. A Result figure is also a file that exists, and
a key finding's figure carries a claim in its stamp. The checker runs beside
a repo's own copy of create_assets.py, which may lag, so it reads the stamp
itself and knows the current version as `CHECK_VERSION`; a test keeps both
equal to create_assets.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from html import unescape
from pathlib import Path, PurePosixPath

from ..document import Document, Entry
from ..document import Section as DocumentSection
from ..rules import WORD, Corpus, check, is_guidance, split_code
from ..spec import Row, Rule
from .record import CITATION, WEIGHT, entries, unwritten

# ===============================
# Defaults
# ===============================

# `![<caption>](<path>)`, an optional title after the path
FIGURE_LINK = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")
ROW = re.compile(r"^- \*\*(.+?)\*\*:\s*(.*)$")
# the stamp create_assets.save_figure writes into a chart that passed its check
STAMP = re.compile(r'<metadata id="labflow-figure-check">(.*?)</metadata>', re.DOTALL)
# create_assets.CHECK_VERSION; a figure checked under older rules is drawn again
CHECK_VERSION = 5
SVG = ".svg"
# the source of a diagram SVG, the same name beside it with one of these suffixes
DIAGRAM_SOURCES = (
    ".mmd",
    ".mermaid",
    ".dot",
    ".gv",
    ".d2",
    ".puml",
    ".plantuml",
    ".drawio",
    ".excalidraw",
    ".tex",
)
# a finding's Result: its first figure or table link, as the record page reads it
RESULT_LINK = re.compile(r"!?\[([^\]]*)\]\(((?:figures|tables)/[^)\s]+)[^)]*\)")
# table_rules.RESULT_STEM, which a test keeps equal: TAB-001 with `.svg` for `.csv`
RESULT_STEM = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-by-[a-z0-9]+(?:-[a-z0-9]+)*$")
FIGURES_FOLDER = "figures"
TABLES_FOLDER = "tables"
# glyph outlines placed by <use>: matplotlib's `DejaVuSans-48`, cairo's `glyph0-1`
GLYPH_USE = re.compile(
    r'<use\b[^>]*href="#(?:glyph\d+-\d+|[A-Z][A-Za-z0-9]*(?:-[A-Za-z]+)*-[0-9a-f]+)"'
)
ARM_COLUMN = "arm"

# ===============================
# Figure row class
# ===============================


@dataclass(frozen=True)
class FigureRow:
    """One written Figure row: the lines it spans and its image links."""

    line: int
    last: int
    links: tuple[tuple[str, str], ...]

    def holds(self, line: int) -> bool:
        return self.line <= line <= self.last


@dataclass(frozen=True)
class CitedFigure:
    """A figure a record row cites: the row's line, the link and the finding it sits in."""

    line: int
    caption: str
    path: str
    finding: Entry | None = None


# ===============================
# Figure row checks
# ===============================


@check("figure-row-file")
def figure_row_file(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each written row links exactly one figure, a file that exists."""
    for row in written_rows(document, part):
        if len(row.links) != 1:
            yield row.line, f"the row links {len(row.links)} figures"
        for _, path in row.links:
            if not (document.folder / path).is_file():
                yield row.line, f"`{path}` does not exist"


@check("figure-row-caption")
def figure_row_caption(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each linked figure has a caption: a word inside the link's `[…]`."""
    for row in written_rows(document, part):
        for caption, path in row.links:
            if not WORD.search(caption):
                yield row.line, f"`{path}`"


@check("figure-row-source")
def figure_row_source(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each linked figure is a stamped chart, or a diagram SVG with its source beside it."""
    for row in written_rows(document, part):
        for _, path in row.links:
            figure = document.folder / path
            if not figure.is_file():
                continue
            problem = stamp_problem(figure)
            if figure.suffix.lower() != SVG:
                yield row.line, f"`{path}` is no SVG"
            elif problem and not has_source(figure):
                yield row.line, f"`{path}` {problem} and has no source beside it"


# ===============================
# Record figure checks
# ===============================


@check("figure-file")
def figure_file(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each cited figure is a file that exists beside the record."""
    for cited in cited_figures(document, part):
        if not (document.folder / cited.path).is_file():
            yield cited.line, f"`{cited.path}` does not exist"


@check("figure-claim")
def figure_claim(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A key finding's cited figure names its claim in the stamp."""
    for cited, figure in existing_figures(document, part):
        # an unreadable or old stamp is figure-stamp's to report
        if not is_key(cited.finding) or stamp_problem(figure):
            continue
        if stamped_claim(figure) is None:
            yield cited.line, f"`{cited.path}` carries no claim"


@check("figure-stamp")
def figure_stamp(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each cited figure carries a stamp at the current check version (FIG-020)."""
    for cited, figure in existing_figures(document, part):
        problem = stamp_problem(figure)
        if problem:
            yield cited.line, f"`{cited.path}` {problem}"


@check("figure-text")
def figure_text(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each cited SVG keeps its text as text, never glyph outlines (FIG-021)."""
    for cited, figure in existing_figures(document, part):
        if figure.suffix.lower() == SVG and GLYPH_USE.search(svg_text(figure)):
            yield cited.line, f"`{cited.path}` draws its text as glyph outlines"


@check("figure-name")
def figure_name(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each cited figure is `figures/<stem>-<quantity>-by-<dimension>.svg` (FIG-022)."""
    for cited in cited_figures(document, part):
        path = PurePosixPath(cited.path)
        if str(path.parent) != FIGURES_FOLDER:
            yield (
                cited.line,
                f"`{cited.path}` sits in {path.parent}/, not {FIGURES_FOLDER}/",
            )
        if path.suffix.lower() != SVG:
            yield cited.line, f"`{cited.path}` is no {SVG}"
        if not RESULT_STEM.match(path.stem):
            yield (
                cited.line,
                f"`{path.name}` is not <stem>-<quantity>-by-<dimension> in lowercase hyphenated words",
            )


@check("figure-caption")
def figure_caption(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each cited figure has a caption: a word inside the link's `[…]` (FIG-023)."""
    for cited in cited_figures(document, part):
        if not WORD.search(cited.caption):
            yield cited.line, f"`{cited.path}`"


@check("figure-clause")
def figure_clause(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The finding cites the clause its figure's stamp claims (FIG-024)."""
    for cited, figure in existing_figures(document, part):
        clause = (stamped_claim(figure) or {}).get("clause")
        if cited.finding is None or not clause:
            continue
        text = "\n".join("\n".join(row.body) for row in cited.finding.rows.values())
        named = sorted(set(CITATION["clause"].findall(text)))
        if clause not in named:
            yield (
                cited.line,
                f"`{cited.path}` claims {clause}; the finding cites {', '.join(named) or 'no clause'}",
            )


@check("figure-arms")
def figure_arms(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each arm the stamp's claim names is an `arm` value of `tables/<same stem>.csv` (FIG-025)."""
    for cited, figure in existing_figures(document, part):
        arms = (stamped_claim(figure) or {}).get("arms") or []
        if not arms:
            continue
        table = figure.parent.parent / TABLES_FOLDER / f"{figure.stem}.csv"
        name = f"{TABLES_FOLDER}/{table.name}"
        if not table.is_file():
            yield cited.line, f"`{cited.path}` names arms, but {name} does not exist"
            continue
        with table.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            values = {row.get(ARM_COLUMN) for row in reader}
        if ARM_COLUMN not in (reader.fieldnames or []):
            yield cited.line, f"{name} has no `{ARM_COLUMN}` column"
            continue
        for arm in arms:
            if arm not in values:
                yield (
                    cited.line,
                    f"`{cited.path}` names arm '{arm}', which {name} lacks",
                )


# ===============================
# Helper functions
# ===============================


def figure_rows(section: DocumentSection | None, label: str) -> list[FigureRow]:
    """Each written `- **<label>**:` row of a section, outside code.

    Args:
        section: The section, or None when the document lacks it.
        label: The row label, `Figure` or `Decision figure`.

    Returns:
        One row per top-level line opening with the label, its indented
        lines joined; a row that holds only `<…>` guidance is left out.
    """
    if section is None:
        return []
    lines = split_code(section.body)[0].split("\n")
    rows, current = [], None
    for number, line in enumerate(lines, start=section.line + 1):
        match = ROW.match(line)
        if match:
            current = (
                [number, number, [match.group(2)]] if match.group(1) == label else None
            )
            if current:
                rows.append(current)
        elif current and (line.startswith((" ", "\t")) or not line.strip()):
            if line.strip():
                current[1] = number
                current[2].append(line.strip())
        else:
            current = None
    return [
        FigureRow(first, last, tuple(FIGURE_LINK.findall(" ".join(parts))))
        for first, last, parts in rows
        if not is_guidance(" ".join(parts))
    ]


def read_stamp(figure: Path) -> dict | None:
    """Reads the check stamp `save_figure` wrote into an SVG, as create_assets.read_stamp does.

    Args:
        figure: The SVG file.

    Returns:
        The stamp, with `check_version`, `rules`, `claim` and `width`, or None
        when the file carries none.

    Raises:
        ValueError: If the stamp is not JSON.
    """
    match = STAMP.search(svg_text(figure))
    return json.loads(unescape(match.group(1))) if match else None


def stamp_problem(figure: Path) -> str:
    """What keeps a figure from a stamp at the current check version; empty when none."""
    if figure.suffix.lower() != SVG:
        return "is no SVG"
    try:
        stamp = read_stamp(figure)
    except ValueError:
        return "carries a stamp that is no JSON"
    if stamp is None:
        return "carries no stamp"
    if not isinstance(stamp, dict):
        return "carries a stamp that is no JSON object"
    version = stamp.get("check_version")
    if version != CHECK_VERSION:
        return f"is stamped at check version {version}, not {CHECK_VERSION}"
    return ""


def stamped_claim(figure: Path) -> dict | None:
    # an unreadable stamp has no claim to compare; figure-stamp reports it
    if stamp_problem(figure):
        return None
    claim = read_stamp(figure).get("claim")
    return claim if isinstance(claim, dict) else None


def is_stamped(figure: Path) -> bool:
    """Whether a figure carries a stamp at all, whatever its version."""
    return STAMP.search(svg_text(figure)) is not None


def svg_text(figure: Path) -> str:
    return figure.read_text(encoding="utf-8", errors="replace")


def has_source(figure: Path) -> bool:
    return any(figure.with_suffix(suffix).is_file() for suffix in DIAGRAM_SOURCES)


def written_rows(document: Document, part: Row) -> list[FigureRow]:
    return figure_rows(document.sections.get(part.heading), part.label)


def cited_figures(document: Document, part: Row) -> list[CitedFigure]:
    """The figures a record part cites: each finding's Result figure, or each Figure row's links.

    Args:
        document: The record.
        part: A row of an entries section (a finding's Result) or a Figure row.

    Returns:
        One figure per written row; a Result whose first link is a table, or
        that holds `<…>` guidance, cites none.
    """
    if not part.in_entries:
        return [
            CitedFigure(row.line, caption, path)
            for row in written_rows(document, part)
            for caption, path in row.links
        ]
    found = []
    for entry in entries(document, part.heading):
        row = entry.rows.get(part.label)
        match = (
            RESULT_LINK.search(row.text) if row and not unwritten(row.text) else None
        )
        if match and match.group(2).startswith(f"{FIGURES_FOLDER}/"):
            found.append(CitedFigure(row.line, match.group(1), match.group(2), entry))
    return found


def existing_figures(document: Document, part: Row) -> list[tuple[CitedFigure, Path]]:
    # a missing file is left to figure-file or figure-row-file
    return [
        (cited, document.folder / cited.path)
        for cited in cited_figures(document, part)
        if (document.folder / cited.path).is_file()
    ]


def is_key(finding: Entry | None) -> bool:
    """Whether a finding's heading weighs it `key`."""
    match = WEIGHT.search(finding.title) if finding else None
    return bool(match) and match.group(1) == "key"


def in_figure_row(document: Document, heading: str, label: str, line: int) -> bool:
    """Whether a line of a section sits inside one of its written `label` rows."""
    return any(
        row.holds(line) for row in figure_rows(document.sections.get(heading), label)
    )


def resolved(document: Document, path: str) -> Path:
    """A linked path, resolved from the document's folder."""
    return (document.folder / path).resolve()
