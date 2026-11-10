"""Read a source document into the parts a spec names, with their line numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ===============================
# Defaults
# ===============================

FENCE = "---"
HEADING = re.compile(r"^## +(.+?)\s*$")
ENTRY = re.compile(r"^### +(.+?)\s*$")
ROW = re.compile(r"^- \*\*(.+?)\*\*:?\s*(.*)$")
LINK_NOISE = re.compile(r"^(\.\./|\./)+|/README\.md$|\.md$|/$")
TIMESTAMP = "tag:yaml.org,2002:timestamp"

# ===============================
# Document classes
# ===============================


class FrontmatterLoader(yaml.SafeLoader):
    """A safe loader that keeps dates as text, so the spec's date rule judges them.

    PyYAML builds a date from `2026-02-30` and raises ValueError, outside any
    rule; as text the date reaches the `type` check, which names its line.
    """


FrontmatterLoader.yaml_implicit_resolvers = {
    first: [(tag, pattern) for tag, pattern in resolvers if tag != TIMESTAMP]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


@dataclass(frozen=True)
class Row:
    """A `- **Label**: text` row, its continuation lines joined.

    `body` keeps the lines as written, indentation and all, the first holding
    the text after the label, so a check can read the nested bullets or
    numbered steps under the row.
    """

    label: str
    line: int
    text: str
    body: tuple[str, ...] = ()


@dataclass(frozen=True)
class Entry:
    """A `### Title` entry inside a section and the rows it holds."""

    title: str
    line: int
    rows: dict[str, Row] = field(default_factory=dict)


@dataclass(frozen=True)
class Section:
    """A `## Heading` section: its body lines, the rows among them and its `###` entries.

    `rows` holds every row of the section; where two entries share a label,
    the later entry's row wins, so a check on entries reads `entries`.
    """

    heading: str
    line: int
    body: tuple[str, ...]
    rows: dict[str, Row] = field(default_factory=dict)
    entries: tuple[Entry, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(self.body).strip()


@dataclass(frozen=True)
class Document:
    """One source file, read into frontmatter and sections.

    `yaml_error` holds the parser's message when the frontmatter is not valid
    YAML; `frontmatter` is then empty.
    """

    path: Path
    kind: str
    id: str
    frontmatter: dict
    key_lines: dict[str, int]
    sections: dict[str, Section]
    yaml_error: str = ""
    yaml_line: int = 1

    @property
    def folder(self) -> Path:
        return self.path.parent


# ===============================
# Main function
# ===============================


def read_document(path: Path, kind: str, experiments: Path, root: str = "") -> Document:
    """Reads one markdown file into its frontmatter and `##` sections.

    Args:
        path: The file to read.
        kind: The document kind its spec declares.
        experiments: The experiments folder, which ids are relative to.
        root: The folder under experiments/ that this kind's ids start from.

    Returns:
        The document, with 1-based line numbers for every key, section and row.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # split off the frontmatter
    end = 0
    if lines and lines[0].strip() == FENCE:
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == FENCE), 0)
    frontmatter, key_lines, error, error_line = (
        _frontmatter(lines[1:end]) if end else ({}, {}, "", 1)
    )

    # read the sections after it
    sections: dict[str, Section] = {}
    heading, start, body = None, 0, []
    for number, line in enumerate(
        lines[end + 1 if end else 0 :], start=end + 2 if end else 1
    ):
        match = HEADING.match(line)
        if match:
            if heading is not None:
                sections[heading] = _section(heading, start, body)
            heading, start, body = match.group(1), number, []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections[heading] = _section(heading, start, body)

    return Document(
        path,
        kind,
        document_id(path, experiments, root),
        frontmatter,
        key_lines,
        sections,
        error,
        error_line,
    )


def document_id(path: Path, experiments: Path, root: str = "") -> str:
    """The id other documents use to name this one: its path under root, without `README.md` or `.md`."""
    relative = (
        path.relative_to(experiments / root) if root else path.relative_to(experiments)
    )
    return normalize_link(relative.as_posix())


def normalize_link(value: str) -> str:
    """Strips what a link may carry around an id: `../`, `./`, `/README.md`, `.md`, a trailing `/`."""
    previous = None
    value = str(value).strip()
    while previous != value:
        previous, value = value, LINK_NOISE.sub("", value)
    return "" if value == "README" else value


# ===============================
# Helper functions
# ===============================


def _frontmatter(lines: list[str]) -> tuple[dict, dict[str, int], str, int]:
    text = "\n".join(lines)
    try:
        node = yaml.compose(text, Loader=FrontmatterLoader)
        data = yaml.load(text, Loader=FrontmatterLoader) or {}
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        return (
            {},
            {},
            str(getattr(error, "problem", error)),
            (mark.line + 2 if mark else 1),
        )
    if not isinstance(data, dict):
        return {}, {}, "frontmatter is not a mapping", 2
    key_lines = (
        {key.value: key.start_mark.line + 2 for key, _ in node.value}
        if node is not None
        else {}
    )
    return data, key_lines, "", 1


def _section(heading: str, line: int, body: list[str]) -> Section:
    # split the body at each `###` entry; the lines before the first stay with the section
    starts = [i for i, content in enumerate(body) if ENTRY.match(content)]
    entries = tuple(
        Entry(
            ENTRY.match(body[start]).group(1),
            line + 1 + start,
            _rows(body[start + 1 : end], line + 2 + start),
        )
        for start, end in zip(starts, [*starts[1:], len(body)])
    )
    return Section(heading, line, tuple(body), _rows(body, line + 1), entries)


def _rows(body: list[str], first: int) -> dict[str, Row]:
    rows: dict[str, Row] = {}
    label, start, lines = None, 0, []
    for number, content in enumerate(body, start=first):
        match = ROW.match(content)
        if match:
            if label is not None:
                rows[label] = _row(label, start, lines)
            label, start, lines = match.group(1), number, [match.group(2)]
        elif label is not None and (content.startswith(" ") or not content.strip()):
            lines.append(content)
        else:
            if label is not None:
                rows[label] = _row(label, start, lines)
            label = None
    if label is not None:
        rows[label] = _row(label, start, lines)
    return rows


def _row(label: str, line: int, lines: list[str]) -> Row:
    # trailing blank lines separate rows; they belong to neither
    while len(lines) > 1 and not lines[-1].strip():
        lines.pop()
    text = " ".join(content.strip() for content in lines).strip()
    return Row(label, line, text, tuple(lines))
