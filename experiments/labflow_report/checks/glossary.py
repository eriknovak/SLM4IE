"""Checks the glossary spec (`specs/glossary.yaml`) names beyond the built-in ones.

A term line is `- **Term** (alias, alias): definition` under a `##` heading:
`## Metrics`, a term group the project names for itself, or `## Other` for
the terms no group takes. `terms` reads them for these checks, the Glossary
page (`glossary_page.py`) and build_report.py, so all three read a line alike.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from ..document import Document
from ..rules import CHECKS, WORD, Corpus, check, is_guidance, row_places
from ..spec import Row, Rule
from ..spec import Section as SpecSection

# ===============================
# Defaults
# ===============================

TERM_LINE = re.compile(r"^- \*\*(.+?)\*\*(?:\s*\((.+?)\))?:\s*(.+)$")
LIST_LINE = re.compile(r"^- ")
# the `##` heading read_document splits sections at
HEADING = re.compile(r"^## +(.+?)\s*$")
METRICS = "Metrics"
OTHER = "Other"
# the heading Other replaced; GLO-003 names the rename
RETIRED_TERMS = "Terms"
DEFINITION_WORDS = 40
# a Metrics row's primary part: after `primary`, up to `; secondary` or `. Secondary`
PRIMARY = re.compile(r"\bprimary\b(.*?)(?:[;.]\s*secondary\b|$)", re.I | re.S)
# a short or symbolic form (`F1`, `P@k`, `κ`) matches as written, words in any case
EXACT_FORM_LENGTH = 4
WORD_FORM = re.compile(r"[^\W\d_]+(?:[ '-][^\W\d_]+)*")

# ===============================
# Term class
# ===============================


@dataclass(frozen=True)
class Term:
    """One term line of the glossary; `section` is Metrics, a group name or Other."""

    name: str
    aliases: tuple[str, ...]
    definition: str
    section: str
    line: int

    @property
    def forms(self) -> tuple[str, ...]:
        """The name and every alias, each once."""
        return tuple(dict.fromkeys((self.name, *self.aliases)))


# ===============================
# Glossary checks
# ===============================


@check("term-lines")
def term_lines(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each list line under a `##` heading parses as a term line."""
    for heading, number, line in _top_lines(document):
        if LIST_LINE.match(line) and not TERM_LINE.match(line):
            yield number, f"'{_quote(line)}' under {heading}"


@check("group-form")
def group_form(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each `##` section but Metrics, a group or Other, holds term lines only."""
    for heading in document.sections:
        if heading == METRICS:
            continue
        rows = SpecSection(heading=heading, form="rows")
        for number, detail in CHECKS["form"](document, rows, rule, corpus):
            yield number, f"{detail} under {heading}"


@check("retired-terms")
def retired_terms(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """No `## Terms` heading: Other holds the terms no group takes."""
    section = document.sections.get(RETIRED_TERMS)
    if section is not None:
        yield section.line, ""


@check("shared-forms")
def shared_forms(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """No name or alias belongs to two terms; case does not tell forms apart."""
    first: dict[str, Term] = {}
    for term in terms(document):
        for form in term.forms:
            owner = first.setdefault(form.casefold(), term)
            if owner is not term:
                yield term.line, f"'{form}' also names {owner.name} (line {owner.line})"


@check("definition-words")
def definition_words(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each definition holds at most DEFINITION_WORDS words."""
    for term in terms(document):
        count = len(WORD.findall(term.definition))
        if count > DEFINITION_WORDS:
            yield term.line, f"{term.name} has {count} words, at most {DEFINITION_WORDS}"


@check("term-sections")
def term_sections(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Every term sits under one `##` heading: none above the first, no heading twice."""
    first: dict[str, int] = {}
    for number, line in enumerate(_lines(document), start=1):
        if match := HEADING.match(line):
            heading = match.group(1)
            if heading in first:
                yield number, f"'## {heading}' again, first at line {first[heading]}"
            first.setdefault(heading, number)
        elif not first and TERM_LINE.match(line):
            yield number, f"'{_quote(line)}' above the first heading"


# ===============================
# Record checks
# ===============================


@check("metric-defined")
def metric_defined(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A record's Design Metrics row names a glossary term in its primary part.

    A row with no `primary` is read whole, so it names at least one term.
    """
    glossary = next((d for d in corpus.documents if d.kind == "glossary"), None)
    forms = [form for term in terms(glossary) for form in term.forms] if glossary else []
    for _, row in row_places(document, part):
        if row is None or is_guidance(row.text):
            continue
        match = PRIMARY.search(row.text)
        primary = match.group(1) if match else row.text
        if not any(names_form(primary, form) for form in forms):
            yield row.line, f"'{_quote(' '.join(primary.split()))}'"


# ===============================
# Helper functions
# ===============================


def names_form(text: str, form: str) -> bool:
    """Whether the text names a term form as a whole word."""
    is_words = len(form) >= EXACT_FORM_LENGTH and WORD_FORM.fullmatch(form)
    flags = re.I if is_words else 0
    return bool(re.search(rf"(?<![\w-]){re.escape(form)}(?![\w-])", text, flags))


def terms(document: Document) -> list[Term]:
    """The term lines under every `##` heading, in file order.

    Args:
        document: The glossary.

    Returns:
        One term per line that parses, each with the heading it sits under
        (Metrics, a group or Other); a line that does not parse is left out.
        A heading used twice keeps the terms of both copies (GLO-008 flags it).
    """
    found = []
    for heading, number, line in _top_lines(document):
        if match := TERM_LINE.match(line):
            aliases = (a.strip() for a in (match.group(2) or "").split(","))
            found.append(
                Term(
                    match.group(1).strip(),
                    tuple(a for a in aliases if a),
                    match.group(3).strip(),
                    heading,
                    number,
                )
            )
    return found


def _lines(document: Document) -> list[str]:
    return document.path.read_text(encoding="utf-8").splitlines()


def _top_lines(document: Document) -> Iterator[tuple[str, int, str]]:
    # read from the file, since document.sections keeps one copy of a repeated heading;
    # indented lines continue an item, so only top-level lines open one
    heading = None
    for number, line in enumerate(_lines(document), start=1):
        if match := HEADING.match(line):
            heading = match.group(1)
        elif heading is not None and line.strip() and not line.startswith((" ", "\t")):
            yield heading, number, line


def _quote(line: str, characters: int = 60) -> str:
    return line if len(line) <= characters else f"{line[: characters - 1]}…"
