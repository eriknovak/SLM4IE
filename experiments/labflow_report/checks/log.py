"""Checks the log spec (`specs/log.yaml`) names: the entries of an exploration log.

Each `### A<n> — <what was tried>` entry is numbered from A1 in order. Its
Result holds a number with its spread, or reads `failed — <why>` for an
attempt that gave none; a digit inside a word (`top-5`, `v2`) is no number.
Its Cost opens with the number spent and its unit: `h` or `m` right after
it, or `$` before it (`1.5h`, `15m`, `$3`), so the plan's
`attempt: … · cost <n>` Log line can repeat the same figure. A row
still holding `<…>` guidance waits; the row's `required` rule reports it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..document import Document
from ..rules import Corpus, check, is_guidance, row_places
from ..spec import Row, Rule, Section

# ===============================
# Defaults
# ===============================

ENTRY_ID = re.compile(r"^A(\d+) — \S")
NUMBER = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?")
SPREAD = re.compile(
    r"(?:±|\+/-)\s*\d"
    r"|\[\s*-?\d+(?:\.\d+)?%?\s*,\s*-?\d+(?:\.\d+)?%?\s*\]"
    r"|\b(?:one|single|1) run\b",
    re.IGNORECASE,
)
COST = re.compile(r"^(?:\$\d+(?:\.\d+)?|\d+(?:\.\d+)?[hm])(?!\.?\w)")
FAILED = re.compile(r"^failed\s*[—–:-]\s*\S", re.IGNORECASE)

# ===============================
# Log checks
# ===============================


@check("log-ids")
def log_ids(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each entry title is `A<n> — <what>`, the n-th entry numbered n."""
    section = document.sections.get(part.heading)
    if section is None:
        return
    for expected, entry in enumerate(section.entries, start=1):
        match = ENTRY_ID.match(entry.title)
        if match is None or int(match.group(1)) != expected:
            yield entry.line, f"'{entry.title}', expected A{expected}"


@check("log-number")
def log_number(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A Result that did not fail holds a number."""
    for row in measured_results(document, part):
        if not NUMBER.search(row.text):
            yield row.line, ""


@check("log-spread")
def log_spread(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A Result with a number gives its spread, or says it is one run."""
    for row in measured_results(document, part):
        if NUMBER.search(row.text) and not SPREAD.search(row.text):
            yield row.line, ""


@check("log-cost")
def log_cost(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A Cost opens with the number spent in hours, minutes or dollars."""
    for _, row in row_places(document, part):
        if row and not is_guidance(row.text) and not COST.match(row.text):
            yield row.line, f"'{row.text}'"


# ===============================
# Helper functions
# ===============================


def measured_results(document: Document, part: Row) -> Iterator:
    """The written Result rows that are not `failed — <why>`."""
    for _, row in row_places(document, part):
        if row is None or is_guidance(row.text):
            continue
        if not FAILED.match(row.text):
            yield row
