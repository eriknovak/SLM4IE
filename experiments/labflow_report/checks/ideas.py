"""Checks the ideas spec (`specs/ideas.yaml`) names.

The document is `reference/ideas.yaml`, the builder's copy of the tracker
issues labelled `idea`. Each idea has a `## Touches` line at least; each
line's relation is one of IDEA_RELATIONS and its `<topic>/<entry>` is an
entry of the reference. A breach sits on the line of the idea's `number:`
key, where the cache holds the issue.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import yaml

from ..document import Document
from ..rules import Corpus, check
from ..spec import Rule
from .topic import IDEAS_FILE, Idea, read_ideas

# ===============================
# Defaults
# ===============================

IDEA_RELATIONS = ("changes", "already has", "does not apply")
ENTRY = "entry"

# ===============================
# Idea checks
# ===============================


@check("idea-touches")
def idea_touches(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each idea has a Touches line; a cache that is no YAML is named here once."""
    try:
        ideas = read_ideas(document.folder)
    except yaml.YAMLError as error:
        yield 1, f"{IDEAS_FILE} is not valid YAML: {error}"
        return
    lines = number_lines(document)
    for idea in ideas:
        if not idea.touches:
            yield lines.get(idea.number, 1), f"idea #{idea.number}"


@check("idea-relation")
def idea_relation(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    lines = number_lines(document)
    for idea in readable_ideas(document):
        for touch in idea.touches:
            if touch.relation not in IDEA_RELATIONS:
                yield (
                    lines.get(idea.number, 1),
                    f"idea #{idea.number} gives {touch.entry} '{touch.relation}'",
                )


@check("idea-entry")
def idea_entry(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    lines = number_lines(document)
    for idea in readable_ideas(document):
        for touch in idea.touches:
            if not corpus.has(ENTRY, touch.entry, document):
                yield lines.get(idea.number, 1), f"idea #{idea.number} touches {touch.entry}"


# ===============================
# Helper functions
# ===============================


def readable_ideas(document: Document) -> list[Idea]:
    """The cached ideas; none when the cache is no YAML, which idea-touches reports."""
    try:
        return read_ideas(document.folder)
    except yaml.YAMLError:
        return []


def number_lines(document: Document) -> dict[int, int]:
    """Issue number → the 1-based line of its `number:` key in the cache."""
    found = {}
    lines = document.path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if match := re.match(r"^-?\s*number:\s*(\d+)\s*$", line):
            found.setdefault(int(match.group(1)), line_number)
    return found
