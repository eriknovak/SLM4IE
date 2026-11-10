"""Checks the topic spec (`specs/topic.yaml`) names beyond the built-in ones.

A topic is `reference/<topic>/README.md`; every other `.md` in its folder is
an entry of the topic. The Question is one sentence ending in `?`; the
Introduction is two to four sentences, due once two entries have a
Description. Each `compare` label is a Facts row (`- **<Label>**: …`) that
every entry past draft writes, matched in any case as the page matches it.
A closed topic has no open idea in `reference/ideas.yaml` whose Touches rows
(`- <topic>/<entry> | <relation> | …`) name one of its entries. The README
cites no table or figure, since the page draws every part from its entries.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..document import Document
from ..document import Section as DocumentSection
from ..rules import (
    GUIDANCE,
    Corpus,
    check,
    is_guidance,
    is_template_guidance,
    split_code,
)
from ..spec import Field, Rule, Section
from .entry import assets

# ===============================
# Defaults
# ===============================

# the sentences a section may hold, (fewest, most)
SENTENCES = {"Question": (1, 1), "Introduction": (2, 4)}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
# a stop, then space and a word: a sentence may open lowercase, with blanked code
# or math; a quote or bracket may sit on either side
SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*\s+(?=[\"'“‘(\[]*\w)")
# a word ending in `.` that ends no sentence
ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "al.", "cf.", "vs.", "fig.", "alg.", "eq.", "sec.", "u.s.", "dr.", "mr.", "mrs."}
)
# an author's initial, as in "A. Vaswani", ends no sentence
INITIAL = re.compile(r"[A-Z]\.")
# a word ending in `.` that ends a sentence only before a capital ("one says no. Neither")
SOFT_ABBREVIATIONS = frozenset({"approx.", "ca.", "no.", "etc."})
MATH_SPAN = re.compile(r"(?<!\\)\$[^$]+\$")
QUESTION_MARK = "?"
# what may close a question after its `?`: emphasis, a quote, a bracket
CLOSING = " \t\n*_\"'”’)]"
DESCRIPTION = "Description"
# a topic needs its Introduction once this many entries have a Description
INTRODUCTION_DESCRIPTIONS = 2
FACTS = "Facts"
FACTS_STAGE = "from-paper"
ENTRY = "entry"
IDEAS_FILE = "ideas.yaml"
OPEN = "open"
CLOSED = "closed"
TOUCHES = re.compile(r"^##\s+Touches\s*$", re.IGNORECASE | re.MULTILINE)
HEADING = re.compile(r"^##\s", re.MULTILINE)
# a Touches row: `- <topic>/<entry> | <relation> | <place> | …`; group 1 is the entry
TOUCHES_ROW = re.compile(r"^[ \t]*-[ \t]+([^|\n]+?)[ \t]*\|.*$", re.MULTILINE)

# ===============================
# Idea classes
# ===============================


@dataclass(frozen=True)
class Touch:
    """One Touches row of an idea: the entry, its relation (lowercase) and the place it touches."""

    entry: str
    relation: str
    place: str


@dataclass(frozen=True)
class Idea:
    """One tracker issue labelled `idea`, as `reference/ideas.yaml` caches it."""

    number: int
    title: str
    url: str
    state: str
    touches: tuple[Touch, ...]


# ===============================
# Topic checks
# ===============================


@check("sentences")
def sentences(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written section holds as many sentences as SENTENCES allows its heading."""
    section = document.sections.get(part.heading)
    text = written_prose(section)
    if not text:
        return
    low, high = SENTENCES[part.heading]
    count = sentence_count(text)
    if not low <= count <= high:
        allowed = (
            NUMBER_WORDS[low]
            if low == high
            else f"{NUMBER_WORDS[low]} to {NUMBER_WORDS[high]}"
        )
        noun = "sentence" if count == 1 else "sentences"
        yield section.line, f"{count} {noun}, {allowed} allowed"


@check("question-end")
def question_end(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written section ends with a question mark."""
    section = document.sections.get(part.heading)
    text = written_prose(section)
    if text and not text.rstrip(CLOSING).endswith(QUESTION_MARK):
        yield section.line, f"it ends '{text.rstrip()[-20:]}'"


@check("introduction-due")
def introduction_due(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The section is present once enough entries of the topic have a Description."""
    section = document.sections.get(part.heading)
    if section is not None and section.text:
        return
    described = [
        entry.id
        for entry in topic_entries(document, corpus)
        if written_prose(entry.sections.get(DESCRIPTION))
    ]
    if len(described) >= INTRODUCTION_DESCRIPTIONS:
        yield (section.line if section else 1), f"{', '.join(described)} have one"


@check("compare-rows")
def compare_rows(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each entry of the topic past draft writes a Facts row per compare label."""
    labels = document.frontmatter.get(part.key)
    if not isinstance(labels, list) or not labels:
        return
    stages = corpus.specs[ENTRY].stages if ENTRY in corpus.specs else ()
    held = stages[stages.index(FACTS_STAGE) :] if FACTS_STAGE in stages else ()
    for entry in topic_entries(document, corpus):
        status = str(entry.frontmatter.get("status") or "")
        if status not in held:
            continue
        facts = entry.sections.get(FACTS)
        written = {
            label.lower()
            for label, row in (facts.rows.items() if facts else ())
            if not is_guidance(row.text)
        }
        missing = [str(label) for label in labels if str(label).lower() not in written]
        if missing:
            yield (
                document.key_lines[part.key],
                f"{entry.id} ({status}) has no {', '.join(missing)}",
            )


@check("closed-ideas")
def closed_ideas(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A closed topic has no open idea touching one of its entries."""
    if document.frontmatter.get(part.key) != CLOSED:
        return
    line = document.key_lines[part.key]
    try:
        ideas = open_ideas(document.folder.parent)
    except yaml.YAMLError as error:
        yield line, f"{IDEAS_FILE} is not valid YAML: {error}"
        return
    prefix = f"{document.id}/"
    for number, entries in ideas:
        touched = [entry for entry in entries if entry.startswith(prefix)]
        if touched:
            yield line, f"idea #{number} touches {', '.join(touched)}"


@check("topic-assets")
def topic_assets(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The topic cites no table or figure: no `![caption](path)` link outside code."""
    for asset in assets(document):
        yield asset.line, f"`{asset.path}`"


# ===============================
# Helper functions
# ===============================


def read_ideas(reference: Path) -> list[Idea]:
    """Every idea in `<reference>/ideas.yaml`, open or closed, with its Touches rows.

    Args:
        reference: The reference folder that holds the ideas cache.

    Returns:
        The ideas in file order; none without a cache.

    Raises:
        yaml.YAMLError: If the cache is not valid YAML.
    """
    path = reference / IDEAS_FILE
    issues = (
        yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    )
    found = []
    for issue in issues if isinstance(issues, list) else []:
        if not isinstance(issue, dict):
            continue
        body = str(issue.get("body") or "")
        touches = []
        if start := TOUCHES.search(body):
            end = HEADING.search(body, start.end())
            rows = body[start.end() : end.start() if end else len(body)]
            for match in TOUCHES_ROW.finditer(rows):
                cells = [c.strip().strip("`").strip() for c in match.group(0).split("|")]
                cells += [""] * (3 - len(cells))
                touches.append(Touch(match.group(1).strip("` "), cells[1].lower(), cells[2]))
        found.append(
            Idea(
                number=int(issue.get("number") or 0),
                title=str(issue.get("title") or ""),
                url=str(issue.get("url") or ""),
                state=str(issue.get("state") or "").lower(),
                touches=tuple(touches),
            )
        )
    return found


def open_ideas(reference: Path) -> list[tuple[int, tuple[str, ...]]]:
    """Each open idea in `<reference>/ideas.yaml`, with the entries its Touches rows name.

    Args:
        reference: The reference folder that holds the ideas cache.

    Returns:
        (issue number, entry keys) per open idea with Touches rows, in file
        order; none without a cache.

    Raises:
        yaml.YAMLError: If the cache is not valid YAML.
    """
    return [
        (idea.number, tuple(touch.entry for touch in idea.touches))
        for idea in read_ideas(reference)
        if idea.state == OPEN and idea.touches
    ]


def topic_entries(document: Document, corpus: Corpus) -> list[Document]:
    """The entries in the topic's folder, sorted by path."""
    prefix = f"{document.id}/"
    return [d for d in corpus.documents if d.kind == ENTRY and d.id.startswith(prefix)]


def written_prose(section: DocumentSection | None) -> str:
    """A section's prose with its code blanked; "" while empty or holding template guidance."""
    if section is None or not section.text or is_guidance(section.text):
        return ""
    prose, _ = split_code(section.body)
    if any(is_template_guidance(m.group(1)) for m in GUIDANCE.finditer(prose)):
        return ""
    return MATH_SPAN.sub("x", prose).strip()


def sentence_count(text: str) -> int:
    """The sentences in prose: stops before a word, abbreviations aside."""
    text = " ".join(text.split())
    ends = 0
    for m in SENTENCE_END.finditer(text):
        word = text[: m.start() + 1].rsplit(" ", 1)[-1].lstrip("\"'“‘([")
        before_capital = text[m.end() :].lstrip("\"'“‘([")[:1].isupper()
        if INITIAL.fullmatch(word) or word.lower() in ABBREVIATIONS:
            continue
        if word.lower() not in SOFT_ABBREVIATIONS or before_capital:
            ends += 1
    return ends + 1 if text else 0
