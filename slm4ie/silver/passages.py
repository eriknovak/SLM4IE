"""Draw real passages of body prose from corpus documents, by a written rule.

A passage is a run of whole paragraphs from one document. A paragraph counts as
body prose when it is long enough, ends like a sentence and is mostly letters,
which leaves out tables of contents, reference lists, tables and headings. One
passage is drawn per document, and per group where a group key is given, so ten
passages never share one template.
"""

import random
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# ===============================
# Defaults
# ===============================

MIN_WORDS = 100
MAX_WORDS = 300

# a body-prose paragraph holds at least this many words
MIN_PARAGRAPH_WORDS = 12
# and at least this share of its characters are letters or spaces
MIN_LETTER_SHARE = 0.85
SENTENCE_END = (".", "!", "?", ":", ";")
# a run of dots is a table-of-contents leader
LEADER = re.compile(r"\.{3,}|…")
# a line that stays inside a paragraph: it holds text and the one before did not end a sentence
BLANK_LINE = re.compile(r"\n\s*\n")
WHITESPACE = re.compile(r"\s+")
# letters that stand in for č, ž and š in documents extracted with a wrong encoding
MISENCODED = frozenset("ĉĈţŢċĊ")

# ===============================
# Passage
# ===============================


@dataclass(frozen=True)
class Passage:
    """One real passage.

    Attributes:
        passage_id: The document id followed by the index of the passage's first paragraph.
        doc_id: The id of the document the passage comes from.
        group: The group the document belongs to, such as a medicine; empty when ungrouped.
        text: The passage, its paragraphs joined by one line break.
        words: Number of whitespace-separated words in `text`.
    """

    passage_id: str
    doc_id: str
    group: str
    text: str
    words: int


# ===============================
# Paragraphs
# ===============================


def prose_paragraphs(text: str, wrapped: bool = False) -> list[str]:
    """Split a document into paragraphs and keep those that read as body prose.

    Args:
        text: The document text.
        wrapped: True when lines are wrapped inside a paragraph and blank lines separate
            paragraphs; False when every line is one sentence or paragraph.

    Returns:
        The body-prose paragraphs in reading order, each with single spaces.
    """
    blocks = BLANK_LINE.split(text) if wrapped else text.split("\n")
    paragraphs = (WHITESPACE.sub(" ", block).strip() for block in blocks)
    return [paragraph for paragraph in paragraphs if _is_body_prose(paragraph)]


def _is_body_prose(paragraph: str) -> bool:
    """Tell whether a paragraph is long enough, ends like a sentence is mostly letters and is not mis-encoded."""
    if len(paragraph.split()) < MIN_PARAGRAPH_WORDS or not paragraph.endswith(SENTENCE_END):
        return False
    if LEADER.search(paragraph) or MISENCODED.intersection(paragraph):
        return False
    letters = sum(character.isalpha() or character == " " for character in paragraph)
    return letters / len(paragraph) >= MIN_LETTER_SHARE


def candidate_windows(
    paragraphs: list[str],
    cue_words: Iterable[str] = (),
    min_cues: int = 1,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[tuple[int, str]]:
    """List the runs of consecutive paragraphs that fit the length range.

    Windows do not overlap: each starts after the one before ends. A window that
    cannot reach `min_words` without passing `max_words` is skipped.

    Args:
        paragraphs: Body-prose paragraphs of one document, in reading order.
        cue_words: Lowercase word stems; when given, a window must contain `min_cues` of them.
        min_cues: Fewest distinct cue stems a window must contain.
        min_words: Fewest words a window may hold.
        max_words: Most words a window may hold.

    Returns:
        `(index of the first paragraph, window text)` pairs in reading order.
    """
    cues = tuple(cue_words)
    windows = []
    start = 0
    while start < len(paragraphs):
        # grow the window until it reaches the minimum length
        end, words = start, 0
        while end < len(paragraphs) and words < min_words:
            words += len(paragraphs[end].split())
            end += 1

        text = "\n".join(paragraphs[start:end])
        fits = min_words <= words <= max_words
        lowered = text.lower()
        has_cues = not cues or sum(cue in lowered for cue in cues) >= min_cues
        if fits and has_cues:
            windows.append((start, text))
            start = end
        else:
            start += 1
    return windows


# ===============================
# Drawing
# ===============================


def draw_passages(
    documents: Iterable[Mapping[str, Any]],
    count: int,
    seed: int,
    keep: Callable[[Mapping[str, Any]], bool] = lambda document: True,
    group_of: Callable[[Mapping[str, Any]], str] = lambda document: "",
    cue_words: Iterable[str] = (),
    min_cues: int = 1,
    wrapped: bool = False,
) -> list[Passage]:
    """Draw `count` passages, one per document and one per group, with a fixed seed.

    Documents are read once, in order; the draw shuffles the kept ones, then takes
    one random window from each until `count` passages are drawn.

    Args:
        documents: Corpus records with `id`, `text` and `metadata`.
        count: Number of passages to draw.
        seed: Seed of the shuffle and of the window choice.
        keep: Predicate on a record; only records it accepts are drawn from.
        group_of: Group key of a record; no two passages share a non-empty group.
        cue_words: Lowercase word stems a passage must contain `min_cues` of.
        min_cues: Fewest distinct cue stems a passage must contain.
        wrapped: Passed to `prose_paragraphs`.

    Returns:
        The drawn passages, in draw order.

    Raises:
        ValueError: If the kept documents yield fewer than `count` passages.
    """
    rng = random.Random(seed)

    # shuffle the kept documents
    kept = [document for document in documents if keep(document)]
    rng.shuffle(kept)

    # take one window per document and group
    passages: list[Passage] = []
    seen_groups: set[str] = set()
    for document in kept:
        if len(passages) == count:
            break
        group = group_of(document)
        if group and group in seen_groups:
            continue
        windows = candidate_windows(prose_paragraphs(document["text"], wrapped), cue_words, min_cues)
        if not windows:
            continue
        index, text = rng.choice(windows)
        passages.append(Passage(f"{document['id']}#{index}", document["id"], group, text, len(text.split())))
        seen_groups.add(group)

    if len(passages) < count:
        raise ValueError(f"drew {len(passages)} passages from {len(kept)} kept documents, {count} wanted")
    return passages
