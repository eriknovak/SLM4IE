"""Zero-shot NER prompt: sentence windows, the chat template, the answer schema and its parser.

A document is read in windows of whole sentences, each one exact slice of the
document text, so a span aligned in a window maps back by adding the window's
start. Every model gets the same two chat turns per window: the label set with
its plain-language descriptions, the rules an answer must follow, and the
window text. It answers with one JSON object, `{"mentions": [{"text", "label"}]}`,
whose mentions `slm4ie.evaluation.scoring.align_mentions` maps to spans.

An answer the schema rejects parses to `None`, which the caller scores as no
mentions for its window and counts as one missed window: the miss rate is the
share of windows whose answer fails to parse.
"""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm4ie.utils.config import load_yaml

# ===============================
# Defaults
# ===============================

MAX_WINDOW_CHARS = 2_000

# names that match no gold mention of any test split
DEFAULT_EXAMPLES = "unseen"

SYSTEM_PROMPT = "You annotate named entities in Slovene text. You answer with JSON only."

USER_PROMPT = """Find every named entity in the text below and give each one a label from this list:

{descriptions}

Rules:
- Copy each mention exactly as it is written in the text, character for character, in the same inflected \
form and letter case. Do not translate, correct, lemmatise or shorten it.
- List every occurrence: a name written three times in the text is listed three times.
- List the mentions in the order they appear in the text.
- A mention is the name itself, without the words or punctuation around it.
- If the text holds no named entity, answer with an empty list.

Answer with one JSON object of this shape and nothing else:
{{"mentions": [{{"text": "<mention copied from the text>", "label": "<one of {label_names}>"}}]}}

Text:
{text}"""

# a run of sentence-final punctuation and closing quotes followed by whitespace
SENTENCE_END = re.compile(r"[.!?…]+[\"'»«”’)\]]*(?=\s)")
CLOSERS = "\"'»«”’)]"
OPENERS = "\"'„“»«(["

# words a period follows without ending the sentence
ABBREVIATIONS = frozenset(
    {
        "dr", "mag", "prof", "doc", "g", "ga", "gdč", "gosp", "ing", "dipl", "univ", "red", "izr", "sv", "st",
        "ul", "št", "tel", "npr", "itd", "ipd", "idr", "oz", "tj", "op", "pribl", "cca", "mio",
        "mrd", "jan", "feb", "mar", "apr", "jun", "jul", "avg", "sep", "okt", "nov", "dec", "mr", "mrs", "ms",
        "jr", "sr", "inc", "ltd", "co", "vs",
    }
)  # fmt: skip

FENCE = re.compile(r"^```[A-Za-z]*\s*\n?(.*?)\n?```$", re.DOTALL)

# ===============================
# Windows
# ===============================


@dataclass(frozen=True)
class Window:
    """A run of whole sentences the model reads in one call.

    Attributes:
        start: Character offset of the window in the document text.
        text: The document text from `start` on, ending at a sentence end.
    """

    start: int
    text: str


def sentence_windows(text: str, max_chars: int = MAX_WINDOW_CHARS) -> list[Window]:
    """Cut a document into windows of whole sentences of up to `max_chars` characters.

    A sentence ends at a line break, or at `.`, `!`, `?` or `…` followed by
    whitespace and a capital letter or an opening quote, unless the period
    follows an abbreviation, an initial or an ordinal number. Sentences are
    packed greedily in reading order; a sentence longer than `max_chars` is
    never cut and forms its own window.

    Args:
        text: The document text.
        max_chars: Most characters a window of two or more sentences may span.

    Returns:
        The windows in reading order; none for a text of only whitespace.
    """
    windows = []
    window_start = window_end = None
    for start, end in _sentence_bounds(text):
        # extend the open window while the sentence still fits
        if window_start is not None and end - window_start <= max_chars:
            window_end = end
            continue

        if window_start is not None:
            windows.append(Window(start=window_start, text=text[window_start:window_end]))
        window_start, window_end = start, end

    if window_start is not None:
        windows.append(Window(start=window_start, text=text[window_start:window_end]))
    return windows


# ===============================
# Prompt and Schema
# ===============================


def build_messages(text: str, descriptions: Mapping[str, str]) -> list[dict[str, str]]:
    """Build the chat turns that ask a model for the named entities of one window.

    Args:
        text: The window text.
        descriptions: Label to plain-language description, in prompt order.

    Returns:
        A system turn and a user turn, each `{"role": str, "content": str}`.
    """
    described = "\n".join(f"- {label}: {description}" for label, description in descriptions.items())
    user = USER_PROMPT.format(descriptions=described, label_names=", ".join(descriptions), text=text)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def answer_schema(labels: Iterable[str]) -> dict[str, Any]:
    """Build the JSON schema an answer must satisfy, for structured-output decoding.

    Args:
        labels: The test set's own labels; no other label validates.

    Returns:
        A JSON schema of an object holding a `mentions` list of `{"text", "label"}` objects.
    """
    mention = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}, "label": {"type": "string", "enum": list(labels)}},
        "required": ["text", "label"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"mentions": {"type": "array", "items": mention}},
        "required": ["mentions"],
        "additionalProperties": False,
    }


def parse_answer(answer: str, labels: Iterable[str]) -> list[dict[str, str]] | None:
    """Read the mentions from a model's answer, or `None` when the schema rejects it.

    One markdown code fence around the JSON is stripped first. The answer is
    accepted only when it satisfies `answer_schema(labels)`: an object with
    exactly the key `mentions`, a list of objects with exactly a non-empty
    string `text` and a `label` from `labels`.

    Args:
        answer: The raw text the model returned.
        labels: The test set's own labels.

    Returns:
        The mentions as `{"text", "label"}` dicts in answer order, or `None` for an answer off the schema.
    """
    allowed = set(labels)

    # strip one code fence and decode
    raw = answer.strip()
    fenced = FENCE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        decoded = json.loads(raw)
    except ValueError:
        return None

    # check the shape against the schema
    if not isinstance(decoded, dict) or set(decoded) != {"mentions"} or not isinstance(decoded["mentions"], list):
        return None
    mentions = []
    for item in decoded["mentions"]:
        if not isinstance(item, dict) or set(item) != {"text", "label"}:
            return None
        text, label = item["text"], item["label"]
        if not isinstance(text, str) or not text or not isinstance(label, str) or label not in allowed:
            return None
        mentions.append({"text": text, "label": label})
    return mentions


# ===============================
# Label Descriptions
# ===============================


def load_label_descriptions(config_path: Path, entry: str, examples: str = DEFAULT_EXAMPLES) -> dict[str, str]:
    """Load the label descriptions of the label set a task registry entry uses.

    Each description is the label's scope followed by the names of one
    example variant, `"<description> Examples: <name>, <name>."`.

    Args:
        config_path: Path to the label description file, `configs/evaluation/ner-labels.yaml`.
        entry: Task registry entry key, such as `ner/suk`.
        examples: Example variant to append, such as `unseen` or `common`.

    Returns:
        Label to description, in the order the file lists them.

    Raises:
        KeyError: If no label set lists `entry`, or a label lacks the `examples` variant.
        ValueError: If more than one label set lists `entry`.
    """
    label_sets = load_yaml(config_path)["label_sets"]
    matches = [name for name, label_set in label_sets.items() if entry in label_set["entries"]]
    if not matches:
        raise KeyError(f"entry '{entry}' is in no label set of {config_path}")
    if len(matches) > 1:
        raise ValueError(f"entry '{entry}' is in several label sets of {config_path}: {matches}")

    descriptions = {}
    for label, spec in label_sets[matches[0]]["labels"].items():
        if examples not in spec["examples"]:
            raise KeyError(f"label '{label}' of {config_path} has no example variant '{examples}'")
        scope = " ".join(str(spec["description"]).split())
        descriptions[str(label)] = f"{scope} Examples: {', '.join(spec['examples'][examples])}."
    return descriptions


# ===============================
# Helpers
# ===============================


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Find the sentences of a text as `(start, end)` offsets, trimmed of whitespace."""
    bounds = []
    for line in re.finditer(r"[^\n]+", text):
        start = line.start()
        for match in SENTENCE_END.finditer(text, line.start(), line.end()):
            if _ends_sentence(text, match, line.start(), line.end()):
                bounds.append((start, match.end()))
                start = match.end()
        bounds.append((start, line.end()))

    # trim the whitespace around each sentence
    trimmed = []
    for start, end in bounds:
        piece = text[start:end]
        if piece.strip():
            start += len(piece) - len(piece.lstrip())
            trimmed.append((start, start + len(piece.strip())))
    return trimmed


def _ends_sentence(text: str, match: re.Match[str], line_start: int, line_end: int) -> bool:
    """Tell whether a run of final punctuation ends a sentence inside one line."""
    # the next sentence opens with a capital letter or a quote
    following = text[match.end() : line_end].lstrip()
    if not following or not (following[0].isupper() or following[0] in OPENERS):
        return False

    # a lone period after an abbreviation, an initial or an ordinal ends nothing
    if match.group().rstrip(CLOSERS) != ".":
        return True
    word = text[line_start : match.start()].rsplit(maxsplit=1)[-1:]
    word = word[0].lstrip(OPENERS) if word else ""
    is_initial = len(word) == 1 and word.isalpha()
    return not (is_initial or word.isdigit() or "." in word or word.lower() in ABBREVIATIONS)
