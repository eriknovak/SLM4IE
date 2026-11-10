"""Checks the entry spec (`specs/entry.yaml`) names beyond the built-in ones.

An entry's Algorithm holds fenced ```` ```algorithm ```` blocks in a subset of
LaTeX algpseudocode, one command per line, each block followed by its Reading
rows `- **<line or range>**: text`. `algorithms` reads them for these checks
and for the entry page, so both number a line alike. A cited code path is
`` `path/from/repo/root.py` `` with an optional `:<line>` or `:<from>-<to>`,
in a Sources row or an algorithm `\\Comment{…}`. A cited table is an image
link `![<caption>](tables/<entry>-<what>.csv)` to a CSV in the topic folder;
a variant table `tables/<entry>-vs-<predecessor>.csv` runs the same forms
through the entry and its predecessor (tables.md, Entry tables). A figure
sits only in the Figure row of Algorithm or Worked example, which
checks/figures.py reads.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..document import Document, normalize_link
from ..document import Section as DocumentSection
from ..rules import CHECKS, Corpus, check, is_guidance, split_code
from ..spec import Artifact, Field, Row, Rule, Section
from .figures import in_figure_row

# ===============================
# Defaults
# ===============================

# the blocks a kind needs from from-paper on, beyond Description, Facts and Sources
KIND_SECTIONS = {
    "algorithm": ("Algorithm",),
    "component": ("Where it sits",),
    "dataset": (),
    "benchmark": (),
}
PATH_REF = re.compile(r"`([\w./-]+\.[A-Za-z]\w*)(?::(\d+)(?:-(\d+))?)?`")
FENCE_OPEN = re.compile(r"^```algorithm[ \t]*$")
FENCE_CLOSE = re.compile(r"^```[ \t]*$")
COMMAND = re.compile(r"\\([A-Za-z]+)\s*(.*)$")
CHANGED = "\\Changed"
COMMENT = "\\Comment"
LABELLED_ROW = re.compile(r"^- \*\*(.+?)\*\*:\s*(.*)$")
LINE_NUMBER = re.compile(r"\d+")
LINE_RANGE = re.compile(r"(\d+)\s*(?:[-–—]|to)\s*(\d+)")
BOLD = "**"
GETS = re.compile(r"\\gets(?![A-Za-z])")
DOLLAR = re.compile(r"(?<!\\)\$")
MATH_SPAN = re.compile(r"(?<!\\)\$([^$]+)\$")
# a symbol: an accent or font command and its argument, a command, or a letter;
# its key is the argument, the command or the letter, so `\tilde{A}` names A
SYMBOL = re.compile(
    r"\\(?:tilde|widetilde|hat|widehat|bar|overline|vec|dot|ddot|boldsymbol"
    r"|mathbf|mathcal|mathrm|mathit|mathsf|mathbb|operatorname|text)(?![A-Za-z])\s*"
    r"(?:\{([^{}]*)\}|(\\[A-Za-z]+|[A-Za-z]))"
    r"|(\\[A-Za-z]+)|([A-Za-z])"
)
NO_SYMBOL = {"\\dots", "\\ldots", "\\cdots"}
# opening command → the closers that end it
OPENERS = {"For", "ForAll", "While", "If"}
CLOSERS = {"EndFor": ("For", "ForAll"), "EndWhile": ("While",), "EndIf": ("If",)}
NUMBERED = {"State", "Return", "ElsIf", "Else", *OPENERS}
SUPPORTED = (
    "\\caption \\Require \\Ensure \\State \\For \\ForAll \\While \\If "
    "\\ElsIf \\Else \\Return \\Comment"
)
COUPLE_KEYS = {"entry", "as"}
# the sections whose Figure row may cite a figure (figures.md, Figure rows)
FIGURE_SECTIONS = ("Algorithm", "Worked example")
FIGURE_ROW = "Figure"
# a cited table or figure: `![<caption>](<path>)`, an optional title after the path
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)")
TABLE_SUFFIX = ".csv"
TABLES_FOLDER = "tables"
TABLE_STEM = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
VARIANT_SEPARATOR = "-vs-"
DIFFERENCE = "Difference from predecessor"
SAME_FORMS = "Same forms"

# ===============================
# Algorithm classes
# ===============================


@dataclass(frozen=True)
class Step:
    """One numbered line of an algorithm; `line` is its line in the file."""

    line: int
    command: str
    text: str
    comment: str
    depth: int
    is_changed: bool


@dataclass(frozen=True)
class ReadingRow:
    """One Reading row under a block: the lines it names and what it says."""

    line: int
    label: str
    text: str


@dataclass(frozen=True)
class Algorithm:
    """One ```` ```algorithm ```` block, numbered from 1 within the entry.

    `line` is the file line of the opening fence; `problems` holds a
    `(line, detail)` pair for each line outside the supported commands.
    """

    number: int
    line: int
    caption: str
    require: tuple[str, ...]
    ensure: tuple[str, ...]
    steps: tuple[Step, ...]
    reading: tuple[ReadingRow, ...]
    problems: tuple[tuple[int, str], ...]

    @property
    def code_refs(self) -> list[tuple[int, re.Match]]:
        """Each cited code path in a `\\Comment`, with the line citing it."""
        return [
            (step.line, match)
            for step in self.steps
            for match in PATH_REF.finditer(step.comment)
        ]


@dataclass(frozen=True)
class Asset:
    """One `![caption](path)` link an entry cites, with its line and section."""

    line: int
    section: str
    path: str

    @property
    def is_table(self) -> bool:
        return self.path.endswith(TABLE_SUFFIX)

    @property
    def stem(self) -> str:
        return PurePosixPath(self.path).stem


# ===============================
# Entry checks
# ===============================


@check("kind-required")
def kind_required(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The section is present and written, when the entry's kind needs it."""
    kind = str(document.frontmatter.get("kind") or "")
    if part.heading in KIND_SECTIONS.get(kind, ()):
        yield from CHECKS["required"](document, part, rule, corpus)


@check("difference-variant")
def difference_variant(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written Difference from predecessor names its predecessor in `variant_of`."""
    section = document.sections.get(part.heading)
    if section is None or document.frontmatter.get("variant_of"):
        return
    if any(not is_guidance(row.text) for row in section.rows.values()):
        yield section.line, ""


@check("variant-difference")
def variant_difference(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """An entry with `variant_of` writes its Difference from predecessor."""
    if not document.frontmatter.get("variant_of"):
        return
    section = document.sections.get(part.heading)
    if section is None or not section.text:
        yield (section.line if section else document.key_lines.get("variant_of", 1)), ""


@check("source-paths")
def source_paths(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each path a section cites exists, and its line range lies in the file."""
    section = document.sections.get(part.heading)
    if section is None:
        return
    for number, line in enumerate(section.body, start=section.line + 1):
        for match in PATH_REF.finditer(line):
            if detail := _path_problem(match, document, corpus):
                yield number, detail


@check("comment-paths")
def comment_paths(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each path an algorithm `\\Comment` cites exists, its line range in the file."""
    for algorithm in _section_algorithms(document, part):
        for number, match in algorithm.code_refs:
            if detail := _path_problem(match, document, corpus):
                yield number, f"Algorithm {algorithm.number}: {detail}"


@check("code-row-path")
def code_row_path(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Some row of the label cites a code path; with no such row, `required` reports it."""
    section = document.sections.get(part.heading)
    if section is None:
        return
    rows = [r for r in _labelled_rows(section, part.label) if not is_guidance(r[1])]
    if rows and not any(PATH_REF.search(text) for _, text in rows):
        yield rows[0][0], ""


@check("checked-comment")
def checked_comment(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Some algorithm `\\Comment` cites code; an entry with no block is left to the block rules."""
    blocks = _section_algorithms(document, part)
    if blocks and not any(algorithm.code_refs for algorithm in blocks):
        yield blocks[0].line, ""


@check("algorithm-blocks")
def algorithm_blocks(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written Algorithm section holds at least one ```` ```algorithm ```` block."""
    section = document.sections.get(part.heading)
    if section is None or not section.text or is_guidance(section.text):
        return
    if not algorithms(section):
        yield section.line, ""


@check("algorithm-parse")
def algorithm_parse(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each block line is a supported command, and every opener is closed."""
    for algorithm in _section_algorithms(document, part):
        for number, detail in algorithm.problems:
            yield number, f"Algorithm {algorithm.number}: {detail}"


@check("algorithm-caption")
def algorithm_caption(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each block holds a `\\caption{…}`."""
    for algorithm in _section_algorithms(document, part):
        if not algorithm.caption:
            yield algorithm.line, f"Algorithm {algorithm.number}"


@check("algorithm-reading")
def algorithm_reading(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each block is followed by at least one Reading row."""
    for algorithm in _section_algorithms(document, part):
        if not algorithm.reading:
            yield algorithm.line, f"Algorithm {algorithm.number}"


@check("reading-lines")
def reading_lines(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each Reading row names a line number, and no line past the block's last."""
    for algorithm in _section_algorithms(document, part):
        count = len(algorithm.steps)
        for row in algorithm.reading:
            # a `<n>` label is template guidance, left to the guidance rule
            if is_guidance(row.label):
                continue
            numbers = [int(n) for n in LINE_NUMBER.findall(row.label)]
            if BOLD in row.label and numbers:
                label = _one_label(row.label, numbers)
                detail = f"'{row.label}' holds **; write one label, as **{label}**"
                yield row.line, f"Algorithm {algorithm.number}: {detail}"
            elif not numbers:
                yield (
                    row.line,
                    f"Algorithm {algorithm.number}: '{row.label}' is no line",
                )
            elif max(numbers) > count:
                detail = f"'{row.label}' names line {max(numbers)}, the algorithm has {count}"
                yield row.line, f"Algorithm {algorithm.number}: {detail}"


@check("reading-symbols")
def reading_symbols(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each Reading row names in its `$…$` math the symbols its lines assign.

    A symbol is assigned on the left of a `\\gets` inside `$…$`; the row names
    it when its math holds the same letter or command (`$f$` names `f(w)`).
    """
    for algorithm in _section_algorithms(document, part):
        count = len(algorithm.steps)
        for row in algorithm.reading:
            if is_guidance(row.label):
                continue

            # the symbols the row's lines assign, by key, first spelling kept
            assigned: dict[str, str] = {}
            for number in _label_lines(row.label):
                if 1 <= number <= count:
                    for spelling, key in _assigned(algorithm.steps[number - 1].text):
                        assigned.setdefault(key, spelling)

            named = {
                _symbol_key(match)
                for span in MATH_SPAN.findall(row.text)
                for match in SYMBOL.finditer(span)
            }
            missing = [f"${s}$" for key, s in assigned.items() if key not in named]
            if missing:
                detail = f"row '{row.label}' names no {', '.join(missing)}"
                yield row.line, f"Algorithm {algorithm.number}, {detail}"


@check("couples-form")
def couples_form(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each item is `<topic>/<entry>` or `{entry: <topic>/<entry>, as: <role>}`."""
    value = document.frontmatter.get(part.key)
    if not isinstance(value, list):
        return
    for item in value:
        if _couple_target(item) is None:
            yield document.key_lines[part.key], f"'{item}'"


@check("table-files")
def table_files(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each table the entry cites exists, from the entry's folder."""
    for asset in assets(document):
        if asset.is_table and not (document.folder / asset.path).is_file():
            yield asset.line, f"`{asset.path}`"


@check("entry-figures")
def entry_figures(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Every image link is a table, but in the Figure row of Algorithm or Worked example."""
    for asset in assets(document):
        if asset.is_table:
            continue
        if asset.section in FIGURE_SECTIONS and in_figure_row(
            document, asset.section, FIGURE_ROW, asset.line
        ):
            continue
        yield asset.line, f"`{asset.path}`"


@check("table-names")
def table_names(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each cited table sits in the topic's tables/ and opens with an entry of the topic."""
    topic = document.id.split("/", 1)[0]
    names = [
        name.split("/", 1)[1]
        for name in corpus.ids.get(document.kind, ())
        if name.startswith(f"{topic}/")
    ]
    for asset in assets(document):
        if not asset.is_table:
            continue
        folder = (document.folder / asset.path).resolve().parent
        if folder != (document.folder / TABLES_FOLDER).resolve():
            yield asset.line, f"`{asset.path}` sits outside {topic}/{TABLES_FOLDER}/"
        elif not TABLE_STEM.match(asset.stem) or not any(
            asset.stem.startswith(f"{name}-") for name in names
        ):
            yield asset.line, f"`{asset.path}` opens with no entry of {topic}"


@check("variant-table")
def variant_table(
    document: Document, part: None, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A variant table names this entry and its `variant_of`, cited in the Same forms row."""
    variant_of = document.frontmatter.get("variant_of")
    predecessor = _entry_name(variant_of) if isinstance(variant_of, str) else ""
    for asset in _variant_tables(document):
        entry, named = asset.stem.split(VARIANT_SEPARATOR, 1)
        if entry != document.path.stem:
            yield asset.line, f"`{asset.path}` names {entry}, not {document.path.stem}"
        elif predecessor and named != predecessor:
            yield (
                asset.line,
                f"`{asset.path}` names {named}, but variant_of is {variant_of}",
            )
        elif asset.section != DIFFERENCE:
            yield asset.line, f"`{asset.path}` is cited under {asset.section}"
        elif (row := _row_at(document.sections[DIFFERENCE], asset.line)) != SAME_FORMS:
            yield asset.line, f"`{asset.path}` is cited in the {row or 'no'} row"


@check("variant-columns")
def variant_columns(
    document: Document, part: Artifact, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A variant table that exists is headed form, <predecessor>, <entry>, why."""
    for asset in _variant_tables(document):
        path = document.folder / asset.path
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        entry, predecessor = asset.stem.split(VARIANT_SEPARATOR, 1)
        expected = ["form", _snake(predecessor), _snake(entry), "why"]
        if header != expected:
            yield asset.line, f"`{asset.path}` has {header}, not {expected}"


@check("input-row")
def input_row(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A section that cites a table holds the row before its first table."""
    section = document.sections.get(part.heading)
    if section is None:
        return
    tables = [
        a.line for a in assets(document) if a.is_table and a.section == part.heading
    ]
    if not tables:
        return
    rows = _labelled_rows(section, part.label)
    if not rows or rows[0][0] > min(tables):
        yield min(tables), ""


# ===============================
# Helper functions
# ===============================


def assets(document: Document) -> list[Asset]:
    """The tables and figures an entry cites, in file order.

    Args:
        document: The entry.

    Returns:
        One asset per `![caption](path)` link outside code, in any section;
        a link whose path holds `<…>` template guidance is left out.
    """
    found = []
    for heading, section in document.sections.items():
        prose = split_code(section.body)[0].split("\n")
        for number, line in enumerate(prose, start=section.line + 1):
            found += [
                Asset(number, heading, match.group(1))
                for match in IMAGE_LINK.finditer(line)
                if "<" not in match.group(1)
            ]
    return found


def _row_at(section: DocumentSection, line: int) -> str:
    # the label of the row a line belongs to: the last row opening at or above it
    opened = [
        label
        for number, label, _ in _rows(section.body, section.line + 1)
        if number <= line
    ]
    return opened[-1] if opened else ""


def _variant_tables(document: Document) -> list[Asset]:
    return [
        asset
        for asset in assets(document)
        if asset.is_table and VARIANT_SEPARATOR in asset.stem
    ]


def _entry_name(link: str) -> str:
    # the file stem a `<topic>/<entry>` or `./<entry>.md` link names
    return normalize_link(link).rsplit("/", 1)[-1]


def _snake(name: str) -> str:
    return name.replace("-", "_")


def algorithms(section: DocumentSection) -> list[Algorithm]:
    """The algorithm blocks of a section, each with its Reading.

    Args:
        section: The entry's Algorithm section.

    Returns:
        One algorithm per closed ```` ```algorithm ```` fence, in file order.
        The Reading of a block is the `- **<label>**: text` rows between its
        closing fence and the next block, or the section's end.
    """
    body = section.body
    fences = []
    index = 0
    while index < len(body):
        if FENCE_OPEN.match(body[index]):
            close = next(
                (j for j in range(index + 1, len(body)) if FENCE_CLOSE.match(body[j])),
                None,
            )
            if close is None:
                break
            fences.append((index, close))
            index = close + 1
        else:
            index += 1

    found = []
    for number, (start, end) in enumerate(fences, start=1):
        after = fences[number][0] if number < len(fences) else len(body)
        first = section.line + 1
        reading = tuple(
            ReadingRow(*row) for row in _rows(body[end + 1 : after], first + end + 1)
        )
        found.append(
            _parse_block(number, first + start, body[start + 1 : end], reading)
        )
    return found


def _parse_block(
    number: int, line: int, source: tuple[str, ...], reading: tuple[ReadingRow, ...]
) -> Algorithm:
    caption, require, ensure, steps, problems = "", [], [], [], []
    stack: list[tuple[str, int]] = []
    for offset, raw in enumerate(source, start=1):
        text, at = raw.strip(), line + offset
        if not text or text.startswith("%"):
            continue

        # split off the marks around the command
        is_changed = text.startswith(CHANGED)
        if is_changed:
            text = text[len(CHANGED) :].strip()
        match = COMMAND.match(text)
        if not match:
            problems.append((at, f"'{_quote(text)}' opens with no command"))
            continue
        command, rest = match.groups()
        comment = ""
        if (position := rest.find(COMMENT)) >= 0:
            comment = _braced(rest[position + len(COMMENT) :])[0]
            rest = rest[:position].strip()

        # read the command
        depth = len(stack)
        if command == "caption":
            caption = _braced(rest)[0]
        elif command == "Require":
            require.append(rest)
        elif command == "Ensure":
            ensure.append(rest)
        elif command in ("ElsIf", "Else") and (not stack or stack[-1][0] != "If"):
            problems.append((at, f"\\{command} outside an \\If"))
        elif command in CLOSERS:
            if stack and stack[-1][0] in CLOSERS[command]:
                stack.pop()
            else:
                problems.append((at, f"\\{command} closes nothing"))
        elif command in NUMBERED:
            if command in OPENERS or command == "ElsIf":
                rest = _braced(rest)[0]
            if command in ("ElsIf", "Else"):
                depth -= 1
            steps.append(Step(at, command, rest, comment, depth, is_changed))
            if command in OPENERS:
                stack.append((command, at))
        else:
            problems.append(
                (at, f"unknown command \\{command} (supported: {SUPPORTED})")
            )
    problems += [(at, f"\\{command} is never closed") for command, at in stack]
    return Algorithm(
        number,
        line,
        caption,
        tuple(require),
        tuple(ensure),
        tuple(steps),
        reading,
        tuple(sorted(problems)),
    )


def _section_algorithms(document: Document, part: Section) -> list[Algorithm]:
    section = document.sections.get(part.heading)
    return algorithms(section) if section else []


def _braced(text: str) -> tuple[str, str]:
    # the content of a leading `{…}` and what follows it; braces inside balance
    text = text.strip()
    if not text.startswith("{"):
        return text, ""
    depth = 0
    for index, character in enumerate(text):
        depth += (character == "{") - (character == "}")
        if depth == 0:
            return text[1:index], text[index + 1 :].strip()
    return text[1:], ""


def _path_problem(match: re.Match, document: Document, corpus: Corpus) -> str:
    # a path is from the repository root, else experiments/, else the entry's folder
    path = match.group(1)
    bases = (corpus.experiments.parent, corpus.experiments, document.folder)
    found = next((base / path for base in bases if (base / path).is_file()), None)
    if found is None:
        return f"`{path}` does not exist"
    if not match.group(2):
        return ""
    first, last = int(match.group(2)), int(match.group(3) or match.group(2))
    count = len(found.read_text(encoding="utf-8", errors="replace").splitlines())
    if first < 1 or last > count:
        span = f"{first}-{last}" if match.group(3) else f"{first}"
        return f"`{path}:{span}` lies outside the file's {count} lines"
    return ""


def _labelled_rows(section: DocumentSection, label: str) -> list[tuple[int, str]]:
    # every row of the label with its line; a label may repeat
    return [
        (number, text)
        for number, name, text in _rows(section.body, section.line + 1)
        if name.lower() == label.lower()
    ]


def _rows(lines: tuple[str, ...], first: int) -> list[tuple[int, str, str]]:
    # each top-level `- **Label**: text` row as (line, label, text), its indented
    # lines joined; a blank line keeps the row open for an indented one after it
    rows: list[tuple[int, str, list[str]]] = []
    is_open = False
    for number, line in enumerate(lines, start=first):
        match = LABELLED_ROW.match(line)
        if match:
            rows.append((number, match.group(1).strip(), [match.group(2)]))
            is_open = True
        elif is_open and (line.startswith((" ", "\t")) or not line.strip()):
            rows[-1][2].append(line.strip())
        else:
            is_open = False
    return [
        (number, label, " ".join(part for part in parts if part).strip())
        for number, label, parts in rows
    ]


def _label_lines(label: str) -> list[int]:
    # the lines a Reading label names: each `<from>-<to>` range, then each number
    lines = []
    for match in LINE_RANGE.finditer(label):
        lines += range(int(match.group(1)), int(match.group(2)) + 1)
    lines += [int(n) for n in LINE_NUMBER.findall(LINE_RANGE.sub("", label))]
    return lines


def _one_label(label: str, numbers: list[int]) -> str:
    # one label for a `**2** and **3**` split: a range when the lines run on
    first, last = numbers[0], numbers[-1]
    if " to " in label or numbers == list(range(first, last + 1)):
        return f"{first}–{last}"
    return ", ".join(str(n) for n in numbers)


def _assigned(text: str) -> list[tuple[str, str]]:
    # each symbol left of a `\gets` inside `$…$`, as (spelling, key); a tuple
    # `(a, b)` or a list `q, k, v` assigns each of its items
    found = []
    previous = -1
    for match in GETS.finditer(text):
        dollars = [d.start() for d in DOLLAR.finditer(text, 0, match.start())]
        if len(dollars) % 2 == 0:
            continue
        start = dollars[-1] + 1
        items = _split_top(text[max(start, previous) : match.start()])
        # a second `\gets` in one span assigns only the item before it
        if previous > start:
            items = items[-1:]
        previous = match.end()
        if len(items) == 1 and items[0].startswith("(") and items[0].endswith(")"):
            items = _split_top(items[0][1:-1])
        for item in items:
            head = SYMBOL.match(item)
            if head and _symbol_key(head) not in NO_SYMBOL:
                found.append((item, _symbol_key(head)))
    return found


def _split_top(text: str) -> list[str]:
    # the items of a `,` or `;` list, brackets kept whole
    items, depth, current = [], 0, ""
    for character in text:
        depth += (character in "({[") - (character in ")}]")
        if character in ",;" and depth == 0:
            items.append(current)
            current = ""
        else:
            current += character
    return [item.strip() for item in [*items, current] if item.strip()]


def _symbol_key(match: re.Match) -> str:
    return next(group for group in match.groups() if group is not None).strip()


def _couple_target(item) -> str | None:
    # the entry a couples_to item names, or None for an item of neither form
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict) and set(item) == COUPLE_KEYS:
        entry, role = item.get("entry"), item.get("as")
        if (
            isinstance(entry, str)
            and entry.strip()
            and isinstance(role, str)
            and role.strip()
        ):
            return entry.strip()
    return None


def _quote(text: str, characters: int = 40) -> str:
    return text if len(text) <= characters else f"{text[: characters - 1]}…"
