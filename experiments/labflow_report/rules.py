"""Run a spec's rules over documents and report each breach as `file:line RULE-ID message`.

Built-in checks read the shape a part declares in the spec (`required`,
`type`, `values`, `words`, `form`, `link`), and `guidance` reports template
guidance (`<what one row is>`) left in a section. A page adds its own checks
with `@check("name")`; a rule names either kind by its `check` key.

A row in a section of `###` entries is checked in every entry: `required`
reports each entry without it, `words` each entry over its cap. A row's
`values` are the words its text may open with, before ` — ` or `:`. A
`link` item may be a map naming its document under the kind's key, as
`{entry: <topic>/<entry>, as: <role>}` does.

The runner itself reports one breach outside any spec: `DOC-001`, when a
document's frontmatter is not valid YAML, since no rule can read it then.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .document import Document, normalize_link, read_document
from .spec import Artifact, DocumentSpec, Field, Row, Rule, Section

# ===============================
# Defaults
# ===============================

YAML_RULE = "DOC-001"
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WORD = re.compile(r"\S+")
PLACEHOLDER = re.compile(r"<[^<>]*>")
# `<` after no word or `<`, a letter, `…`, `≤` or range lead (`3–5 `, never `5 and`),
# then `>` within the paragraph
GUIDANCE = re.compile(
    r"(?<![\w<])<((?:[A-Za-z…≤]|\d+[–-]\d+\s)(?:[^<>\n]|\n(?![ \t]*\n))*?)>"
)
QUOTE_CHARACTERS = 60
INLINE_CODE = re.compile(r"`[^`\n]*`")
# in fenced code a `<path-part>` is often the author's own, so only the template's count
FENCED_GUIDANCE = frozenset({"…", "category", "slug", "sha"})
HTML_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "br",
        "cite",
        "code",
        "del",
        "details",
        "dfn",
        "div",
        "em",
        "figcaption",
        "figure",
        "hr",
        "i",
        "img",
        "ins",
        "kbd",
        "li",
        "mark",
        "ol",
        "p",
        "pre",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "svg",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "time",
        "tr",
        "u",
        "ul",
        "var",
    }
)
WORD_CHARACTER = re.compile(r"\w")
LEAD = re.compile(r" — |:")
FORM_LINE = {
    "bullets": re.compile(r"^- "),
    "rows": re.compile(r"^- \*\*.+?\*\*"),
    "steps": re.compile(r"^\d+\. "),
    "table": re.compile(r"^\|"),
    "entries": re.compile(r"^### "),
}
PYTHON_TYPES = {"str": str, "int": int, "bool": bool, "list": list, "map": dict}

# ===============================
# Runner classes
# ===============================


@dataclass(frozen=True, order=True)
class Breach:
    """One rule a document breaks, at the line that breaks it."""

    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.rule} {self.message}"


@dataclass(frozen=True)
class Corpus:
    """The ids of every document, by kind, so a link check can resolve a name.

    A link that opens with `./` or `../` is relative to the linking document's
    folder; any other link is an id under the kind's root. A link to a kind
    with a root of its own is checked once a document sits under that root,
    so a record may name entries before reference/ is written. `documents` holds
    every document read, for a check that looks inside another one, and
    `specs` the specs they were read by, for a check that asks what one declares.
    """

    experiments: Path
    roots: dict[str, str]
    ids: dict[str, frozenset]
    documents: tuple[Document, ...] = ()
    specs: dict[str, DocumentSpec] = field(default_factory=dict)

    def has(self, kind: str, value: str, source: Document) -> bool:
        value = str(value).strip()
        if value.startswith(("./", "../")):
            base = (self.experiments / self.roots.get(kind, "")).resolve()
            target = (source.folder / value).resolve()
            if not target.is_relative_to(base):
                return False
            value = target.relative_to(base).as_posix()
        return normalize_link(value) in self.ids.get(kind, frozenset())

    def opened(self, kind: str) -> bool:
        """Whether a document sits under the kind's root, or the kind has no root of its own."""
        root = self.roots.get(kind, "")
        return not root or any(self.roots.get(d.kind) == root for d in self.documents)


# a check yields (line, detail) pairs; an empty detail reports the rule's message alone
Check = Callable[
    [Document, "Field | Section | Row | Artifact | None", Rule, Corpus],
    Iterable[tuple[int, str]],
]
CHECKS: dict[str, Check] = {}


def check(name: str) -> Callable[[Check], Check]:
    """Registers a check under the name a spec rule uses in its `check` key."""

    def register(function: Check) -> Check:
        if name in CHECKS:
            raise ValueError(f"check '{name}' is registered twice")
        CHECKS[name] = function
        return function

    return register


# ===============================
# Main functions
# ===============================


def read_corpus(
    experiments: Path, specs: dict[str, DocumentSpec]
) -> tuple[list[Document], Corpus]:
    """Reads every document every spec declares.

    Args:
        experiments: The experiments folder.
        specs: The specs by document kind.

    Returns:
        The documents, sorted by path, and the corpus of their ids.
    """
    documents = []
    for spec in specs.values():
        base = experiments / spec.root if spec.root else experiments
        documents += [
            read_document(path, spec.document, experiments, spec.root)
            for path in sorted(base.glob(spec.path))
        ]
    ids = {kind: frozenset(d.id for d in documents if d.kind == kind) for kind in specs}
    roots = {kind: spec.root for kind, spec in specs.items()}
    documents = sorted(documents, key=lambda d: d.path)
    return documents, Corpus(experiments, roots, ids, tuple(documents), dict(specs))


def run_rules(
    document: Document, spec: DocumentSpec, corpus: Corpus, gate: str = ""
) -> list[Breach]:
    """Runs every rule of a spec that holds at the document's stage.

    Args:
        document: The document to check.
        spec: Its spec.
        corpus: The ids of every document, for link checks.
        gate: A gate whose targets' rules hold at any stage; empty for none.

    Returns:
        The breaches, sorted by line.

    Raises:
        KeyError: If a rule names a check nobody registered.
    """
    path = str(document.path)
    if document.yaml_error:
        return [
            Breach(
                path,
                document.yaml_line,
                YAML_RULE,
                f"frontmatter is not valid YAML: {document.yaml_error}",
            )
        ]

    stage = document_stage(document, spec)
    breaches = []
    for rule in spec.rules:
        if not spec.applies(rule, stage, gate):
            continue
        if rule.check not in CHECKS:
            raise KeyError(
                f"rule {rule.id} uses check '{rule.check}', which is not registered"
            )
        for line, detail in CHECKS[rule.check](
            document, spec.part(rule.target), rule, corpus
        ):
            breaches.append(
                Breach(
                    path,
                    line,
                    rule.id,
                    f"{rule.message}: {detail}" if detail else rule.message,
                )
            )
    return sorted(breaches)


def document_stage(document: Document, spec: DocumentSpec) -> str | None:
    """The stage a document is held to: its stage key, or a derived stage it reached.

    A document at the stage just before a derived stage reaches it once the
    stage's entries section holds an entry whose title is not template guidance.

    Args:
        document: The document.
        spec: Its spec.

    Returns:
        The stage, or None when the spec has no stage key or the document lacks it.
    """
    if not spec.stage_key or spec.stage_key not in document.frontmatter:
        return None
    stage = str(document.frontmatter[spec.stage_key])
    for derived, heading in spec.derived_stages:
        before = spec.stages[spec.stages.index(derived) - 1]
        section = document.sections.get(heading)
        if stage == before and section and _has_written_entry(section):
            stage = derived
    return stage


# ===============================
# Built-in checks
# ===============================


@check("required")
def _required(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    if isinstance(part, Field):
        if _empty(document.frontmatter.get(part.key)):
            yield document.key_lines.get(part.key, 1), ""
    elif isinstance(part, Section):
        section = document.sections.get(part.heading)
        if section is None or not section.text:
            yield (section.line if section else 1), ""
    elif isinstance(part, Row):
        for line, row in row_places(document, part):
            if row is None or is_guidance(row.text):
                yield (row.line if row else line), ""
    elif isinstance(part, Artifact) and not any(document.folder.glob(part.pattern)):
        yield 1, f"no file matches {part.pattern}"


@check("type")
def _type(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    value = document.frontmatter.get(part.key)
    if _empty(value):
        return
    if part.type == "date":
        if not _is_iso_date(value):
            yield document.key_lines[part.key], f"'{value}' is not a YYYY-MM-DD date"
        return
    expected = PYTHON_TYPES[part.type]
    # a bool is an int to Python, never to the spec
    if not isinstance(value, expected) or (
        part.type == "int" and isinstance(value, bool)
    ):
        yield document.key_lines[part.key], f"'{value}' is not a {part.type}"


@check("values")
def _values(
    document: Document, part: Field | Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    allowed = ", ".join(map(str, part.values))
    if isinstance(part, Row):
        for _, row in row_places(document, part):
            # `<…>` template guidance is no value: `open <the word alone…>` reads `open`
            text = without_guidance(row.text) if row else ""
            lead = LEAD.split(text, maxsplit=1)[0].strip().rstrip(".,;")
            if lead and lead not in part.values:
                yield row.line, f"'{lead}' is not one of {allowed}"
        return
    value = document.frontmatter.get(part.key)
    for item in (
        value if isinstance(value, list) else ([] if _empty(value) else [value])
    ):
        if item not in part.values:
            yield document.key_lines[part.key], f"'{item}' is not one of {allowed}"


@check("words")
def _words(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    low, high = part.words
    span = f"at most {high}" if low == 0 else f"{low} to {high}"
    for line, text in _texts(document, part):
        count = len(WORD.findall(text))
        if count < low or count > high:
            yield line, f"{count} words, {span} allowed"


@check("form")
def _form(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    section = document.sections.get(part.heading)
    if section is None:
        return
    in_guidance = False
    for number, line in enumerate(section.body, start=section.line + 1):
        # an entry holds its own rows, so an entries section ends at its first `###`
        if part.form == "entries" and FORM_LINE["entries"].match(line):
            return
        # a `<…>` paragraph is template guidance, from a line opening `<` to one closing `>`
        if in_guidance or line.startswith("<"):
            in_guidance = not line.rstrip().endswith(">")
            continue
        # only top-level lines carry the form; indented lines continue an item
        if not line.strip() or line.startswith((" ", "\t")):
            continue
        is_list = any(pattern.match(line) for pattern in FORM_LINE.values())
        if part.form == "prose" and is_list:
            yield number, "a list line in a prose section"
        elif part.form in FORM_LINE and not FORM_LINE[part.form].match(line):
            yield number, f"a line outside the {part.form} form"


@check("guidance")
def _guidance(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    section = document.sections.get(part.heading)
    if section is None:
        return
    body = "\n".join(section.body)
    prose, fenced = split_code(section.body)
    leftovers = sorted(
        [m for m in GUIDANCE.finditer(prose) if is_template_guidance(m.group(1))]
        + [m for m in GUIDANCE.finditer(fenced) if m.group(1) in FENCED_GUIDANCE],
        key=lambda m: m.start(),
    )
    if leftovers:
        first = leftovers[0]
        line = section.line + 1 + body.count("\n", 0, first.start())
        quote = " ".join(first.group(0).split())
        quote = (
            quote
            if len(quote) <= QUOTE_CHARACTERS
            else f"{quote[: QUOTE_CHARACTERS - len('…>')]}…>"
        )
        more = f" and {len(leftovers) - 1} more" if len(leftovers) > 1 else ""
        yield line, f"{part.heading} keeps '{quote}'{more}"


@check("link")
def _link(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    # a link waits for its kind's root: varies and uses name entries before reference/
    if not corpus.opened(part.links_to):
        return
    value = document.frontmatter.get(part.key)
    for item in (
        value if isinstance(value, list) else ([] if _empty(value) else [value])
    ):
        # a map item names its document under the kind's key: `{entry: <id>, as: <role>}`
        if isinstance(item, dict):
            item = item.get(part.links_to)
            if not isinstance(item, str):
                continue
        if not corpus.has(part.links_to, str(item), document):
            yield document.key_lines[part.key], f"'{item}' names no {part.links_to}"


# ===============================
# Helper functions
# ===============================


def _empty(value) -> bool:
    return value is None or value == "" or value == [] or value == {}


def split_code(body: tuple[str, ...]) -> tuple[str, str]:
    """The body twice, same length: prose with all code blanked, and fenced code alone.

    Inline code in prose is blanked too, since a `<part>` in backticks is the author's own.
    """
    prose, fenced, in_fence = [], [], False
    for line in body:
        is_fence = line.lstrip().startswith("```")
        in_fence = in_fence != is_fence
        blank = " " * len(line)
        if in_fence or is_fence:
            prose.append(blank)
            fenced.append(line)
        else:
            prose.append(INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line))
            fenced.append(blank)
    return "\n".join(prose), "\n".join(fenced)


def is_template_guidance(inner: str) -> bool:
    # an HTML tag or an autolink is content, not guidance
    first = re.split(r"[\s/]", inner, maxsplit=1)[0].lower()
    if first in HTML_TAGS and (first == inner.lower().rstrip("/ ") or "=" in inner):
        return False
    return "://" not in inner and not inner.startswith("mailto:")


def is_guidance(text: str) -> bool:
    """Whether a row holds no word outside `<…>` template guidance, so it is unfilled."""
    return not WORD_CHARACTER.search(without_guidance(text))


def without_guidance(text: str) -> str:
    """The text with every `<…>` template guidance removed, nested ones too."""
    previous = None
    while previous != text:
        previous, text = text, PLACEHOLDER.sub("", text)
    return text


def _has_written_entry(section) -> bool:
    return any(not GUIDANCE.search(entry.title) for entry in section.entries)


def _is_iso_date(value) -> bool:
    # the pattern alone passes 2026-02-30; the calendar does not
    if not (isinstance(value, str) and DATE.match(value)):
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def row_places(document: Document, part: Row):
    """Yields (line, row or None) for each place the row belongs.

    A row of an entries section belongs to every `###` entry, and to nothing
    while there are none; any other row belongs to its section. An absent or
    empty section yields nothing: the section's own `required` rule reports
    it at its stage.
    """
    section = document.sections.get(part.heading)
    if section is None:
        return
    if part.in_entries:
        for entry in section.entries:
            yield entry.line, entry.rows.get(part.label)
    else:
        yield section.line, section.rows.get(part.label)


def _texts(document: Document, part) -> list[tuple[int, str]]:
    if isinstance(part, Field):
        value = document.frontmatter.get(part.key)
        return [] if _empty(value) else [(document.key_lines[part.key], str(value))]
    if isinstance(part, Section):
        section = document.sections.get(part.heading)
        return [(section.line, section.text)] if section else []
    if isinstance(part, Row):
        return [
            (row.line, row.text)
            for _, row in row_places(document, part)
            if row and not is_guidance(row.text)
        ]
    return []


# ===============================
# Page checks
# ===============================

# imported last: each page module registers its checks through `check` above
# `entry` would shadow the loop names in this module, so it takes an alias
from .checks import entry as entry_checks  # noqa: F401
from .checks import book, figures, glossary, ideas, programme, rail, record, topic  # noqa: F401
