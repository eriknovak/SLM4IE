"""The reference entry page on the new builder (ADR-0005).

`entry_page` turns one `reference/<topic>/<entry>.md` into the values
`templates/entry.html` draws: the kicker and title; the status line (the
status square and word, the topic, the predecessor, what it is part of and
each role another entry gives it through `couples_to`); the summary; the
sticky section bar; the Description closed by its Facts table, then the
folded What the paper adds and Why it matters here; then the sections in the
order the entry's kind sets, Evidence, Limits, Related and Sources last. Every
entry the page names shows its title with its status square. A draft shows
each section its kind expects with an empty state; a written entry drops the
empty optional ones. Every algorithm fold starts closed.

Rows the page reads beyond the spec's: a Facts row `<short> — <note>` shows
the short value with the note on hover, a Paper row `<cite>, <id>, <section>
— <role>` puts the role on hover, a Difference step row `<predecessor> |
<this entry> | Alg. <n>, line <m>` is a line of the step table, and a
Definition row `- **$<symbol>$**: <meaning>` is a symbol of the box.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from markupsafe import Markup, escape

# record imports builder, which imports this module: read record's names at call time
from . import markup, record as records
from .builder import ENTRY_STATUS, Model, PageView, Site, entry_page_id, entry_ref, entry_title, view
from .checks import topic as topic_checks
from .checks.entry import KIND_SECTIONS, Algorithm, algorithms
from .checks.figures import DIAGRAM_SOURCES, FigureRow, figure_rows, is_stamped
from .document import Section, read_document
from .rules import is_template_guidance

# ===============================
# Defaults
# ===============================

TEMPLATE = "entry.html"
REFERENCE = "reference"
DIFFERENCE = "Difference from predecessor"
ADDS = ("What the paper adds", "What the source adds")
# the sections after the lead, in the order each kind reads them (board AltEntOrder)
ORDER = {
    "algorithm": (DIFFERENCE, "Definition", "Algorithm", "Worked example", "Where it sits"),
    "component": (DIFFERENCE, "Definition", "Where it sits", "Algorithm", "Worked example"),
    "dataset": (DIFFERENCE, "Definition", "Worked example", "Where it sits", "Algorithm"),
    "benchmark": (DIFFERENCE, "Definition", "Worked example", "Where it sits", "Algorithm"),
}
# heading → (anchor, the colour of its chapter label)
SECTIONS = {
    DIFFERENCE: ("difference", "accent"),
    "Definition": ("definition", "accent"),
    "Algorithm": ("algorithm", "accent"),
    "Worked example": ("worked-example", "accent"),
    "Where it sits": ("where-it-sits", "accent"),
}
# the sections a draft of each kind shows with an empty state (board SkelEntry)
EXPECTED = {
    "algorithm": (DIFFERENCE, "Algorithm", "Worked example"),
    "component": ("Definition", "Where it sits"),
    "dataset": ("Worked example",),
    "benchmark": ("Worked example",),
}
# the Difference rows the spec declares; any other row is a step row
DIFFERENCE_ROWS = ("Line", "Predecessor", "Here", "Change", "Effect", "Same forms")
NOT_WRITTEN = "Not written yet."
NO_PREDECESSOR = "Not written yet. Needs the predecessor the entry is a variant of."
EMPTY = {"Algorithm": "Not written yet. The steps are written as numbered algorithms."}
NO_LIMITS = "None written yet."
NO_RELATION = "No relation recorded yet."
NO_SOURCE = "No paper or code cited yet."
NO_FACT = "not recorded"
NO_FACTS = "no facts recorded yet"
NO_SUMMARY = "Not written yet."
FIGURE = "Figure"
# a Worked example figure is drawn this wide, or at its own width when wider, up to the column
EXAMPLE_FIGURE_PX = 480
NOTE = " — "
LIBRARY = "library"
# (label on the entry that names it, label on the entry it names)
RELATIONS = {
    "variant_of": ("Predecessor", "Variant"),
    "part_of": ("Part of", "Contains"),
    "couples_to": ("Couples to", "Coupled from"),
}
COUPLES_WITH = "Couples with"
# a section's name in the section bar, where its heading is too long for it
BAR_NAMES = {"difference": "Difference"}
# the sections whose item count the section bar shows
BAR_COUNTS = ("algorithm", "limits")
# an algorithm command → the bold words around its text
KEYWORDS = {
    "For": ("for", "do"),
    "ForAll": ("for all", "do"),
    "While": ("while", "do"),
    "If": ("if", "then"),
    "ElsIf": ("else if", "then"),
    "Else": ("else", ""),
    "Return": ("return", ""),
    "State": ("", ""),
}
ROW_LINE = re.compile(r"^- \*\*(.+?)\*\*:\s*(.*)$")
SUB_BULLET = re.compile(r"^\s+[-*]\s+(.*)$")
STEP_CELLS = re.compile(r"\s+\|\s+")
WHERE = re.compile(r"^Alg\.?\s*(\d+),\s*lines?\s*(\d+)(?:\s*[-–]\s*(\d+))?$", re.I)
ALG_NUMBER = re.compile(r"Alg\.?\s*(\d+)", re.I)
IDENT = re.compile(r"(?<![\w/\[`])(arXiv|doi):([\w./()-]*\w)", re.I)
LINK_OR_CODE = re.compile(r"(`[^`\n]*`|\[[^\]]*\]\([^)]*\))")
ENTRY_FILE_LINK = re.compile(r'href="([^"#:]+\.md)"')
CODE_PATH = re.compile(r"<code>([\w./-]+/)([\w.-]+\.[A-Za-z]\w*(?::[\d-]+)?)</code>")
ISSUE = re.compile(r"#?(\d+)")
# a code span holding only a template path, as `<path/to/file.py>:<line>`
CODE_ONLY = re.compile(r"`([^`]+)`")
PLACEHOLDER_CODE = re.compile(r"`<[^`<>]*[/.][^`<>]*>[^`]*`")
EMPTY_EQUATION = re.compile(r"^\s*\$+\s*$")
# a \Comment code span citing a file and, after a colon, its lines
CODE_REF = re.compile(r"`([\w./-]*\w\.[A-Za-z]\w*)(?::(\d+(?:-\d+)?))?`")
# a prose note's kind ("observed: …"), shown as its label
NOTE_KIND = re.compile(r"^([a-z]+):\s+(.+)$", re.S)
# a code span or an equation, where a text-mode \texttt stays as written
CODE_OR_MATH = re.compile(r"(`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]+\$)", re.S)
TEXTTT = re.compile(r"\\texttt\{([^{}]*)\}")

# ===============================
# Main function
# ===============================


@view("entry")
def entry_page(model: Model, site: Site) -> PageView:
    """Builds the entry page's values from one entry and the site around it.

    Args:
        model: The entry.
        site: Every document: the other entries, and the records that vary
            or use this one.

    Returns:
        The page view `templates/entry.html` renders.
    """
    entries = {m.id: m for m in site.of_kind("entry")}
    pid = entry_page_id(model.id)
    title = entry_title(model)
    kind = model.value("kind") or ""
    status = model.value("status") or "draft"
    is_draft = status == "draft"
    topic = read_topic(model, site)
    text = EntryText(model, entries)

    # the lead: description, the folded paper notes and why it matters here
    description = text.block(section_text(model.section("Description")))
    adds = read_adds(model, text)
    why = text.block(section_text(model.section("Why it matters here")))

    # the sections after the lead, in the kind's order
    tables = Tables(model, pid, entries)
    figures = Figures(model, pid, text)
    expected = set(EXPECTED.get(kind, ()))
    if model.value("variant_of"):
        expected.add(DIFFERENCE)
    predecessor = entries.get(model.value("variant_of") or "")
    built = {
        DIFFERENCE: lambda: read_difference(model, text, tables, entries, pid),
        "Definition": lambda: read_definition(model, text),
        "Algorithm": lambda: read_algorithms(model, text, entries, site),
        "Worked example": lambda: read_worked_example(model, text, tables, figures),
        "Where it sits": lambda: read_where_it_sits(model, text),
    }
    sections = []
    for heading in ORDER.get(kind, ORDER["algorithm"]):
        data = built[heading]()
        # the Algorithm's figure sits below its folds, numbered in page order
        figure = figures.algorithm() if heading == "Algorithm" else None
        is_kept = heading in KIND_SECTIONS.get(kind, ()) or (
            is_draft and heading in expected
        )
        if data or figure or is_kept:
            key, color = SECTIONS[heading]
            shown = heading
            if heading == DIFFERENCE:
                shown = f"Difference from {entry_title(predecessor) if predecessor else 'its predecessor'}"
            empty = EMPTY.get(heading, NOT_WRITTEN)
            if heading == DIFFERENCE and predecessor is None:
                empty = NO_PREDECESSOR
            sections.append(_section(key, shown, color, data, empty, figure))

    # evidence always; limits and relations when written, or on a draft
    sections.append(
        _section(
            "evidence",
            "Evidence",
            "muted",
            read_evidence(model, site, entries),
            f"No experiment uses {title} yet.",
        )
    )
    limits = read_limits(model, text, site)
    if limits or is_draft:
        sections.append(_section("limits", "Limits", "h", limits, NO_LIMITS))
    related = read_related(model, entries)
    if related or is_draft:
        name = COUPLES_WITH if kind == "component" else "Related"
        sections.append(_section("related", name, "muted", related, NO_RELATION))
    sections.append(_section("sources", "Sources", "muted", read_sources(model, text), NO_SOURCE))

    # the section bar: the description, then every section in page order
    bar = [{"href": f"#{pid}/description", "name": "Description"}]
    for s in sections:
        link = {"href": f"#{pid}/{s['key']}", "name": BAR_NAMES.get(s["key"], s["title"])}
        if s["key"] in BAR_COUNTS and s["data"]:
            link["n"] = len(s["data"])
        bar.append(link)

    return PageView(
        kind="entry",
        id=pid,
        title=title,
        template=TEMPLATE,
        context={
            "kicker": model.document.path.relative_to(
                site.corpus.experiments.parent
            ).as_posix(),
            "status": {"square": ENTRY_STATUS[status][0], "word": ENTRY_STATUS[status][1]},
            "topic": topic,
            "variant_of": [
                entry_ref(key, entries) for key in entry_keys(model.value("variant_of"))
            ],
            "part_of": [entry_ref(key, entries) for key in entry_keys(model.value("part_of"))],
            "roles": [
                {"role": role, "entry": entry_ref(other.id, entries)}
                for other in entries.values()
                for key, role in couples(other)
                if key == model.id and role
            ],
            "summary": text.inline(model.value("summary") or ""),
            "no_summary": NO_SUMMARY,
            "description": description,
            "adds": adds,
            "why": why,
            "is_draft": is_draft,
            "not_written": NOT_WRITTEN,
            "facts": read_facts(model, text, topic, site),
            "no_facts": NO_FACTS,
            "bar": bar,
            "sections": sections,
        },
    )


# ===============================
# Entry text
# ===============================


class EntryText:
    """Entry markdown as HTML: paper ids linked, a link to an entry file sent
    to its page, a cited code path shown by its file name."""

    def __init__(self, model: Model, entries: dict[str, Model]):
        self.folder = model.document.folder
        self.pages = {m.document.path.resolve(): entry_page_id(m.id) for m in entries.values()}

    def block(self, text: str) -> Markup:
        """A text as block HTML, without its guidance; "" when nothing else is left."""
        text = written_block(text)
        return Markup(self._finish(markup.markdown_html(link_idents(text)))) if text else Markup("")

    def inline(self, text: str, guarded: bool = True) -> Markup:
        """A row value as inline HTML, without the paragraph around one paragraph;
        "" while it still holds template guidance, unless `guarded` is False
        (an algorithm line, whose `<q>` is the author's own token)."""
        text = written(text) if guarded else (text or "").strip()
        return Markup(self._finish(markup.inline_html(link_idents(text)))) if text else Markup("")

    def _finish(self, html_: str) -> str:
        def page(match: re.Match) -> str:
            target = (self.folder / match.group(1)).resolve()
            return f'href="#{self.pages[target]}"' if target in self.pages else match.group(0)

        html_ = ENTRY_FILE_LINK.sub(page, html_)
        return CODE_PATH.sub(
            lambda m: f'<code title="{m.group(1)}{m.group(2)}">{m.group(2)}</code>', html_
        )


# ===============================
# Lead and rail
# ===============================


def read_topic(model: Model, site: Site) -> dict:
    """The entry's topic: its page link, title and compare labels, from its README.md."""
    folder = model.document.folder
    readme = folder / "README.md"
    meta = (
        read_document(readme, "topic", site.corpus.experiments).frontmatter
        if readme.is_file()
        else {}
    )
    compare = meta.get("compare") or []
    return {
        "href": f"#topic-{folder.name}",
        "title": str(meta.get("title") or folder.name.replace("-", " ").capitalize()),
        "compare": [str(label) for label in compare] if isinstance(compare, list) else [],
    }


def read_adds(model: Model, text: EntryText) -> dict | None:
    """What the paper (or source) adds as a fold: its title, digest and body."""
    for heading in ADDS:
        section = model.section(heading)
        if section is None:
            continue
        body = without_rows(section, ("In short",))
        short = written(model.row(heading, "In short"))
        html_ = text.block(body)
        if not (html_ or short):
            continue
        digest = short or markup.first_sentence(markup.plain_text(html_))
        return {"title": heading, "digest": text.inline(digest), "html": html_}
    return None


def read_facts(model: Model, text: EntryText, topic: dict, site: Site) -> list[dict]:
    """The Facts table: each row's short value, its note on hover; then the
    topic's compare labels the entry lacks; then the records varying it."""
    section = model.section("Facts")
    facts = []
    for label, row in (section.rows if section else {}).items():
        value = written(row.text)
        if not written(label) or not value:
            continue
        short, _, note = value.partition(NOTE)
        facts.append(
            {"label": label, "html": text.inline(short), "note": note.strip(), "missing": ""}
        )
    labels = {f["label"].lower() for f in facts}
    facts += [
        {"label": label, "html": "", "note": "", "missing": NO_FACT}
        for label in topic["compare"]
        if label.lower() not in labels
    ]
    varied = [m for m in site.of_kind("record") if model.id in entry_keys(m.value("varies"))]
    if varied:
        facts.append(
            {
                "label": "Varied in",
                "html": Markup(" ").join(records.record_pill(m) for m in varied),
                "note": "",
                "missing": "",
            }
        )
    return facts


def read_sources(model: Model, text: EntryText) -> dict | None:
    """The Sources section: one line per paper, the code files on one line by
    name (the path on hover), any other row, and the Paper vs code card; None
    when nothing is cited."""
    section = model.section("Sources")
    papers, code, other, card = [], [], [], None
    for label, values in ordered_rows(section.body if section else ()):
        values = [v for v in map(written, values) if v]
        if not values:
            continue
        if label == "Paper vs code":
            first, rest = values[0], values[1:]
            if len(values) == 1:
                first = markup.first_sentence(values[0])
                rest = [values[0][len(first) :].strip()] if values[0] != first else []
            card = {
                "first": text.inline(first),
                "rest": [text.inline(r) for r in rest if r],
            }
            continue
        for value in values:
            path = CODE_ONLY.fullmatch(value) if label == "Code" else None
            if path:
                code.append({"name": path.group(1).rsplit("/", 1)[-1], "path": path.group(1)})
            elif label == "Paper":
                shown, _, role = value.partition(NOTE)
                papers.append({"html": text.inline(shown), "note": role.strip()})
            else:
                other.append({"label": label, "html": text.inline(value)})
    if not (papers or code or other or card):
        return None
    return {"papers": papers, "code": code, "other": other, "card": card}


# ===============================
# Sections
# ===============================


class Tables:
    """The tables an entry cites, inlined with the title of each entry of its
    topic in place of that entry's snake-case name, wherever it stands as
    whole words in a column name."""

    def __init__(self, model: Model, pid: str, entries: dict[str, Model]):
        self.folder = model.document.folder
        self.pid = pid
        self.headers = {
            other.document.path.stem.replace("-", "_"): entry_title(other)
            for other in entries.values()
            if other.document.folder == self.folder
        }

    def table(self, src: str, caption: Markup) -> dict:
        """One cited table: its anchor, its caption and the table, or what is missing."""
        found = markup.inline_asset(self.folder, src, self.pid, self.headers)
        if found is None:
            return {"missing": Markup(f"<code>{escape(src)}</code> is missing")}
        return {
            "missing": "",
            "anchor": f"{self.pid}/asset/{re.sub(r'[^A-Za-z0-9_-]', '-', Path(src).stem)}",
            "html": Markup(found[1]),
            "caption": caption,
        }


class Figures:
    """The entry's Figure rows, numbered in page order as the sections read them;
    under each its file and where it comes from."""

    def __init__(self, model: Model, pid: str, text: EntryText):
        self.model = model
        self.folder = model.document.folder
        self.pid = pid
        self.text = text
        self.count = 0

    def algorithm(self) -> dict | None:
        """The Algorithm's Figure row, at the full width; None when it has none."""
        rows = figure_rows(self.model.section("Algorithm"), FIGURE)
        return self.read(rows[0], wide=True) if rows and rows[0].links else None

    def read(self, row: FigureRow, wide: bool = False) -> dict:
        """One written Figure row as the next figure; a narrow one is drawn
        EXAMPLE_FIGURE_PX wide, or at its own width when wider. A missing file
        shows as missing and takes no number."""
        caption, src = row.links[0]
        path = self.folder / src
        n = self.count + 1
        width = "" if wide else f"{max(EXAMPLE_FIGURE_PX, markup.figure_width(path) or 0):.0f}px"
        figure = records.read_figure_row(
            self.folder,
            src,
            self.text.inline(caption),
            f"Figure {n}",
            f"{self.pid}/figure-{n}",
            note=figure_origin(path, src),
            width=width,
        )
        if not figure["missing"]:
            self.count = n
        return figure


def figure_origin(path: Path, src: str) -> str:
    """Where an entry figure comes from: the diagram source beside it, else examples.py for a stamped chart."""
    for suffix in DIAGRAM_SOURCES:
        if path.with_suffix(suffix).is_file():
            return f"drawn from {Path(src).with_suffix(suffix).as_posix()}"
    return "from examples.py" if path.is_file() and is_stamped(path) else ""


def read_difference(
    model: Model, text: EntryText, tables: Tables, entries: dict[str, Model], pid: str
) -> dict | None:
    """The Difference block: the change, the step table, the variant table
    under it, and the effect."""
    section = model.section(DIFFERENCE)
    if section is None:
        return None
    rows = [(label, written(value)) for label, value in _row_texts(section)]
    rows = [(label, value) for label, value in rows if value and written(label)]
    if not rows:
        return None
    found = dict(rows)

    # one table row per step row, and one for a Line, Predecessor and Here triple
    steps = []
    if found.get("Predecessor") or found.get("Here"):
        line = found.get("Line", "")
        steps.append(
            {
                "step": "",
                "before": text.inline(found.get("Predecessor", "")),
                "after": text.inline(found.get("Here", "")),
                "where": {"text": f"line {line}" if line else "", "code": ""},
            }
        )
    for label, value in rows:
        if label in DIFFERENCE_ROWS:
            continue
        cells = [*STEP_CELLS.split(value, maxsplit=2), "", ""][:3]
        steps.append(
            {
                "step": text.inline(label),
                "before": text.inline(cells[0]),
                "after": text.inline(cells[1]),
                "where": where_cell(cells[2], pid),
            }
        )
    variant = model.value("variant_of") or ""
    predecessor = entries.get(variant)

    # the variant table the Same forms row cites
    same = None
    if match := markup.ASSET_LINK.search(found.get("Same forms", "")):
        same = tables.table(match.group(2), text.inline(match.group(1)))

    return {
        "change": text.inline(markup.capitalize(found.get("Change", ""))),
        "title": entry_title(model),
        "head": [
            entry_title(predecessor) if predecessor else "Predecessor",
            entry_title(model),
        ],
        "has_step": any(s["step"] for s in steps),
        "has_where": any(s["where"]["text"] or s["where"]["code"] for s in steps),
        "steps": steps,
        "same": same,
        "effect": text.inline(found.get("Effect", "")),
    }


def where_cell(text: str, pid: str) -> dict:
    """A step row's place in the algorithms: a chip to the algorithm for `Alg. <n>, line <m>`."""
    match = WHERE.match(text.strip())
    if not match:
        return {"text": text.strip(), "code": ""}
    number, first, last = match.groups()
    lines = f"{first}–{last}" if last else first
    return {"text": "", "code": f"Alg {number} · {lines}", "href": f"#{pid}/alg-{number}"}


def read_definition(model: Model, text: EntryText) -> dict | None:
    """The Definition box: the equation, each symbol with its meaning, the rule sentence."""
    section = model.section("Definition")
    if section is None or not written_block(section.text):
        return None
    before, after, symbols = [], [], []
    for label, value, lines in _rows_and_prose(section.body):
        if label is not None and not written(label):
            continue
        if label is not None and "$" in label:
            symbols.append({"symbol": text.inline(label), "meaning": text.inline(value)})
        else:
            (after if symbols else before).extend(lines)
    return {
        "before": text.block("\n".join(before)),
        "symbols": symbols,
        "after": text.block("\n".join(after)),
    }


def read_algorithms(
    model: Model, text: EntryText, entries: dict[str, Model], site: Site
) -> list[dict]:
    """Each algorithm block as a fold: caption, line count, digest, the ruled
    lines with their code and the Reading beside them."""
    section = model.section("Algorithm")
    if section is None:
        return []
    pid = entry_page_id(model.id)
    predecessor = entries.get(model.value("variant_of") or "")
    ideas = read_ideas(site, model.id)
    return [
        algorithm_view(a, pid, text, ideas.get(a.number, []), predecessor, model)
        for a in algorithms(section)
        # the template's own block holds nothing but guidance
        if any(map(written, (a.caption, *a.require, *a.ensure, *(s.text for s in a.steps))))
    ]


def algorithm_view(
    a: Algorithm,
    pid: str,
    text: EntryText,
    ideas: list[int],
    predecessor: Model | None,
    model: Model,
) -> dict:
    """One algorithm block's values for its fold: the head names the file
    most lines cite, each line its own range (and its file when another),
    and the prose notes under it."""
    comments = [read_comment(step.comment) for step in a.steps]
    cited = [ref["path"] for refs, _, _ in comments for ref in refs]
    head = max(cited, key=cited.count) if cited else ""

    # the numbered lines
    steps = []
    for step, (refs, notes, is_library) in zip(a.steps, comments):
        before, after = KEYWORDS.get(step.command, ("", ""))
        steps.append(
            {
                "before": before,
                "html": text.inline(text_mode_code(step.text), guarded=False),
                "after": after,
                "depth": min(step.depth, 4),
                "changed": step.is_changed,
                "library": is_library,
                "refs": [
                    {
                        "file": "" if ref["path"] == head else _file_name(ref["path"]),
                        "lines": ref["lines"].replace("-", "–"),
                        "title": f"{ref['path']}, {_lines_word(ref['lines'])}" if ref["lines"] else ref["path"],
                    }
                    for ref in refs
                ],
                "notes": [{"kind": kind, "html": text.inline(text_mode_code(note), guarded=False)} for kind, note in notes],
            }
        )

    # the digest: open ideas changing it, lines inside a library, the cited file
    topic = model.document.folder.name
    digest = [
        {"text": "changed by the open idea", "number": n, "href": f"#topic-{topic}/idea-{n}"}
        for n in ideas
    ]
    if spans := _spans([i for i, s in enumerate(steps, 1) if s["library"]]):
        digest.append({"text": spans, "href": ""})
    changed = sum(s["changed"] for s in steps)
    whom = entry_title(predecessor) if predecessor else "its predecessor"
    tip = [f"{d['text']} #{d['number']}" if d["href"] else d["text"] for d in digest]
    if head:
        tip.append(f"lines in {_file_name(head)}")

    return {
        "id": f"{pid}/alg-{a.number}",
        "number": a.number,
        "caption": text.inline(text_mode_code(a.caption)),
        "size": _count(len(steps), "line") + (f" · {changed} changed" if changed else ""),
        "digest": digest,
        "digest_tip": " · ".join(tip),
        "file": {"name": _file_name(head), "path": head} if head else None,
        "has_notes": any(s["notes"] for s in steps),
        "shaded": f"Shaded lines are the ones {entry_title(model)} changes from {whom}." if changed else "",
        "require": text.inline(text_mode_code("; ".join(a.require))),
        "ensure": text.inline(text_mode_code("; ".join(a.ensure))),
        "steps": steps,
        "reading": [
            {"label": row.label.replace("-", "–"), "html": text.inline(text_mode_code(row.text))}
            for row in a.reading
            if written(row.label)
        ],
    }


def text_mode_code(text: str) -> str:
    """Algorithm text with each text-mode `\\texttt{X}` as the code span `X`;
    one inside an equation or a code span stays for the equation to set."""
    def code(match: re.Match) -> str:
        fence = "``" if "`" in match.group(1) else "`"
        return f"{fence}{match.group(1)}{fence}"

    parts = CODE_OR_MATH.split(text or "")
    return "".join(part if i % 2 else TEXTTT.sub(code, part) for i, part in enumerate(parts))


def read_comment(comment: str) -> tuple[list[dict], list[tuple[str, str]], bool]:
    """An algorithm line's \\Comment: the code it cites (path and lines), its
    prose notes as (kind, text), and whether it runs inside a library."""
    comment = written(comment.strip())
    if comment.lower() == LIBRARY:
        return [], [], True
    refs = [{"path": m.group(1), "lines": m.group(2) or ""} for m in CODE_REF.finditer(comment)]
    note = CODE_REF.sub("", comment).strip(" ,;")
    if not note:
        return refs, [], False
    kind = NOTE_KIND.match(note)
    return refs, [kind.groups() if kind else ("", note)], False


def read_worked_example(
    model: Model, text: EntryText, tables: Tables, figures: Figures
) -> dict | None:
    """The worked example: its Input line, then prose and tables in place, its figure after them."""
    section = model.section("Worked example")
    if section is None:
        return None
    blocks, prose = [], []

    def flush() -> None:
        if html_ := text.block("\n".join(prose)):
            blocks.append({"html": html_, "table": None, "figure": None})
        prose.clear()

    for line in without_rows(section, ("Input", FIGURE)).splitlines():
        match = markup.ASSET_LINK.fullmatch(line.strip())
        if match and line.strip().startswith("!"):
            flush()
            table = tables.table(match.group(2), text.inline(match.group(1)))
            blocks.append({"html": "", "table": table, "figure": None})
        else:
            prose.append(line)
    flush()

    # the figure shows the toy input's result, so it follows the tables
    rows = figure_rows(section, FIGURE)
    if rows and rows[0].links:
        blocks.append({"html": "", "table": None, "figure": figures.read(rows[0])})

    given = text.inline(model.row("Worked example", "Input"))
    if not (blocks or given):
        return None
    return {"input": given, "blocks": blocks}


def read_where_it_sits(model: Model, text: EntryText) -> dict | None:
    """Where it sits: the chain of the larger process, this entry's step lifted, then the prose."""
    section = model.section("Where it sits")
    if section is None:
        return None
    steps = []
    for part in written(model.row("Where it sits", "Steps")).split("→"):
        part = part.strip()
        if not part:
            continue
        is_here = part.startswith("**") and part.endswith("**")
        steps.append({"html": text.inline(part.strip("*") if is_here else part), "here": is_here})
    html_ = text.block(without_rows(section, ("Steps",)))
    return {"steps": steps, "html": html_} if steps or html_ else None


def read_evidence(model: Model, site: Site, entries: dict[str, Model]) -> dict | None:
    """Key findings of the records that vary the entry, direct apart from
    downstream; else the records that use it."""
    groups = {"direct": [], "downstream": []}
    for record in site.of_kind("record"):
        if model.id not in entry_keys(record.value("varies")):
            continue
        marks = record.value("evidence") or {}
        held = [entry_ref(key, entries) for key in entry_keys(record.value("uses"))]
        slug = records.record_slug(record)
        for finding in records.read_findings(record):
            if finding.weight != "key":
                continue
            mark = marks.get(finding.id) if isinstance(marks.get(finding.id), dict) else {}
            kind = "downstream" if str(mark.get("kind", "")).lower() == "downstream" else "direct"
            on = str(mark.get("measured_on") or "")
            groups[kind].append(
                {
                    "id": finding.id,
                    "href": f"#{slug}/{finding.id.lower()}",
                    "tip": f"{finding.id} — {finding.title}\n{finding.text('Summary')}",
                    "title": Markup(markup.inline_html(finding.title)),
                    "record": records.record_pill(record),
                    "measured_on": entry_ref(on, entries) if on else None,
                    "held": held,
                }
            )
    notes = {
        "direct": "Measured on the entry itself.",
        "downstream": "Measured on something built with it.",
    }
    found = [
        {"name": kind.capitalize(), "note": notes[kind], "rows": rows}
        for kind, rows in groups.items()
        if rows
    ]
    if found:
        return {"groups": found, "used": [], "title": entry_title(model)}
    used = [records.record_pill(r) for r in site.of_kind("record") if model.id in entry_keys(r.value("uses"))]
    return {"groups": [], "used": used, "title": entry_title(model)} if used else None


def read_limits(model: Model, text: EntryText, site: Site) -> list[dict]:
    """One card per limit: its title and paragraph, its question, and its idea issue or a File as issue link."""
    section = model.section("Limits")
    if section is None:
        return []
    titled = [e for e in section.entries if written(e.title)]
    first = section.line + 1
    starts = [e.line - first for e in section.entries] + [len(section.body)]
    cards = []
    for entry in titled:
        index = [e.line for e in section.entries].index(entry.line)
        body = list(section.body[starts[index] + 1 : starts[index + 1]])
        for label in ("Question", "Issue"):
            if row := entry.rows.get(label):
                at = row.line - first - starts[index] - 1
                body[at : at + len(row.body)] = [""] * len(row.body)
        question = written(entry.rows["Question"].text) if "Question" in entry.rows else ""
        issue = ISSUE.search(written(entry.rows["Issue"].text)) if "Issue" in entry.rows else None
        cards.append(
            {
                "title": text.inline(entry.title),
                "html": text.block("\n".join(body)),
                "question": text.inline(question),
                "issue": (
                    {
                        "number": issue.group(1),
                        "href": f"https://github.com/{site.github}/issues/{issue.group(1)}"
                        if site.github
                        else "",
                    }
                    if issue
                    else None
                ),
                "file": (
                    f"https://github.com/{site.github}/issues/new?labels=idea&title={quote(question)}"
                    if question and not issue and site.github
                    else ""
                ),
            }
        )
    return cards


def read_related(model: Model, entries: dict[str, Model]) -> list[dict]:
    """One line per relation, read from both sides: a pair that couples both ways shows once."""
    mine = {
        "variant_of": entry_keys(model.value("variant_of")),
        "part_of": entry_keys(model.value("part_of")),
        "couples_to": [key for key, _ in couples(model)],
    }
    roles = dict(couples(model))
    pointing = {field: [] for field in RELATIONS}
    for other in entries.values():
        if other.id == model.id:
            continue
        pointing["variant_of"] += [other] if model.id in entry_keys(other.value("variant_of")) else []
        pointing["part_of"] += [other] if model.id in entry_keys(other.value("part_of")) else []
        pointing["couples_to"] += [
            (other, role) for key, role in couples(other) if key == model.id
        ]
    mutual = {other.id for other, _ in pointing["couples_to"]} & set(mine["couples_to"])

    lines = []
    for field, (named, naming) in RELATIONS.items():
        if field == "couples_to":
            if mutual:
                lines.append(
                    {"label": COUPLES_WITH, "entries": [entry_ref(k, entries) for k in sorted(mutual)]}
                )
            to = [
                entry_ref(k, entries, roles.get(k, "")) for k in mine[field] if k not in mutual
            ]
            back = [
                entry_ref(o.id, entries, role)
                for o, role in pointing[field]
                if o.id not in mutual
            ]
        else:
            to = [entry_ref(k, entries) for k in mine[field]]
            back = [entry_ref(o.id, entries) for o in pointing[field]]
        if to:
            lines.append({"label": named, "entries": to})
        if back:
            lines.append({"label": naming, "entries": back})
    return lines


# ===============================
# Helper functions
# ===============================


def couples(model: Model) -> list[tuple[str, str]]:
    """The entry's `couples_to` items as (entry, role); the role is "" for a bare `<topic>/<entry>`."""
    found = []
    for item in model.document.frontmatter.get("couples_to") or []:
        if isinstance(item, str) and item.strip():
            found.append((item.strip(), ""))
        elif isinstance(item, dict) and isinstance(item.get("entry"), str):
            found.append((item["entry"].strip(), str(item.get("as") or "").strip()))
    return found


def read_ideas(site: Site, key: str) -> dict[int, list[int]]:
    """The open ideas whose Touches row changes this entry, by the algorithm number the row names."""
    found: dict[int, list[int]] = {}
    for idea in topic_checks.read_ideas(site.corpus.experiments / REFERENCE):
        if idea.state != topic_checks.OPEN:
            continue
        for touch in idea.touches:
            if touch.entry == key and touch.relation == "changes":
                number = ALG_NUMBER.search(touch.place)
                found.setdefault(int(number.group(1)) if number else 1, []).append(idea.number)
    return found


def link_idents(text: str) -> str:
    """`arXiv:<id>` and `doi:<id>` as markdown links to the paper, outside code and links."""

    def link(match: re.Match) -> str:
        base = "https://arxiv.org/abs/" if match.group(1).lower() == "arxiv" else "https://doi.org/"
        return f"[{match.group(1)}:{match.group(2)}]({base}{match.group(2)})"

    parts = LINK_OR_CODE.split(text)
    return "".join(p if i % 2 else IDENT.sub(link, p) for i, p in enumerate(parts))


def written(text: str) -> str:
    """A row value or title as the author wrote it; "" while it holds template
    guidance anywhere, so a half-filled template line never reaches the page."""
    if not text or PLACEHOLDER_CODE.search(text) or _is_guidance_paragraph(text):
        return ""
    kept = markup.unguided(text)
    return kept if kept == text.strip() else ""


def written_block(text: str) -> str:
    """A section's text without its guidance: whole `<…>` paragraphs, inline
    placeholders and an equation left empty by them."""
    paragraphs = re.split(r"\n[ \t]*\n", text or "")
    kept = "\n\n".join(p for p in paragraphs if not _is_guidance_paragraph(p))
    kept = PLACEHOLDER_CODE.sub("", markup.unguided(kept))
    return "\n".join(line for line in kept.splitlines() if not EMPTY_EQUATION.match(line)).strip()


def section_text(section: Section | None) -> str:
    return section.text if section else ""


def without_rows(section: Section, labels: tuple[str, ...]) -> str:
    """A section's text with the named rows and their continuation lines left out."""
    body = list(section.body)
    first = section.line + 1
    for label in labels:
        if row := section.rows.get(label):
            at = row.line - first
            body[at : at + len(row.body)] = [""] * len(row.body)
    return "\n".join(body).strip()


def ordered_rows(body: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """Each `- **Label**:` row in order, a label free to repeat, with its values:
    its sub-bullets, else its text with the continuation lines joined."""
    rows: list[tuple[str, list[str], list[str]]] = []
    for line in body:
        if match := ROW_LINE.match(line):
            rows.append((match.group(1), [match.group(2)], []))
        elif rows and (sub := SUB_BULLET.match(line)):
            rows[-1][2].append(sub.group(1).strip())
        elif rows and line.startswith((" ", "\t")) and line.strip():
            target = rows[-1][2] if rows[-1][2] else rows[-1][1]
            target[-1] = f"{target[-1]} {line.strip()}"
        elif not line.startswith(" ") and line.strip():
            rows.append(("", [], []))
    return [
        (label, subs or [" ".join(text).strip()])
        for label, text, subs in rows
        if label
    ]


def _row_texts(section: Section) -> list[tuple[str, str]]:
    return [(label, " ".join(values)) for label, values in ordered_rows(section.body)]


def _rows_and_prose(body: tuple[str, ...]) -> list[tuple[str | None, str, list[str]]]:
    # each row as (label, text, its lines) and each other line as (None, "", [line])
    found: list[tuple[str | None, str, list[str]]] = []
    for line in body:
        if match := ROW_LINE.match(line):
            found.append((match.group(1), match.group(2), [line]))
        elif found and found[-1][0] is not None and line.startswith((" ", "\t")) and line.strip():
            label, value, lines = found[-1]
            found[-1] = (label, f"{value} {line.strip()}", [*lines, line])
        else:
            found.append((None, "", [line]))
    return found


def _is_guidance_paragraph(text: str) -> bool:
    # a paragraph that is one `<…>` guidance, code spans and all, as the template opens each section
    stripped = text.strip()
    return (
        len(stripped) > 2
        and stripped[0] == "<"
        and stripped[-1] == ">"
        and (stripped[1].isalpha() or stripped[1] == "…" or stripped[1] == "`")
        and is_template_guidance(stripped[1:-1])
    )


def _section(key: str, title: str, color: str, data, empty: str, figure: dict | None = None) -> dict:
    return {"key": key, "title": title, "color": color, "data": data, "empty": empty, "figure": figure}


def entry_keys(value) -> list[str]:
    """A frontmatter value as a list of entry keys: one key, a list, or none."""
    if not value:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def _file_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _lines_word(lines: str) -> str:
    return f"{'lines' if '-' in lines else 'line'} {lines.replace('-', '–')}"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _spans(numbers: list[int]) -> str:
    # consecutive line numbers as ranges: "lines 5 to 10 run inside the library"
    if not numbers:
        return ""
    spans, start = [], numbers[0]
    for previous, current in zip(numbers, [*numbers[1:], None]):
        if current != previous + 1:
            spans.append(str(start) if start == previous else f"{start} to {previous}")
            start = current
    is_one = len(numbers) == 1
    return f"{'line' if is_one else 'lines'} {', '.join(spans)} {'runs' if is_one else 'run'} inside the library"
