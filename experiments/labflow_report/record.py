"""The record page on the new builder: an experiment's answer, findings and setup (ADR-0005).

`record_page` turns one record's `Model` into the values `templates/record.html`
draws, board part by board part: the head, the answer card, the clause trace,
the section bar, findings, discussion, decisions, methods, limitations,
datasets, side documents, references, the question-and-setup, adoption and
reproduce folds, and the records that build on this one. A record before its
verdict draws the same page as a skeleton: planned chips and an empty state
for each section nothing has written yet.

The view reads the record only through `Model.value` and `Model.section`,
the tables the spec's artifacts name, and the other records of the site.
Prose goes through `markup` and then `Prose`, which turns citations into
chips; every other piece of markup is a macro under `templates/components/`.
"""

from __future__ import annotations

import html
import itertools
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from markupsafe import Markup

from . import markup
from .builder import ENVIRONMENT, OUTCOME, Model, PageView, Site, view
from .checks import record as checks
from .document import Entry, Row
from .rules import GUIDANCE, is_template_guidance

# ===============================
# Defaults
# ===============================

TEMPLATE = "record.html"
RECORD_OUTCOMES = (
    "confirmed",
    "refuted",
    "inconclusive",
    "adopted",
    "rejected",
    "answered",
    "partial",
    "unanswered",
)
CLAUSE_KEYWORDS = {
    "H": ("confirmed if", "refuted if"),
    "C": ("better if", "worse if"),
    "Q": ("read on", "robust if"),
    "measured": ("read on", "reported as"),
}
TLDR_STATUS = re.compile(
    rf"^\s*(?:{'|'.join(RECORD_OUTCOMES)}|open|abandoned|leaning \w+)\s*(?:\([^)]*\))?\s*[—–-]\s*",
    re.I,
)
ENTRY_TITLE = re.compile(
    r"^([A-Z]\d+)\s*[—–-]\s*(.*?)(?:\s+· (?:key|supporting(?: F\d+)?|minor))?\s*$"
)
CLAUSE_NAME = r"(?<![\w\[]){id}\b|\[{id}\]"
CITATION = re.compile(r"\[([DFM]\d+|[HCQ]\d+)\]")
# parentheses that hold nothing but citations, which the chips replace
CITED_ONLY = re.compile(
    r"\(\s*((?:\[(?:[DFM]\d+|[HCQ]\d+)\](?:\s*[,;]\s*|\s+and\s+)?)+)\s*\)"
)
# a citation, with the parentheses around it, that a line out of its record drops
BARE_CITATION = re.compile(
    r"\s*\(\s*(?:\[[DFMHCQ]\d+\](?:\s*[,;]\s*)?)+\s*\)|\s*\[[DFMHCQ]\d+\]"
)
CLAUSE_CITATION = re.compile(r"\[([HCQ]\d+)\]")
BIB_CITATION = re.compile(r"\[@([\w:.\-]+)\]")
NUMBER = re.compile(r"(?<![#\w.])\d[\d,.]*%?")
UNIT = re.compile(r"(?:[kMB]|×)(?![A-Za-z])")
PHRASE_CUT = re.compile(r",(?!\d)|(?<!\d),|[;:()—–]|\.(?:\s|$)")
# a number that is a target or threshold, not a result
THRESHOLD_BEFORE = re.compile(
    r"\b(?:bar|target|margin|threshold|floor|limit|cap)(?: of)?$", re.I
)
THRESHOLD_AFTER = re.compile(r"(?:target|bar|margin|threshold|floor|limit|cap)\b", re.I)
# words that make the number after them a name (`level 2`), not a quantity
LABEL_WORDS = frozenset(
    ["level", "levels", "step", "round", "top", "seed", "page", "figure", "table"]
)
# words a key number's context phrase may not start or end on
FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "at",
        "in",
        "on",
        "to",
        "by",
        "for",
        "from",
        "with",
        "than",
        "about",
        "around",
        "nearly",
        "just",
        "only",
        "under",
        "over",
        "above",
        "below",
        "between",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
    ]
)
LINK_VERBS = frozenset(
    [
        "stays",
        "stay",
        "reaches",
        "reach",
        "scores",
        "drops",
        "falls",
        "rises",
        "holds",
        "keeps",
        "gains",
    ]
)
# words that end the phrase after a key number
JOIN_WORDS = frozenset(
    ["and", "but", "while", "whereas", "so", "which", "than", "against"]
)
CONTEXT_WORDS = 6
ISO_DATE = re.compile(r"\b\d{4}-\d\d-\d\d\b")
TIMING = re.compile(r"\((before runs|after \[F\d+\](?:,\s*\[F\d+\])*)(;\s*user)?\)")
REPLACES = re.compile(r"replaces runs ((?:`[0-9a-f]{6,}`(?:,\s*)?)+)")
RULE_NOTE = re.compile(r"Note added (\d{4}-\d\d-\d\d):")
LEAD = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$", re.S)
NESTED_LABEL = re.compile(r"^\*\*(.+?)\*\*:?\s*(.*)$", re.S)
PLANNED_AS = re.compile(r"\s*—\s*planned as ([a-z0-9][a-z0-9-]*)(?: #(\d+))?\s*[.;]?$")
ISSUE = re.compile(r"\s*\(issue #(\d+)\)\s*[.;]?$")
NOT_PLANNED = re.compile(r"\s*—\s*not yet planned(?::.*)?\s*[.;]?$")
DISCLOSURE = re.compile(r"^(earlier runs|pilot|numbers seen):\s*(.*)$", re.S | re.I)
DATASET_META = re.compile(r"^\((?P<meta>[^)]*)\):?\s*(?P<text>.*)$", re.S)
STEP_CODE = re.compile(r"^\s*-?\s*step (\d+):\s*`?([\w./:-]+)`?")
STEP_CODE_INLINE = re.compile(r"`?([\w./:-]+)`?\s*\(step (\d+)\)")
CODE_SPAN = re.compile(r"<code>([^<]+)</code>")
PATH_LIKE = re.compile(r"/|\.(ya?ml|py|csv|json|svg|png|md|txt|toml)(::\w+)?$")
BASH_BLOCK = re.compile(r"(?ms)^```(?:bash|sh|zsh)?\n(.*?)^```[ \t]*$")
SLUG_LEAD = re.compile(r"^(?:`[^`]+`|[\w.]*[-/][\w./-]*)\s*[:—–]\s*")
WHOLE_NUMBER = re.compile(r"^\d[\d,]*$")
TOTAL_ROWS = ("all", "total", "overall")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---", re.S)
RELATIONS = {
    "qualifies": "qualified",
    "supports": "supported",
    "overturns": "overturned",
}
RELATION_BEFORE = re.compile(r"\b(qualifies|supports|overturns)\s+\(?$")

NEEDS = (
    ("People", "People"),
    ("Compute", "Compute"),
    ("Data access", "Data"),
    ("External dependency", "External"),
)
NO_NEED = re.compile(r"^(none|n/a|—|-)\b", re.I)
# a need worth a glance: a hosted model, a quota, a licence, people's hours
HOT_NEED = re.compile(
    r"hosted|\bapi\b|hpc|gpu|cluster|licen|quota|billed|endpoint|annotat|hours|manual|rare|cannot",
    re.I,
)
PREREQUISITE_KINDS = ("Secret", "Login", "Data", "Service", "Frozen")
COUNT_WORDS = (
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)
# an Alternatives item's lead past this many words stays plain
LEAD_WORDS = 14
LEAD_CUT = re.compile(r"(:\s|\s—\s)")
# past this many decisions the section groups them
GROUP_PAST = 6
# intervals an interval strip's axis aims for
STRIP_STEPS = 4
# scripts the Reproduce digest names, and what a script looks like in a step
DIGEST_STEPS = 3
SCRIPT = re.compile(r"[\w./$-]+\.(?:py|sh)\b")
# a reading this large reads in billions or millions
BILLION = 1e9
MILLION = 1e6
TEN_MILLION = 1e7
# runs a finding's runs line names before counting the rest
RUNS_SHOWN = 3
RUN_ID_SHOWN = 8
SHA_SHOWN = 7
INTERVAL_SUFFIXES = (("_ci_low", "_ci_high"), ("_low", "_high"))
YES = ("yes", "true")
SIDE_FOLDERS = ("configs", "prompts")
CONFIG_SUFFIXES = (".yaml", ".yml", ".toml", ".json")

EMPTY = {
    "findings": "No findings yet. The card above gains its verdict line with the first key finding.",
    "discussion": "Nothing to discuss yet. The discussion is written with the verdict.",
    "decisions": "No decision recorded yet.",
    "methods": "No method yet. Each step is written as its code lands.",
    "limits": "None written yet; they are written with the verdict.",
    "datasets": "No dataset described yet. Each input gets a row at the data-ready gate.",
}
SKELETON_STAGES = ("draft", "running", "reporting")

INLINE = ENVIRONMENT.get_template("components/inline.html").module
WORDS = ENVIRONMENT.get_template("components/words.html").module

# ===============================
# Record classes
# ===============================


@dataclass(frozen=True)
class Clause:
    """One clause line of Predictions, read for the page."""

    id: str
    claim: str
    conditions: str
    line: str
    role: str
    standard: str
    round: str
    superseded_by: str = ""

    @property
    def keywords(self) -> tuple[str, str]:
        return CLAUSE_KEYWORDS[
            "measured" if self.standard == "measured" else self.id[0]
        ]

    @property
    def check_clause(self) -> checks.Clause:
        return checks.Clause(
            self.id, 0, self.line, self.role, self.standard, self.round
        )


@dataclass(frozen=True)
class Finding:
    """One `### F<n>` entry with its weight and the key finding it supports."""

    id: str
    title: str
    weight: str
    target: str
    entry: Entry

    def text(self, label: str) -> str:
        row = self.entry.rows.get(label)
        return _unguided(row.text) if row else ""

    def source(self, label: str) -> str:
        row = self.entry.rows.get(label)
        return _unguided(row_source(row)) if row else ""


@dataclass
class Exhibits:
    """The page's figure and table numbers and the anchors already placed."""

    folder: Path
    page_id: str
    counts: dict[str, int] = field(default_factory=lambda: {"Figure": 0, "Table": 0})
    anchors: set[str] = field(default_factory=set)

    def anchor(self, src: str) -> str:
        return f"{self.page_id}/asset/{re.sub(r'[^A-Za-z0-9_-]', '-', Path(src).stem)}"


# ===============================
# Main function
# ===============================


@view("record")
def record_page(model: Model, site: Site) -> PageView:
    """Builds the record page's values from one record and the site around it.

    Args:
        model: The record.
        site: Every document, for parents, later records and their marks.

    Returns:
        The page view `templates/record.html` renders.
    """
    page_id = record_slug(model)
    short = model.value("short") or model.value("title") or page_id

    # what the page reads once
    clauses = read_clauses(model)
    findings = read_findings(model)
    readings = read_readings(model)
    prose = Prose(model, page_id, site, clauses, findings)
    exhibits = Exhibits(model.document.folder, page_id)
    outcome = record_outcome(model)
    stage = model.stage or "draft"
    skeleton = not outcome and stage in SKELETON_STAGES
    words = {c.id: clause_word(c, findings, readings) for c in clauses}

    # the head and the answer, which numbers the first figure
    head = read_head(model, site, page_id)
    answer = read_answer(
        model, site, prose, exhibits, clauses, findings, readings, words, outcome
    )
    trace = (
        read_trace(model, prose, clauses, findings, readings, words)
        if outcome
        else None
    )
    needs = read_needs(model, prose)
    if not skeleton and not any(n["text"] for n in needs):
        needs = []

    # the sections, in page order
    sections = {
        "findings": read_findings_section(
            model, prose, exhibits, clauses, findings, readings, words, answer["shown"]
        ),
        "discussion": read_discussion(model, prose),
        "decisions": read_decisions(model, prose, clauses),
        "methods": read_methods(model, prose),
        "limits": read_limitations(model, prose),
        "datasets": read_datasets(model, prose, exhibits),
        "docs": read_side_documents(model, prose, findings),
        "references": read_references(model),
    }
    folds = {
        "question": read_setup(model, site, prose, clauses),
        "adoption": read_adoption(model, prose),
        "reproduce": read_reproduce(model, prose, findings),
    }
    if skeleton and folds["reproduce"] is None:
        folds["reproduce"] = {
            "id": f"{page_id}/reproduce",
            "meta": "",
            "digest": "",
            "prerequisites": [],
            "steps": [],
            "numbered": [],
            "loose": [],
        }
    builds = read_builds(model, site)
    bar = section_bar(page_id, sections, folds, skeleton)

    return PageView(
        kind="record",
        id=page_id,
        title=short,
        template=TEMPLATE,
        context={
            "head": head,
            "crumb": head["programmes"][0] if head["programmes"] else None,
            "answer": answer,
            "trace": trace,
            "needs": needs,
            "bar": bar,
            "sections": sections,
            "folds": folds,
            "builds": builds,
            "skeleton": skeleton,
            "empty": EMPTY,
            "findings_title": "Findings" if stage == "concluded" else "Findings so far",
        },
    )


# ===============================
# Record helpers
# ===============================


def record_slug(model: Model) -> str:
    """The page anchor: the record's slug, else its folder's name."""
    return model.value("slug") or model.document.folder.name


def record_outcome(model: Model) -> str:
    """The record's outcome word: the Verdict's, else the Hypothesis's; "" while open."""
    for heading in ("Verdict", "Hypothesis"):
        word = _lead_word(model.row(heading, "Outcome"))
        if word in RECORD_OUTCOMES:
            return word
    return ""


def record_word(model: Model) -> str:
    """The word a record's mark carries: its outcome, else running or planned."""
    return record_outcome(model) or (
        "running" if model.stage == "running" else "planned"
    )


def record_pill(model: Model) -> Markup:
    """A record as a pill: its mark and short name, the full title on hover."""
    title = model.value("title") or record_slug(model)
    short = model.value("short")
    return INLINE.pill(
        record_word(model),
        short or title,
        title if short else "",
        f"#{record_slug(model)}",
    )


def row_source(row: Row) -> str:
    """A row's markdown: its first line, then its nested lines dedented."""
    if row is None:
        return ""
    rest = textwrap.dedent("\n".join(row.body[1:]))
    return (row.body[0] + ("\n" + rest if rest.strip() else "")).strip()


def entries(model: Model, heading: str) -> tuple[Entry, ...]:
    section = model.section(heading)
    return (
        tuple(e for e in section.entries if not GUIDANCE.search(e.title))
        if section
        else ()
    )


def section_row(model: Model, heading: str, label: str) -> Row | None:
    """The `- **label**:` row of a section, checked against the spec."""
    model.spec.part(f"section.{heading}.{label}")
    section = model.section(heading)
    return section.rows.get(label) if section else None


def artifact_path(model: Model, name: str, stem: str = "") -> Path:
    """The file an artifact names, its `*` filled with `stem`."""
    pattern = model.spec.part(f"artifact.{name}").pattern
    return model.document.folder / pattern.replace("*", stem)


def items(row: Row | None) -> list[str]:
    """The first-level nested bullets under a row, continuation lines joined."""
    return [_unguided(text) for _, text in checks.items(row) if _unguided(text)]


def read_clauses(model: Model) -> list[Clause]:
    """The clause lines of Predictions, with tags and what superseded them."""
    superseded = {}
    for entry in entries(model, "Rounds"):
        row = entry.rows.get("Earlier decisive")
        superseded.update(checks.SUPERSEDED.findall(row.text) if row else [])
    found = []
    for text in items(section_row(model, "Hypothesis", "Predictions")):
        match = checks.CLAUSE_LINE.match(text)
        if match is None:
            continue
        tags = checks.CLAUSE_TAG.findall(match.group(2))
        found.append(
            Clause(
                id=match.group(1),
                claim=match.group(3).strip(),
                conditions=text[match.end() :].strip(),
                line=text,
                role="diagnostic" if "diagnostic" in tags else "decisive",
                standard="measured" if "measured" in tags else "tested",
                round=next((t for t in tags if t.startswith("R")), "R1"),
                superseded_by=superseded.get(match.group(1), ""),
            )
        )
    return found


def read_findings(model: Model) -> list[Finding]:
    found = []
    for entry in entries(model, "Findings"):
        match = ENTRY_TITLE.match(entry.title)
        if match is None:
            continue
        weight = checks.WEIGHT.search(entry.title)
        found.append(
            Finding(
                id=match.group(1),
                title=match.group(2).strip(),
                weight=weight.group(1) if weight else "key",
                target=(weight.group(2) or "") if weight else "",
                entry=entry,
            )
        )
    return found


def read_readings(model: Model) -> dict[str, list[dict[str, str]]]:
    """`tables/clause-readings.csv` grouped by clause, rows in file order."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in markup.read_table(artifact_path(model, "clause-readings")):
        grouped.setdefault(row.get("clause", ""), []).append(row)
    return grouped


def finding_clauses(finding: Finding, clauses: list[Clause]) -> list[Clause]:
    cited = set(
        CLAUSE_CITATION.findall(finding.text("Reading") + " " + finding.text("Summary"))
    )
    return [c for c in clauses if c.id in cited]


def deciding_findings(clause: Clause, findings: list[Finding]) -> list[Finding]:
    """The key findings whose Reading or Summary names the clause."""
    names = re.compile(CLAUSE_NAME.format(id=clause.id))
    return [
        f
        for f in findings
        if f.weight == "key"
        and names.search(f.text("Reading") + " " + f.text("Summary"))
    ]


def clause_word(clause: Clause, findings: list[Finding], readings: dict) -> str:
    """A clause's outcome: what its deciding finding says, else its one reading word, else planned."""
    for finding in deciding_findings(clause, findings):
        if word := checks.outcome(finding.entry, clause.check_clause):
            return word
    words = {x.get("outcome") for x in readings.get(clause.id, [])} & set(OUTCOME)
    return words.pop() if len(words) == 1 else "planned"


def clause_chip(clause: Clause, readings: dict, word: str, page_id: str) -> Markup:
    """A clause's chip, split when its readings differ by qualifier."""
    href = f"#{page_id}/{clause.id.lower()}"
    split = reading_split(readings.get(clause.id, []))
    if len(split) > 1:
        return WORDS.split_chip(clause.id, split, href)
    return WORDS.chip(clause.id, word, clause.id.startswith("C"), href)


def reading_split(readings: list[dict]) -> list[tuple[str, str]]:
    """(outcome, qualifier) per distinct outcome; readings sharing one join their qualifiers."""
    groups: dict[str, list[str]] = {}
    for x in readings:
        if x.get("outcome") in OUTCOME:
            groups.setdefault(x["outcome"], []).append(x.get("qualifier", ""))
    return [
        (word, "; ".join(dict.fromkeys(q for q in quals if q)))
        for word, quals in groups.items()
    ]


def deciding_keys(model: Model, findings: list[Finding]) -> list[Finding]:
    """The key findings the Verdict's Evidence cites, its first bullet's when a list; else every key."""
    evidence = section_row(model, "Verdict", "Evidence")
    bullets = items(evidence)
    text = (
        bullets[0]
        if bullets and CITATION.search(bullets[0])
        else _unguided(evidence.text if evidence else "")
    )
    cited = set(re.findall(r"\[(F\d+)\]", text))
    keys = [f for f in findings if f.weight == "key"]
    return [f for f in keys if f.id in cited] or keys


# ===============================
# Prose class
# ===============================


class Prose:
    """Record prose as HTML, its citations as chips and its code spans labelled.

    A `[D<n>]`, `[F<n>]`, `[M<n>]` or clause citation becomes a chip linked to
    its entry with a hover card; one naming no entry loses its brackets. A
    `[<slug>:F<n>]` links the other record's finding, a `[@key]` the record's
    reference, and a link to one of the record's tables or figures its
    rendering on the page.
    """

    def __init__(
        self,
        model: Model,
        page_id: str,
        site: Site,
        clauses: list[Clause],
        findings: list[Finding],
    ):
        self.page_id = page_id
        self.mlflow = (model.value("mlflow") or "").rstrip("/")
        self.slugs = {record_slug(m) for m in site.of_kind("record")}
        self.bib = read_bibliography(model)
        self.tips = self._tips(model, clauses, findings)

    def block(self, text: str, capital: bool = True) -> Markup:
        if not text:
            return Markup("")
        source = markup.capitalize(text) if capital else text
        return Markup(self._link(markup.markdown_html(source)))

    def inline(self, text: str, capital: bool = False) -> Markup:
        if not text:
            return Markup("")
        source = markup.capitalize(text) if capital else text
        return Markup(self._link(markup.inline_html(source)))

    def chip(self, code: str) -> Markup:
        """A chip for one of this record's entries or clauses."""
        kind = {"D": "d", "F": "f", "M": "m"}.get(code[0], "h")
        return INLINE.ct(
            kind, code, f"#{self.page_id}/{code.lower()}", self.tips.get(code, "")
        )

    def chips(self, text: str) -> Markup:
        """A chip for every citation in a text, in order, each once."""
        return Markup("").join(
            self.chip(c)
            for c in dict.fromkeys(CITATION.findall(text))
            if c in self.tips
        )

    # =================================
    # Private methods
    # =================================

    def _tips(
        self, model: Model, clauses: list[Clause], findings: list[Finding]
    ) -> dict[str, str]:
        def tip(head: str, body: str, label: str = "") -> str:
            inline = markup.inline_html(body)
            inline = re.sub(
                r'<span class="math[^"]*">.*?</span>', "", inline, flags=re.S
            )
            return (
                f"{head}\n" + (f"<strong>{label}:</strong> " if label else "") + inline
            )

        tips = {}
        for entry in entries(model, "Decisions"):
            if match := ENTRY_TITLE.match(entry.title):
                row = entry.rows.get("Decision")
                tips[match.group(1)] = tip(
                    entry.title, _unguided(row.text) if row else ""
                )
        for entry in entries(model, "Methods"):
            if match := ENTRY_TITLE.match(entry.title):
                row = entry.rows.get("Output")
                tips[match.group(1)] = tip(
                    entry.title, _unguided(row.text) if row else "", "Output"
                )
        for finding in findings:
            tips[finding.id] = tip(
                f"{finding.id} — {finding.title}", finding.text("Summary")
            )
        for clause in clauses:
            tips[clause.id] = tip(f"{clause.id} — {clause.claim}", clause.conditions)
        return tips

    def _link(self, html_: str) -> str:
        def citation(match: re.Match) -> str:
            code = match.group(1)
            return str(self.chip(code)) if code in self.tips else code

        def other_record(match: re.Match) -> str:
            slug, code = match.groups()
            if slug not in self.slugs:
                return match.group(0)
            return str(
                INLINE.ct(
                    "f" if code.startswith("F") else "h",
                    f"{slug}:{code}",
                    f"#{slug}/{code.lower()}",
                )
            )

        def reference(match: re.Match) -> str:
            key = match.group(1)
            label = self.bib[key]["label"] if key in self.bib else key
            return str(INLINE.cite(label, f"#{self.page_id}/ref-{key}"))

        def bare(match: re.Match) -> str:
            codes = CITATION.findall(match.group(1))
            if not all(c in self.tips for c in codes):
                return match.group(0)
            return " ".join(f"[{c}]" for c in codes)

        def text(run: str) -> str:
            run = checks.RECORD_CITATION.sub(other_record, run)
            run = CITED_ONLY.sub(bare, run)
            run = CITATION.sub(citation, run)
            return BIB_CITATION.sub(reference, run)

        html_ = _prose_runs(html_, text)
        html_ = CODE_SPAN.sub(self._code, html_)
        # a link to one of the record's tables or figures is a chip to its rendering
        return re.sub(
            r'<a href="((?:tables|figures)/[^"]+)">',
            lambda m: (
                f'<a class="tlink" href="#{self.page_id}/asset/{re.sub(r"[^A-Za-z0-9_-]", "-", Path(m.group(1)).stem)}">'
                f'<span class="tic">{"▦" if m.group(1).endswith(".csv") else "◩"}</span>'
            ),
            html_,
        )

    def _code(self, match: re.Match) -> str:
        """A code span labelled as a run id, a commit or a file."""
        value = match.group(1)
        found = checks.hex_ids(value)
        if found and found[0][0] == value:
            if not found[0][1] and len(value) == checks.RUN_ID_LENGTH:
                chip = (
                    f'<code class="run" title="{value}">{value[:RUN_ID_SHOWN]}</code>'
                )
                return (
                    f'<a href="{html.escape(self.mlflow)}/runs/{value}">{chip}</a>'
                    if self.mlflow
                    else chip
                )
            if found[0][1]:
                return f'<code class="commit">{value}</code>'
        if PATH_LIKE.search(value):
            return f'<code class="file">{value}</code>'
        return match.group(0)


def read_bibliography(model: Model) -> dict[str, dict[str, str]]:
    """references.bib as {key: label, title, venue}; the label is `Author et al. year`."""
    path = model.document.folder / checks.BIBLIOGRAPHY
    if not path.is_file():
        return {}
    found = {}
    entry_pattern = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\s*\}", re.S)
    for match in entry_pattern.finditer(path.read_text(encoding="utf-8")):
        fields = {
            key.lower(): re.sub(r"[{}]", "", value).strip()
            for key, value in re.findall(
                r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*$", match.group(2), re.M
            )
        }
        names = [
            a.split(",")[0].strip() if "," in a else a.split()[-1]
            for a in re.split(r"\s+and\s+", fields.get("author", ""))
            if a.strip()
        ]
        if len(names) == 1:
            who = names[0]
        elif len(names) == 2:
            who = f"{names[0]} and {names[1]}"
        else:
            who = f"{names[0]} et al." if names else match.group(1)
        found[match.group(1)] = {
            "label": f"{who} {fields.get('year', '')}".strip(),
            "title": fields.get("title", ""),
            "venue": fields.get("journal")
            or fields.get("booktitle")
            or fields.get("publisher")
            or "",
        }
    return found


# ===============================
# Head and answer
# ===============================


def read_head(model: Model, site: Site, page_id: str) -> dict:
    """Kicker, short name and title, the parents, the muted tail and the reference entries."""
    experiments = site.corpus.experiments
    title = model.value("title") or page_id
    short = model.value("short")

    # the tail: category, run types, links, ticket, PR and the date or the stage
    tail = [model.value("category") or model.document.folder.parent.name]
    tail += [str(t) for t in model.value("types") or []]
    if mlflow := model.value("mlflow"):
        tail.append(INLINE.link("MLflow", mlflow))
    elif any(
        not is_commit
        for f in read_findings(model)
        for _, is_commit in checks.hex_ids(f.text("Runs"))
    ):
        tail.append(INLINE.missing("no MLflow address"))
    tail.append(_ticket(model.value("ticket")))
    if pr := model.value("pr"):
        tail.append(_pr(str(pr)))
    concluded = model.value("concluded")
    tail.append(
        f"concluded {concluded.isoformat()}" if concluded else (model.stage or "draft")
    )

    # the reference entries the record varies or uses
    named = []
    for key in ("varies", "uses"):
        found = []
        for value in model.value(key) or []:
            topic, _, name = str(value).partition("/")
            exists = (experiments / "reference" / topic / f"{name}.md").is_file()
            found.append((name or topic, f"#ref-{topic}-{name}" if exists else ""))
        if found:
            named.append((key, found))

    folder = model.document.folder.resolve()
    programmes = [
        {"name": g["short"], "title": g["title"], "href": f"#programme-{g['slug']}"}
        for g in read_programmes(experiments)
        if folder in g["members"]
    ]
    return {
        "kicker": model.document.path.relative_to(experiments.parent).as_posix(),
        "programmes": programmes,
        "short": short or title,
        "title": title if short and short != title else "",
        "parents": [record_pill(p) for p in parents(model, site)],
        "tail": tail,
        "entries": named,
    }


def read_programmes(experiments: Path) -> list[dict]:
    """Each `programmes/<slug>.md`: slug, short name, title, member folders and planned short names."""
    found = []
    for path in sorted((experiments / "programmes").glob("*.md")):
        match = FRONTMATTER.match(path.read_text(encoding="utf-8"))
        meta = yaml.safe_load(match.group(1)) if match else None
        if not isinstance(meta, dict):
            continue
        slug = str(meta.get("slug") or path.stem)
        title = str(meta.get("title") or slug)
        planned = [p for p in meta.get("planned") or [] if isinstance(p, dict)]
        found.append(
            {
                "slug": slug,
                "short": str(meta.get("short") or title),
                "title": title,
                "members": {
                    (path.parent / str(m)).resolve() for m in meta.get("members") or []
                },
                "planned": {
                    str(p["slug"]): (str(p.get("short") or p["slug"]), str(p.get("title") or ""))
                    for p in planned
                    if p.get("slug")
                },
            }
        )
    return found


def parents(model: Model, site: Site) -> list[Model]:
    folder = model.document.folder
    found = []
    for value in model.value("builds_on") or []:
        target = (folder / str(value)).resolve()
        hit = next(
            (
                m
                for m in site.of_kind("record")
                if m.document.folder.resolve() == target
            ),
            None,
        )
        if hit is not None:
            found.append(hit)
    return found


def children(model: Model, site: Site) -> list[Model]:
    return [
        m
        for m in site.of_kind("record")
        if m is not model and any(p is model for p in parents(m, site))
    ]


def read_answer(
    model: Model,
    site: Site,
    prose: Prose,
    exhibits: Exhibits,
    clauses: list[Clause],
    findings: list[Finding],
    readings: dict,
    words: dict[str, str],
    outcome: str,
) -> dict:
    """The answer card: question, verdict line, evidence, strips, figure, notes, what it opened.

    Before a verdict the card is the question and the clause table alone.
    `shown` names the figure the card draws, which its finding's card then
    points to instead of drawing it twice.
    """
    answer = {
        "question": read_question(model, prose, clauses),
        "verdict": None,
        "shown": "",
    }
    answer |= {"arms": None, "strips": [], "figure": None, "notes": [], "opened": []}
    if not outcome:
        answer["clause_table"] = read_clause_table(
            model, prose, clauses, findings, readings, words, set(), anchored=True
        )
        return answer

    # the strips, and the figure unless it plots only what the strips draw
    strips, drawn = read_strips(model, clauses, readings, words)
    for finding in deciding_keys(model, findings):
        match = markup.ASSET_LINK.search(finding.text("Result"))
        if match and not match.group(2).lower().endswith(".csv"):
            if not plotted(model, match.group(2), readings, drawn):
                answer["shown"] = match.group(2)
                answer["figure"] = read_exhibit(
                    prose, exhibits, match.group(2), match.group(1), ""
                )
            break

    answer |= {
        "verdict": read_verdict_line(model, prose, outcome),
        "arms": read_arms(model, prose, clauses, readings, words, drawn),
        "strips": strips,
        "notes": read_notes(model, site, prose),
        "opened": read_opened(model, site, prose),
    }
    answer["clause_table"] = (
        []
        if answer["arms"]
        else read_clause_table(
            model, prose, clauses, findings, readings, words, drawn, anchored=False
        )
    )
    return answer


def read_question(model: Model, prose: Prose, clauses: list[Clause]) -> dict:
    """The answer's first line: the Question; else Decision it informs; else a drafted question."""
    question = _unguided(model.row("Hypothesis", "Question"))
    decision = _unguided(model.row("Hypothesis", "Decision it informs"))
    if question:
        return {
            "label": "Question",
            "text": prose.inline(question, capital=True),
            "decision": prose.inline(decision),
            "note": "",
        }
    if decision:
        return {
            "label": "Decision it informs",
            "text": prose.inline(decision, capital=True),
            "decision": "",
            "note": "The record has no Question row.",
        }
    claims = "; ".join(c.claim for c in clauses if not c.superseded_by)
    return {
        "label": "Question",
        "text": prose.inline(f"Does it hold that {claims}?")
        if claims
        else INLINE.missing("not written yet"),
        "decision": "",
        "note": "Drafted from the record’s clauses; the record has no Question row."
        if claims
        else "",
    }


def read_verdict_line(model: Model, prose: Prose, outcome: str) -> dict:
    """The outcome mark, the one-line answer, the Outcome's reading and the amended-rule flags."""
    discussion = items(section_row(model, "Verdict", "Discussion"))
    answer = _lead(discussion[0])[0] if discussion else ""
    answer = answer or TLDR_STATUS.sub("", _unguided(model.row("TL;DR", "Hypothesis")))
    reading = _unguided(model.row("Verdict", "Outcome"))
    reading = re.sub(rf"^\s*{re.escape(outcome)}\b\s*", "", reading, flags=re.I)
    reading = re.sub(r"^[—–:;,.\s-]+", "", reading)
    return {
        "word": outcome,
        "answer": prose.inline(answer, capital=True),
        "reading": prose.inline(reading),
        "flags": read_flags(model, prose),
    }


def read_flags(model: Model, prose: Prose) -> list[dict]:
    """A flag for each `Note added <date>:` on a Predictions line dated at or after the first result."""
    history = " ".join(f.text("History") for f in read_findings(model))
    dates = ISO_DATE.findall(history)
    first = min(dates) if dates else None
    concluded = model.value("concluded")
    flags = []
    for text in items(section_row(model, "Hypothesis", "Predictions")):
        clause = checks.CLAUSE_LINE.match(text)
        label = NESTED_LABEL.match(text)
        where = (
            f"{clause.group(1)} bar"
            if clause
            else (label.group(1) if label else "Predictions")
        )
        notes = list(RULE_NOTE.finditer(text))
        for i, note in enumerate(notes):
            date = note.group(1)
            if first is None or date < first:
                continue
            end = notes[i + 1].start() if i + 1 < len(notes) else len(text)
            at = (
                "at conclusion"
                if concluded and date == concluded.isoformat()
                else "after the first result"
            )
            flags.append(
                {
                    "title": f"{where} amended {at} · {date}",
                    "text": prose.inline(text[note.end() : end].strip(), capital=True),
                }
            )
    return flags


def read_clause_table(
    model: Model,
    prose: Prose,
    clauses: list[Clause],
    findings: list[Finding],
    readings: dict,
    words: dict[str, str],
    drawn: set[str],
    anchored: bool,
) -> list[dict]:
    """The answer's clause rows: chip, claim, condition, a Q clause's answer and what was measured."""
    rows = []
    for clause in clauses:
        yes, _ = clause.keywords
        condition = clause.conditions.split("; ")[0]
        if not condition.lower().startswith(yes.split()[0]):
            condition = f"{yes} {condition}"
        deciding = deciding_findings(clause, findings)
        answer = ""
        if clause.id.startswith("Q") and deciding:
            answer = Markup("{} {}").format(
                prose.inline(deciding[0].text("Summary"), capital=True),
                prose.chip(deciding[0].id),
            )
        measured = ""
        if (
            not answer
            and clause.id not in drawn
            and (readings.get(clause.id) or deciding)
        ):
            measured = read_measured(prose, clause, deciding, readings)
        rows.append(
            {
                "id": f"{prose.page_id}/{clause.id.lower()}" if anchored else "",
                "chip": clause_chip(clause, readings, words[clause.id], prose.page_id),
                "claim": prose.inline(clause.claim),
                "condition": prose.inline(_trace_free(condition)),
                "answer": answer,
                "measured": measured,
                "planned": bool(clause.superseded_by),
            }
        )
    return rows


def read_measured(
    prose: Prose, clause: Clause, deciding: list[Finding], readings: dict
) -> Markup:
    """What a clause measured: each numeric reading with its interval; else the deciding Summary."""
    texts = []
    for x in readings.get(clause.id, []):
        if markup.to_float(x.get("value", "")) is None:
            continue
        interval = (
            f" [{_compact(x['ci_low'])}, {_compact(x['ci_high'])}]"
            if x.get("ci_low") and x.get("ci_high")
            else ""
        )
        qualifier = f"{x['qualifier']} " if x.get("qualifier") else ""
        texts.append(f"{qualifier}{_compact(x['value'])}{interval}")
    if texts:
        return Markup(html.escape("; ".join(texts)))
    if deciding:
        return prose.inline(deciding[0].text("Summary"), capital=True)
    return INLINE.missing("not read yet")


def read_strips(
    model: Model, clauses: list[Clause], readings: dict, words: dict
) -> tuple[list[dict], set[str]]:
    """Interval strips: one per H clause and bar, one for every C reading against its margin.

    Returns:
        The strips, and the clauses they draw.
    """
    bars: dict[tuple[str, float], list[dict]] = {}
    margins: list[dict] = []
    for clause in clauses:
        for x in readings.get(clause.id, []):
            value = markup.to_float(x.get("value", ""))
            if value is None:
                continue
            low, high = (
                markup.to_float(x.get("ci_low", "")),
                markup.to_float(x.get("ci_high", "")),
            )
            low, high = (value, value) if low is None or high is None else (low, high)
            word = x.get("outcome") if x.get("outcome") in OUTCOME else "inconclusive"
            interval = (
                f" [{_compact(x['ci_low'])}, {_compact(x['ci_high'])}]"
                if x.get("ci_low") and x.get("ci_high")
                else ""
            )
            point = {
                "chip": WORDS.chip(
                    clause.id,
                    word,
                    clause.id.startswith("C"),
                    f"#{record_slug(model)}/{clause.id.lower()}",
                ),
                "qualifier": x.get("qualifier", ""),
                "value": value,
                "low": low,
                "high": high,
                "kind": OUTCOME[word],
                "text": f"{_compact(x['value'])}{interval}",
            }
            if (bar := markup.to_float(x.get("bar", ""))) is not None:
                bars.setdefault((clause.id, bar), []).append(point)
            elif (margin := markup.to_float(x.get("margin", ""))) is not None:
                margins.append(point | {"margin": margin})

    strips = []
    for (_, bar), points in bars.items():
        lo, hi, ticks = axis_ticks(
            min([bar] + [p["low"] for p in points]),
            max([bar] + [p["high"] for p in points]),
        )
        strips.append(_strip(points, lo, hi, ticks, bar=bar))
    if margins:
        lo, hi, ticks = axis_ticks(
            min(min(p["low"], -p["margin"]) for p in margins),
            max(max(p["high"], p["margin"]) for p in margins),
        )
        strips.append(_strip(margins, lo, hi, ticks))
    drawn = {cid for cid, xs in readings.items() for x in xs if _drawn(x)}
    return strips, drawn if strips else set()


def axis_ticks(lo: float, hi: float) -> tuple[float, float, list[float]]:
    """A strip's axis: about STRIP_STEPS steps of 1, 2, 2.5 or 5 times a power of ten."""
    if hi <= lo:
        lo, hi = lo - (abs(lo) or 1) / 2, hi + (abs(hi) or 1) / 2
    raw = (hi - lo) / STRIP_STEPS
    magnitude = 10 ** int(f"{raw:e}".split("e")[1])
    step = next(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw)
    start, end = (lo // step) * step, -((-hi) // step) * step
    ticks = [
        round(start + i * step, 10) for i in range(int(round((end - start) / step)) + 1)
    ]
    return round(start, 10), round(end, 10), ticks


def plotted(model: Model, src: str, readings: dict, drawn: set[str]) -> bool:
    """Whether a figure plots only values the strips draw, by its same-named table."""
    table = markup.read_table(
        model.document.folder / "tables" / f"{Path(src).stem}.csv"
    )
    values = {
        round(v, 4)
        for cid in drawn
        for x in readings.get(cid, [])
        if (v := markup.to_float(x.get("value", ""))) is not None
    }
    if not values or not table:
        return False
    rows = [
        {round(v, 4) for c in row.values() if (v := markup.to_float(c)) is not None}
        for row in table
    ]
    return all(row & values for row in rows)


def read_arms(
    model: Model,
    prose: Prose,
    clauses: list[Clause],
    readings: dict,
    words: dict,
    drawn: set[str],
) -> dict | None:
    """`verdict-by-arm.csv` joined with each `<quantity>-by-arm.csv` on `arm`, as the arms table.

    Columns equal across arms are dropped, an interval's columns join their
    value, the baseline row (`role`) is lifted to the top and the adopted arm
    badged. Columns group by the table they come from; a table with several
    rows per arm is left out.
    """
    verdict = markup.read_table(artifact_path(model, "verdict-by-arm"))
    if not verdict:
        return None
    by_clause = {c.id: c for c in clauses}

    # the column groups: the verdict table's, then each per-arm table's
    groups = [("", verdict)]
    for path in sorted(
        artifact_path(model, "arm-table", "").parent.glob("*-by-arm.csv")
    ):
        if path.name == Path(model.spec.part("artifact.verdict-by-arm").pattern).name:
            continue
        table = markup.read_table(path)
        per_arm = [row.get("arm", "") for row in table]
        # a table with several rows per arm (settings, levels) has no one cell per arm
        if len(per_arm) == len(set(per_arm)):
            stem = path.stem.removesuffix("-by-arm")
            groups.append((markup.column_header(stem.replace("-", "_")), table))
    arms = [row.get("arm", "") for row in verdict]
    baseline = [
        i for i, row in enumerate(verdict) if row.get("role", "").lower() == "baseline"
    ]
    order = baseline + [i for i in range(len(arms)) if i not in baseline]

    columns = []
    for name, table in groups:
        by_arm = {row.get("arm", ""): row for row in table}
        rows = [by_arm.get(arm, {}) for arm in arms]
        keys = [
            k
            for k in (table[0].keys() if table else ())
            if k not in ("arm", "role", "adopted")
        ]
        bounds = _interval_columns(keys)
        for key in keys:
            values = [row.get(key, "") for row in rows]
            # a column another table already gave, by name or by values, shows once
            again = name and any(
                markup.column_header(c["key"]).lower()
                == markup.column_header(key).lower()
                or c["values"] == values
                for c in columns
            )
            if key in {b for pair in bounds.values() for b in pair} or again:
                continue
            if key != "clause" and len(set(values)) < 2:
                continue
            columns.append(
                {
                    "group": name,
                    "key": key,
                    "values": values,
                    "bounds": bounds.get(key),
                    "rows": rows,
                }
            )

    # one cell per column per arm, the baseline first
    body = []
    for i in order:
        cells = []
        for column in columns:
            value = column["values"][i]
            if column["key"] == "clause" and value in by_clause:
                cells.append(
                    (
                        "chip",
                        clause_chip(
                            by_clause[value], readings, words[value], prose.page_id
                        ),
                    )
                )
            elif value in OUTCOME:
                cells.append(("word", Markup("{} {}").format(WORDS.mark(value), value)))
            elif value in ("True", "False"):
                cells.append(("text", "yes" if value == "True" else "no"))
            elif markup.NUMERIC.match(value.strip() or "x"):
                low, high = (
                    column["rows"][i].get(b, "") for b in (column["bounds"] or ("", ""))
                )
                cells.append(
                    (
                        "number",
                        (
                            markup.number(value),
                            markup.number(low) if low else "",
                            markup.number(high) if high else "",
                        ),
                    )
                )
            else:
                cells.append(("text", value))
        adopted = verdict[i].get("adopted", "").lower() in YES
        body.append(
            {
                "arm": arms[i],
                "cells": cells,
                "adopted": adopted,
                "baseline": i in baseline,
            }
        )
    heads = [
        {
            "group": c["group"],
            "name": markup.column_header(c["key"]),
            "numeric": all(markup.NUMERIC.match(v.strip() or "—") for v in c["values"]),
        }
        for c in columns
    ]
    spans = [
        (g, sum(1 for c in columns if c["group"] == g))
        for g in dict.fromkeys(c["group"] for c in columns)
    ]
    rule = next(
        (
            i
            for i in items(section_row(model, "Hypothesis", "Predictions"))
            if checks.VERDICT_RULE.match(i)
        ),
        "",
    )
    rule = RULE_NOTE.split(re.sub(r"^\*\*Verdict rule\*\*:\s*", "", rule))[0].strip()
    return {
        "heads": heads,
        "groups": spans if len(spans) > 1 else [],
        "rows": body,
        "rule": prose.inline(rule, capital=True),
    }


def read_notes(model: Model, site: Site, prose: Prose) -> list[dict]:
    """Caveat, Disclosure tagged with its kind, and Later evidence from later records."""
    notes = []
    if caveat := _unguided(model.row("Verdict", "Caveat")):
        notes.append(
            {
                "kind": "caveat",
                "title": "Caveat",
                "tag": "",
                "text": prose.inline(caveat, capital=True),
                "refs": [],
            }
        )
    if disclosure := _unguided(model.row("Hypothesis", "Disclosure")):
        match = DISCLOSURE.match(disclosure)
        tag, text = (
            (match.group(1).lower(), match.group(2)) if match else ("", disclosure)
        )
        notes.append(
            {
                "kind": "disclosure",
                "title": "Disclosure",
                "tag": tag,
                "text": prose.inline(text, capital=True),
                "refs": [],
            }
        )

    # later records that cite a finding here, and those that only build on it
    slug = record_slug(model)
    cites = []
    for other in site.of_kind("record"):
        if other is model:
            continue
        for finding in read_findings(other):
            text = "\n".join(row_source(r) for r in finding.entry.rows.values())
            for match in checks.RECORD_CITATION.finditer(text):
                if match.group(1) == slug and match.group(2).startswith("F"):
                    relation = RELATION_BEFORE.search(text[: match.start()])
                    cites.append(
                        (
                            other,
                            finding,
                            match.group(2),
                            relation.group(1) if relation else "",
                        )
                    )
    later = _unique([c[0] for c in cites] + children(model, site))
    if not later:
        return notes
    texts = []
    for other, finding, code, relation in cites:
        verb = f" {relation} {code}" if relation else f" cites {code}"
        texts.append(
            Markup("<b>{}</b>{}: {}").format(
                other.value("short") or record_slug(other),
                verb,
                prose.inline(finding.text("Summary")),
            )
        )
    only = [c for c in children(model, site) if not any(c is x[0] for x in cites)]
    if only:
        count = _count(len(only), "later record", spelled=True)
        texts.append(
            f"{count[0].upper()}{count[1:]} list{'s' if len(only) == 1 else ''} this one in builds_on without citing a finding."
        )
    kinds = list(dict.fromkeys(RELATIONS[c[3]] for c in cites if c[3]))
    title = f"Later evidence · {_count(len(later), 'experiment')} build{'s' if len(later) == 1 else ''} on this"
    refs = [record_pill(m) for m in later] + [
        prose.chip(c) for c in dict.fromkeys(x[2] for x in cites) if c in prose.tips
    ]
    notes.append(
        {
            "kind": "later",
            "title": title,
            "tag": ", ".join(kinds),
            "text": Markup(" ").join(texts),
            "refs": refs,
        }
    )
    return notes


def read_opened(model: Model, site: Site, prose: Prose) -> list[dict]:
    """The Verdict's Next items as What it opened: planned pill, issue pill, finding chips or none."""
    by_slug = {record_slug(m): m for m in site.of_kind("record")}
    planned_names = {
        slug: names
        for g in read_programmes(site.corpus.experiments)
        for slug, names in g["planned"].items()
    }
    opened = []
    for item in items(section_row(model, "Verdict", "Next")):
        planned, issue = PLANNED_AS.search(item), ISSUE.search(item)
        text = NOT_PLANNED.sub("", PLANNED_AS.sub("", ISSUE.sub("", item))).strip()
        shown = markup.first_sentence(text)
        if planned and planned.group(1) in by_slug:
            other = by_slug[planned.group(1)]
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": record_word(other),
                    "link": record_pill(other),
                    "rests": False,
                }
            )
        elif planned:
            short, title = planned_names.get(planned.group(1), (planned.group(1), ""))
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": "planned",
                    "link": INLINE.pill("planned", short, title, ""),
                    "rests": False,
                }
            )
        elif issue:
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": "",
                    "link": INLINE.iss(issue.group(1), ""),
                    "rests": False,
                }
            )
        else:
            cited = CITATION.findall(shown)
            chips = Markup("").join(
                prose.chip(c)
                for c in CITATION.findall(text)
                if c not in cited and c in prose.tips
            )
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": "",
                    "link": chips,
                    "rests": bool(cited),
                }
            )
    return opened


# ===============================
# Clause trace and needs
# ===============================


def read_trace(
    model: Model,
    prose: Prose,
    clauses: list[Clause],
    findings: list[Finding],
    readings: dict,
    words: dict,
) -> dict | None:
    """How each clause was settled: chip and claim, Decided and Measured by, Found, So.

    Decided and Measured by show only when every clause line cites both;
    otherwise a note names the decisions the Predictions framing cites. Found
    gives each clause its own readings; a So that several clauses in a row
    share spans their rows instead of repeating.
    """
    if not clauses:
        return None
    linked = all(
        re.search(r"\[D\d+\]", c.line) and re.search(r"\[M\d+\]", c.line)
        for c in clauses
    )
    rounds = len(entries(model, "Rounds"))

    rows = []
    for clause in clauses:
        deciding = deciding_findings(clause, findings)
        chips = Markup("").join(prose.chip(f.id) for f in deciding)
        found = read_measured(prose, clause, deciding, readings)
        tags = [
            w for w in (clause.role, clause.standard) if w not in ("decisive", "tested")
        ]
        tags += [clause.round] if rounds > 1 else []
        tags += (
            [f"superseded by {clause.superseded_by}"] if clause.superseded_by else []
        )
        rows.append(
            {
                "id": f"{prose.page_id}/{clause.id.lower()}",
                "chip": clause_chip(clause, readings, words[clause.id], prose.page_id),
                "claim": prose.inline(clause.claim),
                "tags": " · ".join(tags),
                "decided": prose.chips(" ".join(re.findall(r"\[D\d+\]", clause.line))),
                "measured_by": prose.chips(
                    " ".join(re.findall(r"\[M\d+\]", clause.line))
                ),
                "found": found,
                "chips": chips,
                "so": clause_so(prose, clause, deciding),
                "planned": bool(clause.superseded_by),
            }
        )
    # a So shared by the rows after it spans them
    for i, row in enumerate(rows):
        shared = itertools.takewhile(lambda r: r["so"] == row["so"], rows[i:])
        first = i == 0 or rows[i - 1]["so"] != row["so"]
        row["so_span"] = len(list(shared)) if first else 0

    framing = section_row(model, "Hypothesis", "Predictions")
    under = prose.chips(
        " ".join(re.findall(r"\[D\d+\]", framing.body[0] if framing else ""))
    )
    return {
        "rows": rows,
        "linked": linked,
        "has_found": any(r["found"] for r in rows),
        "under": under,
        "rests_on": read_rests_on(model, prose),
    }


def clause_so(prose: Prose, clause: Clause, deciding: list[Finding]) -> Markup:
    """The So of a clause: its own Implication bullet, else the first sentence of the Implication."""
    for finding in deciding:
        row = finding.entry.rows.get("Implication")
        for text in items(row):
            if text.startswith(f"[{clause.id}]:"):
                return prose.inline(text.split(":", 1)[1].strip(), capital=True)
    for finding in deciding:
        if implication := finding.text("Implication"):
            return prose.inline(
                markup.first_sentence(implication.split(" - ")[0]), capital=True
            )
    return INLINE.missing("no implication recorded")


def read_rests_on(model: Model, prose: Prose) -> list[dict]:
    """The decisions and methods the Limitations lean on, each with its label."""
    rests = []
    for text in items(section_row(model, "Verdict", "Limitations")):
        label, _, _ = text.partition(" — ")
        cited = re.findall(r"\[([DM]\d+)\]", text)
        if " — " in text and cited:
            rests.append(
                {
                    "label": prose.inline(label),
                    "chips": Markup("").join(
                        prose.chip(c) for c in cited if c in prose.tips
                    ),
                }
            )
    return rests


def read_needs(model: Model, prose: Prose) -> list[dict]:
    """People, Compute, Data and External as their short values; a heavy one marked hot."""
    needs = []
    for label, name in NEEDS:
        text = _unguided(model.row("Design", label))
        short = text.split(" — ")[0] if " — " in text else markup.first_sentence(text)
        plain = markup.plain_text(markup.inline_html(short))
        needs.append(
            {
                "name": name,
                "text": prose.inline(short) if text else "",
                "hot": bool(plain)
                and not NO_NEED.match(plain)
                and bool(HOT_NEED.search(plain)),
            }
        )
    return needs


# ===============================
# Findings
# ===============================


def read_findings_section(
    model: Model,
    prose: Prose,
    exhibits: Exhibits,
    clauses: list[Clause],
    findings: list[Finding],
    readings: dict,
    words: dict,
    shown: str,
) -> dict:
    """Deciding key findings as open cards with their supporting findings; the rest as small cards."""
    if not findings:
        return {"n": 0, "cards": [], "others": []}
    deciding = deciding_keys(model, findings)

    def supporting(key: Finding) -> list[Finding]:
        return [f for f in findings if f.weight == "supporting" and f.target == key.id]

    def small(finding: Finding) -> dict:
        return read_small_card(
            prose, exhibits, clauses, finding, readings, words, shown
        )

    cards = []
    for key in deciding:
        card = read_card(model, prose, exhibits, clauses, key, readings, words, shown)
        card["supporting"] = [small(f) for f in supporting(key)]
        cards.append(card)
    nested = {f.id for key in deciding for f in supporting(key)}
    rest = [f for f in findings if f not in deciding and f.id not in nested]
    keys = {f.id for f in rest if f.weight != "supporting"}
    others = []
    for finding in rest:
        if finding.weight == "supporting" and finding.target in keys:
            continue
        others.append(small(finding))
        others += [
            small(f)
            for f in rest
            if f.weight == "supporting" and f.target == finding.id
        ]
    return {"n": len(findings), "cards": cards, "others": others}


def read_card(
    model: Model,
    prose: Prose,
    exhibits: Exhibits,
    clauses: list[Clause],
    finding: Finding,
    readings: dict,
    words: dict,
    shown: str,
) -> dict:
    """A finding in full: key number, claim, reading, could overturn it, So, runs, exhibit, history."""
    cited = finding_clauses(finding, clauses)
    number, word, note = key_number(finding, cited, readings)

    # the exhibit, or a pointer to the answer card's figure
    match = markup.ASSET_LINK.search(finding.text("Result"))
    src = match.group(2) if match else ""
    if src and src == shown:
        exhibit, pointer = None, True
    else:
        exhibit, pointer = read_result(prose, exhibits, finding), False
    more = read_more_exhibits(prose, exhibits, finding, src)

    return {
        "id": finding.id,
        "anchor": f"{prose.page_id}/{finding.id.lower()}",
        "chip": prose.chip(finding.id),
        "answers": [
            clause_chip(c, readings, words[c.id], prose.page_id) for c in cited
        ],
        "number": number,
        "number_kind": OUTCOME.get(word, word),
        "number_note": note,
        "claim": prose.inline(finding.text("Summary"), capital=True),
        **read_reading(prose, finding),
        "runs": read_runs(prose, finding),
        "exhibit": exhibit,
        "pointer": pointer,
        "more": more,
        "supporting": [],
    }


def read_small_card(
    prose: Prose,
    exhibits: Exhibits,
    clauses: list[Clause],
    finding: Finding,
    readings: dict,
    words: dict,
    shown: str,
) -> dict:
    """A supporting or other finding: chip, title, clauses, Summary, exhibit and runs; its reading folds."""
    match = markup.ASSET_LINK.search(finding.text("Result"))
    pointer = bool(match and match.group(2) == shown)
    implication = finding.entry.rows.get("Implication")
    reading = read_reading(prose, finding)
    if not (implication and _unguided(implication.text)):
        reading["so"] = ""
    return {
        **reading,
        "anchor": f"{prose.page_id}/{finding.id.lower()}",
        "chip": prose.chip(finding.id),
        "title": prose.inline(finding.title),
        "minor": finding.weight == "minor",
        "answers": [
            clause_chip(c, readings, words[c.id], prose.page_id)
            for c in finding_clauses(finding, clauses)
        ],
        "summary": prose.inline(finding.text("Summary"), capital=True),
        "exhibit": None if pointer else read_result(prose, exhibits, finding),
        "pointer": pointer,
        "runs": read_runs(prose, finding),
    }


def read_reading(prose: Prose, finding: Finding) -> dict:
    """A finding's Reading as paragraphs, what could overturn it, its So and its dated history."""
    reading = finding.entry.rows.get("Reading")
    blocks = items(reading)
    lead = _unguided(reading.body[0]) if reading else ""
    paragraphs = ([lead] if lead and blocks else []) + (
        blocks[:2] if blocks else [finding.source("Reading")]
    )
    overturn = blocks[2] if len(blocks) > 2 else ""
    implication = finding.entry.rows.get("Implication")
    return {
        "reading": [prose.inline(p, capital=True) for p in paragraphs if p],
        "overturn": prose.inline(overturn, capital=True),
        "so": prose.block(row_source(implication))
        if implication and _unguided(implication.text)
        else INLINE.missing("no implication recorded"),
        "history": read_history(prose, finding.entry.rows.get("History")),
    }


def key_number(
    finding: Finding, cited: list[Clause], readings: dict
) -> tuple[str, str, str]:
    """A key finding's number, the outcome word that colours it, and what it counts.

    The Key number row first; else the one numeric reading of the one clause
    it reads, with its qualifier, bar or margin and interval; else the
    Summary's first number with the words around it that say what it counts.
    """
    word = ""
    if cited:
        word = checks.outcome(finding.entry, cited[0].check_clause)
    if row := finding.text("Key number"):
        value, _, note = row.partition(" — ")
        return value.strip(), word, note.strip()
    numeric = [
        x
        for c in cited
        for x in readings.get(c.id, [])
        if markup.to_float(x.get("value", "")) is not None
    ]
    if len(cited) == 1 and len(numeric) == 1:
        x = numeric[0]
        limit = (
            f"bar {x['bar']}"
            if x.get("bar")
            else (f"margin ±{x['margin']}" if x.get("margin") else "")
        )
        interval = (
            f"interval {x['ci_low']} to {x['ci_high']}"
            if x.get("ci_low") and x.get("ci_high")
            else ""
        )
        return (
            _value(x["value"]),
            x.get("outcome", ""),
            "; ".join(p for p in (x.get("qualifier", ""), limit, interval) if p),
        )
    value, note = number_in_context(finding.text("Summary"))
    return (value, word, note) if value else ("", "", "")


def number_in_context(text: str) -> tuple[str, str]:
    """The Summary's first reported quantity and the words that say what it counts.

    The words after the number come first (`0.06 accuracy`), else the words
    before it (`accuracy on held-out pages stays at 0.91`); `N of the M <noun>`
    reads as `N / M` and `N to M` as a range. Skipped: a number naming a thing
    (`KPI 4`, `level 2`, `Opus 5`) and a target or threshold (`the 0.70
    target`, `the bar of 650`), so a Summary holding only a bar gives none.
    """
    plain = re.sub(r"`[^`]*`|\[[^\]]*\]|\$[^$]*\$", " ", text)
    for match in NUMBER.finditer(plain):
        if plain[match.end() : match.end() + 1].isalpha() and not UNIT.match(
            plain, match.end()
        ):
            continue
        raw = match.group(0).rstrip(",.")
        end = match.start() + len(raw)
        before = plain[: match.start()].split()
        after_text = plain[end:].lstrip()

        # skip names, targets and thresholds
        last = before[-1] if before else ""
        if last.lower() in LABEL_WORDS or (last.isupper() and last.isalpha()):
            continue
        bare = (
            raw.replace(",", "").isdigit()
            and not UNIT.match(plain, end)
            and not after_text.startswith("%")
        )
        if bare and last[:1].isupper() and last.isalpha():
            continue
        if THRESHOLD_BEFORE.search(" ".join(before[-3:])) or THRESHOLD_AFTER.match(
            after_text
        ):
            continue

        # the number with its sign, unit, whole or range
        start_at = match.start()
        value = raw
        if (
            start_at
            and plain[start_at - 1] in "+-−"
            and (start_at < 2 or not plain[start_at - 2].isalnum())
        ):
            start_at -= 1
            value = ("+" if plain[start_at] == "+" else "−") + value
        if unit := UNIT.match(plain, end):
            value, end = value + unit.group(0), unit.end()
        cut = PHRASE_CUT.search(plain, end)
        after = plain[end : cut.start() if cut else len(plain)].split()
        if after and after[0] == "%":
            value, after = value + "%", after[1:]
        joined = re.match(
            r"(of (?:the )?|to )(\d[\d,.]*(?:%|×|[kMB](?![A-Za-z]))?)", " ".join(after)
        )
        if joined:
            value += (
                " / " if joined.group(1).startswith("of") else " to "
            ) + joined.group(2)
            after = " ".join(after)[joined.end() :].split()

        # the words after the number, else the words before it in its phrase
        after = list(itertools.takewhile(lambda w: w.lower() not in JOIN_WORDS, after))
        if after and (
            after[0].lower() not in FUNCTION_WORDS
            or (after[0].lower() == "of" and len(after) > 2)
        ):
            after = after[:CONTEXT_WORDS]
            while after and after[-1].lower() in FUNCTION_WORDS:
                after.pop()
            return value, " ".join(after)
        phrase = max(
            (m.end() for m in PHRASE_CUT.finditer(plain, 0, start_at)), default=0
        )
        words = _context_words(plain[phrase:start_at].split(), last=True)
        if not words:
            first = PHRASE_CUT.search(plain)
            words = _context_words(
                plain[: first.start() if first else start_at].split(), last=False
            )
        return value, " ".join(words)
    return "", ""


def read_runs(prose: Prose, finding: Finding) -> dict:
    """The runs line: run ids and commits (the first three of each) and the replaced runs."""
    found = checks.hex_ids(finding.text("Runs"))
    runs = [value for value, is_commit in dict.fromkeys(found) if not is_commit]
    commits = [value for value, is_commit in dict.fromkeys(found) if is_commit]
    replaced = [
        run
        for match in REPLACES.finditer(finding.text("History"))
        for run in re.findall(r"`([0-9a-f]+)`", match.group(1))
    ]
    return {
        "ids": [
            (run[:RUN_ID_SHOWN], f"{prose.mlflow}/runs/{run}" if prose.mlflow else "")
            for run in runs[:RUNS_SHOWN]
        ],
        "more": max(len(runs) - RUNS_SHOWN, 0),
        "commits": [c[:SHA_SHOWN] for c in commits[:RUNS_SHOWN]],
        "more_commits": max(len(commits) - RUNS_SHOWN, 0),
        "linked": bool(prose.mlflow),
        "replaced": [
            (run[:RUN_ID_SHOWN], f"{prose.mlflow}/runs/{run}" if prose.mlflow else "")
            for run in replaced
        ],
    }


def read_history(prose: Prose, row: Row | None) -> list[dict]:
    """History bullets as dated lines, each with its timing tag and whose call it was."""
    history = []
    for text in items(row):
        match = re.match(r"(\d{4}-\d\d-\d\d)\s*(.*)$", text, re.S)
        date, rest = (match.group(1), match.group(2)) if match else ("", text)
        tag = TIMING.search(rest)
        if tag:
            rest = TIMING.sub("", rest, count=1).strip()
        history.append(
            {
                "date": date,
                "after": None if not tag else tag.group(1).startswith("after"),
                "user": bool(tag and tag.group(2)),
                "text": prose.inline(rest),
            }
        )
    return history


def read_result(prose: Prose, exhibits: Exhibits, finding: Finding) -> dict | None:
    """A finding's Result: a figure or table file, or an inline markdown table, captioned."""
    source = finding.source("Result")
    if not source:
        return _exhibit(missing=INLINE.missing("no result recorded"))
    match = markup.ASSET_LINK.search(source)
    if match:
        rest = " ".join(
            line.strip()
            for line in source.replace(match.group(0), "").splitlines()
            if line.strip()
        )
        return read_exhibit(prose, exhibits, match.group(2), match.group(1), rest)

    # an inline markdown table, its prose the caption
    table = "\n".join(
        line.strip() for line in source.splitlines() if line.lstrip().startswith("|")
    )
    if not table:
        return _exhibit(missing=INLINE.missing("no figure or table"))
    caption = " ".join(
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("|")
    )
    exhibits.counts["Table"] += 1
    return _exhibit(
        label=f"Table {exhibits.counts['Table']}",
        caption=prose.inline(caption),
        html=Markup(markup.markdown_html(table)),
        kind="Table",
    )


def read_exhibit(
    prose: Prose, exhibits: Exhibits, src: str, lead: str, rest: str
) -> dict:
    """One figure or table file, numbered page-wide, its caption the link text and prose."""
    inlined = markup.inline_asset(
        exhibits.folder, src, f"{prose.page_id}-{Path(src).stem}-"
    )
    if inlined is None:
        return _exhibit(missing=INLINE.missing(f"missing: {src}"))
    kind, html_ = inlined
    exhibits.counts[kind] += 1
    anchor = exhibits.anchor(src)
    placed = anchor not in exhibits.anchors
    exhibits.anchors.add(anchor)
    lead = lead.strip()
    if lead and not lead.endswith((".", "?", "!", ":", ";", "—")):
        lead += "."
    caption = " ".join(x for x in (lead, rest.strip()) if x)
    return _exhibit(
        anchor=anchor if placed else "",
        label=f"{kind} {exhibits.counts[kind]}",
        caption=prose.inline(caption),
        html=Markup(html_),
        src=src,
        kind=kind,
    )


def read_more_exhibits(
    prose: Prose, exhibits: Exhibits, finding: Finding, result: str
) -> list[dict]:
    """Tables and figures the Reading or Implication links beyond the Result."""
    seen, found = {result}, []
    for label in ("Reading", "Implication"):
        for lead, src in markup.ASSET_LINK.findall(finding.source(label)):
            if src not in seen:
                seen.add(src)
                found.append(read_exhibit(prose, exhibits, src, lead, ""))
    return found


# ===============================
# Discussion, decisions, methods, limitations
# ===============================


def read_discussion(model: Model, prose: Prose) -> dict:
    """The Discussion blocks after the first, which the verdict line uses, as folds."""
    folds = []
    for i, block in enumerate(items(section_row(model, "Verdict", "Discussion"))[1:]):
        lead, rest = _lead(block)
        n = len(set(re.findall(r"\[(F\d+)\]", block)))
        folds.append(
            {
                "id": f"{prose.page_id}/discussion-{i + 1}",
                "title": prose.inline(lead or markup.first_sentence(rest)),
                "meta": _count(n, "finding") if n else "no citation",
                "text": prose.inline(rest, capital=True) if lead else "",
                "open": i == 0,
            }
        )
    return {"n": len(folds), "folds": folds}


def read_decisions(model: Model, prose: Prose, clauses: list[Clause]) -> dict:
    """Each decision a fold with its timing, last date and change count; grouped past GROUP_PAST."""
    found = []
    for entry in entries(model, "Decisions"):
        match = ENTRY_TITLE.match(entry.title)
        if match is None:
            continue
        history_row = entry.rows.get("History")
        history = read_history(prose, history_row)
        dates = ISO_DATE.findall(history_row.text if history_row else "")
        rows = [
            (
                label,
                prose.block(
                    _bold_leads(row_source(entry.rows[label]))
                    if label == "Alternatives"
                    else row_source(entry.rows[label])
                ),
            )
            for label in ("Decision", "Why", "Alternatives")
            if label in entry.rows and _unguided(entry.rows[label].text)
        ]
        last = items(history_row)[-1] if len(history) > 1 else ""
        found.append(
            {
                "id": match.group(1),
                "anchor": f"{prose.page_id}/{match.group(1).lower()}",
                "chip": prose.chip(match.group(1)),
                "title": prose.inline(match.group(2)),
                "after": any(h["after"] for h in history),
                "date": dates[-1][5:] if dates else "",
                "changes": max(len(history) - 1, 0),
                "last": _last_change(last),
                "rows": rows,
                "history": history,
            }
        )
    if len(found) <= GROUP_PAST:
        return {"n": len(found), "groups": [("", found)] if found else []}

    # the groups: cited by a clause, beside the readings, setup
    predictions = items(section_row(model, "Hypothesis", "Predictions"))
    beside_text = " ".join(t for t in predictions if t.startswith("**Read beside"))
    beside = set(re.findall(r"\[(D\d+)\]", beside_text))
    cited = set(re.findall(r"\[(D\d+)\]", " ".join(c.line for c in clauses))) - beside
    groups = [
        ("Cited by a clause", [d for d in found if d["id"] in cited]),
        ("Beside the readings", [d for d in found if d["id"] in beside]),
        ("Setup", [d for d in found if d["id"] not in cited | beside]),
    ]
    return {"n": len(found), "groups": [g for g in groups if g[1]]}


def read_methods(model: Model, prose: Prose) -> dict:
    """Each method step a fold: in → out, How as steps with their code, Code, Settings."""
    found = []
    for entry in entries(model, "Methods"):
        match = ENTRY_TITLE.match(entry.title)
        if match is None:
            continue
        code_row = entry.rows.get("Code")
        code = row_source(code_row) if code_row else ""
        symbols = [s or Path(p).name for p, s in checks.CODE_REFERENCE.findall(code)]
        settings_row = entry.rows.get("Settings")
        found.append(
            {
                "id": match.group(1),
                "anchor": f"{prose.page_id}/{match.group(1).lower()}",
                "chip": prose.chip(match.group(1)),
                "title": prose.inline(match.group(2)),
                "meta": symbols[0] if symbols else "",
                "input": prose.block(row_source(entry.rows.get("Input"))),
                "output": prose.block(row_source(entry.rows.get("Output"))),
                "how": method_how(prose, row_source(entry.rows.get("How")), code),
                "code": prose.block(code) if _unguided(code) else "",
                "settings": read_settings(prose, settings_row)
                if settings_row
                else None,
            }
        )
    return {"n": len(found), "methods": found}


def method_how(prose: Prose, how: str, code: str) -> Markup:
    """A How as numbered steps, each carrying the symbol the Code row names for it."""
    html_ = str(prose.block(how)).replace("<ol>", '<ol class="steps2">', 1)
    per_step = {
        int(n): symbol
        for line in code.splitlines()
        for n, symbol in STEP_CODE.findall(line)
    }
    per_step |= {int(n): symbol for symbol, n in STEP_CODE_INLINE.findall(code)}
    head, separator, rest = html_.partition('<ol class="steps2">')
    if not per_step or not separator:
        return Markup(html_)
    count = iter(range(1, 1000))

    def tag(match: re.Match) -> str:
        symbol = per_step.get(next(count), "")
        return (
            f'<span class="stepcode">{html.escape(symbol.split("::")[-1])}</span></li>'
            if symbol
            else match.group(0)
        )

    return Markup(head + separator + re.sub(r"</li>", tag, rest))


def read_settings(prose: Prose, row: Row) -> dict:
    """A Settings row as key, what it sets and value; the value is the gloss's last part with a digit."""
    text = _unguided(row_source(row))
    if not text or text.lower().startswith("none"):
        return {
            "rows": [],
            "text": prose.inline(text, capital=True) if text else "None.",
            "where": "",
        }
    found = re.findall(r"`([^`]+)`\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text)
    if not found:
        return {"rows": [], "text": prose.block(text), "where": ""}
    rows = []
    for key, gloss in found:
        what, _, tail = gloss.rpartition(", ")
        what, value = (what, tail) if what and re.search(r"\d", tail) else (gloss, "")
        rows.append(
            {
                "key": key,
                "what": prose.inline(what),
                "value": prose.inline(value) if value else "",
            }
        )
    places = list(dict.fromkeys(re.findall(r"\bin\s+(`[^`]+`)", text)))
    return {
        "rows": rows,
        "text": "",
        "where": prose.inline(", ".join(places)) if places else "",
    }


def read_limitations(model: Model, prose: Prose) -> dict:
    row = section_row(model, "Verdict", "Limitations")
    found = items(row) or ([_unguided(row.text)] if row and _unguided(row.text) else [])
    return {
        "n": len(found),
        "items": [prose.inline(text, capital=True) for text in found],
    }


# ===============================
# Datasets, side documents and references
# ===============================


def read_datasets(model: Model, prose: Prose, exhibits: Exhibits) -> dict:
    """One fold per Datasets row: its row with size and chips, opening to stats, fields and tables."""
    section = model.section("Datasets")
    rows = []
    for n, (name, row) in enumerate((section.rows if section else {}).items(), 1):
        source = _unguided(row_source(row))
        if not source:
            continue
        meta = DATASET_META.match(source)
        kind, labels, text = "", "", source
        if meta:
            kind, _, labels = (
                part.strip() for part in meta.group("meta").partition(";")
            )
            text = meta.group("text")
        one = re.match(r"one item is (.+?[.;])(?=\s|$)", text.strip(), re.S | re.I)
        links = markup.ASSET_LINK.findall(text)
        fid = f"{prose.page_id}/dataset-{n}"

        # the standard tables the row links: summary, fields, statistics
        stem = _dataset_stem(links)
        summary = (
            markup.read_table(artifact_path(model, "dataset-summary", stem))
            if stem
            else []
        )
        fields = (
            markup.read_table(artifact_path(model, "dataset-fields", stem))
            if stem
            else []
        )
        drawn = (
            {f"tables/dataset-{stem}-{part}.csv" for part in ("summary", "fields")}
            if stem
            else set()
        )
        statistics = (
            markup.read_table(artifact_path(model, "dataset-statistics", stem))
            if stem and not summary
            else []
        )

        rest = re.sub(
            rf"(?:\s*{markup.ASSET_LINK.pattern})+\s*$",
            "",
            text[one.end() :] if one else text,
            flags=re.I,
        ).strip()
        rows.append(
            {
                "id": fid,
                "name": prose.inline(name),
                "kind": kind,
                "labels": ""
                if labels.removesuffix(" labels") in ("", "none")
                else labels.removesuffix(" labels"),
                "one_item": prose.inline(one.group(1).rstrip(".;")) if one else "",
                "size": _dataset_size(summary)
                or _statistics_size(statistics, one.group(1) if one else ""),
                "chips": [
                    INLINE.tbl(src, lead or src, f"#{fid}") for lead, src in links
                ],
                "prose": prose.block(rest),
                "stats": [
                    {
                        "value": _abbreviate(s.get("value", "")),
                        "statistic": s.get("statistic", ""),
                        "note": s.get("note", ""),
                    }
                    for s in summary
                ],
                "fields": [
                    {
                        "field": f.get("field", ""),
                        "type": f.get("type", ""),
                        "distinct": f.get("distinct", ""),
                        "values": [v for v in f.get("values", "").split("; ") if v],
                    }
                    for f in fields
                ],
                "exhibits": [
                    read_exhibit(prose, exhibits, src, lead, "")
                    for lead, src in links
                    if src not in drawn
                ],
            }
        )
    return {"n": len(rows), "rows": rows}


def read_side_documents(model: Model, prose: Prose, findings: list[Finding]) -> dict:
    """Markdown files beside the README, and the configs and prompts the README names."""
    folder = model.document.folder
    sections = [model.section(s.heading) for s in model.spec.sections]
    text = "\n".join(s.text for s in sections if s)
    files = [p for p in sorted(folder.glob("*.md")) if p.name != "README.md"]
    for name in SIDE_FOLDERS:
        if (folder / name).is_dir():
            files += [
                p
                for p in sorted((folder / name).rglob("*"))
                if p.is_file() and p.name in text
            ]
    cited = [(f.id, "\n".join(r.text for r in f.entry.rows.values())) for f in findings]
    for heading in ("Decisions", "Methods"):
        for entry in entries(model, heading):
            if match := ENTRY_TITLE.match(entry.title):
                cited.append(
                    (match.group(1), "\n".join(r.text for r in entry.rows.values()))
                )
    documents = []
    for path in files:
        body = path.read_text(encoding="utf-8", errors="replace")
        lines = body.splitlines()
        opening = [
            line[2:].strip()
            for line in lines[
                : next(
                    (i for i, l in enumerate(lines) if not l.startswith("# ")),
                    len(lines),
                )
            ]
        ]
        if path.suffix != ".md" and opening:
            what = markup.first_sentence(" ".join(opening))
        else:
            what = next(
                (l.lstrip("# ").strip() for l in lines if l.startswith("# ")), ""
            )
        what = what or (
            "config"
            if path.suffix in CONFIG_SUFFIXES
            else "prompt"
            if "prompt" in str(path)
            else path.suffix.lstrip(".")
        )
        documents.append(
            {
                "path": path.relative_to(folder).as_posix(),
                "what": what,
                "lines": len(lines),
                "chips": Markup("").join(
                    prose.chip(code)
                    for code, text in cited
                    if path.name in text and code in prose.tips
                ),
            }
        )
    return {"n": len(documents), "documents": documents}


def read_references(model: Model) -> dict:
    """Cited references.bib entries with the row they were cited for; the uncited counted."""
    bib = read_bibliography(model)
    if not bib:
        return {"n": 0, "cited": [], "uncited": 0}
    where: dict[str, str] = {}
    for heading in ("Hypothesis", "Design", "Verdict"):
        section = model.section(heading)
        for label, row in (section.rows if section else {}).items():
            for key in BIB_CITATION.findall(row_source(row)):
                where.setdefault(key, f"the {heading} {label}")
    for heading in ("Decisions", "Methods", "Findings"):
        for entry in entries(model, heading):
            for key in BIB_CITATION.findall(
                "\n".join(row_source(r) for r in entry.rows.values())
            ):
                where.setdefault(key, entry.title.split(" ")[0])
    cited = [
        {
            "id": f"{record_slug(model)}/ref-{key}",
            "label": bib[key]["label"],
            "title": bib[key]["title"],
            "venue": bib[key]["venue"],
            "where": where[key],
        }
        for key in bib
        if key in where
    ]
    return {"n": len(cited), "cited": cited, "uncited": len(bib) - len(cited)}


# ===============================
# Folds, builds on this and the section bar
# ===============================


def read_setup(
    model: Model, site: Site, prose: Prose, clauses: list[Clause]
) -> dict | None:
    """The Question and setup fold: Builds on, Statement, Rationale, Predictions, rule, Design, Rounds."""
    rows = []
    found_parents = parents(model, site)
    builds_on = _unguided(model.row("Hypothesis", "Builds on"))
    if found_parents or builds_on:
        pills = Markup(" ").join(
            record_pill(p) for p in found_parents
        ) or INLINE.missing("builds on nothing")
        # the pills name each parent, so its bullet's slug lead goes; one paragraph
        row = section_row(model, "Hypothesis", "Builds on")
        said = [SLUG_LEAD.sub("", i) for i in items(row)] or [builds_on]
        rows.append(
            (
                "Builds on",
                Markup("<p>{}</p>{}").format(
                    pills,
                    prose.block(" ".join(t for t in said if t)) if builds_on else "",
                ),
            )
        )
    for label in ("Statement", "Rationale"):
        if _unguided(model.row("Hypothesis", label)):
            rows.append(
                (
                    label,
                    prose.block(row_source(section_row(model, "Hypothesis", label))),
                )
            )

    # the Predictions framing; the clauses sit in the answer card and the trace
    predictions = section_row(model, "Hypothesis", "Predictions")
    if predictions and _unguided(predictions.body[0]):
        link = (
            Markup(
                ' The clauses are under <a href="#{}/answer">the answer</a>.'
            ).format(prose.page_id)
            if clauses
            else ""
        )
        rows.append(
            (
                "Predictions",
                Markup("<p>{}{}</p>").format(
                    prose.inline(predictions.body[0], capital=True), link
                ),
            )
        )
    for text in items(predictions):
        label = NESTED_LABEL.match(text)
        if label and not checks.CLAUSE_LINE.match(text):
            name = (
                "Read beside"
                if label.group(1).startswith("Read beside")
                else label.group(1)
            )
            rows.append((name, prose.block(label.group(2))))

    # the Design rows past the needs, Inputs only without Datasets
    has_datasets = bool(
        model.section("Datasets") and _unguided(model.section("Datasets").text)
    )
    for label in ("Inputs", "Method / factors", "Metrics", "Protocol"):
        if label == "Inputs" and has_datasets:
            continue
        if _unguided(model.row("Design", label)):
            rows.append(
                (label, prose.block(row_source(section_row(model, "Design", label))))
            )

    # the rounds, one dated line each
    rounds = []
    for entry in entries(model, "Rounds"):
        match = ENTRY_TITLE.match(entry.title)
        if match is None:
            continue
        opened = _unguided(entry.rows["Opened"].text) if "Opened" in entry.rows else ""
        closed = _unguided(entry.rows["Closed"].text) if "Closed" in entry.rows else ""
        rounds.append(
            {
                "id": match.group(1),
                "anchor": f"{prose.page_id}/{match.group(1).lower()}",
                "title": prose.inline(match.group(2)),
                "opened": opened,
                "closed": prose.inline(closed),
            }
        )
    if not rows and not rounds:
        return None
    digest = ", ".join(
        dict.fromkeys(
            label.lower() for label, _ in rows + ([("rounds", "")] if rounds else [])
        )
    )
    return {
        "id": f"{prose.page_id}/question",
        "meta": _count(len(clauses), "clause") if clauses else "",
        "digest": digest,
        "rows": rows,
        "rounds": rounds,
    }


def read_adoption(model: Model, prose: Prose) -> dict | None:
    """The Adoption candidate fold: the candidate, what it carries and what stays out."""
    row = section_row(model, "Verdict", "Adoption candidate")
    text = _unguided(row.body[0]) if row else ""
    if not text or text.lower().startswith("none"):
        return None
    rows = [("Candidate", prose.block(text))]
    for item in items(row):
        if label := NESTED_LABEL.match(item):
            rows.append((label.group(1), prose.block(label.group(2))))
    return {
        "id": f"{prose.page_id}/adoption",
        "digest": markup.plain_text(markup.inline_html(markup.first_sentence(text))),
        "rows": rows,
    }


def read_reproduce(model: Model, prose: Prose, findings: list[Finding]) -> dict | None:
    """The Reproduce fold: prerequisites by kind, the bash block as steps, other prose as written."""
    section = model.section("Reproduce")
    text = _unguided(section.text) if section else ""
    if not text:
        return None

    # prerequisites, by kind
    prerequisites: dict[str, list[Markup]] = {}
    for item in items(section.rows.get("Prerequisites")):
        match = re.match(r"\*\*(.+?)\*\*\s*(.*)$", item, re.S)
        kind, rest = (
            (match.group(1), match.group(2))
            if match and match.group(1) in PREREQUISITE_KINDS
            else ("Other", item)
        )
        prerequisites.setdefault(kind, []).append(prose.inline(rest))
    ordered = [
        (k, prerequisites[k])
        for k in (*PREREQUISITE_KINDS, "Other")
        if k in prerequisites
    ]

    # steps and loose prose
    ids = {f.id for f in findings}
    steps, loose = [], []
    parts = BASH_BLOCK.split(text)
    for i, part in enumerate(parts):
        if i % 2:
            steps += read_steps(prose, part, ids)
        else:
            rest = re.sub(
                r"(?ms)^- \*\*Prerequisites\*\*:.*?(?=^\S|\Z)", "", part
            ).strip()
            if rest:
                loose.append(prose.block(rest, capital=False))
    # the digest names the prerequisites and the scripts the steps run; the count sits on the right
    numbered = [s for s in steps if not s["env"]]
    manual = sum(s["by_hand"] for s in numbered)
    scripts = list(
        dict.fromkeys(
            Path(m).name
            for s in numbered
            for m in SCRIPT.findall(markup.plain_text(str(s["code"])))
        )
    )
    if len(scripts) > DIGEST_STEPS:
        more = _count(len(scripts) - DIGEST_STEPS, "more script")
        named = ", ".join(scripts[:DIGEST_STEPS]) + f" and {more}"
    elif len(scripts) > 1:
        named = ", ".join(scripts[:-1]) + f" and {scripts[-1]}"
    else:
        named = "".join(scripts)
    digest = [", ".join(k.lower() for k, _ in ordered), named]
    digest += [f"{manual} by hand"] if manual else []
    return {
        "id": f"{prose.page_id}/reproduce",
        "meta": _count(len(numbered), "step"),
        "digest": " · ".join(d for d in digest if d),
        "prerequisites": ordered,
        "steps": steps,
        "numbered": numbered,
        "loose": loose,
    }


def read_steps(prose: Prose, code: str, ids: set[str]) -> list[dict]:
    """A bash block as steps: lines before the first comment are the environment; each comment titles a step."""
    groups, title, commands = [], [], []
    for line in code.splitlines():
        if line.startswith("#"):
            if commands:
                groups.append((title, commands))
                title, commands = [], []
            title.append(line.lstrip("#").strip())
        elif line.strip():
            commands.append(line.rstrip())
    if title or commands:
        groups.append((title, commands))

    steps, n = [], 0
    for title, commands in groups:
        body = Markup("\n").join(
            Markup('{}<span class="c">{}</span>').format(
                c[: m.start() + 1], c[m.start() + 1 :]
            )
            if (m := re.search(r"\s#.*$", c))
            else c
            for c in commands
        )
        if not title:
            steps.append(
                {
                    "env": True,
                    "n": "env",
                    "title": "Environment",
                    "commits": [],
                    "by_hand": False,
                    "code": body,
                }
            )
            continue
        n += 1
        text = " ".join(title)
        by_hand = bool(re.match(r"by hand:\s*", text, re.I))
        text = re.sub(r"^by hand:\s*", "", text, flags=re.I)
        text = re.sub(r"^(?:step\s*)?\d+[.:)]?\s+", "", text, flags=re.I)
        commits = [s.strip() for s in checks.STEP_COMMIT.findall(text)]
        text = (
            re.sub(r"\s+([;,.])", r"\1", checks.STEP_COMMIT.sub("", text))
            .strip(" ;,")
            .replace(" -- ", " — ")
        )
        head = Markup(html.escape(text))
        head = Markup(
            re.sub(
                r"\bF(\d+)\b",
                lambda m: (
                    str(prose.chip(m.group(0))) if m.group(0) in ids else m.group(0)
                ),
                str(head),
            )
        )
        steps.append(
            {
                "env": False,
                "n": n,
                "title": head,
                "commits": [c[:SHA_SHOWN] for c in commits],
                "by_hand": by_hand,
                "code": body,
            }
        )
    return steps


def read_builds(model: Model, site: Site) -> dict:
    """The records that build on this one, each with what it found."""
    found = []
    for child in children(model, site):
        line = TLDR_STATUS.sub("", _unguided(child.row("TL;DR", "Hypothesis")))
        if not record_outcome(child) or not line:
            line = _unguided(child.row("Hypothesis", "Statement")) or line
        found.append(
            {
                "pill": record_pill(child),
                "text": markup.plain_text(
                    markup.inline_html(markup.capitalize(BARE_CITATION.sub("", line)))
                ),
            }
        )
    return {"n": len(found), "records": found}


def section_bar(
    page_id: str, sections: dict, folds: dict, skeleton: bool
) -> list[dict]:
    """The section bar: every section below it with its count, then the folds."""
    names = {
        "findings": "Findings",
        "discussion": "Discussion",
        "decisions": "Decisions",
        "methods": "Methods",
        "limits": "Limitations",
        "datasets": "Datasets",
        "docs": "Side documents",
        "references": "References",
    }
    shown_empty = (
        {"findings", "decisions", "methods", "limits", "datasets", "discussion"}
        if skeleton
        else set()
    )
    bar = [
        {"href": f"#{page_id}/{key}", "name": name, "n": sections[key]["n"]}
        for key, name in names.items()
        if sections[key]["n"] or key in shown_empty
    ]
    for key, name in (
        ("question", "Question and setup"),
        ("adoption", "Adoption"),
        ("reproduce", "Reproduce"),
    ):
        if folds[key] or (key == "reproduce" and skeleton):
            n = len(folds[key]["numbered"]) if key == "reproduce" else 0
            bar.append({"href": f"#{page_id}/{key}", "name": name, "n": n})
    return bar


# ===============================
# Helper functions
# ===============================


def _unguided(text: str) -> str:
    """A text without the template's `<…>` guidance; "" when nothing else is left."""
    if not text:
        return ""
    stripped = GUIDANCE.sub(
        lambda m: "" if is_template_guidance(m.group(1)) else m.group(0), text
    )
    return stripped.strip() if re.search(r"\w", stripped) else ""


def _exhibit(**values) -> dict:
    """An exhibit's values, every key present so the template can read each one."""
    return {
        "anchor": "",
        "label": "",
        "caption": "",
        "html": "",
        "src": "",
        "kind": "",
        "missing": "",
    } | values


def _unique(models: list[Model]) -> list[Model]:
    """The models in order, each once; a model is unhashable, so it is told by identity."""
    found: list[Model] = []
    for model in models:
        if not any(model is seen for seen in found):
            found.append(model)
    return found


def _lead_word(text: str) -> str:
    words = re.sub(r"<.*?>", "", text or "").strip().split()
    return words[0].lower().strip(".,;:—") if words else ""


def _lead(block: str) -> tuple[str, str]:
    match = LEAD.match(block.strip())
    return (match.group(1), match.group(2)) if match else ("", block)


def _count(n: int, noun: str, spelled: bool = False) -> str:
    word = COUNT_WORDS[n] if spelled and n < len(COUNT_WORDS) else str(n)
    return f"{word} {noun}{'' if n == 1 else 's'}"


def _value(value: str) -> str:
    """A reading's value as the table holds it, a large whole number with separators."""
    text = str(value).strip()
    return f"{int(text):,}" if re.fullmatch(r"-?\d{5,}", text) else text


def _compact(value: str) -> str:
    """A reading's value in a line of text: `22.5 B`, `22.1 M`, `5,941`, `0.8131`."""
    number = markup.to_float(value)
    if number is None:
        return str(value).strip()
    if abs(number) >= BILLION:
        return f"{number / BILLION:.1f}\u00a0B"
    if abs(number) >= TEN_MILLION:
        return f"{number / MILLION:.1f}\u00a0M"
    if number == int(number) and "." not in str(value):
        return f"{int(number):,}"
    return markup.number(str(value).strip())


def _bold_leads(text: str) -> str:
    """List items as `**lead** — rest`, the lead being the words before the first `: ` or ` — `."""

    def lead(match: re.Match) -> str:
        indent, body = match.groups()
        if body.startswith("**"):
            return match.group(0)
        parts = LEAD_CUT.split(body, maxsplit=1)
        if len(parts) < 3 or len(parts[0].split()) > LEAD_WORDS:
            return match.group(0)
        head, _, rest = parts
        return f"{indent}**{head.strip()}** — {rest.strip()}"

    return re.sub(r"(?m)^(\s*[-*]\s+)(.+)$", lead, text)


def _last_change(text: str) -> str:
    """A decision's last History line, to its first `;` or bracket, as its fold's subline."""
    match = re.match(r"(\d{4}-\d\d-\d\d)\s*(.*)$", text, re.S)
    if not match:
        return ""
    said = markup.plain_text(
        markup.inline_html(BARE_CITATION.sub("", TIMING.sub("", match.group(2))))
    )
    said = re.split(r"[;(]|\.\s", said, maxsplit=1)[0].strip(" ;,.")
    return f"the last change, on {match.group(1)[5:]}, {said}" if said else ""


def _trace_free(condition: str) -> str:
    """A condition without its trailing `([D<n>]; measured by [M<n>])`, which the trace shows."""
    return checks.CLAUSE_TRACE.sub("", condition).strip().rstrip(".").strip()


def _ticket(ticket) -> Markup | str:
    text = str(ticket or "").strip()
    if not text:
        return INLINE.missing("no ticket")
    if re.match(r"https?://", text):
        return INLINE.link("issue", text)
    number = text.lstrip("#")
    return f"issue #{number}" if number.isdigit() else text


def _pr(pr: str) -> Markup | str:
    text = pr.strip()
    if re.match(r"https?://", text):
        match = re.search(r"/(?:pull|merge_requests)/(\d+)", text)
        return INLINE.link(f"PR #{match.group(1)}" if match else "PR", text)
    number = text.lstrip("#")
    return f"PR #{number}" if number.isdigit() else text


def _prose_runs(html_: str, function) -> str:
    """Applies `function` to the text runs outside code, links, headings, svg and math."""
    skip_tags = ("code", "a", "pre", "h1", "h2", "h3", "svg", "math", "style", "script")
    out, skip = [], []
    for token in re.split(r"(<[^>]+>)", html_):
        if token.startswith("<"):
            match = re.match(r"</?([a-zA-Z0-9]+)", token)
            name = match.group(1).lower() if match else ""
            if token.startswith("</"):
                if skip and skip[-1] == name:
                    skip.pop()
            elif (
                not token.endswith("/>")
                and (skip or name in skip_tags)
                and name not in ("br", "img", "hr", "wbr")
            ):
                skip.append(name)
            out.append(token)
        else:
            out.append(token if skip else function(token))
    return "".join(out)


def _strip(
    points: list[dict],
    lo: float,
    hi: float,
    ticks: list[float],
    bar: float | None = None,
) -> dict:
    """An interval strip's rows and axis as positions along the track."""

    def at(value: float) -> str:
        return f"{(value - lo) / (hi - lo) * 100:.2f}%"

    rows = []
    for p in points:
        margin = p.get("margin")
        rows.append(
            {
                "chip": p["chip"],
                "qualifier": p["qualifier"],
                "kind": p["kind"],
                "low": at(p["low"]),
                "width": f"calc({at(p['high'])} - {at(p['low'])})",
                "point": at(p["value"]),
                "text": p["text"],
                "band": (at(-margin), f"calc({at(margin)} - {at(-margin)})", at(0))
                if margin is not None
                else None,
                "bar": at(bar) if bar is not None else None,
            }
        )
    return {
        "rows": rows,
        "ticks": [(at(t), _tick(t)) for t in ticks],
        "legend": "dashed line: the bar"
        if bar is not None
        else "grey band: the margin around zero",
        "wide": any(p["qualifier"] for p in points),
    }


def _tick(value: float) -> str:
    """A strip's axis label: `5 B`, `250 M`, `750,000`, `0.05`."""
    if abs(value) >= BILLION:
        return f"{value / BILLION:g}\u00a0B"
    if abs(value) >= TEN_MILLION:
        return f"{value / MILLION:g}\u00a0M"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:g}"


def _drawn(reading: dict) -> bool:
    return markup.to_float(reading.get("value", "")) is not None and (
        markup.to_float(reading.get("bar", "")) is not None
        or markup.to_float(reading.get("margin", "")) is not None
    )


def _interval_columns(keys: list[str]) -> dict[str, tuple[str, str]]:
    """Value column → its interval columns: `<q>_low/_high`, `<q>_ci_low/_ci_high`, or `ci_low/ci_high` after it."""
    bounds = {}
    for i, key in enumerate(keys):
        for low, high in INTERVAL_SUFFIXES:
            if f"{key}{low}" in keys and f"{key}{high}" in keys:
                bounds[key] = (f"{key}{low}", f"{key}{high}")
                break
        else:
            following = keys[i + 1 : i + 3]
            if following == ["ci_low", "ci_high"]:
                bounds[key] = ("ci_low", "ci_high")
    return bounds


def _context_words(words: list[str], last: bool) -> list[str]:
    """A phrase cut to CONTEXT_WORDS words, its first or last, without loose function words."""
    while words and words[-1].lower() in FUNCTION_WORDS | LINK_VERBS:
        words.pop()
    words = words[-CONTEXT_WORDS:] if last else words[:CONTEXT_WORDS]
    while words and words[-1].lower() in FUNCTION_WORDS | LINK_VERBS:
        words.pop()
    while words and words[0].lower() in FUNCTION_WORDS:
        words.pop(0)
    return words


def _dataset_stem(links: list[tuple[str, str]]) -> str:
    for _, src in links:
        match = re.match(
            r"tables/dataset-(.+)-(?:summary|fields|statistics)\.csv$", src
        )
        if match:
            return match.group(1)
    return ""


def _dataset_size(summary: list[dict]) -> str:
    """The head size: the first two summary tiles, as `40.9 M documents · 18 sources`."""
    return " · ".join(
        f"{_abbreviate(s.get('value', ''))} {s.get('statistic', '')}".strip()
        for s in summary[:2]
    )


def _statistics_size(rows: list[dict], one_item: str) -> str:
    """The head size from a statistics table: the item column the one-item text names, summed.

    A row cut `all`, `total` or `overall` gives the size itself; a table whose
    columns the one-item text does not name gives none.
    """
    if not rows:
        return ""
    named = [
        key
        for key in rows[0]
        if re.search(rf"\b{re.escape(key)}\b", one_item, re.I)
        and all(WHOLE_NUMBER.match(r.get(key, "") or "0") for r in rows)
    ]
    if not named:
        return ""
    key = named[0]
    first = next(iter(rows[0]))
    totals = [r for r in rows if r.get(first, "").lower() in TOTAL_ROWS]
    count = (
        int(totals[0][key].replace(",", ""))
        if totals
        else sum(int((r.get(key) or "0").replace(",", "")) for r in rows)
    )
    return f"{count:,} {key}"


def _abbreviate(value: str) -> str:
    number = markup.to_float(value)
    if number is None:
        return value
    if abs(number) >= 1e9:
        return f"{number / 1e9:.1f} B"
    if abs(number) >= 1e6:
        return f"{number / 1e6:.1f} M"
    if number == int(number):
        return f"{int(number):,}"
    return markup.number(value)
