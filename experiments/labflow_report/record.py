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

import fnmatch
import html
import itertools
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml
from markupsafe import Markup

from . import markup
from .builder import ENVIRONMENT, OUTCOME, Model, PageView, Site, entry_ref, view
from .checks import record as checks
from .checks.figures import figure_rows
from .document import Entry, Row
from .rules import GUIDANCE

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
# the stage, and the word, of an exploration ended without a confirmation
DROPPED = "dropped"
# a question's reading words; any other word answers it in part
QUESTION_WORDS = {
    "answered": "answered",
    "confirmed": "answered",
    "unanswered": "unanswered",
    "refuted": "unanswered",
}
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
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
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
POINTER_TEXT = re.compile(
    r"^(?:the|its|this|that|a)\b(?:(?:\s+\S+){0,2}|.*\b(?:table|figure|chart|plot|heatmap))$"
    # a bare noun ("table", "seed table") or one with a short tail ("the figure by outer pair")
    r"|^(?:(?:the|its|this|that|a)\s+)?(?:drawn\s+as\s+a\s+)?(?:\S+\s+){0,2}"
    r"(?:table|figure|chart|plot|heatmap|crosstab)\b(?:\s+\S+){0,3}$",
    re.I,
)
# `(M2)` or `(D3, M4)` written without brackets, which the row's chips already show
BARE_CODES = re.compile(r"\s*\((?:[DFM]\d+(?:,\s*)?)+\)")
ROUND_CITATION = re.compile(r"\[(R\d+)\]")
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
TIMING = re.compile(
    r"\((before runs|after runs|after \[F\d+\](?:,\s*\[F\d+\])*)(;\s*user)?\)"
)
# when a History line was set, from its timing tag, in the order a reader weighs it
WHEN = {"before runs": "before", "after runs": "runs"}
WHEN_ORDER = ("before", "runs", "results")
REPLACES = re.compile(r"replaces runs ((?:`[0-9a-f]{6,}`(?:,\s*)?)+)")
RULE_NOTE = re.compile(r"Note added (\d{4}-\d\d-\d\d):")
# a nested `- <date> …:` bullet under a Predictions line, joined into its text
DATED_NOTE = re.compile(r"(?:^|\s)- (\d{4}-\d\d-\d\d)\b[^:]{0,80}:\s*")
LEAD = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$", re.S)
NESTED_LABEL = re.compile(r"^\*\*(.+?)\*\*:?\s*(.*)$", re.S)
PLANNED_AS = re.compile(r"\s*—\s*planned as ([a-z0-9][a-z0-9-]*)(?: #(\d+))?\s*[.;]?$")
ISSUE = re.compile(r"\s*\(issue #(\d+)\)\s*[.;]?$")
# a bare `#12` in prose, never an entity such as `&#39;` or part of a word or path
ISSUE_NUMBER = re.compile(r"(?<![\w&/#])#(\d+)\b")
NOT_PLANNED = re.compile(r"\s*—\s*not yet planned(?::.*)?\s*[.;]?$")
DONE_IN = re.compile(r"\s*—\s*done in `?([a-z0-9][a-z0-9-]*)`?\s*[.;]?$")
DISCLOSURE = re.compile(r"^(earlier runs|pilot|numbers seen):\s*(.*)$", re.S | re.I)
DATASET_META = re.compile(r"^\((?P<meta>[^)]*)\):?\s*(?P<text>.*)$", re.S)
CODE_SPAN = re.compile(r"<code>([^<]+)</code>")
FILE_BREAK = re.compile(r"(/|::|_)")
CODE_PATH_BREAK = re.compile(r"(/|::)")
METHOD_FILES_SHOWN = 3  # a closed method names this many source files, then +n
PATH_LIKE = re.compile(r"/|\.(ya?ml|py|csv|json|svg|png|md|txt|toml)(::\w+)?$")
BASH_BLOCK = re.compile(r"(?ms)^```(?:bash|sh|zsh)?\n(.*?)^```[ \t]*$")
SLUG_LEAD = re.compile(r"^(?:`[^`]+`|[\w.]*[-/][\w./-]*)\s*[:—–]\s*")
WHOLE_NUMBER = re.compile(r"^\d[\d,]*$")
STATED_COUNT = re.compile(
    r"(?P<about>(?:about|some|roughly|over|under) )?(?P<count>\d[\d,.]*)"
    r"(?: (?P<scale>million|thousand|billion))?(?: [\w-]+){0,2}? (?P<noun>[a-z]+s)\b",
    re.I,
)
SCALE_UNITS = {"thousand": "k", "million": "M", "billion": "B"}
ITEM_WORDS = 5
TOTAL_ROWS = ("all", "total", "overall")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---", re.S)
# a figure whose width over height is under this takes half the row
NARROW_FIGURE = 1.8
DECISION_FIGURE = "Decision figure"

RELATION_BEFORE = re.compile(r"\b(qualifies|supports|overturns)\s+\(?$")

NEEDS = (
    ("People", "People"),
    ("Compute", "Compute"),
    ("Data access", "Data"),
    ("External dependency", "External"),
)
NO_NEED = re.compile(r"^(none|n/a|—|-)\b", re.I)
# a need worth a glance: a hosted model, a quota, a licence, people's hours; any external dependency
HOT_NEED = re.compile(
    r"hosted|claude|\bapi\b|hpc|gpu|cluster|licen|quota|billed|endpoint|annotat|hours|manual|rare|cannot",
    re.I,
)
PREREQUISITE_KINDS = ("Secret", "Login", "Data", "Service", "Frozen")
# each prerequisite kind as the Reproduce digest names it
PREREQUISITE_NAMES = {
    "Secret": "secrets",
    "Login": "sign-ins",
    "Data": "data",
    "Service": "services",
    "Frozen": "kept by hand",
    "Other": "other",
}
# each prerequisite kind as its group's heading
PREREQUISITE_HEADINGS = {
    "Secret": "Secrets",
    "Login": "Sign-ins",
    "Data": "Read in place",
    "Service": "Services",
    "Frozen": "Kept by hand",
    "Other": "Other",
}
# a prose prerequisite's kind, tried in order; anything else is data read in place
PREREQUISITE_SIGNS = (
    ("Secret", re.compile(r"`[A-Z][A-Z0-9]*_[A-Z0-9_]+`|\.env\b")),
    ("Login", re.compile(r"\b(?:signed|logged) in\b|\blog-?in\b|subscription", re.I)),
    (
        "Frozen",
        re.compile(
            r"not regenera|cannot be regenera|by hand|out of band|licensed|notebook writes",
            re.I,
        ),
    ),
    ("Service", re.compile(r"\bserver\b|\bservice\b|\bvLLM\b|reachable|\bAPI\b", re.I)),
)
NEEDS_LEAD = re.compile(
    r"^(?:(?:it|a rerun|each run|this) (?:also )?)?(?:needs|requires)\s+", re.I
)
PROSE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`])")
# a step a person does: tagged `by hand:`, or a person who validates, labels or checks
BY_HAND = re.compile(
    r"\bby hand\b|\bmanual(?:ly)?\b"
    r"|\b(?:human|person|annotator|author)s?\b[^.;]*?\b(?:validat|label|check|annotat|review)",
    re.I,
)
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
# the arms table's gain column is narrow, so its axis takes fewer steps
GAIN_STEPS = 2
# a reading this large reads in billions or millions
BILLION = 1e9
MILLION = 1e6
THOUSAND = 1e3
# which way an arms column's best value lies, by its header
HIGHER_BETTER = re.compile(r"\bper\b|/s\b|throughput", re.I)
LOWER_BETTER = re.compile(
    r"\((?:s|ms|min)\)|\b(?:latency|time|seconds?|loss|error|cost|p9\d)\b", re.I
)
NO_BEST = re.compile(r"\b(?:gap|difference|delta|seed|runs?)\b", re.I)
# a bar this large is a count target, so Found reads a reading as its multiple
COUNT_BAR = 10
# runs a finding's runs line names before counting the rest
RUNS_SHOWN = 3
RUN_ID_SHOWN = 8
SHA_SHOWN = 7
INTERVAL_SUFFIXES = (("_ci_low", "_ci_high"), ("_low", "_high"))
INTERVAL_END = re.compile(r"^(.*?)[_ ]?(low|high)$", re.I)
YES = ("yes", "true")
SIDE_FOLDERS = ("configs", "prompts")
CONFIG_SUFFIXES = (".yaml", ".yml", ".toml", ".json")

EMPTY = {
    "findings": "No findings yet. The card above gains its verdict line with the first key finding.",
    "discussion": "Nothing to discuss yet. The discussion is written with the verdict.",
    "decisions": "No decision recorded yet.",
    "methods": "No method yet. Each step is written as its code lands.",
    "limits": "None written yet; they are written with the verdict.",
    "datasets": "No dataset described yet. Each input gets a row before its final files are fixed.",
}
SKELETON_STAGES = ("draft", "running", "reporting")
# a tracking server on one of these hosts opens only on the author's machine
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
FIGURE_SUFFIXES = (".svg", *markup.IMAGE_TYPES)

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
        return markup.unguided(row.text) if row else ""

    def source(self, label: str) -> str:
        row = self.entry.rows.get(label)
        return markup.unguided(row_source(row)) if row else ""


@dataclass
class Exhibits:
    """The page's figure and table numbers, the findings drawing each file and the anchors already placed."""

    folder: Path
    page_id: str
    counts: dict[str, int] = field(default_factory=lambda: {"Figure": 0, "Table": 0})
    anchors: set[str] = field(default_factory=set)
    placements: dict[str, int] = field(default_factory=dict)
    numbers: dict[str, str] = field(default_factory=dict)
    cited_by: dict[str, list[str]] = field(default_factory=dict)

    def anchor(self, src: str) -> str:
        return f"{self.page_id}/asset/{re.sub(r'[^A-Za-z0-9_-]', '-', Path(src).stem)}"

    def label(self, key: str, kind: str) -> str:
        """The exhibit's `Table <n>` or `Figure <n>`: its number once given, else the next one."""
        if key not in self.numbers:
            self.counts[kind] += 1
            self.numbers[key] = f"{kind} {self.counts[kind]}"
        return self.numbers[key]


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

    # the Decision figure is Figure 1, so it takes its number before any finding's
    decision = read_decision_figure(model, prose, exhibits) if outcome else None

    head = read_head(model, site, page_id)
    answer = read_answer(
        model, site, prose, clauses, findings, readings, words, outcome, decision
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
            model, prose, exhibits, clauses, findings, readings, words
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
    """The word a record's mark carries: dropped for a dropped exploration, its
    outcome, else running or planned."""
    if model.stage == DROPPED:
        return DROPPED
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
    return [markup.unguided(text) for _, text in checks.items(row) if markup.unguided(text)]


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
    """The findings whose Reading or Summary names the clause: key ones, and
    supporting ones too for a diagnostic clause (REC-150)."""
    names = re.compile(CLAUSE_NAME.format(id=clause.id))
    weights = ("key", "supporting") if clause.role == "diagnostic" else ("key",)
    return [
        f
        for f in findings
        if f.weight in weights
        and names.search(f.text("Reading") + " " + f.text("Summary"))
    ]


def clause_word(clause: Clause, findings: list[Finding], readings: dict) -> str:
    """A clause's outcome: its readings' one word, else what its deciding finding says, else planned.

    A question's readings never split: they answer it, leave it open, or
    answer it in part.
    """
    words = [
        x["outcome"] for x in readings.get(clause.id, []) if x.get("outcome") in OUTCOME
    ]
    if clause.id.startswith("Q") and words:
        answers = {QUESTION_WORDS.get(w, "partial") for w in words}
        return answers.pop() if len(answers) == 1 else "partial"
    if len(set(words)) == 1:
        return words[0]
    for finding in deciding_findings(clause, findings):
        if word := checks.outcome(finding.entry, clause.check_clause):
            return word
    return "planned"


def clause_chip(clause: Clause, readings: dict, word: str, page_id: str) -> Markup:
    """A clause's chip, a cut circle when its readings differ by qualifier; a question's never cuts."""
    href = f"#{page_id}/{clause.id.lower()}"
    split = (
        [] if clause.id.startswith("Q") else reading_split(readings.get(clause.id, []))
    )
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
        else markup.unguided(evidence.text if evidence else "")
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
    reference, a bare `#<n>` its GitHub issue when the repo is known, and a
    link to one of the record's tables or figures its rendering on the page.
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
        self.github = site.github
        address = (model.value("mlflow") or "").rstrip("/")
        # a server only the author's machine reaches gets no links, its run ids stay text
        self.mlflow_local = bool(address) and is_local(address)
        self.mlflow = "" if self.mlflow_local else address
        self.concluded = model.stage == "concluded"
        self.slugs = {record_slug(m) for m in site.of_kind("record")}
        self.bib = read_bibliography(model)
        self.tips = self._tips(model, clauses, findings)
        self.rounds = {
            m.group(1)
            for e in entries(model, "Rounds")
            if (m := ENTRY_TITLE.match(e.title))
        }

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

    def record_decision(self, slug: str, code: str) -> Markup:
        """A chip for `[<slug> D<n>]`, a decision of another record; text when no record has the slug."""
        if slug not in self.slugs:
            return Markup("{} {}").format(slug, code)
        return INLINE.ct("d", f"{slug} {code}", f"#{slug}/{code.lower()}")

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
                    entry.title, markup.unguided(row.text) if row else ""
                )
        for entry in entries(model, "Methods"):
            if match := ENTRY_TITLE.match(entry.title):
                row = entry.rows.get("Output")
                tips[match.group(1)] = tip(
                    entry.title, markup.unguided(row.text) if row else "", "Output"
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

        def round_(match: re.Match) -> str:
            code = match.group(1)
            if code not in self.rounds:
                return match.group(0)
            return str(INLINE.ct("i", code, f"#{self.page_id}/{code.lower()}"))

        def issue(match: re.Match) -> str:
            number = match.group(1)
            return str(
                INLINE.link(f"#{number}", github_url(self.github, "issues", number))
            )

        def text(run: str) -> str:
            if self.github:
                run = ISSUE_NUMBER.sub(issue, run)
            run = ROUND_CITATION.sub(round_, run)
            run = checks.RECORD_CITATION.sub(other_record, run)
            run = checks.RECORD_DECISION.sub(
                lambda m: str(self.record_decision(*m.groups())), run
            )
            run = CITED_ONLY.sub(bare, run)
            run = CITATION.sub(citation, run)
            return BIB_CITATION.sub(reference, run)

        # a markdown link to an entry's GitHub heading anchor is that entry's chip
        html_ = re.sub(
            r'<a href="#[dfm]\d+[^"]*">([DFM]\d+)</a>',
            lambda m: (
                str(self.chip(m.group(1))) if m.group(1) in self.tips else m.group(1)
            ),
            html_,
        )
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
                    f'<code class="run full" title="{value}">{value}</code>'
                    if self.mlflow_local
                    else f'<code class="run" title="{value}">{value[:RUN_ID_SHOWN]}</code>'
                )
                return (
                    f'<a href="{html.escape(self.mlflow)}/runs/{value}">{chip}</a>'
                    if self.mlflow
                    else chip
                )
            if found[0][1]:
                return f'<code class="commit">{value}</code>'
        if PATH_LIKE.search(value):
            # a long path breaks after a slash, `::` or `_` before it breaks inside a name
            broken = FILE_BREAK.sub(r"\1<wbr>", value)
            return f'<code class="file">{broken}</code>'
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

    # the tail: links, ticket, PR and the date or the stage; the kicker names the category
    tail = []
    mlflow = model.value("mlflow")
    if mlflow and is_local(mlflow):
        tail.append(INLINE.local("MLflow · local", mlflow))
    elif mlflow:
        tail.append(INLINE.link("MLflow", mlflow))
    elif any(
        not is_commit
        for f in read_findings(model)
        for _, is_commit in checks.hex_ids(f.text("Runs"))
    ):
        tail.append(INLINE.missing("no MLflow address"))
    tail.append(_ticket(model.value("ticket"), site.github))
    if pr := model.value("pr"):
        tail.append(_pr(str(pr), site.github))
    # a dropped exploration writes its concluded date when it is dropped
    concluded = model.value("concluded")
    ended = DROPPED if model.stage == DROPPED else "concluded"
    tail.append(
        f"{ended} {concluded.isoformat()}" if concluded else (model.stage or "draft")
    )

    # the reference entries the record varies or uses
    known = {m.id: m for m in site.of_kind("entry")}
    named = [
        (key, [entry_ref(str(value), known) for value in model.value(key)])
        for key in ("varies", "uses")
        if model.value(key)
    ]

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


def is_local(address: str) -> bool:
    """Whether a tracking address opens only on the author's machine: a local host, a file or a bare path."""
    parsed = urlparse(address)
    if parsed.scheme in ("http", "https"):
        return (parsed.hostname or "") in LOCAL_HOSTS
    if parsed.scheme:
        return parsed.scheme == "file"
    host = address.split("/")[0].split(":")[0]
    return not host or host.startswith(".") or host in LOCAL_HOSTS


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
                    str(p["slug"]): (
                        str(p.get("short") or p["slug"]),
                        str(p.get("title") or ""),
                    )
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
    clauses: list[Clause],
    findings: list[Finding],
    readings: dict,
    words: dict[str, str],
    outcome: str,
    decision: dict | None = None,
) -> dict:
    """The answer card: question, verdict line, arms table, clause list, Decision figure, notes, later evidence, what it opened.

    Before a verdict the card is the question and the clause list alone. An
    arms record's arms table carries its arms' clauses, which then leave the
    clause list.
    """
    answer = {
        "question": read_question(model, prose, clauses),
        "verdict": None,
        "arms": None,
        "figure": decision,
        "notes": [],
        "later": [],
        "opened": [],
    }
    if not outcome:
        answer["clauses"] = read_clause_list(
            model, prose, clauses, findings, readings, words, set(), anchored=True
        )
        return answer

    arms = read_arms(model, prose, clauses, readings, words)
    answer |= {
        "verdict": read_verdict_line(model, prose, outcome),
        "arms": arms,
        "clauses": read_clause_list(
            model,
            prose,
            clauses,
            findings,
            readings,
            words,
            arms["carried"] if arms else set(),
            anchored=False,
        ),
        "notes": read_notes(model, prose),
        "later": read_later(model, site, prose),
        "opened": read_opened(model, site, prose),
    }
    return answer


def read_question(model: Model, prose: Prose, clauses: list[Clause]) -> dict:
    """The answer's first line: the Question; else Decision it informs; else a drafted question."""
    question = markup.unguided(model.row("Hypothesis", "Question"))
    decision = markup.unguided(model.row("Hypothesis", "Decision it informs"))
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
    answer = answer or TLDR_STATUS.sub("", markup.unguided(model.row("TL;DR", "Hypothesis")))
    reading = markup.unguided(model.row("Verdict", "Outcome"))
    reading = re.sub(rf"^\s*{re.escape(outcome)}\b\s*", "", reading, flags=re.I)
    reading = re.sub(r"^[—–:;,.\s-]+", "", reading)
    return {
        "word": outcome,
        "answer": prose.inline(answer, capital=True),
        "reading": prose.inline(reading),
        "flags": read_flags(model, prose),
        "evidence": read_evidence(model, prose),
    }


def read_evidence(model: Model, prose: Prose) -> dict:
    """The Verdict's Evidence: its lead line, then one item per bullet; a plain row is the lead alone."""
    row = section_row(model, "Verdict", "Evidence")
    bullets = items(row)
    lead = markup.unguided(row.body[0] if bullets else row.text) if row else ""
    return {
        "lead": prose.inline(lead, capital=True),
        "items": [prose.inline(b, capital=True) for b in bullets],
    }


def read_decision_figure(model: Model, prose: Prose, exhibits: Exhibits) -> dict | None:
    """The Verdict's Decision figure for the answer card; None while the row is absent or guidance.

    It is noted as drawn, so the files no card draws leave it out, and the
    finding whose Result shows the same file keeps the one number.
    """
    rows = figure_rows(model.section("Verdict"), DECISION_FIGURE)
    if not rows or not rows[0].links:
        return None
    caption, src = rows[0].links[0]
    exhibits.cited_by.setdefault(src, [])
    return read_exhibit(prose, exhibits, src, caption, "")


def read_flags(model: Model, prose: Prose) -> list[dict]:
    """A flag for each dated note on the Verdict rule or a clause line, at or after the first result.

    A note is `Note added <date>:` inside the line, or a nested `- <date> …:`
    bullet under it; notes of one line and date share a flag.
    """
    history = " ".join(f.text("History") for f in read_findings(model))
    dates = ISO_DATE.findall(history)
    first = min(dates) if dates else None
    concluded = model.value("concluded")
    flags: dict[str, dict] = {}
    for text in items(section_row(model, "Hypothesis", "Predictions")):
        clause = checks.CLAUSE_LINE.match(text)
        if not clause and not checks.VERDICT_RULE.match(text):
            continue
        where = f"{clause.group(1)} bar" if clause else "Verdict rule"
        notes = sorted(
            [*RULE_NOTE.finditer(text), *DATED_NOTE.finditer(text)],
            key=lambda m: m.start(),
        )
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
            title = f"{where} amended {at} · {date}"
            said = prose.inline(text[note.end() : end].strip(), capital=True)
            if title in flags:
                flags[title]["text"] = Markup("{} {}").format(
                    flags[title]["text"], said
                )
            else:
                flags[title] = {"title": title, "text": said}
    return list(flags.values())


def read_clause_list(
    model: Model,
    prose: Prose,
    clauses: list[Clause],
    findings: list[Finding],
    readings: dict,
    words: dict[str, str],
    carried: set[str],
    anchored: bool,
) -> list[dict]:
    """The answer's clauses: chip, claim and condition, then the readings table or the answer in words.

    A clause the arms table carries is left out. Numeric readings with a bar
    or a margin go in a table on the clause's own axis; a question, and any
    clause without such a reading, is answered in words with its finding chips.
    """
    rounds = len(entries(model, "Rounds"))
    rows = []
    for clause in clauses:
        if clause.id in carried:
            continue
        yes, _ = clause.keywords
        condition = clause.conditions.split("; ")[0]
        if not condition.lower().startswith(yes.split()[0]):
            condition = f"{yes} {condition}"
        table = (
            None if clause.id.startswith("Q") else read_readings_table(clause, readings)
        )
        rows.append(
            {
                "id": f"{prose.page_id}/{clause.id.lower()}" if anchored else "",
                "chip": clause_chip(clause, readings, words[clause.id], prose.page_id),
                "claim": prose.inline(clause.claim, capital=True),
                "condition": prose.inline(_trace_free(condition)),
                "tags": clause_tags(prose, clause, rounds),
                "table": table,
                "answer": ""
                if table
                else read_answer_words(prose, clause, findings, readings),
                "planned": bool(clause.superseded_by),
            }
        )
    return rows


def clause_tags(prose: Prose, clause: Clause, rounds: int) -> Markup:
    """A clause's tags beside its claim: role, standard, round, and the clause that superseded it as a chip."""
    tags = [
        w for w in (clause.role, clause.standard) if w not in ("decisive", "tested")
    ]
    tags += [clause.round] if rounds > 1 else []
    if clause.superseded_by:
        later = clause.superseded_by
        named = prose.chip(later) if later in prose.tips else later
        tags.append(Markup("superseded by {}").format(named))
    return Markup(" · ").join(tags)


def read_answer_words(
    prose: Prose, clause: Clause, findings: list[Finding], readings: dict
) -> Markup:
    """A clause answered in words: the deciding Summary, else its readings, several as a list; its finding chips apart."""
    deciding = deciding_findings(clause, findings)
    if deciding:
        return Markup('{} <span class="fchips">{}</span>').format(
            prose.inline(deciding[0].text("Summary"), capital=True),
            Markup("").join(prose.chip(f.id) for f in deciding),
        )
    texts = []
    for x in readings.get(clause.id, []):
        if not x.get("value"):
            continue
        text = Markup("{}{}").format(
            f"{x['qualifier']} " if x.get("qualifier") else "", _value(x["value"])
        )
        if x.get("ci_low") and x.get("ci_high"):
            text += Markup(' <span class="muted">[{}, {}]</span>').format(
                _value(x["ci_low"]), _value(x["ci_high"])
            )
        texts.append(text)
    if len(texts) > 1:
        return Markup('<ul class="rlist">{}</ul>').format(
            Markup("").join(Markup("<li>{}</li>").format(t) for t in texts)
        )
    return texts[0] if texts else Markup("")


def read_readings_table(clause: Clause, readings: dict) -> dict | None:
    """A clause's numeric readings on its own axis, when one is read against a bar or a margin."""
    points = [p for x in readings.get(clause.id, []) if (p := _reading_point(x))]
    if not any(p["bar"] is not None or p["margin"] is not None for p in points):
        return None
    edges = [e for p in points for e in (p["low"], p["high"])]
    edges += [p["bar"] for p in points if p["bar"] is not None]
    edges += [
        m
        for p in points
        if p["margin"] is not None
        for m in (-p["margin"], p["margin"])
    ]
    lo, hi, ticks = axis_ticks(min(edges), max(edges))
    return {
        "rows": [
            _track(p, lo, hi)
            | {
                "word": p["word"],
                "qualifier": p["qualifier"] or "the reading",
                "value": p["text"],
                "interval": p["interval"],
            }
            for p in points
        ],
        "ticks": [(_at(t, lo, hi), _tick(t)) for t in ticks],
        "legend": "dashed line: the bar"
        if any(p["bar"] is not None for p in points)
        else "grey band: the margin",
        "value_head": "Value [95% interval]"
        if any(p["interval"] for p in points)
        else "Value",
    }


def axis_ticks(
    lo: float, hi: float, steps: int = STRIP_STEPS
) -> tuple[float, float, list[float]]:
    """A track's axis: about `steps` steps of 1, 2, 2.5 or 5 times a power of ten."""
    if hi <= lo:
        lo, hi = lo - (abs(lo) or 1) / 2, hi + (abs(hi) or 1) / 2
    raw = (hi - lo) / steps
    magnitude = 10 ** int(f"{raw:e}".split("e")[1])
    step = next(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw)
    start, end = (lo // step) * step, -((-hi) // step) * step
    ticks = [
        round(start + i * step, 10) for i in range(int(round((end - start) / step)) + 1)
    ]
    return round(start, 10), round(end, 10), ticks


def read_arms(
    model: Model,
    prose: Prose,
    clauses: list[Clause],
    readings: dict,
    words: dict,
) -> dict | None:
    """`verdict-by-arm.csv` joined with each `<quantity>-by-arm.csv` on `arm`, as the arms table.

    Each arm's comparison clause sits in its row as its chip and its gain on
    one shared axis with the margin band. Columns equal across arms are
    dropped, an interval's columns join their value, the baseline row
    (`role`) comes first, the adopted arm is badged and each column's best
    value is bold. Columns group by the table they come from; a table with
    several rows per arm is left out.
    """
    verdict = markup.read_table(artifact_path(model, "verdict-by-arm"))
    if not verdict:
        return None
    by_clause = {c.id: c for c in clauses}

    # the column groups: the verdict table's, then each per-arm table's
    groups = [("Verdict", verdict)]
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

    # each arm's clause, which carries its word, so an outcome column says it twice
    arm_clauses = [
        by_clause.get(row.get("clause", "")) if i not in baseline else None
        for i, row in enumerate(verdict)
    ]
    carried = {c.id for c in arm_clauses if c}
    skipped = {"arm", "role", "adopted"} | ({"clause", "outcome"} if carried else set())

    columns = []
    for name, table in groups:
        by_arm = {row.get("arm", ""): row for row in table}
        rows = [by_arm.get(arm, {}) for arm in arms]
        keys = [k for k in (table[0].keys() if table else ()) if k not in skipped]
        bounds = _interval_columns(keys)
        for key in keys:
            values = [row.get(key, "") for row in rows]
            # a column another table already gave, by name or by values, shows once,
            # with this table's interval when the first had none
            again = next(
                (
                    c
                    for c in columns
                    if markup.column_header(c["key"]).lower()
                    == markup.column_header(key).lower()
                    or c["values"] == values
                ),
                None,
            )
            if again and key in bounds and not again["bounds"]:
                again["bounds"], again["bound_rows"] = bounds[key], rows
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
                    "bound_rows": rows,
                    "best": _best_values(markup.column_header(key), values),
                }
            )

    # each arm's gain over the baseline, on one axis with the margin band
    points = {
        c.id: p
        for c in arm_clauses
        if c
        for p in [_reading_point(x) for x in readings.get(c.id, [])][:1]
        if p
    }
    gain = None
    if points:
        edges = [
            e
            for p in points.values()
            for e in (p["low"], p["high"], -(p["margin"] or 0), p["margin"] or 0)
        ]
        lo, hi, ticks = axis_ticks(min(edges), max(edges), GAIN_STEPS)
        gain = {
            "ticks": [(_at(t, lo, hi), _tick(t)) for t in ticks],
            "legend": "grey band: the margin around zero",
            "rows": {cid: _track(p, lo, hi) for cid, p in points.items()},
        }

    # one cell per column per arm, the baseline first
    body = []
    for i in order:
        cells = []
        for column in columns:
            value = column["values"][i]
            best = value in column["best"]
            if column["key"] == "clause" and value in by_clause:
                cells.append(
                    (
                        "chip",
                        clause_chip(
                            by_clause[value], readings, words[value], prose.page_id
                        ),
                        False,
                    )
                )
            elif value in OUTCOME:
                cells.append(
                    ("word", Markup("{} {}").format(WORDS.mark(value), value), False)
                )
            elif value in ("True", "False"):
                cells.append(("text", "yes" if value == "True" else "no", False))
            elif markup.NUMERIC.match(value.strip() or "x"):
                low, high = (
                    column["bound_rows"][i].get(b, "")
                    for b in (column["bounds"] or ("", ""))
                )
                cells.append(
                    (
                        "number",
                        (
                            markup.number(value),
                            markup.number(low) if low else "",
                            markup.number(high) if high else "",
                        ),
                        best,
                    )
                )
            elif not value:
                cells.append(("text", Markup(markup.EMPTY_CELL), False))
            else:
                cells.append(("text", value, False))
        clause = arm_clauses[i]
        body.append(
            {
                "arm": arms[i],
                "clause": clause_chip(clause, readings, words[clause.id], prose.page_id)
                if clause
                else "",
                "gain": gain["rows"].get(clause.id) if gain and clause else None,
                "cells": cells,
                "adopted": verdict[i].get("adopted", "").lower() in YES,
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
    rule = re.sub(r"^\*\*Verdict rule\*\*:\s*", "", rule)
    rule = DATED_NOTE.split(RULE_NOTE.split(rule)[0])[0].strip()
    return {
        "heads": heads,
        "groups": spans if len(spans) > 1 else [],
        "rows": body,
        "carried": carried,
        "with_clause": bool(carried),
        "gain": gain,
        "rule": prose.inline(rule, capital=True),
    }


def read_notes(model: Model, prose: Prose) -> list[dict]:
    """Caveat, and Disclosure tagged with its kind."""
    notes = []
    if caveat := markup.unguided(model.row("Verdict", "Caveat")):
        notes.append(
            {
                "kind": "caveat",
                "title": "Caveat",
                "tag": "",
                "text": prose.inline(caveat, capital=True),
                "refs": [],
            }
        )
    if disclosure := markup.unguided(model.row("Hypothesis", "Disclosure")):
        match = DISCLOSURE.match(disclosure)
        tag, text = (
            (match.group(1).lower(), match.group(2)) if match else ("", disclosure)
        )
        notes.append(
            {
                "kind": "disclosure",
                "title": f"Disclosure · {tag}" if tag else "Disclosure",
                "tag": "",
                "text": prose.inline(text, capital=True),
                "refs": [],
            }
        )

    return notes


def read_later(model: Model, site: Site, prose: Prose) -> list[dict]:
    """Later evidence: one line per later record citing a finding here, then those that only build on it.

    A line is the later record's pill, what it does (`supports`, `qualifies`,
    `overturns`, `cites`, or `builds on this`), the finding it names and the
    date it concluded.
    """
    slug = record_slug(model)

    # later records that cite a finding here, each citation once
    cites = []
    for other in site.of_kind("record"):
        if other is model:
            continue
        for finding in read_findings(other):
            text = "\n".join(row_source(r) for r in finding.entry.rows.values())
            for match in checks.RECORD_CITATION.finditer(text):
                if match.group(1) == slug and match.group(2).startswith("F"):
                    relation = RELATION_BEFORE.search(text[: match.start()])
                    cite = (
                        other,
                        match.group(2),
                        relation.group(1) if relation else "",
                    )
                    if not any(c[0] is other and c[1:] == cite[1:] for c in cites):
                        cites.append(cite)

    # one line per citation, then the records that only build on this one
    lines = [(other, relation or "cites", code) for other, code, relation in cites]
    lines += [
        (other, "builds on this", "")
        for other in children(model, site)
        if not any(c[0] is other for c in cites)
    ]
    return [
        {
            "link": record_pill(other),
            "word": word,
            "chip": prose.chip(code) if code in prose.tips else "",
            "date": concluded.isoformat()
            if (concluded := other.value("concluded"))
            else "",
        }
        for other, word, code in lines
    ]


def read_opened(model: Model, site: Site, prose: Prose) -> list[dict]:
    """The Verdict's Next items as What it opened, each with one status mark.

    A done item takes the done mark, a planned one the status of the record it
    names (the dashed planned mark while that record does not exist yet); an
    issue, a not-yet-planned item and an item with no ending take the planned
    mark, since no experiment has started on them. `arrow` marks a link that
    names where the item went: a record, a planned name, a commit or an issue.
    """
    by_slug = {record_slug(m): m for m in site.of_kind("record")}
    planned_names = {
        slug: names
        for g in read_programmes(site.corpus.experiments)
        for slug, names in g["planned"].items()
    }
    opened = []
    for item in items(section_row(model, "Verdict", "Next")):
        planned, issue, done = (
            PLANNED_AS.search(item),
            ISSUE.search(item),
            DONE_IN.search(item),
        )
        text = DONE_IN.sub("", item)
        text = NOT_PLANNED.sub("", PLANNED_AS.sub("", ISSUE.sub("", text))).strip()
        shown = markup.first_sentence(text)
        if done:
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": "done",
                    "link": done_link(done.group(1), by_slug, site.github),
                    "arrow": True,
                    "rests": False,
                }
            )
        elif planned and planned.group(1) in by_slug:
            other = by_slug[planned.group(1)]
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": record_word(other),
                    "link": record_pill(other),
                    "arrow": True,
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
                    "arrow": True,
                    "rests": False,
                }
            )
        elif issue:
            opened.append(
                {
                    "text": prose.inline(shown),
                    "word": "planned",
                    "link": INLINE.iss(
                        issue.group(1),
                        "",
                        github_url(site.github, "issues", issue.group(1))
                        if site.github
                        else "",
                    ),
                    "arrow": True,
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
                    "word": "planned",
                    "link": chips,
                    "arrow": False,
                    "rests": bool(cited),
                }
            )
    return opened


def done_link(target: str, by_slug: dict[str, Model], github: str) -> Markup | str:
    """Where a done Next item was done: a record's pill, else a commit, else the name."""
    if target in by_slug:
        return record_pill(by_slug[target])
    if checks.SHA.match(target):
        return commit_chip(target, github)
    return target


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
    otherwise a note names the decisions the Predictions framing cites. No
    result shows twice: Found gives a count clause its multiple of the bar and
    the deciding finding chips, never a value the answer card already draws.
    """
    if not clauses:
        return None
    linked = all(
        (re.search(r"\[D\d+\]", c.line) or checks.RECORD_DECISION.search(c.line))
        and re.search(r"\[M\d+\]", c.line)
        for c in clauses
    )
    rounds = len(entries(model, "Rounds"))

    rows, said = [], {}
    for clause in clauses:
        deciding = deciding_findings(clause, findings)
        chips = Markup("").join(prose.chip(f.id) for f in deciding)
        found = bar_multiples(readings.get(clause.id, []))
        tags = clause_tags(prose, clause, rounds)

        # a So another clause already says points back to that clause, never repeats
        so = clause_so(prose, clause, deciding)
        key = markup.plain_text(str(so))
        if key and key in said and 'class="missing"' not in so:
            so = Markup("{} {}").format(INLINE.missing("as"), prose.chip(said[key]))
        else:
            said.setdefault(key, clause.id)
        rows.append(
            {
                "id": f"{prose.page_id}/{clause.id.lower()}",
                "chip": clause_chip(clause, readings, words[clause.id], prose.page_id),
                "claim": prose.inline(clause.claim),
                "tags": tags,
                "decided": prose.chips(" ".join(re.findall(r"\[D\d+\]", clause.line)))
                + Markup("").join(
                    prose.record_decision(slug, code)
                    for slug, code in checks.RECORD_DECISION.findall(clause.line)
                ),
                "measured_by": prose.chips(
                    " ".join(re.findall(r"\[M\d+\]", clause.line))
                ),
                "found": found,
                "chips": chips,
                "so": so,
                "planned": bool(clause.superseded_by),
            }
        )
    framing = section_row(model, "Hypothesis", "Predictions")
    under = prose.chips(
        " ".join(re.findall(r"\[D\d+\]", framing.body[0] if framing else ""))
    )
    return {
        "rows": rows,
        "linked": linked,
        # finding chips alone join So, and the Found column goes
        "has_found": any(r["found"] for r in rows),
        "concluded": prose.concluded,
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
            lead = implication.split(" - ")[0]
            naming = [
                sentence
                for sentence in SENTENCE_SPLIT.split(lead)
                if re.search(CLAUSE_NAME.format(id=clause.id), sentence)
            ]
            return prose.inline(
                naming[0] if naming else markup.first_sentence(lead), capital=True
            )
    # a concluded record leaves the cell empty rather than read as unfinished
    if prose.concluded:
        return Markup("")
    if not deciding:
        return INLINE.missing("no finding decides it yet")
    return INLINE.missing("no implication recorded")


def bar_multiples(readings: list[dict]) -> str:
    """Each count reading as a multiple of its bar, `0.59× by source`; "" when none reads against a count."""
    texts = []
    for x in readings:
        value, bar = (
            markup.to_float(x.get("value", "")),
            markup.to_float(x.get("bar", "")),
        )
        if value is None or bar is None or bar < COUNT_BAR:
            continue
        ratio = value / bar
        shown = f"{ratio:.2f}" if ratio < 10 else f"{ratio:,.0f}"
        where = x.get("qualifier", "")
        texts.append(f"{shown}× {where}".strip())
    return "; ".join(texts)


def read_rests_on(model: Model, prose: Prose) -> list[dict]:
    """The decisions and methods the Limitations lean on, each with a short label.

    The label is the text before ` — `; else the sentence when it is short,
    else the part of it that holds the citation.
    """
    rests = []
    for text in items(section_row(model, "Verdict", "Limitations")):
        cited = [c for c in re.findall(r"\[([DM]\d+)\]", text) if c in prose.tips]
        if not cited:
            continue
        label = text.partition(" — ")[0] if " — " in text else _rests_label(text)
        rests.append(
            {
                "label": prose.inline(label),
                "chips": Markup("").join(prose.chip(c) for c in cited),
            }
        )
    return rests


def read_needs(model: Model, prose: Prose) -> list[dict]:
    """People, Compute, Data and External as their short values; a heavy one marked hot."""
    needs = []
    for label, name in NEEDS:
        text = markup.unguided(model.row("Design", label))
        short = text.split(" — ")[0] if " — " in text else markup.first_sentence(text)
        plain = markup.plain_text(markup.inline_html(short))
        needs.append(
            {
                "name": name,
                "text": prose.inline(short) if text else "",
                "hot": bool(plain)
                and not NO_NEED.match(plain)
                and (name == "External" or bool(HOT_NEED.search(plain))),
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
) -> dict:
    """Deciding key findings as open cards with their supporting findings; the rest as small cards.

    Tables and figures are numbered in page order, as the cards draw them: a
    key card draws its Result and the files its Reading and Implication link,
    a small card only its Result. The files nothing on the page draws follow
    in a closed fold.
    """
    if not findings:
        return {"n": 0, "cards": [], "others": [], "more": None}
    deciding = deciding_keys(model, findings)

    def supporting(key: Finding) -> list[Finding]:
        return [f for f in findings if f.weight == "supporting" and f.target == key.id]

    # the small cards in page order: every other finding with its supporting ones
    nested = {f.id for key in deciding for f in supporting(key)}
    rest = [f for f in findings if f not in deciding and f.id not in nested]
    keys = {f.id for f in rest if f.weight != "supporting"}
    others = []
    for finding in rest:
        if finding.weight == "supporting" and finding.target in keys:
            continue
        others.append(finding)
        others += [
            f for f in rest if f.weight == "supporting" and f.target == finding.id
        ]

    # who draws each file, so a shared file names the other findings drawing it
    for key in deciding:
        note_drawn(exhibits, key, full=True)
        for finding in supporting(key):
            note_drawn(exhibits, finding, full=False)
    for finding in others:
        note_drawn(exhibits, finding, full=False)

    def small(finding: Finding) -> dict:
        return read_small_card(prose, exhibits, clauses, finding, readings, words)

    cards = []
    for key in deciding:
        card = read_card(prose, exhibits, clauses, key, readings, words)
        card["supporting"] = [small(f) for f in supporting(key)]
        cards.append(card)
    return {
        "n": len(findings),
        "cards": cards,
        "others": [small(f) for f in others],
        "more": read_uncited(model, prose, exhibits),
    }


def note_drawn(exhibits: Exhibits, finding: Finding, full: bool) -> None:
    """Notes the finding against each file its card draws: the Result, and in full every link too."""
    result = [src for _, src in markup.ASSET_LINK.findall(finding.source("Result"))]
    drawn = result if full else result[:1]
    if full:
        drawn += [
            src
            for label in ("Reading", "Implication")
            for _, src in markup.ASSET_LINK.findall(finding.source(label))
        ]
    for src in dict.fromkeys(drawn):
        if _exhibit_kind(exhibits.folder, src):
            exhibits.cited_by.setdefault(src, []).append(finding.id)


def read_uncited(model: Model, prose: Prose, exhibits: Exhibits) -> dict | None:
    """The tables and figures on disk that no finding card or dataset row draws."""
    folder = exhibits.folder
    datasets = model.section("Datasets")
    linked = {
        src for _, src in markup.ASSET_LINK.findall(datasets.text if datasets else "")
    }

    # the standard tables the answer card and the dataset rows draw
    drawn = [
        model.spec.part(f"artifact.{name}").pattern
        for name in (
            "clause-readings",
            "verdict-by-arm",
            "dataset-summary",
            "dataset-fields",
            "dataset-statistics",
        )
    ]
    if (folder / drawn[1]).is_file():
        drawn.append(model.spec.part("artifact.arm-table").pattern)

    files = [
        path.relative_to(folder).as_posix()
        for sub in ("tables", "figures")
        for path in sorted((folder / sub).glob("*"))
        if _exhibit_kind(folder, path.relative_to(folder).as_posix())
    ]
    uncited = [
        src
        for src in files
        if src not in exhibits.cited_by
        and src not in linked
        and not any(fnmatch.fnmatch(src, pattern) for pattern in drawn)
    ]
    if not uncited:
        return None
    shown = [read_exhibit(prose, exhibits, src, "", "") for src in uncited]
    kinds = [e["kind"] for e in shown]
    return {
        "id": f"{prose.page_id}/more-exhibits",
        "meta": " · ".join(
            _count(kinds.count(kind), noun)
            for kind, noun in (("Table", "table"), ("Figure", "figure"))
            if kind in kinds
        ),
        "exhibits": shown,
    }


def read_card(
    prose: Prose,
    exhibits: Exhibits,
    clauses: list[Clause],
    finding: Finding,
    readings: dict,
    words: dict,
) -> dict:
    """A finding in full: key number, claim, reading, could overturn it, So, runs, history, exhibits."""
    cited = finding_clauses(finding, clauses)
    number, word, note = key_number(finding, cited, readings)
    match = markup.ASSET_LINK.search(finding.source("Result"))
    exhibit = read_result(prose, exhibits, finding)
    more = read_more_exhibits(prose, exhibits, finding, match.group(2) if match else "")

    return {
        "id": finding.id,
        "anchor": f"{prose.page_id}/{finding.id.lower()}",
        "chip": prose.chip(finding.id),
        "answers": [
            clause_chip(c, readings, words[c.id], prose.page_id) for c in cited
        ],
        "number": number,
        "number_kind": OUTCOME.get(word, word),
        "number_note": prose.inline(note) if note else "",
        "claim": prose.inline(finding.text("Summary"), capital=True),
        **read_reading(prose, finding),
        "runs": read_runs(prose, finding),
        "exhibit": exhibit,
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
) -> dict:
    """A supporting or other finding: chip, title, clauses, Summary, exhibit and runs; its reading folds."""
    reading = read_reading(prose, finding)
    if not _said(finding.text("Implication")):
        reading["so"] = ""
    return {
        **reading,
        "id": finding.id,
        "anchor": f"{prose.page_id}/{finding.id.lower()}",
        "chip": prose.chip(finding.id),
        "title": prose.inline(finding.title),
        "answers": [
            clause_chip(c, readings, words[c.id], prose.page_id)
            for c in finding_clauses(finding, clauses)
        ],
        "summary": prose.inline(finding.text("Summary"), capital=True),
        "exhibit": read_result(prose, exhibits, finding),
        "runs": read_runs(prose, finding),
    }


def read_reading(prose: Prose, finding: Finding) -> dict:
    """A finding's Reading as paragraphs, what could overturn it, its So and its dated history."""
    reading = finding.entry.rows.get("Reading")
    blocks = items(reading)
    lead = markup.unguided(reading.body[0]) if reading else ""
    paragraphs = ([lead] if lead and blocks else []) + (
        blocks[:2] if blocks else [finding.source("Reading")]
    )
    overturn = blocks[2] if len(blocks) > 2 else ""
    implication = finding.entry.rows.get("Implication")
    if not implication or not markup.unguided(implication.text):
        so = "" if prose.concluded else INLINE.missing("no implication recorded")
    else:
        so = prose.block(row_source(implication)) if _said(implication.text) else ""
    return {
        "reading": [
            prose.inline(_unbold(p).strip(), capital=True) for p in paragraphs if p
        ],
        "overturn": prose.inline(_unbold(overturn).strip(), capital=True),
        "so": so,
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
    # the first cited clause given a word colours it; its readings beat the prose
    word = ""
    for clause in cited:
        words = {
            x["outcome"]
            for x in readings.get(clause.id, [])
            if x.get("outcome") in OUTCOME
        }
        word = (
            words.pop()
            if len(words) == 1
            else checks.outcome(finding.entry, clause.check_clause)
        )
        if word:
            break
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
    """The runs line: run ids, lost runs and commits (the first three of each) and the replaced runs."""
    found = checks.hex_ids(finding.text("Runs"))
    lost = list(dict.fromkeys(checks.lost_runs(finding.text("Runs"))))
    runs = [value for value, is_commit in dict.fromkeys(found) if not is_commit]
    commits = [value for value, is_commit in dict.fromkeys(found) if is_commit]
    replaced = [
        run
        for match in REPLACES.finditer(finding.text("History"))
        for run in re.findall(r"`([0-9a-f]+)`", match.group(1))
    ]

    def shown(run: str) -> tuple[str, str, str]:
        """(id shown, link, full id); a local server's id shows whole, clipped by the style."""
        if prose.mlflow_local:
            return run, "", run
        return (
            run[:RUN_ID_SHOWN],
            f"{prose.mlflow}/runs/{run}" if prose.mlflow else "",
            "",
        )

    return {
        "ids": [shown(run) for run in runs[:RUNS_SHOWN]],
        "more": max(len(runs) - RUNS_SHOWN, 0),
        "lost": [(run[:RUN_ID_SHOWN], run) for run in lost[:RUNS_SHOWN]],
        "more_lost": max(len(lost) - RUNS_SHOWN, 0),
        "commits": [c[:SHA_SHOWN] for c in commits[:RUNS_SHOWN]],
        "more_commits": max(len(commits) - RUNS_SHOWN, 0),
        "addressed": bool(prose.mlflow) or prose.mlflow_local,
        "replaced": [shown(run) for run in replaced],
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
                "when": None if not tag else WHEN.get(tag.group(1), "results"),
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
        return read_exhibit(
            prose, exhibits, match.group(2), match.group(1), rest, finding.id
        )

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
    return _exhibit(
        label=exhibits.label(f"{finding.id}/result", "Table"),
        caption=prose.inline(caption),
        html=Markup(markup.markdown_html(table)),
        kind="Table",
    )


def read_exhibit(
    prose: Prose, exhibits: Exhibits, src: str, lead: str, rest: str, owner: str = ""
) -> dict:
    """One figure or table file, numbered page-wide, its caption the link text and prose.

    A file several findings cite keeps its one number in each, and names the
    other findings that cite it beside its caption.
    """
    # each placement of a file prefixes its SVG ids apart, so no two copies share one
    placement = exhibits.placements.get(src, 0) + 1
    exhibits.placements[src] = placement
    prefix = f"{prose.page_id}-{Path(src).stem}-" + (
        f"{placement}-" if placement > 1 else ""
    )
    inlined = markup.inline_asset(exhibits.folder, src, prefix)
    if inlined is None:
        return _exhibit(missing=INLINE.missing(f"missing: {src}"))
    kind, html_ = inlined
    anchor = exhibits.anchor(src)
    placed = anchor not in exhibits.anchors
    exhibits.anchors.add(anchor)
    lead = lead.strip()
    # a link text that only points ("the table") says nothing; the file name says what it holds
    if POINTER_TEXT.match(lead) or not lead:
        lead = markup.column_header(Path(src).stem.replace("-", "_"))
    if lead and not lead.endswith((".", "?", "!", ":", ";", "—")):
        lead += "."
    rest = rest.strip().lstrip(",;:—– ")
    caption = " ".join(x for x in (lead, rest) if x)

    # a figure narrower than NARROW_FIGURE takes half the row; an unread size, all of it
    shape = ""
    if kind == "Figure":
        ratio = markup.figure_ratio(exhibits.folder / src)
        shape = "narrow" if ratio and ratio < NARROW_FIGURE else "wide"

    also = [f for f in exhibits.cited_by.get(src, []) if f != owner]
    return _exhibit(
        anchor=anchor if placed else "",
        label=exhibits.label(src, kind),
        title=prose.inline(lead.rstrip(".?!:;—"), capital=True),
        caption=prose.inline(caption),
        html=Markup(html_),
        src=src,
        kind=kind,
        shape=shape,
        also=Markup(" ").join(prose.chip(f) for f in also),
    )


def read_figure_row(
    folder: Path,
    src: str,
    caption: Markup,
    label: str,
    anchor: str,
    note: str = "",
    width: str = "",
    shown: str = "",
) -> dict:
    """A Figure row's figure on a programme or entry page, as `evidence.exhibit` draws it.

    Args:
        folder: The document's folder, which `src` is relative to.
        src: The linked file.
        caption: The link text as HTML.
        label: The page-wide `Figure <n>`.
        anchor: The figure's id, which also sets the prefix of the ids in its SVG.
        note: Words after the file name under the figure (`drawn from x.dot`).
        width: The widest the figure is drawn, a CSS length; "" for the column.
        shown: The file line's path; "" for `src` as written.

    Returns:
        The exhibit's values, or a missing note when the file is absent or no figure.
    """
    prefix = re.sub(r"[^A-Za-z0-9_-]", "-", anchor) + "-"
    inlined = markup.inline_asset(folder, src, prefix)
    if inlined is None or inlined[0] != "Figure":
        return _exhibit(missing=INLINE.missing(f"missing: {src}"))
    return _exhibit(
        anchor=anchor,
        label=label,
        caption=caption,
        html=Markup(inlined[1]),
        src=" · ".join(x for x in (shown or src, note) if x),
        kind="Figure",
        shape="frow",
        width=width,
    )


def figure_place(model: Model, site: Site, path: Path) -> tuple[str, str]:
    """Where a record's page draws one of its figure files.

    Args:
        model: The record.
        site: Every document, which the record page reads.
        path: The figure file, resolved.

    Returns:
        The finding whose card first draws it ("" for the answer card or the
        fold of files no card draws) and its `Figure <n>`; ("", "") when the
        page draws it nowhere.
    """
    context = record_page(model, site).context
    folder = model.document.folder

    def draws(exhibit: dict | None) -> bool:
        return bool(exhibit and exhibit["src"] and (folder / exhibit["src"]).resolve() == path)

    # the answer card, then the cards in page order, then the files no card draws
    if draws(context["answer"].get("figure")):
        return "", context["answer"]["figure"]["label"]
    findings = context["sections"]["findings"]
    drawn = []
    for key in findings["cards"]:
        drawn += [(key["id"], exhibit) for exhibit in (key["exhibit"], *key["more"])]
        drawn += [(card["id"], card["exhibit"]) for card in key["supporting"]]
    drawn += [(card["id"], card["exhibit"]) for card in findings["others"]]
    for finding, exhibit in drawn:
        if draws(exhibit):
            return finding, exhibit["label"]
    for exhibit in (findings["more"] or {}).get("exhibits", []):
        if draws(exhibit):
            return "", exhibit["label"]
    return "", ""


def read_more_exhibits(
    prose: Prose, exhibits: Exhibits, finding: Finding, result: str
) -> list[dict]:
    """Tables and figures the Result lists after its first, then the Reading or Implication links."""
    seen, found = {result}, []
    for label in ("Result", "Reading", "Implication"):
        for lead, src in markup.ASSET_LINK.findall(finding.source(label)):
            if src not in seen:
                seen.add(src)
                found.append(read_exhibit(prose, exhibits, src, lead, "", finding.id))
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
                "meta": _count(n, "finding")
                if n
                else ("" if prose.concluded else "no citation"),
                "text": prose.inline(rest, capital=True) if lead else "",
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
            if label in entry.rows and markup.unguided(entry.rows[label].text)
        ]
        # only a change made after runs began names itself under the title
        last = (
            items(history_row)[-1] if len(history) > 1 and history[-1]["after"] else ""
        )
        found.append(
            {
                "id": match.group(1),
                "anchor": f"{prose.page_id}/{match.group(1).lower()}",
                "chip": prose.chip(match.group(1)),
                "title": prose.inline(match.group(2)),
                "when": max(
                    (h["when"] for h in history if h["when"]),
                    key=WHEN_ORDER.index,
                    default="before",
                ),
                "date": dates[0][5:] if dates else "",
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
    """Each method a fold: in → out, How as numbered steps, Code, Settings."""
    found = []
    for entry in entries(model, "Methods"):
        match = ENTRY_TITLE.match(entry.title)
        if match is None:
            continue
        code_row = entry.rows.get("Code")
        code = row_source(code_row) if code_row else ""
        files = list(dict.fromkeys(p for p, _ in checks.CODE_REFERENCE.findall(code)))
        settings_row = entry.rows.get("Settings")
        found.append(
            {
                "id": match.group(1),
                "anchor": f"{prose.page_id}/{match.group(1).lower()}",
                "chip": prose.chip(match.group(1)),
                "title": prose.inline(match.group(2)),
                "meta": method_files(files),
                "input": prose.block(row_source(entry.rows.get("Input"))),
                "output": prose.block(row_source(entry.rows.get("Output"))),
                "how": method_how(prose, row_source(entry.rows.get("How"))),
                "code": prose.block(code) if markup.unguided(code) else "",
                "settings": read_settings(prose, settings_row)
                if settings_row
                else None,
            }
        )
    return {"n": len(found), "methods": found}


def method_files(files: list[str]) -> Markup:
    """A closed method's source files, as the Code row writes them, then +n past the shown ones."""
    shown = [
        Markup(CODE_PATH_BREAK.sub(r"\1<wbr>", html.escape(f)))
        for f in files[:METHOD_FILES_SHOWN]
    ]
    rest = len(files) - METHOD_FILES_SHOWN
    return Markup(" · ").join(shown + ([Markup(f"+{rest}")] if rest > 0 else []))


def method_how(prose: Prose, how: str) -> Markup:
    """A How as numbered steps; the Code row, not the steps, names the code."""
    return Markup(str(prose.block(how)).replace("<ol>", '<ol class="steps2">', 1))


def read_settings(prose: Prose, row: Row) -> dict:
    """A Settings row as key, what it sets and value; the value is the gloss's last part with a digit."""
    text = markup.unguided(row_source(row))
    if not text or text.lower().startswith("none"):
        said = _said(text)
        return {
            "rows": [],
            "text": prose.inline(text, capital=True) if said else "No settings.",
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
                # a long key breaks after an underscore, never inside a word
                "key": Markup("_<wbr>").join(key.split("_")),
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
    found = items(row) or ([markup.unguided(row.text)] if row and markup.unguided(row.text) else [])
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
        source = markup.unguided(row_source(row))
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
            rf"(?:[\s,;]*{markup.ASSET_LINK.pattern})+[\s.]*$",
            "",
            text[one.end() :] if one else text,
            flags=re.I,
        ).strip()
        # a bullet that held only the links is left empty, and a lone `-` reads as a heading rule
        rest = re.sub(r"(?:^|\n)\s*[-*+]\s*$", "", rest).strip()
        # the kind slot holds one word; what it is derived from or split of opens the description
        kind, _, origin = kind.partition(" ")
        if origin:
            rest = f"{markup.capitalize(kind)} {origin.rstrip('.')}.\n\n{rest}".strip()
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
                or _statistics_size(
                    statistics, one.group(1) if one else "", markup.plain_text(name)
                )
                or _stated_size(text, one.group(1) if one else ""),
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
        kind = (
            "config"
            if path.suffix in CONFIG_SUFFIXES
            else "prompt"
            if "prompt" in str(path)
            else ""
        )
        codes = [c for c, t in cited if path.name in t and c in prose.tips]
        # a bare `(D5)` in the description joins the row's chips
        for bare in BARE_CODES.findall(what):
            named = [c for c in re.findall(r"[DFM]\d+", bare) if c in prose.tips]
            if named:
                codes += [c for c in named if c not in codes]
                what = what.replace(bare, "")
        what = re.sub(r"\s+([.,;:])", r"\1", what)
        codes.sort(key=lambda c: ("DMF".index(c[0]), int(c[1:])))
        documents.append(
            {
                "path": path.relative_to(folder).as_posix(),
                "what": prose.inline(what or kind or path.suffix.lstrip(".")),
                "kind": kind if what else "",
                "lines": len(lines),
                "chips": Markup("").join(prose.chip(c) for c in codes),
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
    return {"n": len(bib), "cited": cited, "uncited": len(bib) - len(cited)}


# ===============================
# Folds, builds on this and the section bar
# ===============================


def read_setup(
    model: Model, site: Site, prose: Prose, clauses: list[Clause]
) -> dict | None:
    """The Question and setup fold: Builds on, Statement, Rationale, Predictions, rule, Design, Rounds."""
    rows = []
    found_parents = parents(model, site)
    builds_on = markup.unguided(model.row("Hypothesis", "Builds on"))
    if found_parents or builds_on:
        pills = Markup(" ").join(
            record_pill(p) for p in found_parents
        ) or INLINE.missing("builds on nothing")
        # the pills name each parent, so its bullet's slug lead goes; one paragraph
        row = section_row(model, "Hypothesis", "Builds on")
        said = [markup.capitalize(SLUG_LEAD.sub("", i)) for i in items(row)] or [
            builds_on
        ]
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
        if markup.unguided(model.row("Hypothesis", label)):
            rows.append(
                (
                    label,
                    prose.block(row_source(section_row(model, "Hypothesis", label))),
                )
            )

    # the Predictions framing; the clauses sit in the answer card and the trace
    predictions = section_row(model, "Hypothesis", "Predictions")
    if predictions and markup.unguided(predictions.body[0]):
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
        model.section("Datasets") and markup.unguided(model.section("Datasets").text)
    )
    for label in ("Inputs", "Method / factors", "Metrics", "Protocol"):
        if label == "Inputs" and has_datasets:
            continue
        if markup.unguided(model.row("Design", label)):
            rows.append(
                (label, prose.block(row_source(section_row(model, "Design", label))))
            )

    # the rounds, one dated line each
    rounds = []
    for entry in entries(model, "Rounds"):
        match = ENTRY_TITLE.match(entry.title)
        if match is None:
            continue
        opened = markup.unguided(entry.rows["Opened"].text) if "Opened" in entry.rows else ""
        closed = markup.unguided(entry.rows["Closed"].text) if "Closed" in entry.rows else ""
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
        "meta": "",
        "digest": digest,
        "rows": rows,
        "rounds": rounds,
    }


def read_adoption(model: Model, prose: Prose) -> dict | None:
    """The Adoption candidate fold: the candidate, what it carries and what stays out."""
    row = section_row(model, "Verdict", "Adoption candidate")
    text = markup.unguided(row.body[0]) if row else ""
    if not text or text.lower().startswith("none"):
        return None
    rows = [("Candidate", prose.block(text))]
    for item in items(row):
        if label := NESTED_LABEL.match(item):
            rows.append((label.group(1), prose.block(label.group(2))))
    return {
        "id": f"{prose.page_id}/adoption",
        "digest": prose.inline(markup.first_sentence(text), capital=True),
        "rows": rows,
    }


def read_reproduce(model: Model, prose: Prose, findings: list[Finding]) -> dict | None:
    """The Reproduce fold: prerequisites by kind, the bash block as steps, other prose as written."""
    section = model.section("Reproduce")
    text = markup.unguided(section.text) if section else ""
    if not text:
        return None

    # prerequisites, by kind: the row's bullets, else the `Needs …` sentences of the prose
    prerequisites: dict[str, list[Markup]] = {}
    for item in items(section.rows.get("Prerequisites")):
        match = re.match(r"\*\*(.+?)\*\*\s*(.*)$", item, re.S)
        kind, rest = (
            (match.group(1), match.group(2))
            if match and match.group(1) in PREREQUISITE_KINDS
            else ("Other", item)
        )
        prerequisites.setdefault(kind, []).append(prose.inline(rest))

    # steps and loose prose
    ids = {f.id for f in findings}
    steps, loose = [], []
    parts = BASH_BLOCK.split(text)
    for i, part in enumerate(parts):
        if i % 2:
            steps += read_steps(prose, part, ids)
            continue
        rest = re.sub(r"(?ms)^- \*\*Prerequisites\*\*:.*?(?=^\S|\Z)", "", part).strip()
        if i == 0 and not prerequisites:
            needed, rest = prose_prerequisites(rest)
            for kind, things in needed.items():
                prerequisites[kind] = [prose.inline(t) for t in things]
        if rest:
            loose.append(prose.block(rest, capital=False))
    ordered = [
        (k, PREREQUISITE_HEADINGS.get(k, k), prerequisites[k])
        for k in (*PREREQUISITE_KINDS, "Other")
        if k in prerequisites
    ]

    # the digest names the prerequisite kinds and the steps done by hand; the count sits on the right
    numbered = [s for s in steps if not s["env"]]
    manual = sum(s["by_hand"] for s in numbered)
    digest = [", ".join(PREREQUISITE_NAMES.get(k, k.lower()) for k, _, _ in ordered)]
    digest += [f"{manual} by hand"] if manual else []
    return {
        "id": f"{prose.page_id}/reproduce",
        "meta": _count(len(numbered), "step"),
        "digest": " · ".join(d for d in digest if d),
        "prerequisites": ordered,
        "steps": steps,
        "numbered": numbered,
        "loose": loose,
        "concluded": prose.concluded,
    }


def prose_prerequisites(text: str) -> tuple[dict[str, list[str]], str]:
    """The `Needs …` or `Requires …` sentences of Reproduce prose as prerequisites by kind.

    Each such sentence is a list, split at its top-level commas; each item's
    kind follows `PREREQUISITE_SIGNS`. Other sentences stay as prose.

    Returns:
        The items by kind, and the prose left once the lists are taken out.
    """
    found: dict[str, list[str]] = {}
    kept = []
    for sentence in _code_safe_split(PROSE_SENTENCE, text.strip()):
        lead = NEEDS_LEAD.match(sentence)
        if not lead:
            kept.append(sentence)
            continue
        body = sentence[lead.end() :].rstrip(" .")
        for item in _code_safe_split(re.compile(r",\s+(?:and\s+)?|;\s+"), body):
            item = re.sub(r"^and\s+", "", item.strip())
            if not item:
                continue
            kind = next(
                (k for k, sign in PREREQUISITE_SIGNS if sign.search(item)), "Data"
            )
            found.setdefault(kind, []).append(item)
    return found, " ".join(kept)


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
        by_hand = bool(BY_HAND.search(text))
        text = re.sub(r"^by hand:\s*", "", text, flags=re.I)
        text = re.sub(r"^(?:step\s*)?\d+[.:)]?\s+", "", text, flags=re.I)
        commits = [s.strip() for s in checks.STEP_COMMIT.findall(text)]
        text = (
            re.sub(r"\s+([;,.])", r"\1", checks.STEP_COMMIT.sub("", text))
            .strip(" ;,")
            .replace(" -- ", " — ")
        )
        head = prose.inline(text)
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
                "commits": [
                    (c[:SHA_SHOWN], _github_commit(prose.github, c)) for c in commits
                ],
                "by_hand": by_hand,
                "code": body,
            }
        )
    return steps


def read_builds(model: Model, site: Site) -> dict:
    """The records that build on this one, each with what it found."""
    found = []
    for child in children(model, site):
        line = TLDR_STATUS.sub("", markup.unguided(child.row("TL;DR", "Hypothesis")))
        if not record_outcome(child) or not line:
            line = markup.unguided(child.row("Hypothesis", "Statement")) or line
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
    """The section bar: every section below it with its count, then the setup and Reproduce folds."""
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
        ("reproduce", "Reproduce"),
    ):
        if folds[key] or (key == "reproduce" and skeleton):
            n = len(folds[key]["numbered"]) if key == "reproduce" else 0
            bar.append({"href": f"#{page_id}/{key}", "name": name, "n": n})
    return bar


# ===============================
# Helper functions
# ===============================


def _exhibit(**values) -> dict:
    """An exhibit's values, every key present so the template can read each one."""
    return {
        "anchor": "",
        "label": "",
        "title": "",
        "caption": "",
        "html": "",
        "src": "",
        "kind": "",
        "shape": "",
        "width": "",
        "missing": "",
        "also": "",
    } | values


def _exhibit_kind(folder: Path, src: str) -> str:
    """`Table` for a CSV file, `Figure` for an image; "" for a file absent or not drawn."""
    path = folder / src
    if not path.is_file():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "Table"
    return "Figure" if suffix in FIGURE_SUFFIXES else ""


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


def _unbold(block: str) -> str:
    """A block with its bold lead as plain words; a lead followed by punctuation keeps it close."""
    lead, rest = _lead(block)
    if not lead:
        return block
    return lead + ("" if re.match(r"[,;:.)]", rest) else " ") + rest


def _lead(block: str) -> tuple[str, str]:
    match = LEAD.match(block.strip())
    return (match.group(1), match.group(2)) if match else ("", block)


def _count(n: int, noun: str, spelled: bool = False) -> str:
    word = COUNT_WORDS[n] if spelled and n < len(COUNT_WORDS) else str(n)
    return f"{word} {noun}{'' if n == 1 else 's'}"


def _value(value: str) -> str:
    """A reading's value in full: `22,539,297,662`, `5,941`, a long float to four decimals."""
    text = str(value).strip()
    if re.fullmatch(r"-?\d{4,}", text):
        return f"{int(text):,}"
    return markup.number(text)


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


def _rests_label(text: str) -> str:
    """A Limitations item cut to the words a rests-on label needs, citations dropped."""
    sentence = markup.first_sentence(text)
    if not re.search(r"\[[DM]\d+\]", sentence):
        sentence = text
    parts = re.split(r",\s+|;\s+", sentence)
    if len(sentence.split()) > LEAD_WORDS:
        sentence = next(
            (p for p in reversed(parts) if re.search(r"\[[DM]\d+\]", p)), parts[-1]
        )
    sentence = re.sub(r"^(?:and|but|while|so)\s+", "", sentence.strip())
    return BARE_CITATION.sub("", sentence).strip(" .;,")


def _said(text: str) -> bool:
    """Whether a row says something: not empty, guidance, or a bare `none`."""
    return markup.unguided(text).strip(" .").lower() not in ("", "none", "n/a")


def _trace_free(condition: str) -> str:
    """A condition without its trailing `([D<n>]; measured by [M<n>])`, which the trace shows."""
    return checks.CLAUSE_TRACE.sub("", condition).strip().rstrip(".").strip()


def _ticket(ticket, github: str) -> Markup | str:
    """The ticket as `issue #<n>`, linked to the GitHub issue when the repo is known."""
    text = str(ticket or "").strip()
    if not text:
        return INLINE.missing("no ticket")
    if re.match(r"https?://", text):
        return INLINE.link("issue", text)
    number = text.lstrip("#")
    if not number.isdigit():
        return text
    if github:
        return INLINE.link(f"issue #{number}", github_url(github, "issues", number))
    return f"issue #{number}"


def _pr(pr: str, github: str) -> Markup | str:
    """The PR as `PR #<n>`, linked to the GitHub pull request when the repo is known."""
    text = pr.strip()
    if re.match(r"https?://", text):
        match = re.search(r"/(?:pull|merge_requests)/(\d+)", text)
        return INLINE.link(f"PR #{match.group(1)}" if match else "PR", text)
    number = text.lstrip("#")
    if not number.isdigit():
        return text
    if github:
        return INLINE.link(f"PR #{number}", github_url(github, "pull", number))
    return f"PR #{number}"


def commit_chip(sha: str, github: str) -> Markup:
    """A commit, linked to its GitHub page when the repo is known."""
    return INLINE.commit(sha[:SHA_SHOWN], _github_commit(github, sha))


def _github_commit(github: str, sha: str) -> str:
    """The GitHub page of commit `sha` in the `owner/name` repo; "" when no repo is known."""
    return f"https://github.com/{github}/commit/{sha}" if github else ""


def github_url(github: str, kind: str, number: str) -> str:
    """The page of issue or pull request `number` in the `owner/name` repo."""
    return f"https://github.com/{github}/{kind}/{number}"


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


def _reading_point(reading: dict) -> dict | None:
    """One numeric row of clause-readings.csv as a point with its interval, bar and margin."""
    value = markup.to_float(reading.get("value", ""))
    if value is None:
        return None
    low = markup.to_float(reading.get("ci_low", ""))
    high = markup.to_float(reading.get("ci_high", ""))
    has_interval = low is not None and high is not None
    word = reading.get("outcome")
    return {
        "word": word if word in OUTCOME else "inconclusive",
        "qualifier": reading.get("qualifier", ""),
        "value": value,
        "low": low if has_interval else value,
        "high": high if has_interval else value,
        "bar": markup.to_float(reading.get("bar", "")),
        "margin": markup.to_float(reading.get("margin", "")),
        "text": _value(reading["value"]),
        "interval": f"[{_value(reading['ci_low'])}, {_value(reading['ci_high'])}]"
        if has_interval
        else "",
    }


def _at(value: float, lo: float, hi: float) -> str:
    return f"{(value - lo) / (hi - lo) * 100:.2f}%"


def _track(point: dict, lo: float, hi: float) -> dict:
    """A reading's point, interval, bar and margin band as positions along a track."""
    margin = point["margin"]
    return {
        "kind": OUTCOME[point["word"]],
        "low": _at(point["low"], lo, hi),
        "width": f"calc({_at(point['high'], lo, hi)} - {_at(point['low'], lo, hi)})",
        "point": _at(point["value"], lo, hi),
        "band": (
            _at(-margin, lo, hi),
            f"calc({_at(margin, lo, hi)} - {_at(-margin, lo, hi)})",
            _at(0, lo, hi),
        )
        if margin is not None
        else None,
        "bar": _at(point["bar"], lo, hi) if point["bar"] is not None else None,
    }


def _tick(value: float) -> str:
    """An axis label, each in its own unit as the board writes them: `5 B`, `250 k`, `0.05`."""
    for size, unit in ((BILLION, "B"), (MILLION, "M"), (THOUSAND, "k")):
        if abs(value) >= size:
            return f"{value / size:g}\u00a0{unit}"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:g}"


def _best_values(header: str, values: list[str]) -> set[str]:
    """The best value of an arms column, bold in the table; none for counts or no clear direction."""
    numbers = [markup.to_float(v) for v in values]
    if not values or any(n is None for n in numbers):
        return set()
    if all(WHOLE_NUMBER.match(v.strip()) for v in values) or NO_BEST.search(header):
        return set()
    lower = not HIGHER_BETTER.search(header) and bool(LOWER_BETTER.search(header))
    best = min(numbers) if lower else max(numbers)
    return {v for v, n in zip(values, numbers) if n == best}


def _interval_columns(keys: list[str]) -> dict[str, tuple[str, str]]:
    """Value column → its interval columns: `<q>_low/_high`, `<q>_ci_low/_ci_high`, or a low/high pair right after it."""
    bounds = {}
    for i, key in enumerate(keys):
        for low, high in INTERVAL_SUFFIXES:
            if f"{key}{low}" in keys and f"{key}{high}" in keys:
                bounds[key] = (f"{key}{low}", f"{key}{high}")
                break
        else:
            # `ci_low, ci_high` or an author's `F1 95% low, F1 95% high` right after it
            following = keys[i + 1 : i + 3]
            ends = [INTERVAL_END.match(k) for k in following]
            if (
                len(ends) == 2
                and all(ends)
                and ends[0].group(1) == ends[1].group(1)
                and (ends[0].group(2).lower(), ends[1].group(2).lower())
                == ("low", "high")
            ):
                bounds[key] = (following[0], following[1])
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


def _statistics_size(rows: list[dict], one_item: str, name: str = "") -> str:
    """The head size from a statistics table: the item column the one-item text names, summed.

    The column may be named in the singular (`one document` names
    `documents`). A row cut `all`, `total` or `overall` gives the size itself.
    A table that holds several datasets keeps the rows whose cell is the
    dataset name's first word. Rows are summed only when one text column keys
    them, so a cross table never counts an item twice; otherwise no size.
    """
    if not rows:
        return ""
    named = [
        key
        for key in rows[0]
        if re.search(rf"\b{re.escape(key.removesuffix('s'))}s?\b", one_item, re.I)
        and all(WHOLE_NUMBER.match(r.get(key, "") or "0") for r in rows)
    ]
    if not named:
        return ""
    key = named[0]
    first = next(iter(rows[0]))
    totals = [r for r in rows if r.get(first, "").lower() in TOTAL_ROWS]
    if totals:
        return f"{int(totals[0][key].replace(',', '')):,} {key}"

    # the rows of this dataset, when the table names it in a column
    word = name.split()[0].lower() if name.split() else ""
    columns = list(rows[0])
    for column in columns:
        mine = [r for r in rows if r.get(column, "").lower() == word]
        if mine and len(mine) < len(rows):
            rows = mine
            columns.remove(column)
            break
    texts = [
        c
        for c in columns
        if not all(
            WHOLE_NUMBER.match(r.get(c, "") or "0")
            or markup.to_float(r.get(c, "")) is not None
            for r in rows
        )
    ]
    if not any(len({r.get(c, "") for r in rows}) == len(rows) for c in texts):
        return ""
    count = sum(int((r.get(key) or "0").replace(",", "")) for r in rows)
    return f"{count:,} {key}"


def _stated_size(text: str, one_item: str) -> str:
    """The head size the row's prose states: `about 30 M documents`, `295 rows`; "" when none.

    The count must name a noun of the one-item text's first words (`one
    document` counts `documents`) or rows, at most two words after the number.
    """
    words = re.findall(r"[a-z]+", one_item.lower())[:ITEM_WORDS]
    nouns = {"rows", "items"} | {w.removesuffix("s") + "s" for w in words}
    for match in STATED_COUNT.finditer(text):
        if match.group("noun").lower() in nouns:
            about, count, scale = match.group("about", "count", "scale")
            shown = f"{count}\u00a0{SCALE_UNITS[scale.lower()]}" if scale else count
            return f"{about or ''}{shown} {match.group('noun')}".strip()
    return ""


def _code_safe_split(pattern: re.Pattern, text: str) -> list[str]:
    """`pattern.split(text)`, never inside a code span or brackets."""
    guarded = re.sub(
        r"`[^`]*`|\([^()]*\)", lambda m: re.sub(r"[^`()]", "x", m.group(0)), text
    )
    parts, start = [], 0
    for match in pattern.finditer(guarded):
        parts.append(text[start : match.start()])
        start = match.end()
    parts.append(text[start:])
    return [p for p in parts if p.strip()]


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
