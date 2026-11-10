"""Checks the record spec (`specs/record.yaml`) names beyond the built-in ones.

Most of them cross-reference one part of a record against another: a cited
`[D<n>]` against the Decisions entries, a clause's round against Rounds, a
Code row against the file it names. Each yields `(line, detail)` pairs, as
every check does (see `rules.check`).
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ..document import Document, Entry, normalize_link
from ..document import Row as DocumentRow
from ..rules import (
    CHECKS,
    GUIDANCE,
    INLINE_CODE,
    WORD,
    Corpus,
    check,
    is_guidance,
    is_template_guidance,
    row_places,
    split_code,
)
from ..ship import skill_references
from ..spec import Artifact, Field, Row, Rule, Section

# ===============================
# Defaults
# ===============================

# a template placeholder alone in inline code: `<parent-slug>`, or
# `<file>.py::<function>` with an optional `step <n>: ` lead; a one-word
# `<unk>` or a path part like `configs/<arm>.yaml` is the author's own
CODE_PLACEHOLDER = re.compile(
    r"^(?:step <n>: )?(?:<[a-z]+(?:[ -][a-z]+)+>|<[a-z]+>\.py::<[a-z]+>)$"
)

DATASET_ROW = re.compile(
    r"^- \*\*(?P<name>.+?)\*\*(?: \((?P<meta>[^)]*)\))?:\s*(?P<text>.*)$"
)
DATASET_KIND = re.compile(r"^(benchmark|external|own|split of \S.*|derived from \S.*)$")
DATASET_KINDS = "benchmark, split of <dataset>, external, own or derived from <dataset>"
LABEL_QUALITIES = ("gold", "silver", "none")
ONE_ITEM = "one item is "

# the entry heading of each entries section; a finding also carries its weight
DASH = r"\s*[—–-]\s*"
ENTRY_HEADINGS = {
    "Rounds": (re.compile(rf"^(R\d+){DASH}\S"), "R<n> — <what the round tests>"),
    "Methods": (re.compile(rf"^(M\d+){DASH}\S"), "M<n> — <what the step does>"),
    "Decisions": (re.compile(rf"^(D\d+){DASH}\S"), "D<n> — <the choice made>"),
    "Findings": (
        re.compile(rf"^(F\d+){DASH}.+? · (?:key|supporting F\d+|minor)$"),
        "F<n> — <title> · key | supporting F<k> | minor",
    ),
}
ENTRY_ID = re.compile(r"^([A-Z]\d+)\b")
ENTRY_TITLE = re.compile(rf"^[A-Z]\d+{DASH}(.+)$")
# a decision or method title shorter than this names a topic, not the choice or the step
TITLE_MIN_WORDS = 4
WEIGHT = re.compile(r" · (key|supporting|minor)(?: (F\d+))?$")
MINOR_ROWS = ("Summary", "Runs", "Result", "Reading", "History")
# a decision's History line: `(before runs)`, `(after runs)` or `(after [F2], [F3])`,
# each with an optional `; user`
TIMING = re.compile(
    r"\((?:before runs|after runs|after \[F\d+\](?:,\s*\[F\d+\])*)(?:;\s*user)?\)"
)

# a clause line under Predictions: `- **H2 · diagnostic · measured · R2 — <claim>**: …`
CLAUSE_LINE = re.compile(r"^\*\*([HCQ]\d+)((?: · [^*—]+?)*) — ([^*]+)\*\*:")
CLAUSE_TAG = re.compile(r"decisive|diagnostic|tested|measured|R\d+")
VERDICT_RULE = re.compile(r"^\*\*Verdict rule\*\*:")
# the category folder of records that build data rather than vary a method
DATA_CATEGORY = "data"
# a bar set by a decision of this record, `[D3]`, or of another, `[<slug> D3]`
RECORD_DECISION = re.compile(r"\[([a-z0-9][a-z0-9-]*) (D\d+)\]")
CLAUSE_TRACE = re.compile(
    r"\((?:\[D\d+\]|\[[a-z0-9][a-z0-9-]* D\d+\])[^()]*;\s*measured by \[M\d+\][^()]*\)"
)
NOTE_ADDED = re.compile(r"Note added(?! \d{4}-\d{2}-\d{2}:)")
SUPERSEDED = re.compile(r"\b([HCQ]\d+) superseded by ([HCQ]\d+)\b")
STANDS = re.compile(r"\b([HCQ]\d+) stands\b")
# the outcome words a reading may give a clause, by clause type
OUTCOME_WORDS = {
    "H": ("inconclusive", "refuted", "confirmed"),
    "C": ("undecided", "equivalent", "worse", "better"),
    "Q": ("unanswered", "partial", "answered"),
    "measured": ("unanswered", "answered"),
}
OUTCOME_STEMS = {"confirmed": "confirm", "refuted": "refut"}
# by clause letter: the two verdicts a split holds, and the record Outcomes
# that name one side, so a split under them should read `partial`
SPLIT_SIDES = {
    "H": (("confirmed", "refuted"), ("confirmed", "refuted")),
    "C": (("better", "worse"), ("adopted", "rejected")),
}
CLAUSE_NAME = re.compile(r"\[?\b[HCQ]\d+\b")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?]\*\*)\s+|\n+")
ABBREVIATION = re.compile(
    r"\b(e\.g|i\.e|vs|et al|cf|fig|no|approx|ca)\.", re.IGNORECASE
)

# citations: `[D3]`, `[H1]`, and `[<slug>:F2]` for another record's finding or clause
CITATION = {
    "D": re.compile(r"\[(D\d+)\]"),
    "clause": re.compile(r"\[([HCQ]\d+)\]"),
    "F": re.compile(r"\[(F\d+)\]"),
    "M": re.compile(r"\[(M\d+)\]"),
}
CITED_SECTION = {"D": "Decisions", "F": "Findings", "M": "Methods"}
# `([D10] there)` cites the entry of another record the sentence names
ELSEWHERE = re.compile(r"\s+there\)")
RECORD_CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*):(F\d+|[HCQ]\d+)\]")
RELATION = re.compile(r"\b(?:qualifies|supports|overturns)\s+\(?$")
BIB_CITATION = re.compile(r"\[[^\]\n]*@[^\]\n]*\]")
BIB_KEY = re.compile(r"@([\w:.\-]+)")
BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
BIBLIOGRAPHY = "references.bib"

# a TL;DR line that is no `- **Label**:` row is a key finding's line
TLDR_KEY_LINE = re.compile(r"^\*\*F\d+ — [^*\n]+?[.!?]\*\* \S")
ROW_LINE = re.compile(r"^- \*\*[^*]+\*\*:")
# a finding reads cold, so it cites no `#<n>` ticket
TICKET = re.compile(r"(?<![\w`])#\d+\b")
TICKET_ROWS = ("Summary", "Reading", "Implication")
# a finding titled as a fix is a History line on the entry it corrects
FIX_WORDS = re.compile(
    r"\b(bug|bugs|defect|defects|fix|fixed|fixes|rewrite|rewritten|typo|notebook)\b",
    re.IGNORECASE,
)
# a `[` inside an image's caption ends the link early, so markdown draws no image
CAPTION_BRACKET = re.compile(r"!\[[^\]]*\[")

# past this many words a prose row is bold-lead blocks, and a How numbered steps
BLOCK_WORDS = 60
LEAD_BLOCK = re.compile(r"^\s+- \*\*[^*\n]+\*\*")
NUMBERED_STEP = re.compile(r"^\s*\d+\. ")
NEEDS_ROWS = ("Compute", "People", "Data access", "External dependency")

# a Code row names `<file>.py` or `<file>.py::<symbol>`
CODE_REFERENCE = re.compile(r"([\w./-]+\.py)(?:::(\w+))?")
FILE_EXTENSION = re.compile(r"\.(ya?ml|py|csv|json|svg|png|md|txt|toml)(::\w+)?$")
SETTINGS_KEY = re.compile(r"`([^`]+)`(\s*(?:\(|in\b|:)|)")
# a run id is MLflow's full 32-character hex; a commit follows `at` or `commit`,
# or is a short sha of 7 to 12 characters or a full one of 40
HEX = re.compile(r"(?<![\w-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,64}(?![\w-])")
COMMIT_LEAD = re.compile(r"\b(?:at|commit)\s+`?$")
RUN_ID_LENGTH = 32
# a run whose tracking server is gone, by the prefix the record kept:
# `run <hex> (server lost)`, or `runs <hex>, <hex> and <hex> (server lost)`
LOST_HEX = r"`?[0-9a-f]{7,31}`?"
LOST_RUNS = re.compile(
    rf"\bruns? ({LOST_HEX}(?:(?:,\s*|,?\s+and\s+){LOST_HEX})*)\s*\(server lost\)"
)
SHORT_SHA_LENGTHS = range(7, 13)
FULL_SHA_LENGTH = 40
SHA = re.compile(r"^[0-9a-f]{7,40}$")
STEP_COMMIT = re.compile(r"\(commit ([^)]*)\)")

ADOPTION_ROWS = ("Carries", "Stays out")
NEXT_ENDING = re.compile(
    r"(?:— planned as [\w-]+(?: #\d+)?|\(issue #\d+\)|— not yet planned"
    r"|— done in (?:`[0-9a-f]{7,40}`|[a-z0-9][a-z0-9-]*))[.;]?$"
)
PREREQUISITE = re.compile(r"^\*\*(Secret|Login|Data|Service|Frozen)\*\*")
SIDE_FOLDERS = ("configs", "prompts")
# a lock file is generated from the config the record names
LOCK_FILE = re.compile(r"\.lock\.ya?ml$")

# the table rules live in labflow:analyze's table_rules.py, beside the report
# skill this package is in or under the home folder; a repo's copy is never run
ANALYZE_TABLE_RULES = Path("analyze") / "references" / "table_rules.py"
PACKAGE_REFERENCES = Path(__file__).resolve().parents[2]
CLAUSE_READINGS = "clause-readings"
VERDICT_BY_ARM = "verdict-by-arm"
# the arm-table rule every `*-by-arm.csv` takes, beside the verdict table's own rules
ARM_FIRST = "TAB-050"
ADOPTED = "adopted"
WARNING = "labflow_report check: warning:"

# ===============================
# Clause class
# ===============================


@dataclass(frozen=True)
class Clause:
    """One clause line of Predictions, with its tags defaulted as the template says."""

    id: str
    line: int
    text: str
    role: str
    standard: str
    round: str


# ===============================
# Template guidance
# ===============================


@check("template-guidance")
def template_guidance(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Guidance left in a section: `<…>` outside inline code (the `guidance`
    check), else a template placeholder alone in inline code."""
    found = list(CHECKS["guidance"](document, part, rule, corpus))
    yield from found or code_placeholders(document, part.heading)


@check("added-section-guidance")
def added_section_guidance(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Guidance left in a `##` section the document's spec does not declare."""
    declared = {section.heading for section in corpus.specs[document.kind].sections}
    for heading in document.sections:
        if heading not in declared:
            yield from template_guidance(document, Section(heading), rule, corpus)


# ===============================
# Datasets
# ===============================


@check("dataset-row")
def dataset_row(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each Datasets row: `- **<name>** (<kind>; <quality> labels): one item is …`."""
    section = document.sections.get(part.heading)
    if section is None:
        return
    for number, line in enumerate(section.body, start=section.line + 1):
        if not line.startswith("- **"):
            continue
        match = DATASET_ROW.match(line)
        if match is None or match.group("meta") is None:
            yield number, "no (<kind>; <labels> labels) after the name"
            continue

        # the kind, then the label quality, then the one-item sentence
        kind, _, labels = (text.strip() for text in match.group("meta").partition(";"))
        if not DATASET_KIND.match(kind):
            yield number, f"kind '{kind}' is not {DATASET_KINDS}"
        quality, _, word = labels.rpartition(" ")
        if word != "labels" or quality not in LABEL_QUALITIES:
            yield number, f"'{labels}' is not gold, silver or none labels"
        if not match.group("text").startswith(ONE_ITEM):
            yield number, f"the row does not open '{ONE_ITEM.strip()}'"


# ===============================
# Citations
# ===============================


def _cited(kind: str):
    """A check that each citation of one kind names an entry or clause the record has."""

    def cited(
        document: Document, part, rule: Rule, corpus: Corpus
    ) -> Iterator[tuple[int, str]]:
        known = (
            {c.id for c in clauses(document)}
            if kind == "clause"
            else entry_ids(document, CITED_SECTION[kind])
        )
        reported = set()
        for number, text in prose_lines(document):
            for match in CITATION[kind].finditer(text):
                cited_id = match.group(1)
                if text[: match.start()].endswith("(") and ELSEWHERE.match(
                    text, match.end()
                ):
                    continue
                if cited_id not in known and cited_id not in reported:
                    reported.add(cited_id)
                    yield number, f"[{cited_id}] names no {cited_id}"

    return cited


check("cited-decision")(_cited("D"))
check("cited-clause")(_cited("clause"))
check("cited-finding")(_cited("F"))
check("cited-method")(_cited("M"))


@check("cited-record")
def cited_record(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each `[<slug>:F<n>]` names a record with that slug, holding that finding or clause."""
    records = same_kind(document, corpus)
    for number, text in prose_lines(document):
        for match in RECORD_CITATION.finditer(text):
            slug, cited_id = match.groups()
            other = records.get(slug)
            if other is None:
                yield number, f"[{slug}:{cited_id}] names no record '{slug}'"
                continue
            known = (
                entry_ids(other, "Findings")
                if cited_id.startswith("F")
                else {c.id for c in clauses(other)}
            )
            if cited_id not in known:
                yield number, f"[{slug}:{cited_id}] names no {cited_id} in {slug}"


@check("relation-word")
def relation_word(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A `[<slug>:F<n>]` in the section follows qualifies, supports or overturns."""
    for number, text in prose_lines(document, part.heading):
        for match in RECORD_CITATION.finditer(text):
            if not RELATION.search(text[: match.start()]):
                yield number, f"{match.group(0)} follows no relation word"


@check("bibliography")
def bibliography(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each `[@key]` has an entry in the record's references.bib."""
    path = document.folder / BIBLIOGRAPHY
    entries = (
        set(BIB_ENTRY.findall(path.read_text(encoding="utf-8")))
        if path.is_file()
        else set()
    )
    reported = set()
    for number, text in prose_lines(document):
        for citation in BIB_CITATION.findall(text):
            for key in BIB_KEY.findall(citation):
                key = key.rstrip(".:")
                if key not in entries and key not in reported:
                    reported.add(key)
                    yield number, f"'@{key}' has no entry in {BIBLIOGRAPHY}"


# ===============================
# Frontmatter and Hypothesis
# ===============================


@check("mlflow-for-runs")
def mlflow_for_runs(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A record whose findings cite run ids names its MLflow experiment."""
    if document.frontmatter.get("mlflow"):
        return
    for entry in entries(document, "Findings"):
        row = entry.rows.get("Runs")
        if row and any(not is_commit for _, is_commit in hex_ids(row.text)):
            yield (
                document.key_lines.get("mlflow", 1),
                f"{entry_id(entry)} cites run ids",
            )
            return


@check("builds-on")
def builds_on(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The Builds on row has one bullet per `builds_on` parent, in that order."""
    parents = document.frontmatter.get("builds_on")
    if not isinstance(parents, list) or not parents:
        return
    slugs = [normalize_link(str(p)).rsplit("/", 1)[-1] for p in parents]
    section = document.sections.get(part.heading)
    row = section.rows.get(part.label) if section else None
    if row is None:
        yield document.key_lines["builds_on"], "builds_on is set but no Builds on row"
        return
    leads = [
        match.group(1)
        for _, text in items(row)
        if (match := re.match(r"^`([^`]+)`", text))
    ]
    missing = [slug for slug in slugs if slug not in leads]
    if missing:
        yield row.line, f"no bullet opens `{missing[0]}`"
    elif [lead for lead in leads if lead in slugs] != slugs:
        yield row.line, f"bullets are not in builds_on order {slugs}"


@check("varies-due")
def varies_due(
    document: Document, part: Field, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Once reference/ holds a topic or an entry, a methods or validation
    record names what it varies; a data record builds data and varies none."""
    if not (corpus.ids.get("topic") or corpus.ids.get("entry")):
        return
    if document.path.parent.parent.name == DATA_CATEGORY:
        return
    if not document.frontmatter.get(part.key):
        yield document.key_lines.get(part.key, 1), ""


@check("question-mark")
def question_mark(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row and not unwritten(row.text) and not row.text.rstrip().endswith("?"):
            yield row.line, ""


@check("clause-lines")
def clause_lines(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    row = predictions(document)
    if row and not unwritten(row.text) and not clauses(document):
        yield row.line, ""


@check("verdict-rule")
def verdict_rule(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Several decisive clauses still standing need a `- **Verdict rule**:` line."""
    row = predictions(document)
    gone = superseded(document)
    deciding = [
        c.id for c in clauses(document) if c.role == "decisive" and c.id not in gone
    ]
    if len(deciding) > 1 and not any(VERDICT_RULE.match(t) for _, t in items(row)):
        yield row.line, f"decisive clauses {', '.join(deciding)}"


@check("clause-trace")
def clause_trace(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each clause line ends with its trace; a `[<slug> D<n>]` in it names a
    decision another record holds."""
    records = same_kind(document, corpus)
    for clause in clauses(document):
        if unwritten(clause.text):
            continue
        trace = CLAUSE_TRACE.search(clause.text)
        if trace is None:
            yield clause.line, clause.id
            continue
        for slug, decision in RECORD_DECISION.findall(trace.group(0)):
            other = records.get(slug)
            if other is None:
                yield clause.line, f"[{slug} {decision}] names no record '{slug}'"
            elif decision not in entry_ids(other, "Decisions"):
                yield clause.line, f"[{slug} {decision}] names no {decision} in {slug}"


@check("note-added")
def note_added(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for number, text in prose_lines(document, part.heading):
        if NOTE_ADDED.search(text):
            yield number, ""


@check("outcome-agrees")
def outcome_agrees(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """The Hypothesis Outcome word is the Verdict Outcome's first word."""
    words = {}
    for heading in ("Hypothesis", "Verdict"):
        section = document.sections.get(heading)
        row = section.rows.get("Outcome") if section else None
        if row is None or unwritten(row.text):
            return
        words[heading] = (row, lead_word(row.text))
    (hypothesis, said), (_, verdict) = words["Hypothesis"], words["Verdict"]
    if said != verdict:
        yield hypothesis.line, f"Hypothesis '{said}', Verdict '{verdict}'"


@check("split-outcome")
def split_outcome(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Decisive clauses of one letter that split between yes and no read
    `partial`, never the Outcome word of one side, unless the Verdict rule
    written before the runs gives that word when any one clause does."""
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text):
            continue
        said = lead_word(row.text)
        rule_lines = [t for _, t in items(predictions(document)) if VERDICT_RULE.match(t)]
        if said and any(re.search(rf"\b{re.escape(said)}\s+(?:when|if)\s+any\b", t) for t in rule_lines):
            continue
        gone = superseded(document)
        keys = [
            entry
            for entry in entries(document, "Findings")
            if (match := WEIGHT.search(entry.title)) and match.group(1) == "key"
        ]
        deciding = [
            c for c in clauses(document) if c.role == "decisive" and c.id not in gone
        ]
        for letter, (sides, one_sided) in SPLIT_SIDES.items():
            if said not in one_sided:
                continue

            # each clause's verdict is the word its first reading finding gives it
            verdicts = {
                c.id: next((word for e in keys if (word := outcome(e, c))), "")
                for c in deciding
                if c.id.startswith(letter)
            }
            if all(side in verdicts.values() for side in sides):
                split = ", ".join(f"{cid} {word}" for cid, word in verdicts.items() if word)
                yield row.line, f"{split} under Outcome '{said}'"


# ===============================
# TL;DR
# ===============================


@check("tldr-key-line")
def tldr_key_line(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each top-level bullet that is no row is a key finding's line; a line
    still holding `<…>` guidance waits for the guidance rule."""
    section = document.sections.get(part.heading)
    if section is None:
        return

    # join each top-level bullet with its continuation lines
    bullets: list[list] = []
    for number, text in enumerate(section.body, start=section.line + 1):
        if unwritten(text):
            continue
        if text.startswith("- "):
            bullets.append([number, text])
        elif bullets and text.startswith("  ") and text.strip():
            bullets[-1][1] += " " + text.strip()
    for number, text in bullets:
        if not ROW_LINE.match(text) and not TLDR_KEY_LINE.match(text[2:].strip()):
            yield number, f"'{text[2:52].strip()}'"


# ===============================
# Rounds
# ===============================


@check("entry-heading")
def entry_heading(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each `###` entry opens with its id and a dash, numbers never repeat."""
    pattern, form = ENTRY_HEADINGS[part.heading]
    seen = set()
    for entry in entries(document, part.heading):
        match = pattern.match(entry.title)
        if match is None:
            yield entry.line, f"'{entry.title[:50]}' is not `### {form}`"
        elif match.group(1) in seen:
            yield entry.line, f"{match.group(1)} is used twice"
        else:
            seen.add(match.group(1))


@check("title-words")
def title_words(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each entry's title after its id is at least TITLE_MIN_WORDS words."""
    for entry in entries(document, part.heading):
        match = ENTRY_TITLE.match(entry.title)
        if match is None or unwritten(match.group(1)):
            continue
        count = len(match.group(1).split())
        if count < TITLE_MIN_WORDS:
            yield entry.line, f"{entry_id(entry)} '{match.group(1)}' is {count} words"


@check("clause-round")
def clause_round(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    rounds = entry_ids(document, part.heading)
    if not rounds:
        return
    for clause in clauses(document):
        if clause.round not in rounds:
            yield clause.line, f"{clause.id} belongs to {clause.round}"


@check("motivated-by")
def motivated_by(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A round after the first cites the finding that called for it."""
    for entry in entries(document, part.heading)[1:]:
        row = entry.rows.get("Motivated by")
        if row and unwritten(row.text):
            continue
        if row is None or not CITATION["F"].search(row.text):
            yield (row.line if row else entry.line), entry_id(entry)


@check("earlier-decisive")
def earlier_decisive(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A later round labels each earlier decisive clause `stands` or `superseded by`."""
    rounds = [
        (match.group(1), entry)
        for entry in entries(document, part.heading)
        if (match := ENTRY_ID.match(entry.title))
    ]
    order = [round_id for round_id, _ in rounds]
    known = {c.id for c in clauses(document)}
    for index, (round_id, entry) in enumerate(rounds):
        if index == 0:
            continue
        row = entry.rows.get("Earlier decisive")
        if row and unwritten(row.text):
            continue
        text = row.text if row else ""
        line = row.line if row else entry.line
        labelled = set(STANDS.findall(text)) | {c for c, _ in SUPERSEDED.findall(text)}
        earlier = [
            c.id
            for c in clauses(document)
            if c.role == "decisive"
            and c.round in order
            and order.index(c.round) < index
        ]
        for clause_id in earlier:
            if clause_id not in labelled:
                yield line, f"{round_id} leaves {clause_id} unlabelled"
        for clause_id, by in SUPERSEDED.findall(text):
            if by not in known:
                yield line, f"{clause_id} superseded by {by}, which is not a clause"


# ===============================
# Design and Methods
# ===============================


@check("needs-row")
def needs_row(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each needs row opens with a short value, then ` — ` and a sentence."""
    section = document.sections.get(part.heading)
    for label in NEEDS_ROWS:
        row = section.rows.get(label) if section else None
        if row and not unwritten(row.text) and " — " not in row.text:
            yield row.line, label


@check("known-rows")
def known_rows(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each entry holds only the rows the spec declares for its section."""
    declared = {row.label for row in part.rows}
    for entry in entries(document, part.heading):
        for label, row in entry.rows.items():
            if label not in declared and "<" not in label:
                yield row.line, f"'{label}' in {entry_id(entry)}"


@check("numbered-steps")
def numbered_steps(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row and is_long(row) and not any(map(NUMBERED_STEP.match, row.body)):
            yield row.line, f"{len(WORD.findall(row.text))} words in one paragraph"


@check("lead-blocks")
def lead_blocks(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row and is_long(row) and not any(map(LEAD_BLOCK.match, row.body)):
            yield row.line, f"{len(WORD.findall(row.text))} words in one paragraph"


@check("code-row")
def code_row(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A Code row names a file that exists and, where it names a symbol, one it defines."""
    bases = (document.folder, corpus.experiments, corpus.experiments.parent)
    for _, row in row_places(document, part):
        # the template's `<file>.py::<function>` sits in inline code
        if row is None or unwritten(row.text) or code_guidance(row.text):
            continue
        references = CODE_REFERENCE.findall(row.text)
        if not references:
            yield row.line, "no `<file>.py` or `<file>.py::<symbol>`"
        for path, symbol in references:
            found = next((b / path for b in bases if (b / path).is_file()), None)
            if found is None:
                yield row.line, f"{path} does not exist"
            elif symbol and not re.search(
                rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b",
                found.read_text(encoding="utf-8"),
            ):
                yield row.line, f"{path} defines no {symbol}"


@check("settings-gloss")
def settings_gloss(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each Settings key carries a gloss: `` `clusters` (how many groups) ``."""
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text) or lead_word(row.text) == "none":
            continue
        for match in SETTINGS_KEY.finditer(row.text):
            key, glossed = match.groups()
            # a name inside a parenthesis is part of a gloss: an arm, a value
            before = row.text[: match.start()]
            if before.count("(") > before.count(")"):
                continue
            # a file or a section is the place the keys live, not a key
            if not ("/" in key or FILE_EXTENSION.search(key) or glossed):
                yield row.line, f"`{key}`"


# ===============================
# Decisions and Findings
# ===============================


@check("history-timing")
def history_timing(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each History line says when: `(before runs)`, `(after runs)` or `(after [F<n>])`."""
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text):
            continue
        for number, text in items(row) or [(row.line, row.text)]:
            if not TIMING.search(text):
                yield number, f"'{text[:40]}'"


@check("supporting-target")
def supporting_target(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    keys = {entry_id for entry_id, weight, _ in weights(document) if weight == "key"}
    for entry in entries(document, part.heading):
        match = WEIGHT.search(entry.title)
        if match and match.group(1) == "supporting" and match.group(2) not in keys:
            yield entry.line, f"{match.group(2)} is not a key finding"


@check("minor-rows")
def minor_rows(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for entry in entries(document, part.heading):
        match = WEIGHT.search(entry.title)
        extra = [label for label in entry.rows if label not in MINOR_ROWS]
        if match and match.group(1) == "minor" and extra:
            yield entry.line, ", ".join(extra)


@check("finding-ticket")
def finding_ticket(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for entry in entries(document, part.heading):
        for label in TICKET_ROWS:
            row = entry.rows.get(label)
            if row is None or unwritten(row.text):
                continue
            if found := TICKET.findall(row.text):
                yield row.line, f"{entry_id(entry)} {label} cites {', '.join(found)}"


@check("fix-words")
def fix_words(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A finding's title and Summary use no word that names a fix."""
    for entry in entries(document, part.heading):
        summary = entry.rows.get("Summary")
        texts = [entry.title] + ([summary.text] if summary else [])
        found = {
            word.lower()
            for text in texts
            if not unwritten(text)
            for word in FIX_WORDS.findall(text)
        }
        if found:
            yield entry.line, f"{entry_id(entry)}: {', '.join(sorted(found))}"


@check("caption-bracket")
def caption_bracket(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row and CAPTION_BRACKET.search(row.text):
            yield row.line, ""


@check("key-reading")
def key_reading(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A key Reading cites a clause in a sentence that gives its outcome word."""
    standing = {c.id: c for c in clauses(document)}
    if not standing:
        return
    for entry in entries(document, part.heading):
        match = WEIGHT.search(entry.title)
        if match is None or match.group(1) != "key" or "Reading" not in entry.rows:
            continue
        text = reading_text(entry)
        cited = [c for c in CITATION["clause"].findall(text) if c in standing]
        if not cited:
            yield entry.line, "it cites no [H<n>], [C<n>] or [Q<n>]"
        elif not any(outcome(entry, standing[c]) for c in cited):
            yield entry.line, f"no sentence citing {', '.join(cited)} says its outcome"


@check("clauses-read")
def clauses_read(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each clause no later round superseded is read by a key finding; a
    diagnostic clause may be read by a supporting finding instead."""
    gone = superseded(document)
    readers = {
        weight: [
            entry
            for entry in entries(document, "Findings")
            if (match := WEIGHT.search(entry.title)) and match.group(1) == weight
        ]
        for weight in ("key", "supporting")
    }
    for clause in clauses(document):
        if clause.id in gone or unwritten(clause.text):
            continue
        allowed = readers["key"] + (
            readers["supporting"] if clause.role == "diagnostic" else []
        )
        if not any(outcome(e, clause) for e in allowed):
            yield clause.line, clause.id


@check("run-ids")
def run_ids(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Runs name full 32-character run ids, the commit a script ran at, or
    runs whose tracking server is lost by their prefix."""
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text):
            continue
        found = hex_ids(row.text)
        if not found and not lost_runs(row.text):
            yield row.line, "no run id and no `at <sha>`"
        for value, is_commit in found:
            if not is_commit and len(value) != RUN_ID_LENGTH:
                yield row.line, f"'{value}' is not a full {RUN_ID_LENGTH}-character id"


# ===============================
# Verdict and Reproduce
# ===============================


@check("adoption-rows")
def adoption_rows(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text) or lead_word(row.text) == "none":
            continue
        labels = {m.group(1) for line in row.body if (m := _nested_label(line))}
        missing = [label for label in ADOPTION_ROWS if label not in labels]
        if missing:
            yield row.line, f"no {' or '.join(missing)} row"


@check("caveat-cites")
def caveat_cites(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row and not unwritten(row.text) and not re.search(r"\[[DF]\d+\]", row.text):
            yield row.line, ""


@check("next-items")
def next_items(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each Next item ends planned as, (issue #n), not yet planned or done in."""
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text):
            continue
        for number, text in items(row) or [(row.line, row.text)]:
            if not NEXT_ENDING.search(text.rstrip()):
                yield number, f"'{text[:40]}'"


@check("prerequisite-kind")
def prerequisite_kind(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text):
            continue
        for number, text in items(row):
            if not PREREQUISITE.match(text):
                yield number, f"'{text[:40]}'"


@check("step-commit")
def step_commit(
    document: Document, part: Section, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A Reproduce step's `(commit <sha>)` is a sha."""
    section = document.sections.get(part.heading)
    if section is None:
        return
    _, fenced = split_code(section.body)
    for number, line in enumerate(fenced.split("\n"), start=section.line + 1):
        for value in STEP_COMMIT.findall(line):
            # a `<sha>` placeholder is the guidance rule's to report
            if not SHA.match(value.strip()) and not is_placeholder(value.strip()):
                yield number, f"'{value}'"


@check("side-files-named")
def side_files_named(
    document: Document, part, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Each file under configs/ or prompts/ beside the record is named in it."""
    text = document.path.read_text(encoding="utf-8")
    for folder in SIDE_FOLDERS:
        for path in sorted((document.folder / folder).rglob("*")):
            if (
                path.is_file()
                and not LOCK_FILE.search(path.name)
                and path.name not in text
            ):
                yield 1, f"{path.relative_to(document.folder).as_posix()}"


# ===============================
# Tables
# ===============================


@check("clause-readings-table")
def clause_readings_table(
    document: Document, part: Artifact, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written clause readings table passes the clause readings rules of tables.md
    (TAB-040..044), as save_table runs them."""
    yield from table_breaches(document, part, rule, CLAUSE_READINGS)


@check("verdict-by-arm-table")
def verdict_by_arm_table(
    document: Document, part: Artifact, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A written verdict table passes the arm and verdict table rules of tables.md
    (TAB-050..054), as save_table runs them."""
    yield from table_breaches(document, part, rule, VERDICT_BY_ARM, (ARM_FIRST,))


@check("adopted-arm")
def adopted_arm(
    document: Document, part: Artifact, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """A record whose Outcome is adopted marks exactly one arm `adopted` = `yes`
    in its verdict table, so the overview ledger names the adopted arm."""
    row = next(
        (
            section.rows["Outcome"]
            for heading in ("Verdict", "Hypothesis")
            if (section := document.sections.get(heading)) and "Outcome" in section.rows
        ),
        None,
    )
    if row is None or unwritten(row.text) or lead_word(row.text) != ADOPTED:
        return
    path = document.folder / part.pattern
    if not path.is_file():
        yield row.line, f"Outcome is adopted, and {part.pattern} is missing"
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    adopted = [r for r in rows if (r.get("adopted") or "").strip() == "yes"]
    if len(adopted) != 1:
        yield row.line, f"Outcome is adopted, and {part.pattern} marks {len(adopted)} arms adopted = yes"


# ===============================
# Helper functions
# ===============================


def table_breaches(
    document: Document, part: Artifact, rule: Rule, kind: str, extra: tuple[str, ...] = ()
) -> Iterator[tuple[int, str]]:
    """The breaches of a standard table's own rules, and of `extra`, in each written copy.

    The rules are labflow:analyze's table_rules.py, which save_table runs:
    `check_table` runs them, and `KIND_CHECKS` names the kind's own.
    """
    tables = sorted(document.folder.glob(part.pattern))
    if not tables:
        return
    assets = table_rules(rule.id)
    if assets is None:
        return
    kept = {tab for tab, _ in assets.KIND_CHECKS[kind]} | set(extra)
    for path in tables:
        for breach in assets.check_table(path):
            if breach.rule in kept:
                yield 1, str(breach)


# labflow:analyze's table_rules.py by path, loaded once a run
_TABLE_RULES: dict[Path, ModuleType] = {}
# the warnings already given, so a run warns once
_WARNED: set[str] = set()


def table_rules(rule_id: str) -> ModuleType | None:
    """labflow:analyze's own table_rules.py as a module; a repo's copy is never run.

    It uses the standard library only, so it loads wherever the checker runs.
    The copy beside the report skill this package is in comes first, then the
    one under the home folder. When neither exists, one warning goes to
    stderr and the rule checks nothing.

    Args:
        rule_id: The rule asking, named in the warning.

    Returns:
        The loaded module, or None when no copy exists.
    """
    candidates = [
        references.parent.parent / ANALYZE_TABLE_RULES
        for references in (PACKAGE_REFERENCES, skill_references())
    ]
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if path is None:
        reason = f"no table rules at {' or '.join(map(str, candidates))}"
        if reason not in _WARNED:
            _WARNED.add(reason)
            print(f"{WARNING} {rule_id} checks no table: {reason}", file=sys.stderr)
        return None

    # load it once; a dataclass looks its module up in sys.modules while the file runs
    if path not in _TABLE_RULES:
        name = f"_labflow_table_rules_{len(_TABLE_RULES)}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _TABLE_RULES[path] = module
    return _TABLE_RULES[path]


def entries(document: Document, heading: str) -> tuple[Entry, ...]:
    section = document.sections.get(heading)
    return section.entries if section else ()


def entry_id(entry: Entry) -> str:
    match = ENTRY_ID.match(entry.title)
    return match.group(1) if match else entry.title[:20]


def entry_ids(document: Document, heading: str) -> set[str]:
    return {
        match.group(1)
        for entry in entries(document, heading)
        if (match := ENTRY_ID.match(entry.title))
    }


def same_kind(document: Document, corpus: Corpus) -> dict[str, Document]:
    """The documents of the corpus of the document's kind, by slug."""
    return {
        str(d.frontmatter.get("slug") or d.folder.name): d
        for d in corpus.documents
        if d.kind == document.kind
    }


def weights(document: Document) -> list[tuple[str, str, str]]:
    """(id, weight, target) of each finding whose heading carries a weight."""
    return [
        (ENTRY_ID.match(entry.title).group(1), match.group(1), match.group(2) or "")
        for entry in entries(document, "Findings")
        if ENTRY_ID.match(entry.title) and (match := WEIGHT.search(entry.title))
    ]


def items(row: DocumentRow | None) -> list[tuple[int, str]]:
    """(line, text) of each first-level nested bullet under a row, continuation joined."""
    if row is None:
        return []
    found, indent = [], None
    for offset, line in enumerate(row.body[1:], start=1):
        stripped = line.lstrip()
        depth = len(line) - len(stripped)
        if stripped.startswith("- ") and (indent is None or depth <= indent):
            indent = depth
            found.append([row.line + offset, stripped[2:]])
        elif found and stripped:
            found[-1][1] += " " + stripped
    return [(line, text) for line, text in found]


def predictions(document: Document) -> DocumentRow | None:
    section = document.sections.get("Hypothesis")
    return section.rows.get("Predictions") if section else None


def clauses(document: Document) -> list[Clause]:
    """The clause lines of Predictions, each with its role, standard and round."""
    found = []
    for line, text in items(predictions(document)):
        match = CLAUSE_LINE.match(text)
        if match is None:
            continue
        tags = CLAUSE_TAG.findall(match.group(2))
        found.append(
            Clause(
                match.group(1),
                line,
                text,
                "diagnostic" if "diagnostic" in tags else "decisive",
                "measured" if "measured" in tags else "tested",
                next((tag for tag in tags if tag.startswith("R")), "R1"),
            )
        )
    return found


def superseded(document: Document) -> dict[str, str]:
    """Clause id → the clause that replaced it, from every round's Earlier decisive."""
    gone = {}
    for entry in entries(document, "Rounds"):
        row = entry.rows.get("Earlier decisive")
        gone.update(SUPERSEDED.findall(row.text) if row else [])
    return gone


def reading_text(entry: Entry) -> str:
    rows = [entry.rows.get("Reading"), entry.rows.get("Summary")]
    return "\n".join("\n".join(row.body) for row in rows if row)


def outcome(entry: Entry, clause: Clause) -> str:
    """The outcome word a finding gives the clause, in a sentence that names it."""
    names = re.compile(rf"(?<![\w\[]){clause.id}\b|\[{clause.id}\]")
    words = OUTCOME_WORDS["measured" if clause.standard == "measured" else clause.id[0]]
    text = ABBREVIATION.sub(
        lambda m: m.group(0).replace(".", "\x00"), reading_text(entry)
    )
    for sentence in SENTENCE_END.split(text):
        named = names.search(sentence)
        if not named:
            continue

        # the word after the clause's own name, before the next clause named
        after = sentence[named.end() :]
        other = CLAUSE_NAME.search(after)
        for segment in (after[: other.start()] if other else after, sentence):
            found = [
                (match.start(), word)
                for word in words
                if (
                    match := re.search(
                        rf"\b{OUTCOME_STEMS.get(word, word)}\w*", segment, re.IGNORECASE
                    )
                )
            ]
            if found:
                return min(found)[1]
    return ""


def hex_ids(text: str) -> list[tuple[str, bool]]:
    """(value, is_commit) of each hex id in a Runs row, lost runs left out.

    A commit follows `at` or `commit`, or has a sha's length: 7 to 12
    characters, or 40. The builder reads Runs rows through this function too.
    """
    lost = [match.span() for match in LOST_RUNS.finditer(text)]
    return [
        (
            match.group(0),
            bool(COMMIT_LEAD.search(text[: match.start()]))
            or len(match.group(0)) in SHORT_SHA_LENGTHS
            or len(match.group(0)) == FULL_SHA_LENGTH,
        )
        for match in HEX.finditer(text)
        if not any(start <= match.start() < end for start, end in lost)
    ]


def lost_runs(text: str) -> list[str]:
    """The prefix of each run a Runs row names `run <hex> (server lost)`.

    The builder shows them as text: their tracking server is gone.
    """
    return [
        prefix
        for match in LOST_RUNS.finditer(text)
        for prefix in re.findall(r"[0-9a-f]{7,31}", match.group(1))
    ]


def prose_lines(
    document: Document, heading: str | None = None
) -> Iterator[tuple[int, str]]:
    """(line, text) of every section line, or one section's, with code blanked.

    A line that holds `<…>` template guidance comes back blank: the guidance
    rule reports the leftover, so a check on the line's content waits for it.
    """
    if heading is None:
        sections = list(document.sections.values())
    else:
        sections = [document.sections[heading]] if heading in document.sections else []
    for section in sections:
        code_blanked, _ = split_code(_skip_guidance_blocks(section.body))

        # blank template guidance, inner `<…>` first, keeping the line breaks
        prose, previous = code_blanked, None
        while previous != prose:
            previous, prose = prose, GUIDANCE.sub(_blank_guidance, prose)

        lines = zip(code_blanked.split("\n"), prose.split("\n"))
        for number, (before, after) in enumerate(lines, start=section.line + 1):
            yield number, "" if before != after else after


def unwritten(text: str) -> bool:
    """Whether a text is empty or still holds `<…>` template guidance outside code.

    The guidance rule reports the leftover once its section's stage is reached;
    until then a check on the text's content waits for it to be written.
    """
    return is_guidance(text) or any(
        is_template_guidance(m.group(1))
        for m in GUIDANCE.finditer(INLINE_CODE.sub(" ", text))
    )


def code_placeholders(document: Document, heading: str) -> Iterator[tuple[int, str]]:
    """The first template placeholder alone in a section's inline code, if any,
    as the `guidance` check reports a leftover."""
    section = document.sections.get(heading)
    if section is None:
        return
    _, fenced = split_code(section.body)
    found = [
        (number, code)
        for number, (line, fence) in enumerate(
            zip(section.body, fenced.split("\n")), start=section.line + 1
        )
        if not fence.strip()
        for code in INLINE_CODE.findall(line)
        if CODE_PLACEHOLDER.match(code.strip("`").strip())
    ]
    if found:
        number, code = found[0]
        more = f" and {len(found) - 1} more" if len(found) > 1 else ""
        yield number, f"{heading} keeps '{code}'{more}"


def code_guidance(text: str) -> bool:
    """Whether an inline code span of the text holds `<…>` template guidance."""
    return any(
        is_template_guidance(m.group(1))
        for code in INLINE_CODE.findall(text)
        for m in GUIDANCE.finditer(code.strip("`"))
    )


def is_placeholder(text: str) -> bool:
    match = GUIDANCE.fullmatch(text)
    return bool(match) and is_template_guidance(match.group(1))


def is_long(row: DocumentRow) -> bool:
    return not unwritten(row.text) and len(WORD.findall(row.text)) > BLOCK_WORDS


def lead_word(text: str) -> str:
    words = re.split(r"[\s—:,.;]+", text.strip(), maxsplit=1)
    return words[0].lower() if words else ""


def _skip_guidance_blocks(body: tuple[str, ...]) -> tuple[str, ...]:
    """The body with each top-level `<…>` block blanked, from a line opening `<`
    to one closing `>`, since such a block may span several paragraphs."""
    kept, inside = [], False
    for line in body:
        if inside or line.startswith("<"):
            inside = not line.rstrip().endswith(">")
            kept.append("")
        else:
            kept.append(line)
    return tuple(kept)


def _blank_guidance(match: re.Match) -> str:
    if not is_template_guidance(match.group(1)):
        return match.group(0)
    return re.sub(r"[^\n]", " ", match.group(0))


def _nested_label(line: str) -> re.Match | None:
    return re.match(r"^\s+- \*\*([^*]+)\*\*", line)
