"""Checks the record spec (`specs/record.yaml`) names beyond the built-in ones.

Most of them cross-reference one part of a record against another: a cited
`[D<n>]` against the Decisions entries, a clause's round against Rounds, a
Code row against the file it names. Each yields `(line, detail)` pairs, as
every check does (see `rules.check`).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from ..document import Document, Entry, normalize_link
from ..document import Row as DocumentRow
from ..rules import (
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
from ..spec import Row, Rule, Section

# ===============================
# Defaults
# ===============================

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
WEIGHT = re.compile(r" · (key|supporting|minor)(?: (F\d+))?$")
MINOR_ROWS = ("Summary", "Runs", "Result", "Reading", "History")
# a decision's History line: `(before runs)`, `(after [F2], [F3])`, either with `; user`
TIMING = re.compile(r"\((?:before runs|after \[F\d+\](?:,\s*\[F\d+\])*)(?:;\s*user)?\)")

# a clause line under Predictions: `- **H2 · diagnostic · measured · R2 — <claim>**: …`
CLAUSE_LINE = re.compile(r"^\*\*([HCQ]\d+)((?: · [^*—]+?)*) — ([^*]+)\*\*:")
CLAUSE_TAG = re.compile(r"decisive|diagnostic|tested|measured|R\d+")
VERDICT_RULE = re.compile(r"^\*\*Verdict rule\*\*:")
CLAUSE_TRACE = re.compile(r"\(\[D\d+\][^()]*;\s*measured by \[M\d+\][^()]*\)")
NOTE_ADDED = re.compile(r"Note added(?! \d{4}-\d{2}-\d{2}:)")
SUPERSEDED = re.compile(r"\b([HCQ]\d+) superseded by ([HCQ]\d+)\b")
STANDS = re.compile(r"\b([HCQ]\d+) stands\b")
# the outcome words a reading may give a clause, by clause type; checked in order
OUTCOME_WORDS = {
    "H": ("inconclusive", "refuted", "confirmed"),
    "C": ("undecided", "equivalent", "worse", "better"),
    "Q": ("unanswered", "partial", "answered"),
    "measured": ("unanswered", "answered"),
}
OUTCOME_STEMS = {"confirmed": "confirm", "refuted": "refut"}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
ABBREVIATION = re.compile(r"\b(e\.g|i\.e|vs|et al|cf|fig|no|approx|ca)\.", re.I)

# citations: `[D3]`, `[H1]`, and `[<slug>:F2]` for another record's finding or clause
CITATION = {
    "D": re.compile(r"\[(D\d+)\]"),
    "clause": re.compile(r"\[([HCQ]\d+)\]"),
    "F": re.compile(r"\[(F\d+)\]"),
    "M": re.compile(r"\[(M\d+)\]"),
}
CITED_SECTION = {"D": "Decisions", "F": "Findings", "M": "Methods"}
RECORD_CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*):(F\d+|[HCQ]\d+)\]")
RELATION = re.compile(r"\b(?:qualifies|supports|overturns)\s+\(?$")
BIB_CITATION = re.compile(r"\[[^\]\n]*@[^\]\n]*\]")
BIB_KEY = re.compile(r"@([\w:.\-]+)")
BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
BIBLIOGRAPHY = "references.bib"

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
SHORT_SHA_LENGTHS = range(7, 13)
FULL_SHA_LENGTH = 40
SHA = re.compile(r"^[0-9a-f]{7,40}$")
STEP_COMMIT = re.compile(r"\(commit ([^)]*)\)")

ADOPTION_ROWS = ("Carries", "Stays out")
NEXT_ENDING = re.compile(
    r"(?:— planned as [\w-]+(?: #\d+)?|\(issue #\d+\)|— not yet planned)[.;]?$"
)
PREREQUISITE = re.compile(r"^\*\*(Secret|Login|Data|Service|Frozen)\*\*")
SIDE_FOLDERS = ("configs", "prompts")

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
    records = {
        str(d.frontmatter.get("slug") or d.folder.name): d
        for d in corpus.documents
        if d.kind == document.kind
    }
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
    for clause in clauses(document):
        if not unwritten(clause.text) and not CLAUSE_TRACE.search(clause.text):
            yield clause.line, clause.id


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
    """Each History line says when: `(before runs)` or `(after [F<n>])`."""
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
    """Each clause no later round superseded is read by a key finding."""
    gone = superseded(document)
    keys = [
        entry
        for entry in entries(document, "Findings")
        if (match := WEIGHT.search(entry.title)) and match.group(1) == "key"
    ]
    for clause in clauses(document):
        if clause.id in gone or unwritten(clause.text):
            continue
        if not any(outcome(e, clause) for e in keys):
            yield clause.line, clause.id


@check("run-ids")
def run_ids(
    document: Document, part: Row, rule: Rule, corpus: Corpus
) -> Iterator[tuple[int, str]]:
    """Runs name full 32-character run ids, or the commit a script ran at."""
    for _, row in row_places(document, part):
        if row is None or unwritten(row.text):
            continue
        found = hex_ids(row.text)
        if not found:
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
    """Each Next item ends planned as, (issue #n) or not yet planned."""
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
            if path.is_file() and path.name not in text:
                yield 1, f"{path.relative_to(document.folder).as_posix()}"


# ===============================
# Helper functions
# ===============================


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
        if names.search(sentence):
            for word in words:
                if re.search(
                    rf"\b{OUTCOME_STEMS.get(word, word)}\w*", sentence.lower()
                ):
                    return word
    return ""


def hex_ids(text: str) -> list[tuple[str, bool]]:
    """(value, is_commit) of each hex id in a Runs row.

    A commit follows `at` or `commit`, or has a sha's length: 7 to 12
    characters, or 40. The builder reads Runs rows through this function too.
    """
    return [
        (
            match.group(0),
            bool(COMMIT_LEAD.search(text[: match.start()]))
            or len(match.group(0)) in SHORT_SHA_LENGTHS
            or len(match.group(0)) == FULL_SHA_LENGTH,
        )
        for match in HEX.finditer(text)
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
