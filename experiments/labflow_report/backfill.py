"""Backfill: rewrite documents toward their spec's rules, and list what a person must write.

    python -m labflow_report backfill [paths] [--experiments DIR] [--write] [--list FILE]

A fixer is registered under the rule id it repairs, for one document kind
(`@fixer("REC-139", "record")`). It reads a document as it stands on disk and
returns its `Change`s, each a range of lines and the lines that replace it,
and the `Item`s a person must write that no checker rule reports. A change is
`auto` only when its new text follows from the record alone (a form change, a
rename, a moved row) or from a source in the repo (git, an `mlruns/` folder);
anything that needs a judgement or a fact the record lacks is a `person` item,
never guessed. Every fixer reads the file as it stands, so two fixers never
see each other's edits; their changes are applied together.

Without `--write` the changes are printed and nothing is written; with it,
the files are rewritten. Then the checker runs, and each breach left, with the
fixers' own items, is printed as `file:line RULE-ID — what the author must
write`. `--list FILE` writes that list as markdown, grouped by record, with
the breach counts before and after on top. A later page adds its fixers here.

Exit codes: 0 done, 2 the backfill could not run (as `check`).
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from .check import EXIT_ERROR, CheckError, find_breaches
from .checks.record import (
    COMMIT_LEAD,
    HEX,
    NEEDS_ROWS,
    SHORT_SHA_LENGTHS,
    entries,
    items,
)
from .document import Document, Row, normalize_link
from .rules import Breach, read_corpus
from .spec import DocumentSpec, SpecError, load_specs

# ===============================
# Defaults
# ===============================

EXIT_DONE = 0
EXPERIMENTS_NAME = "experiments"
DASH = " — "

# REC-016: a PR number written as a YAML int
PR_NUMBER = re.compile(r"^(pr:\s*)\d+")

# REC-122: a Builds on text that already opens with a slug in backticks
BUILDS_ON_LEAD = re.compile(r"^`[^`]+`")

# REC-135: a needs row's short value, before the `;` (or `. ` and a lowercase
# word or a number) that ends it
NEEDS_VALUE = re.compile(r"^(?P<value>[^;.,—()]+?)(?:;|\.(?= [a-z0-9]))\s+(?P<rest>\S)")
NEEDS_VALUE_WORDS = 4

# REC-139: an inline Code row, `` `a.py::f` (step 1), `g` and `h` (steps 2 to 4) ``
INLINE_STEPS = re.compile(r"\(steps? \d")
STEP_LINE = re.compile(r"^\s*- step \d+:")
CODE_REFERENCES = re.compile(r"^`[^`]+`(?:(?:,? and |, | with )`[^`]+`)*$")
CODE_REFERENCE = re.compile(r"`([^`]+)`")
CODE_NOTE = re.compile(r"^(?P<refs>.*?)\s*\((?P<note>[^()]*)\)$")
STEP_NOTE = re.compile(r"^steps? (\d+(?:(?:, | and | to |–|, and )\d+)*)$")
SYMBOL = re.compile(r"^\w+$")

# REC-149: an MLflow run id in full, as `mlruns/<experiment>/<run>/` names it
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
MLRUNS = "mlruns"
PREFIXES_SHOWN = 4

# TAB-010 in tables.md: a column name is lowercase snake case
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
COLUMNS_SHOWN = 6

# rows a page no longer reads, and where their content belongs
LEGACY_RULE = "LEGACY-ROW"
LEGACY_DESTINATIONS = {
    ("Design", "Data"): "the Inputs row (one sentence ending `described in "
    "## Datasets`) and one `## Datasets` row per dataset",
    (
        "Design",
        "Gateway",
    ): "the Protocol row, or a `### D<n>` decision when it sets a bar",
    (
        "Design",
        "Stages",
    ): "the Method / factors row, one nested bullet per step citing `[M<n>]`",
    ("Verdict", "Open"): "the Next row, one item per open thread ending "
    "`— planned as <slug>`, `(issue #<n>)` or `— not yet planned`",
    ("TL;DR", "Caveat"): "the Verdict's Caveat row",
}
# entries sections whose undeclared rows a known-rows rule already reports
KNOWN_ROWS_CHECKED = frozenset({"Methods", "Decisions"})
TLDR_FINDING = re.compile(r"^F\d+\b")

# what the author must write for each rule a fixer leaves to a person
ADVICE = {
    "REC-005": "add `short:`, the title in 2 to 5 plain words",
    "REC-016": 'set `pr:` to the PR as text, `"#<n>"` or its URL',
    "REC-028": "make the TL;DR bullets only: fold the paragraph into the Hypothesis line or drop it",
    "REC-033": "add `- **Question**:`, the title's question as one plain sentence ending in `?`",
    "REC-034": "add `- **Decision it informs**:`, what will be done differently on each answer",
    "REC-035": "open Disclosure with its kind and a colon: `earlier runs:`, `pilot:` or `numbers seen:`",
    "REC-041": "set the Hypothesis Outcome to the word alone; its reading goes in the "
    "Verdict's Outcome after ` — `",
    "REC-050": "add `- **Inputs**:`, the datasets the runs consume, ending `described in ## Datasets`",
    "REC-060": "add `- **People**:`, a short value, ` — ` and a sentence (`none — …` when none)",
    "REC-062": "add `- **Data access**:`, a short value, ` — ` and a sentence (`none — …` when none)",
    "REC-064": "add `- **External dependency**:`, a short value, ` — ` and a sentence "
    "(`none — …` when none)",
    "REC-066": "add `## Datasets`, one row per dataset the runs consume",
    "REC-073": "shorten How to at most 240 words; detail goes in Settings or a decision",
    "REC-078": "shorten the Decision row to its cap; the reason goes in Why",
    "REC-080": "shorten Why to its cap; rejected options go in Alternatives",
    "REC-083": "Findings holds only `### F<n>` entries: move or drop the paragraph above the first",
    "REC-098": "shorten Discussion to at most 200 words",
    "REC-103": "add `(<kind>; <gold|silver|none> labels)` after the name, kind benchmark, "
    "split of <dataset>, external, own or derived from <dataset>, and open the text "
    "`one item is`",
    "REC-114": "the `[D<n>]` names no decision here; if it is another record's, name that "
    "record in words (a known false breach)",
    "REC-121": "set `mlflow:` to the experiment's MLflow URL, since Findings cite run ids",
    "REC-122": "write Builds on as one nested bullet per builds_on parent, "
    "`` `<slug>`: what it left open ``",
    "REC-126": "end the clause line `([D<n>]; measured by [M<n>])`: the decision that set its "
    "bar and the method that measures it",
    "REC-127": "write the dated change as `Note added YYYY-MM-DD: <what changed and why>`",
    "REC-135": "open the row with a short value, then ` — ` and a sentence (`none — …` when none)",
    "REC-138": "turn How into numbered steps, one per action",
    "REC-140": "gloss the Settings key: `` `key` (what it steers, value) ``",
    "REC-143": "tag the History line `(before runs)` or `(after [F<n>])` naming the first "
    "finding in hand, with `; user` inside when the user made the call",
    "REC-149": "name full 32-character MLflow run ids, or `at <sha>` for a finding no run made",
    "REC-150": "read this clause in a key finding's Reading, or mark it superseded",
    "REC-152": "add nested `- **Carries**:` and `- **Stays out**:` rows under the Adoption "
    "candidate",
    "REC-154": "end the Next item `— planned as <slug>`, `(issue #<n>)` or `— not yet planned`",
    "REC-156": "write the step's commit as `(commit <sha>)` alone; move the note outside",
    "REC-157": "name the file in the record, in a Settings row or a Reproduce step",
}

# ===============================
# Backfill classes
# ===============================


@dataclass(frozen=True, order=True)
class Change:
    """Lines `line` to `line + len(old) - 1` of a file, and the lines that replace them."""

    path: Path
    line: int
    rule: str
    old: tuple[str, ...]
    new: tuple[str, ...]

    def __str__(self) -> str:
        removed = [f"  - {text}" for text in self.old]
        added = [f"  + {text}" for text in self.new]
        return "\n".join([f"{self.path}:{self.line} {self.rule}", *removed, *added])


@dataclass(frozen=True, order=True)
class Item:
    """One thing a person must write, at the line it belongs."""

    path: Path
    line: int
    rule: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} {self.rule} — {self.text}"


@dataclass(frozen=True)
class Context:
    """What a fixer may read beyond the document: the specs and the repo's sources.

    `is_commit` says whether a short hex id is a commit of the repo; `run_ids`
    holds every full MLflow run id an `mlruns/` folder in the repo names.
    """

    specs: dict[str, DocumentSpec]
    is_commit: Callable[[str], bool] = field(default=lambda value: True)
    run_ids: tuple[str, ...] = ()


Fixer = Callable[[Document, list[str], Context], tuple[list[Change], list[Item]]]
FIXERS: dict[str, tuple[str, Fixer]] = {}


def fixer(rule: str, kind: str) -> Callable[[Fixer], Fixer]:
    """Registers a fixer under the rule id it repairs, for one document kind."""

    def register(function: Fixer) -> Fixer:
        if rule in FIXERS:
            raise ValueError(f"fixer '{rule}' is registered twice")
        FIXERS[rule] = (kind, function)
        return function

    return register


# ===============================
# Main functions
# ===============================


def plan(
    documents: list[Document], context: Context
) -> tuple[list[Change], list[Item]]:
    """Runs every fixer of each document's kind over the document as it stands.

    Args:
        documents: The documents to backfill.
        context: The specs and the repo's sources.

    Returns:
        The changes, sorted by file and line, and the items a person must write.
    """
    changes, found = [], []
    for document in documents:
        lines = document.path.read_text(encoding="utf-8").splitlines()
        for kind, function in FIXERS.values():
            if kind == document.kind:
                document_changes, document_items = function(document, lines, context)
                changes += document_changes
                found += document_items
    return sorted(changes), sorted(found)


def apply_changes(text: str, changes: list[Change]) -> str:
    """The text with each change's lines replaced, from the last change up.

    Raises:
        ValueError: If two changes cover the same line, or a change's old
            lines are not the text's.
    """
    lines = text.splitlines()
    end = len(lines) + 1
    for change in sorted(changes, reverse=True):
        start = change.line - 1
        if change.line + len(change.old) > end:
            raise ValueError(
                f"{change.path}:{change.line} {change.rule} overlaps a change"
            )
        if tuple(lines[start : start + len(change.old)]) != change.old:
            raise ValueError(
                f"{change.path}:{change.line} {change.rule} no longer matches"
            )
        lines[start : start + len(change.old)] = change.new
        end = change.line
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def write_changes(changes: list[Change]) -> None:
    """Rewrites each changed file with its changes applied."""
    for path in sorted({change.path for change in changes}):
        text = path.read_text(encoding="utf-8")
        mine = [change for change in changes if change.path == path]
        path.write_text(apply_changes(text, mine), encoding="utf-8")


def person_items(
    breaches: list[Breach],
    changes: list[Change],
    fixer_items: list[Item],
    messages: dict[str, str],
) -> list[Item]:
    """Each breach no change covers, as what the author must write, with the fixers' items.

    Args:
        breaches: The checker's breaches.
        changes: The changes planned; a breach on a line they replace is fixed.
        fixer_items: The items the fixers found.
        messages: Each rule's message by id, so the breach's detail can be cut out.

    Returns:
        The items, sorted by file and line.
    """
    covered = {
        (change.path.resolve(), change.rule, line)
        for change in changes
        for line in range(change.line, change.line + len(change.old))
    }
    left = [
        Item(Path(b.path), b.line, b.rule, _advice(b, messages))
        for b in breaches
        if (Path(b.path).resolve(), b.rule, b.line) not in covered
    ]
    return sorted(left + fixer_items)


def person_list(
    found: list[Item], experiments: Path, before: int, after: int, applied: int
) -> str:
    """The person items as markdown, grouped by record, with the counts on top."""
    lines = [
        "# Backfill person list",
        "",
        f"- Breaches before: {before}",
        f"- Breaches after: {after}",
        f"- Auto changes: {applied}",
        f"- Person items: {len(found)}",
    ]
    groups: dict[str, list[Item]] = {}
    for item in found:
        groups.setdefault(_record_of(item.path, experiments), []).append(item)
    for record, record_items in sorted(groups.items()):
        lines += ["", f"## {record}", ""]
        lines += [
            f"- `{_relative(i.path, experiments)}:{i.line}` {i.rule} — {i.text}"
            for i in record_items
        ]
    return "\n".join(lines) + "\n"


def repo_context(experiments: Path, specs: dict[str, DocumentSpec]) -> Context:
    """The context of the repo that holds the experiments folder: git and mlruns/.

    Outside a git repo every short hex id counts as a commit, so none is
    reported as a run-id prefix on no evidence.
    """
    root = experiments.resolve().parent
    is_repo = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-dir"], capture_output=True
        ).returncode
        == 0
    )

    @cache
    def is_commit(value: str) -> bool:
        if not is_repo:
            return True
        command = ["git", "-C", str(root), "cat-file", "-e", f"{value}^{{commit}}"]
        return subprocess.run(command, capture_output=True).returncode == 0

    runs = sorted(
        {
            run.name
            for folder in (root / MLRUNS, experiments / MLRUNS)
            if folder.is_dir()
            for run in folder.glob("*/*")
            if run.is_dir() and RUN_ID.match(run.name)
        }
    )
    return Context(specs, is_commit, tuple(runs))


def main(
    paths: list[Path],
    experiments: Path | None,
    spec_folder: Path,
    write: bool = False,
    list_path: Path | None = None,
) -> int:
    """Prints the changes and the person items; with `write`, rewrites the files.

    Args:
        paths: The files to backfill; empty means every declared document.
        experiments: The experiments folder, or None to take the nearest
            folder named `experiments` above the first path, else
            `./experiments`, else the current folder when it is one.
        spec_folder: The folder of `*.yaml` specs.
        write: Whether to rewrite the files.
        list_path: Where to write the person list as markdown, if anywhere.

    Returns:
        The exit code: 0 done, 2 the backfill could not run.
    """
    try:
        folder = experiments or _experiments_folder(paths)
        before, _ = find_breaches(paths, folder, spec_folder)
        specs = load_specs(spec_folder)
        context = repo_context(folder, specs)
        changes, found = plan(_documents(paths, folder, specs), context)
        if write:
            write_changes(changes)
            # the files moved, so the items are read again from the new lines
            _, found = plan(_documents(paths, folder, specs), context)
            after, _ = find_breaches(paths, folder, spec_folder)
            left = person_items(after, [], found, _messages(specs))
        else:
            left = person_items(before, changes, found, _messages(specs))
            after = [item for item in left if item not in found]
    except (CheckError, SpecError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"labflow_report backfill: {error}", file=sys.stderr)
        return EXIT_ERROR

    # report the changes, then what a person must write
    for change in changes:
        print(change)
    for item in left:
        print(item)
    verb = "applied" if write else "planned"
    print(
        f"breaches before {len(before)}, after {len(after)}; "
        f"{len(changes)} auto changes {verb}; {len(left)} person items"
    )
    if list_path:
        markdown = person_list(left, folder, len(before), len(after), len(changes))
        list_path.write_text(markdown, encoding="utf-8")
    return EXIT_DONE


# ===============================
# Record fixers
# ===============================


@fixer("REC-016", "record")
def pr_as_text(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """`pr: 7` → `pr: "#7"`, the form every other record writes its PR in."""
    value = document.frontmatter.get("pr")
    if not isinstance(value, int) or isinstance(value, bool):
        return [], []
    line = document.key_lines["pr"]
    old = lines[line - 1]
    new = PR_NUMBER.sub(rf'\g<1>"#{value}"', old, count=1)
    return [Change(document.path, line, "REC-016", (old,), (new,))], []


@fixer("REC-041", "record")
def outcome_word(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """The Hypothesis Outcome keeps its word alone when the Verdict's Outcome already holds its reading."""
    hypothesis = _row(document, "Hypothesis", "Outcome")
    verdict = _row(document, "Verdict", "Outcome")
    if hypothesis is None or verdict is None or DASH not in hypothesis.text:
        return [], []
    if not _plain(verdict.text).startswith(_plain(hypothesis.text)):
        return [], []
    word = hypothesis.text.partition(DASH)[0].strip()
    return [
        _replace_row(document, lines, hypothesis, "REC-041", [f"- **Outcome**: {word}"])
    ], []


@fixer("REC-122", "record")
def builds_on_bullet(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """A one-parent Builds on paragraph becomes the bullet `` `<slug>`: <the same text> ``."""
    parents = document.frontmatter.get("builds_on")
    row = _row(document, "Hypothesis", "Builds on")
    if not isinstance(parents, list) or len(parents) != 1 or row is None:
        return [], []
    if items(row) or BUILDS_ON_LEAD.match(row.text) or not row.text:
        return [], []
    slug = normalize_link(str(parents[0])).rsplit("/", 1)[-1]
    first = row.body[0].strip().replace(f" ({slug})", "", 1)
    rest = [f"  {line}" if line.strip() else line for line in row.body[1:]]
    new = ["- **Builds on**:", f"  - `{slug}`: {first}", *rest]
    return [_replace_row(document, lines, row, "REC-122", new)], []


@fixer("REC-135", "record")
def needs_dash(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """A needs row's short value before its first `;` takes ` — ` instead."""
    changes = []
    for label in NEEDS_ROWS:
        row = _row(document, "Design", label)
        if row is None or DASH in row.text:
            continue
        old = lines[row.line - 1]
        prefix = f"- **{label}**:"
        match = NEEDS_VALUE.match(old[len(prefix) :].strip())
        if match is None or len(match.group("value").split()) > NEEDS_VALUE_WORDS:
            continue
        value = match.group("value").strip()
        rest = old[len(prefix) :].strip()[match.start("rest") :]
        new = f"{prefix} {value}{DASH}{rest}"
        changes.append(Change(document.path, row.line, "REC-135", (old,), (new,)))
    return changes, []


@fixer("REC-139", "record")
def code_steps(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """An inline Code row, `` `a.py::f` (step 1), `g` (steps 2 and 3) ``, becomes one bullet per step.

    A bare symbol takes the file of the reference before it. A row with a
    part that names no step stays, as a person item.
    """
    changes, found = [], []
    for entry in entries(document, "Methods"):
        row = entry.rows.get("Code")
        if row is None or not INLINE_STEPS.search(row.text):
            continue
        if any(STEP_LINE.match(line) for line in row.body):
            continue
        steps, problem = _code_steps(row.text)
        if problem:
            text = (
                "write Code as one nested bullet per step, "
                f"`step <n>: <file>.py::<function>`; '{problem}' names no step"
            )
            found.append(Item(document.path, row.line, "REC-139", text))
            continue
        new = ["- **Code**:"] + [
            f"  - step {n}: " + ", ".join(f"`{ref}`" for ref in refs)
            for n, refs in sorted(steps.items())
        ]
        changes.append(_replace_row(document, lines, row, "REC-139", new))
    return changes, found


@fixer("REC-149", "record")
def run_prefixes(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """A short hex in Runs that is no commit of the repo is a run-id prefix.

    The fixer writes the full id when exactly one `mlruns/` run starts with
    the prefix, else lists the prefixes for a person.
    """
    changes, found = [], []
    for entry in entries(document, "Findings"):
        row = entry.rows.get("Runs")
        if row is None:
            continue
        prefixes = sorted(
            {
                match.group(0)
                for match in HEX.finditer(row.text)
                if len(match.group(0)) in SHORT_SHA_LENGTHS
                and not COMMIT_LEAD.search(row.text[: match.start()])
                and not context.is_commit(match.group(0))
            }
        )
        full = {}
        for prefix in prefixes:
            candidates = [run for run in context.run_ids if run.startswith(prefix)]
            if len(candidates) == 1:
                full[prefix] = candidates[0]

        # write what resolves, list what does not
        if full:
            old = lines[row.line - 1 : row.line - 1 + len(row.body)]
            new = [_replace_ids(line, full) for line in old]
            changes.append(_replace_row(document, lines, row, "REC-149", new))
        missing = [prefix for prefix in prefixes if prefix not in full]
        if missing:
            shown = ", ".join(f"`{p}`" for p in missing[:PREFIXES_SHOWN])
            more = (
                f" and {len(missing) - PREFIXES_SHOWN} more"
                if len(missing) > PREFIXES_SHOWN
                else ""
            )
            text = (
                f"replace the {len(missing)} run-id prefixes ({shown}{more}) with full "
                "32-character MLflow run ids; no mlruns/ in the repo holds them"
            )
            found.append(Item(document.path, row.line, "REC-149", text))
    return changes, found


@fixer("TAB-010", "record")
def table_columns(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """Lists each table whose columns are not lowercase snake case; renames none."""
    found = []
    for table in sorted((document.folder / "tables").glob("*.csv")):
        with table.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        bad = [name for name in header if not SNAKE_CASE.match(name)]
        if not bad:
            continue
        shown = ", ".join(f"'{name}'" for name in bad[:COLUMNS_SHOWN])
        more = (
            f" and {len(bad) - COLUMNS_SHOWN} more" if len(bad) > COLUMNS_SHOWN else ""
        )
        text = (
            f"columns {shown}{more} are not lowercase snake case; rename them where "
            "analysis.py writes the table, once the TAB-010 decision is made"
        )
        found.append(Item(table, 1, "TAB-010", text))
    return [], found


@fixer(LEGACY_RULE, "record")
def legacy_rows(
    document: Document, lines: list[str], context: Context
) -> tuple[list[Change], list[Item]]:
    """Lists each row the spec does not declare, which the page no longer shows; changes none."""
    found = []
    for section in context.specs[document.kind].sections:
        declared = {row.label for row in section.rows}
        written = document.sections.get(section.heading)
        if not declared or written is None or section.heading in KNOWN_ROWS_CHECKED:
            continue
        rows = (
            [row for entry in written.entries for row in entry.rows.values()]
            if written.entries
            else list(written.rows.values())
        )
        for row in rows:
            if row.label in declared or (
                section.heading == "TL;DR" and TLDR_FINDING.match(row.label)
            ):
                continue
            destination = LEGACY_DESTINATIONS.get(
                (section.heading, row.label),
                f"a row the template declares for `## {section.heading}`",
            )
            text = (
                f"`## {section.heading}` declares no '{row.label}' row, so the page "
                f"drops it; move its content into {destination}"
            )
            found.append(Item(document.path, row.line, LEGACY_RULE, text))
    return [], found


# ===============================
# Helper functions
# ===============================


def _row(document: Document, heading: str, label: str) -> Row | None:
    section = document.sections.get(heading)
    return section.rows.get(label) if section else None


def _replace_row(
    document: Document, lines: list[str], row: Row, rule: str, new: list[str]
) -> Change:
    old = tuple(lines[row.line - 1 : row.line - 1 + len(row.body)])
    return Change(document.path, row.line, rule, old, tuple(new))


def _plain(text: str) -> str:
    return " ".join(text.split()).rstrip(".")


def _code_steps(text: str) -> tuple[dict[int, list[str]], str]:
    """The references of an inline Code row by step number, or the part that names no step."""
    steps: dict[int, list[str]] = {}
    pending, current_file = [], None
    for part in _top_level_parts(text):
        match = CODE_NOTE.match(part.strip())
        refs_text = match.group("refs") if match else part.strip()
        if not CODE_REFERENCES.match(refs_text):
            return {}, part.strip()

        # qualify each bare symbol with the file named before it
        refs = []
        for ref in CODE_REFERENCE.findall(refs_text):
            if ".py" in ref and " " not in ref:
                current_file = ref.partition("::")[0]
                refs.append(ref)
            elif SYMBOL.match(ref) and current_file:
                refs.append(f"{current_file}::{ref}")
            else:
                return {}, part.strip()

        # refs without a note share the step of the next part that has one
        if match is None:
            pending += refs
            continue
        numbers = _step_numbers(match.group("note"))
        if not numbers:
            return {}, part.strip()
        for n in numbers:
            steps.setdefault(n, []).extend(pending + refs)
        pending = []
    if pending:
        return {}, ", ".join(f"`{ref}`" for ref in pending)
    return steps, ""


def _top_level_parts(text: str) -> list[str]:
    # split at `, ` outside backticks and parentheses
    parts, depth, in_code, start = [], 0, False, 0
    for index, character in enumerate(text):
        if character == "`":
            in_code = not in_code
        elif not in_code and character == "(":
            depth += 1
        elif not in_code and character == ")":
            depth -= 1
        elif not in_code and depth == 0 and text.startswith(", ", index):
            parts.append(text[start:index])
            start = index + 2
    parts.append(text[start:])
    return [part for part in parts if part.strip()]


def _step_numbers(note: str) -> list[int]:
    match = STEP_NOTE.match(note.strip())
    if match is None:
        return []
    numbers, tokens = [], re.findall(r"\d+|to|–", match.group(1))
    for index, token in enumerate(tokens):
        if token in ("to", "–"):
            numbers += range(numbers[-1] + 1, int(tokens[index + 1]))
        else:
            numbers.append(int(token))
    return sorted(set(numbers))


def _replace_ids(line: str, full: dict[str, str]) -> str:
    for prefix, run in full.items():
        line = re.sub(rf"(?<![\w-]){prefix}(?![\w-])", run, line)
    return line


def _advice(breach: Breach, messages: dict[str, str]) -> str:
    message = messages.get(breach.rule, "")
    detail = (
        breach.message[len(message) + 2 :]
        if message and breach.message.startswith(f"{message}: ")
        else ""
    )
    advice = ADVICE.get(breach.rule, f"fix: {breach.message}")
    return f"{advice} ({detail})" if detail and breach.rule in ADVICE else advice


def _messages(specs: dict[str, DocumentSpec]) -> dict[str, str]:
    return {rule.id: rule.message for spec in specs.values() for rule in spec.rules}


def _documents(
    paths: list[Path], experiments: Path, specs: dict[str, DocumentSpec]
) -> list[Document]:
    documents, _ = read_corpus(experiments, specs)
    wanted = {path.resolve() for path in paths}
    return [d for d in documents if not wanted or d.path.resolve() in wanted]


def _experiments_folder(paths: list[Path]) -> Path:
    if paths:
        for parent in paths[0].resolve().parents:
            if parent.name == EXPERIMENTS_NAME:
                return parent
        raise CheckError(f"no experiments folder above {paths[0]}; pass --experiments")
    for folder in (Path(EXPERIMENTS_NAME), Path.cwd()):
        if folder.resolve().name == EXPERIMENTS_NAME and folder.is_dir():
            return folder.resolve()
    raise CheckError(f"no experiments folder at {Path.cwd()}; pass --experiments")


def _record_of(path: Path, experiments: Path) -> str:
    parts = _relative(path, experiments).split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def _relative(path: Path, experiments: Path) -> str:
    resolved, base = path.resolve(), experiments.resolve()
    return (
        resolved.relative_to(base).as_posix()
        if resolved.is_relative_to(base)
        else str(path)
    )
