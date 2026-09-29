#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.6", "pyyaml>=6.0", "latex2mathml>=3.77"]
# ///
"""Render every labflow experiment record into one self-contained HTML report.

Reads the book (experiments/README.md) and every record
(experiments/<category>/<slug>/README.md) and writes experiments/report.html:
an app-style page with a sidebar of every experiment, ordered by family (a
root record plus everything whose `builds_on` chain reaches it, each follow-up
indented under its parent), one report page
per record, and a rail of that page's sections. Figures and `tables/*.csv`
are inlined, so the file needs no data source. Requires `markdown` and
`pyyaml`.

    uv run experiments/build_report.py [--out experiments/report.html]

`$…$` and `$$…$$` in a record become MathML, so equations need no script or
font at read time. Under plain `python` that needs `latex2mathml` installed
beside `markdown` and `pyyaml`; `uv run` fetches all three itself.

Exit code 1 if a record has a hole: a missing figure or table, a finding
whose Result is neither, a `### F<n>` or `### M<n>` heading the parser
cannot read, a concluded record without Hypothesis, Design, Methods, Verdict or
Reproduce, a method step missing a label or naming a file or symbol that does
not exist, or prose
over the record template's word caps (CAPS below), prose outside the rows of
Design or Verdict, a finding titled as a fix (FIX_WORDS — a fix revises the
entry it corrects, it is not an entry), a supporting finding naming no key
finding, a key finding whose Reading cites no `[H<n>]` clause, a Summary citing a
ticket or carrying more than a pair of numbers, a concluded record whose
Verdict has no Discussion, or prose the voice rules reject (VOICE below): a
sentence over SENTENCE_WORDS words or with more than SENTENCE_SEMICOLONS
semicolons, a `[D<n>]` / `[F<n>]` / `[M<n>]` citation with fewer than
CITATION_WORDS words of its own sentence before it, a Reading with more than
READING_NUMBERS numbers beyond its Summary's, a long Reading, Rationale or
Discussion that is not bold-lead blocks, a long How that is not numbered
steps, a Settings key with no gloss, a minor finding with no Reading, or a
TL;DR key-finding line not shaped `**F<n> — title.** Summary`. Standard
methods named in prose but not in the glossary are listed alongside the
undefined abbreviations.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import markdown
import yaml

try:
    from latex2mathml.converter import convert as latex_to_mathml
except ModuleNotFoundError:  # only an equation needs it; build() fails loudly if one appears
    latex_to_mathml = None

EXPERIMENTS = Path(__file__).resolve().parent
REPO_URL = None  # set in build()
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
KV = re.compile(r"^- \*\*(.+?)\*\*:\s*(.*)$")
# `#32` at line or bullet start is a ticket ref, not a heading; escape it for markdown
HASH_REF = re.compile(r"(?m)^([ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?)#(?=\d)")
REQUIRED_WHEN_CONCLUDED = {
    "hypothesis": "Hypothesis",
    "design": "Design",
    "methods": "Methods",
    "verdict": "Verdict",
    "reproduce": "Reproduce",
}
SECTION_TITLES = {
    "hypothesis": "Question",
    "design": "Setup",
    "datasets": "Datasets",
    "methods": "Methods",
    "findings": "Findings",
    "verdict": "Verdict",
    "decisions": "Decisions",
    "reproduce": "Reproduce",
}
PAGE_ORDER = ["findings", "verdict", "hypothesis", "design", "datasets", "methods", "decisions", "reproduce", "builds"]
# Design rows that say what an experiment needs beyond code; the page shows them as a strip, a programme as a table
CONSTRAINT_ROWS = ("People", "Compute", "Data access", "External dependency")
NO_NEED = re.compile(r"^(none|n/a|—|-)\b", re.I)
# a need worth a glance: anything beyond none, local compute or the parent's data, or naming a hosted model, quota, licence or people
HOT_NEED = re.compile(
    r"hosted|\bapi\b|hpc|gpu|cluster|licen|quota|billed|endpoint|annotat|hours|manual|rare|cannot", re.I
)


def needs_glance(text: str) -> bool:
    t = plain(text).strip()
    if not t or NO_NEED.match(t):
        return False
    return bool(HOT_NEED.search(t))


# a clause's conditions: `confirmed if …; refuted if … — note`
CONDITIONS = re.compile(r"confirmed if (.*?)[;.]\s*refuted if (.*?)(?:\s+—\s+(.*))?\s*$", re.I | re.S)
HCITE = re.compile(r"\[(H\d+)\]")
TLDR_STATUS = re.compile(r"^\s*(?:confirmed|refuted|inconclusive|open|leaning \w+)\s*(?:\([^)]*\))?\s*[—–-]\s*", re.I)
# lint that reads as advice on prose, not a broken record: counted per record, listed with --warnings, never fails the build
WARNING = re.compile(
    r"\(cap \d+\)|sentence|cites \[|numbers beyond|one paragraph|TL;DR line|carries no gloss|is minor but|"
    r"is supporting but|content outside|cites a ticket|constraints|will scroll|Predictions has no|names a topic"
)
# a decision or method title shorter than this names a topic, not the choice or the step
TITLE_MIN_WORDS = 4
# a programme is a question several experiments answer together: experiments/programmes/<slug>.md
PROGRAMME_SECTIONS = {
    "question": "Question",
    "what-we-now-believe": "What we now believe",
    "threads": "Threads",
    "open": "Open",
}
XREF = re.compile(
    r"\[([a-z0-9][a-z0-9-]*):([FH]\d+)\]"
)  # a finding or clause of a member experiment, cited from a programme
# word caps from the record template; Alternatives is per bullet, Design per row
CAPS = {
    "Summary": 25,
    "Reading": 240,
    "Implication": 50,
    "Decision": 60,
    "Why": 60,
    "Alternatives": 25,
    "Design": 120,
    "Method / factors": 200,
    "Discussion": 200,
    "How": 240,
}
SUMMARY_NUMBERS = 2  # a Summary carries its one number, at most a pair
TABLE_ROWS = 20  # a table longer than this scrolls vertically under a sticky header (the CSS hard-codes the same 20)
TICKET = re.compile(r"(?<![\w`])#\d+\b")
NUMBER = re.compile(r"(?<![#\w.])\d[\d,.]*%?")


MODEL_VERSION = re.compile(r"\b(?:Haiku|Sonnet|Opus|Fable|Claude|GPT|Gemini|Llama|Mistral|Qwen)[- ]?\d[\d.]*", re.I)


def numbers(text: str) -> list[str]:
    """Numbers in prose, trailing punctuation dropped so `0.92,` and `0.92` match;
    a model version (`Sonnet 4.5`) is a name, not a number."""
    return [n.rstrip(",.") for n in NUMBER.findall(MODEL_VERSION.sub(" ", text))]


PREDICTION = re.compile(r"\[H\d+\]")  # a key Reading cites the clause it decides
MINOR_LABELS = {"Summary", "Runs", "Result", "Reading", "History"}
# voice rules from dev-tools/references/voice.md, "In records"
SENTENCE_WORDS = 35
SENTENCE_SEMICOLONS = 2
CITATION_WORDS = 3  # words of its own sentence a citation needs before it: "with score rules removed ([D3])"
READING_NUMBERS = 2  # beyond the Summary's; the rest belong in the Result
BLOCK_WORDS = 60  # past this a Reading, Rationale or Discussion is bold-lead blocks and a How is numbered steps
CITATION = re.compile(r"\[([DFMH]\d+)\]")
LEAD_BULLET = re.compile(r"(?m)^\s*- \*\*[^*\n]+\*\*")
NUMBERED_STEP = re.compile(r"(?m)^\s*\d+\. ")
TLDR_LINE = re.compile(r"^\*\*F\d+ — [^*\n]+?[.!?]\*\* \S")
ABBREV_BEFORE_STOP = re.compile(r"\b(e\.g|i\.e|vs|et al|cf|fig|no|approx|ca)\.", re.I)
# standard methods a reader outside the domain needs a glossary line for; matched as whole words, any case
METHOD_TERMS = [
    "bootstrap",
    "BCa",
    "SVD",
    "TF-IDF",
    "k-means",
    "kmeans",
    "embedding",
    "embeddings",
    "regularisation",
    "regularization",
    "logistic regression",
    "Kendall",
    "tau-b",
    "cosine",
    "z-score",
    "log-odds",
    "percentile",
    "quantile",
    "kappa",
    "argmax",
    "softmax",
    "tokeniser",
    "tokenizer",
    "transformer",
    "fine-tuning",
    "distillation",
    "perplexity",
    "BLEU",
    "ROUGE",
    "BERTScore",
    "TPE",
    "Optuna",
    "cross-validation",
    "AUC",
    "ROC",
    "PCA",
    "t-SNE",
    "UMAP",
    "LLM",
    "few-shot",
    "zero-shot",
    "chain-of-thought",
]
DECISION_LABELS = {"Decision", "Why", "Alternatives", "History"}
METHOD_LABELS = {"Input", "Output", "How", "Code", "Settings"}
METHOD_REQUIRED = ("Input", "Output", "How", "Code")
# a Code row points at what runs the step: `file.py`, or `file.py::symbol`
CODE_REF = re.compile(r"([\w./-]+\.py)(?:::(\w+))?")
FILE_EXT = re.compile(r"\.(ya?ml|py|csv|json|svg|png|md|txt|toml)(::\w+)?$")
# a finding titled as a fix is a History line on the entry it corrects, not an entry
FIX_WORDS = re.compile(r"\b(bug|bugs|defect|defects|fix|fixed|fixes|rewrite|rewritten|typo|notebook)\b", re.I)


MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\w$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])", re.S)
MATH_TOKEN = re.compile(r"mathx(\d+)x")
MATH_HTML: list[str] = []
MATH_SRC: list[str] = []


def stash_math(m: re.Match) -> str:
    """One `$…$` or `$$…$$` becomes a plain token, so markdown, the reference
    linker and the glossary linker all pass over it; build() puts the MathML
    back once every substitution has run."""
    latex, block = (m.group(1), True) if m.group(1) is not None else (m.group(2), False)
    latex = latex.strip()
    if latex_to_mathml is None:
        rendered = f"<code>{esc(latex)}</code>"
    else:
        rendered = latex_to_mathml(latex, display="block" if block else "inline")
        rendered = f'<span class="math{" block" if block else ""}">{rendered}</span>'
    MATH_HTML.append(rendered)
    MATH_SRC.append(latex)
    return f"mathx{len(MATH_HTML) - 1}x"


def restore_math(html_: str, source: bool = False) -> str:
    """Tokens back to MathML, or to the LaTeX they came from for a hover card."""
    pool = MATH_SRC if source else MATH_HTML
    return MATH_TOKEN.sub(lambda m: pool[int(m.group(1))] if int(m.group(1)) < len(pool) else m.group(0), html_)


NUMERIC = re.compile(r"^[-+−]?[\d,.]+%?$|^—$")
FLOAT = re.compile(r"^[-+]?\d+\.\d+$")
HEXID = re.compile(r"^[0-9a-f]{16,}$")


def num(cell: str) -> str:
    """Display form of a numeric cell: floats to four decimals, `750.0` to `750`."""
    c = cell.strip()
    if FLOAT.match(c):
        f = float(c)
        return (
            str(int(f))
            if f == int(f) and c.endswith(".0")
            else f"{f:.4f}".rstrip("0").rstrip(".") if len(c.split(".")[1]) > 4 else c
        )
    return cell


def md(text: str) -> str:
    """Markdown → HTML; every table gets the report's table styling and a scroll wrapper.
    A spaced `--` in prose renders as a dash; code is left alone."""
    text = "".join(
        p if i % 2 else MATH.sub(stash_math, re.sub(r"(?<=\s)--(?=\s)", "—", p))
        for i, p in enumerate(re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S))
    )
    text = HASH_REF.sub(r"\1\\#", text)
    # markdown needs a blank line where a table or list starts after prose, and
    # where prose or a list follows a table; record authors rarely leave one
    text = re.sub(r"(?m)^(?![ \t]*\|)(\S.*)\n(?=[ \t]*\|)", r"\1\n\n", text)
    text = re.sub(r"(?m)^([ \t]*\|.*)\n(?=[ \t]*[^|\s])", r"\1\n\n", text)
    text = re.sub(r"(?m)^(?![ \t]*(?:[-*] |\d+\. ))(\S.*)\n(?=[ \t]*(?:[-*] |\d+\. ))", r"\1\n\n", text)
    out = markdown.markdown(text, extensions=["tables", "fenced_code"])
    out = re.sub(
        r"<td([^>]*)>([^<]*)</td>",
        lambda m: (
            f'<td class="n"{m.group(1)}>{num(m.group(2))}</td>' if NUMERIC.match(m.group(2).strip()) else m.group(0)
        ),
        out,
    )

    def wrap(m: re.Match) -> str:
        tall = " tall" if m.group(0).count("<tr>") - 1 > TABLE_ROWS else ""
        return f'<div class="scroll{tall}"><table class="data{tall}">{m.group(1)}</table></div>'

    return re.sub(r"<table>(.*?)</table>", wrap, out, flags=re.S)


def cap(text: str) -> str:
    """A row value starts a sentence once its label is a heading, so its first
    letter is upper-cased; code, links and emphasis at the start are left alone."""
    t = text.lstrip()
    return t[0].upper() + t[1:] if t and t[0].islower() else text


def md_inline(text: str) -> str:
    out = md(text).strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S) if out.count("<p>") == 1 else out


def repo_url() -> str | None:
    """https URL of origin, for PR links given as a bare number."""
    try:
        url = subprocess.run(
            ["git", "-C", str(EXPERIMENTS), "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
    url = re.sub(r"^ssh://git@", "https://", url)
    return re.sub(r"\.git$", "", url) or None


def esc(v) -> str:
    return html.escape(str(v)) if v is not None else ""


def prose(text: str) -> str:
    """Prose only: code, equations, tables and link targets removed, link text and
    citations kept, emphasis markers dropped — the text the voice rules read."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = MATH.sub(" ", re.sub(r"`[^`\n]*`", "x", text))
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("|"))
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("*", "")


def sentences(text: str) -> list[str]:
    """Sentences of a prose run: split at a full stop, question or exclamation
    mark followed by space, and at every line break, so a bullet is its own
    sentence. A stop inside `e.g.` or a decimal does not split."""
    text = ABBREV_BEFORE_STOP.sub(lambda m: m.group(0).replace(".", "\x00"), prose(text))
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.replace("\x00", ".").strip() for s in parts if s.strip()]


def voice_problems(where: str, text: str, length: bool = True) -> list[str]:
    """What the voice rules reject in one prose field: a sentence too long or
    semicolon-stacked, and a citation with nothing of its own sentence before
    it — `see [F3]`, `per [D6]`, `([D2]) lifts` — the reader needs the words."""
    out = []
    for s in sentences(text):
        n = len(s.split())
        if length and n > SENTENCE_WORDS:
            out.append(f"{where} has a {n}-word sentence (cap {SENTENCE_WORDS}): {s[:50]}…")
        if length and s.count(";") > SENTENCE_SEMICOLONS:
            out.append(f"{where} has a sentence with {s.count(';')} semicolons (cap {SENTENCE_SEMICOLONS}): {s[:50]}…")
        for m in CITATION.finditer(s):
            if m.group(1).startswith("H"):
                continue  # a clause id reads on its own: "confirms [H2]"
            clause = re.split(r"[;:—]", s[: m.start()])[-1]  # the gloss belongs to the citation's own clause
            before = re.sub(r"\[[DFMH]\d+\]|[^\w\s]", " ", clause).split()
            if len(before) < CITATION_WORDS:
                out.append(f"{where} cites [{m.group(1)}] with no words saying what it is: {s[:50]}…")
    return out


def words(text: str) -> int:
    """Prose word count: table rows, code fences, equations and link targets do not count."""
    text = MATH.sub(" ", re.sub(r"```.*?```", "", text, flags=re.S))
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("|"))
    return len(re.sub(r"\]\([^)]*\)", "]", text).split())


class Entry:
    """One `### F<n> — title · weight [F<k>]` or `### D<n> — title` entry.
    A heading that does not parse keeps an empty id and the raw heading as title."""

    def __init__(self, kind: str, head: str, body: str):
        m = re.match(rf"({kind}\d+)\s*[—–-]\s*(.+?)(?:\s*·\s*(\w+)(?:\s+(F\d+))?)?$", head.strip())
        self.id, self.title, self.weight, self.target = (
            (m.group(1), m.group(2), m.group(3) or "", m.group(4) or "") if m else ("", head.strip(), "", "")
        )
        self.kv = parse_kv(body)


def parse_entries(block: str, kind: str) -> list[Entry]:
    return [Entry(kind, *chunk.partition("\n")[::2]) for chunk in re.split(r"^### ", block, flags=re.M)[1:]]


def outside_rows(block: str) -> list[str]:
    """Lines of a rows-only section that are neither a `- **Label**:` row nor its
    indented continuation — headings and loose prose the row caps would miss."""
    block = re.sub(r"(?ms)^<.*?>$", "", block)
    return [l.strip() for l in block.splitlines() if l.strip() and not l.startswith("  ") and not KV.match(l)]


class Record:
    def __init__(self, path: Path):
        self.path = path
        self.dir = path.parent
        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER.match(text)
        if not match:
            raise ValueError(f"{path}: missing YAML frontmatter")
        self.meta = yaml.safe_load(match.group(1)) or {}
        self.sections = self._split(text[match.end() :])
        self.slug = self.meta.get("slug") or self.dir.name
        self.category = self.meta.get("category") or self.dir.parent.name
        self.title = self.meta.get("title") or self.slug
        self.status = self.meta.get("status", "draft")
        self.builds_on = [(self.dir / p).resolve() for p in (self.meta.get("builds_on") or [])]
        self.children: list[Record] = []
        self.problems: list[str] = []
        self.counts = {"Figure": 0, "Table": 0}
        self.anchors: set[str] = set()  # assets already inlined on the page
        hyp = parse_kv(self.sections.get("hypothesis", ""))
        outcome = re.sub(r"<.*?>", "", hyp.get("Outcome", "")).strip().split()[:1]
        self.outcome = outcome[0].lower() if outcome and outcome[0].lower() != "open" else ""
        if self.status == "concluded":
            for key, name in REQUIRED_WHEN_CONCLUDED.items():
                if not strip_placeholders(self.sections.get(key, "")):
                    self.problems.append(f"concluded record has no ## {name}")
        self.findings = parse_entries(strip_placeholders(self.sections.get("findings", "")), "F")
        self.decisions = parse_entries(strip_placeholders(self.sections.get("decisions", "")), "D")
        self.methods = parse_entries(strip_placeholders(self.sections.get("methods", "")), "M")
        for f in self.findings:
            if not f.id:
                self.problems.append(f"finding heading not `### F<n> — title · weight`: {f.title}")
        for m in self.methods:
            if not m.id:
                self.problems.append(f"method step heading not `### M<n> — title`: {m.title}")
        self.lint()

    def lint(self) -> None:
        """Enforce the record template's caps so the report stays scannable."""

        def over(eid: str, label: str, text: str, cap: int) -> None:
            if (n := words(text)) > cap:
                self.problems.append(f"{eid} {label} is {n} words (cap {cap})")

        # a title of a few words names a topic ("Outputs"); the fold heading must say the step or the choice
        for e in self.decisions + self.methods:
            if e.id and words(e.title) < TITLE_MIN_WORDS:
                what = "the choice made" if e.id.startswith("D") else "what the step does"
                self.problems.append(f"{e.id} title names a topic, not {what}: {e.title}")

        keys = {f.id for f in self.findings if f.weight == "key"}
        clause_ids = {c["id"] for c in clause_rows(self)}
        if self.status == "concluded":
            for c in clause_rows(self):
                if not any(clause_outcome(f, c["id"]) for f in c["findings"]):
                    self.problems.append(
                        f"{c['id']} is decided by no key finding: no Reading sentence names [{c['id']}] with confirmed, refuted or inconclusive"
                    )
        for f in self.findings:
            fid, kv = f.id, f.kv
            if not fid:
                continue
            for label in ("Summary", "Reading", "Implication"):
                over(fid, label, kv.get(label, ""), CAPS[label])
                if refs := TICKET.findall(kv.get(label, "")):
                    self.problems.append(f"{fid} {label} cites a ticket ({', '.join(refs)}): findings read cold")
            if len(nums := numbers(kv.get("Summary", ""))) > SUMMARY_NUMBERS:
                self.problems.append(
                    f"{fid} Summary carries {len(nums)} numbers (cap {SUMMARY_NUMBERS}): the rest belong in the Result"
                )
            if f.weight == "minor" and (extra := sorted(set(kv) - MINOR_LABELS)):
                self.problems.append(f"{fid} is minor but has {', '.join(extra)}")
            if f.weight and not strip_placeholders(kv.get("Reading", "")):
                self.problems.append(
                    f"{fid} is {f.weight} but has no Reading: a number with no meaning is not a result"
                )
            reading = kv.get("Reading", "")
            if (extra_nums := [n for n in numbers(reading) if n not in numbers(kv.get("Summary", ""))]) and len(
                extra_nums
            ) > READING_NUMBERS:
                self.problems.append(
                    f"{fid} Reading carries {len(extra_nums)} numbers beyond the Summary's (cap {READING_NUMBERS}): the rest belong in the Result"
                )
            if words(reading) > BLOCK_WORDS and not LEAD_BULLET.search(reading):
                self.problems.append(
                    f"{fid} Reading is one paragraph of {words(reading)} words: split it into bold-lead blocks"
                )
            for label in ("Summary", "Reading", "Implication"):
                self.problems += voice_problems(f"{fid} {label}", kv.get(label, ""))
            if f.weight == "key" and kv.get("Reading") and clause_ids:
                cited = set(HCITE.findall(reading + " " + kv.get("Summary", "")))
                if not cited:
                    self.problems.append(f"{fid} is key but its Reading cites no clause as [H<n>]")
                elif not any(clause_outcome(f, cid) for cid in cited):
                    self.problems.append(
                        f"{fid} names {', '.join(sorted(cited))} but no sentence with the citation says confirmed, refuted or inconclusive"
                    )
            if re.search(r"!\[[^\]]*\[", kv.get("Result", "")):
                self.problems.append(f"{fid} Result image alt text carries a bracket, so the image does not render")
            if f.weight == "supporting" and f.target not in keys:
                self.problems.append(f"{fid} is supporting but names no key finding (`· supporting F<k>`)")
            if hits := sorted({w.lower() for w in FIX_WORDS.findall(f"{f.title} {kv.get('Summary', '')}")}):
                self.problems.append(f"{fid} reads as a fix ({', '.join(hits)}): revise the F<n> it corrects instead")
        for d in self.decisions:
            did, kv = d.id, d.kv
            if not did:
                continue
            over(did, "Decision", kv.get("Decision", ""), CAPS["Decision"])
            over(did, "Why", kv.get("Why", ""), CAPS["Why"])
            for alt in bullets(kv.get("Alternatives", "")) or [kv.get("Alternatives", "")]:
                over(did, "Alternatives bullet", alt, CAPS["Alternatives"])
            if extra := sorted(set(kv) - DECISION_LABELS):
                self.problems.append(f"{did} has labels outside the template: {', '.join(extra)}")
            for label in ("Decision", "Why", "Alternatives"):
                self.problems += voice_problems(f"{did} {label}", kv.get(label, ""))
        for m in self.methods:
            mid, kv = m.id, m.kv
            if not mid:
                continue
            over(mid, "How", kv.get("How", ""), CAPS["How"])
            if extra := sorted(set(kv) - METHOD_LABELS):
                self.problems.append(f"{mid} has labels outside the template: {', '.join(extra)}")
            if missing := [l for l in METHOD_REQUIRED if not strip_placeholders(kv.get(l, ""))]:
                self.problems.append(f"{mid} has no {', '.join(missing)}: a step reads cold or not at all")
            self.check_code(mid, kv.get("Code", ""))
            how = kv.get("How", "")
            if words(how) > BLOCK_WORDS and not NUMBERED_STEP.search(how):
                self.problems.append(f"{mid} How is one paragraph of {words(how)} words: write it as numbered steps")
            for label in ("Input", "Output", "How"):
                self.problems += voice_problems(f"{mid} {label}", kv.get(label, ""))
            self.check_settings(mid, kv.get("Settings", ""))
        for key, name in (("design", "Design"), ("datasets", "Datasets"), ("verdict", "Verdict")):
            block = strip_placeholders(self.sections.get(key, ""))
            for line in outside_rows(block):
                self.problems.append(f"{name} has content outside `- **Label**:` rows: {line[:60]}")
            for label, value in parse_kv(block).items():
                over(name, label, value, CAPS.get(label, CAPS["Design"]))
                if not (name == "Verdict" and label == "Evidence"):  # Evidence is the list of citations by design
                    self.problems += voice_problems(f"{name} {label}", value)
                if (
                    label in ("Discussion", "Rationale")
                    and words(value) > BLOCK_WORDS
                    and not LEAD_BULLET.search(value)
                ):
                    self.problems.append(
                        f"{name} {label} is one paragraph of {words(value)} words: split it into bold-lead blocks"
                    )
        for label, value in parse_kv(strip_placeholders(self.sections.get("hypothesis", ""))).items():
            self.problems += voice_problems(f"Hypothesis {label}", value, length=label != "Predictions")
            if label == "Rationale" and words(value) > BLOCK_WORDS and not LEAD_BULLET.search(value):
                self.problems.append(
                    f"Hypothesis Rationale is one paragraph of {words(value)} words: split it into bold-lead blocks"
                )
        for line in self.key_lines():
            if not TLDR_LINE.match(line):
                self.problems.append(f"TL;DR line is not `**F<n> — <title>.** <Summary>`: {line[:50]}…")
        if self.status == "concluded" and not strip_placeholders(
            parse_kv(self.sections.get("verdict", "")).get("Discussion", "")
        ):
            self.problems.append("concluded record has no Verdict Discussion")
        predictions = strip_placeholders(parse_kv(self.sections.get("hypothesis", "")).get("Predictions", ""))
        if predictions:
            clauses = CLAUSE.findall(predictions)
            if not clauses:
                self.problems.append(
                    "Predictions has no `- **H<n> — <claim>**:` line: one per clause, so findings can cite H<n>"
                )
            elif len(clauses) > 1 and not VERDICT_RULE.search(predictions):
                self.problems.append("Predictions has several clauses but no `- **Verdict rule**:` line")

    def check_settings(self, mid: str, value: str) -> None:
        """Every config key in a Settings row carries a gloss in parentheses —
        `clusters` (how many groups of sources) — so the row reads without the
        repo open. A file path is the place the keys live, not a key."""
        value = strip_placeholders(value)
        if not value or value.strip().lower() == "none":
            return
        for m in re.finditer(r"`([^`]+)`(\s*(?:\(|in\b|:)|)", value):
            key = m.group(1)
            if "/" in key or FILE_EXT.search(key) or m.group(2):  # glossed, or a section named as the place
                continue
            self.problems.append(f"{mid} Settings key `{key}` carries no gloss: write `{key}` (what it steers)")

    def check_code(self, mid: str, value: str) -> None:
        """A Code row must resolve: the file exists and, when a symbol is named, that
        file defines it. A step pointing at code that moved is worse than no step."""
        if not (refs := CODE_REF.findall(strip_placeholders(value))):
            self.problems.append(f"{mid} Code names no `<file>.py` or `<file>.py::<symbol>`")
            return
        for path, symbol in refs:
            for base in (self.dir, EXPERIMENTS, EXPERIMENTS.parent):
                if (found := base / path).is_file():
                    break
            else:
                self.problems.append(f"{mid} Code points at a file that does not exist: {path}")
                continue
            if symbol and not re.search(
                rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b", found.read_text(encoding="utf-8")
            ):
                self.problems.append(f"{mid} Code names {symbol}, which {path} does not define")

    def key_findings(self) -> list[Entry]:
        return [f for f in self.findings if f.weight == "key"]

    @staticmethod
    def _split(body: str) -> dict[str, str]:
        sections, key, buf = {}, "_head", []
        for line in body.splitlines():
            if line.startswith("## "):
                sections[key] = "\n".join(buf).strip()
                key, buf = line[3:].strip().lower().replace(";", "").replace(" ", "-"), []
            else:
                buf.append(line)
        sections[key] = "\n".join(buf).strip()
        return sections

    def key_lines(self) -> list[str]:
        """TL;DR bullets that are not `- **Label**:` rows — one per key finding."""
        return [b for b in bullets(self.sections.get("tldr", "")) if not KV.match("- " + b)]


def bullets(block: str) -> list[str]:
    """Top-level `- ` bullets with their indented continuation lines joined."""
    out = []
    for line in block.splitlines():
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif out and line.startswith("  ") and line.strip():
            out[-1] += " " + line.strip()
    return out


CLAUSE = re.compile(r"^\s*- \*\*H\d+ — [^*]+\*\*:", re.M)
CLAUSE_LINE = re.compile(r"^\s*- \*\*(H\d+) — ([^*]+)\*\*:\s*(.*)$")


def clauses(predictions: str) -> tuple[str, list[tuple[str, str, str]], str]:
    """A Predictions row split into its framing prose, its `(id, claim, conditions)`
    clauses, and whatever follows them (the Verdict rule and read-beside lines)."""
    before, items, after = [], [], []
    for line in predictions.splitlines():
        m = CLAUSE_LINE.match(line)
        if m:
            items.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
        elif items and line.strip():
            after.append(line)
        elif not items:
            before.append(line)
    return "\n".join(before).strip(), items, "\n".join(after).strip()


VERDICT_RULE = re.compile(r"^\s*- \*\*Verdict rule\*\*:", re.M)


def parse_kv(block: str) -> dict[str, str]:
    """`- **Label**: value` bullets → dict. Continuation lines keep their nesting
    (only the bullet's own two-space indent is removed) and blank lines survive,
    so a value may hold sub-lists, tables and several paragraphs."""
    out, label = {}, None
    for line in block.splitlines():
        m = KV.match(line)
        if m:
            label, out[label] = m.group(1), m.group(2)
        elif line.startswith("- "):
            label = None
        elif label and (line.startswith("  ") or not line.strip()):
            out[label] += "\n" + line[2:].rstrip()
    return {k: v.strip() for k, v in out.items()}


def strip_placeholders(text: str) -> str:
    return "" if re.fullmatch(r"\s*<.*>\s*", text, flags=re.S) else text


def load_records() -> list[Record]:
    records = [Record(p) for p in sorted(EXPERIMENTS.glob("*/*/README.md"))]
    by_dir = {r.dir.resolve(): r for r in records}
    for r in records:
        for parent in r.builds_on:
            if parent in by_dir:
                by_dir[parent].children.append(r)
    return records


def families(records: list[Record]) -> list[list[Record]]:
    """Each family is a root (nothing it builds on exists here) plus its descendants."""
    by_dir = {r.dir.resolve(): r for r in records}
    roots = [r for r in records if not any(p in by_dir for p in r.builds_on)]

    def walk(r: Record, depth: int, seen: set) -> list[tuple[Record, int]]:
        if r.dir in seen:
            return []
        seen.add(r.dir)
        out = [(r, depth)]
        for c in sorted(r.children, key=lambda c: (str(c.meta.get("concluded") or "9999"), c.slug)):
            out += walk(c, depth + 1, seen)
        return out

    roots.sort(key=lambda r: (str(r.meta.get("concluded") or "9999"), r.slug))
    return [walk(r, 0, set()) for r in roots]


FIG_PATCH = re.compile(r'(<g id="(?:figure|axes)_\d+">\s*<g id="patch_\d+">\s*<path[^>]*?fill: )#ffffff')


def transparent_bg(svg: str) -> str:
    """Matplotlib paints the figure and each axes white; the page supplies the
    background, so those patches become transparent and the figure sits on either theme."""
    return FIG_PATCH.sub(r"\1none", svg)


def inline_asset(record: Record, src: str) -> str | None:
    target = (record.dir / src).resolve()
    if not target.is_file():
        record.problems.append(f"missing asset: {src}")
        return None
    suffix = target.suffix.lower()
    if suffix == ".svg":
        svg = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", target.read_text(encoding="utf-8")).strip()
        svg = transparent_bg(svg)
        return unique_ids(svg, re.sub(r"[^A-Za-z0-9_-]", "-", f"{record.slug}-{target.stem}") + "-")
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix[1:]}"
        data = base64.b64encode(target.read_bytes()).decode()
        return f'<img src="data:{mime};base64,{data}" alt="{esc(src)}">'
    if suffix == ".csv":
        with target.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return "<table></table>"
        numeric = [
            any(NUMERIC.match(r[i].strip()) for r in rows[1:] if i < len(r))
            and all(not r[i].strip() or NUMERIC.match(r[i].strip()) for r in rows[1:] if i < len(r))
            for i in range(len(rows[0]))
        ]
        cls = lambda i, c="": ' class="n"' if numeric[i] else (' class="id"' if HEXID.match(c.strip()) else "")
        head = "".join(f"<th{cls(i)}>{esc(c)}</th>" for i, c in enumerate(rows[0]))
        body = "".join(
            "<tr>" + "".join(f"<td{cls(i, c)}>{esc(num(c))}</td>" for i, c in enumerate(row)) + "</tr>"
            for row in rows[1:]
        )
        check_width(record, rows, src)
        wide = " wide" if is_wide(rows) else ""
        tall = " tall" if len(rows) - 1 > TABLE_ROWS else ""
        return f'<table class="data{wide}{tall}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return None


def unique_ids(svg: str, prefix: str) -> str:
    """Prefix every id (and its references) so several inlined SVGs do not collide."""
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', svg)
    return re.sub(
        r'(href="#|url\(#)([^")]+)',
        lambda m: m.group(1) + prefix + m.group(2) if m.group(2) in ids else m.group(0),
        svg,
    )


def check_width(record: Record, rows: list[list[str]], what: str) -> None:
    if len(rows[0]) > 8:
        record.problems.append(
            f"{what} has {len(rows[0])} columns; it will scroll — consider transposing or splitting it"
        )


def is_wide(rows: list[list[str]]) -> bool:
    """A table takes the full width of its finding when it has more than three
    columns or a cell of prose length; otherwise it sits beside the text."""
    return len(rows[0]) > 3 or any(len(c) > 60 for r in rows for c in r)


SENTENCE_END = (".", "?", "!", ":", ";", "—")


def caption(record: Record, kind: str, lead: str, prose: str) -> str:
    """A numbered caption: `Figure 3 — <lead>. <prose>`. The lead is the result's
    own one-line description and the prose whatever the record wrote around it;
    the two are separate sentences, so the lead gets a full stop of its own."""
    record.counts[kind] += 1
    label = f'<b class="lbl">{kind} {record.counts[kind]}</b>'
    lead = lead.strip()
    if lead and not lead.endswith(SENTENCE_END):
        lead += "."
    body = " ".join(x for x in (lead, prose.strip()) if x)
    return f'<figcaption>{label}{" — " + body if body else ""}</figcaption>'


ASSET_LINK = re.compile(r"!?\[([^\]]*)\]\(((?:figures|tables)/[^)]+\.(?:csv|svg|png|jpe?g|gif|webp))\)", re.I)


def asset_anchor(record: Record, src: str) -> str:
    """In-page id of an inlined table or figure, so prose that links the file
    reaches the rendering instead of a path the self-contained page cannot open."""
    return f"{record.slug}/asset/{re.sub(r'[^A-Za-z0-9_-]', '-', Path(src).stem)}"


def render_asset(record: Record, src: str, lead: str, prose: str) -> tuple[str, bool]:
    """One inlined figure or table with its numbered caption. Returns (html, wide)."""
    inlined = inline_asset(record, src)
    if inlined is None:
        return f'<p class="missing">missing: {esc(src)}</p>', False
    wide = 'class="data wide"' in inlined
    kind = "Table" if inlined.startswith("<table") else "Figure"
    cap_html = caption(record, kind, esc(lead), md_inline(prose))
    if inlined.startswith("<table"):
        inlined = f'<div class="scroll{" tall" if "tall" in inlined[:40] else ""}">{inlined}</div>'
    else:
        inlined = f'<div class="fig">{inlined}</div>'
    anchor = asset_anchor(record, src)
    aid = ""
    if anchor not in record.anchors:
        record.anchors.add(anchor)
        aid = f' id="{anchor}"'
    return f"<figure{aid}>{inlined}{cap_html}</figure>", wide


def render_result(record: Record, value: str) -> tuple[str, bool]:
    """A finding's Result: a figure or table from figures/ or tables/, or an inline
    markdown table; prose around either becomes the caption. Returns (html, wide)."""
    # only an inlinable asset counts as the Result; a tables/*.md link is a pointer, not the result
    m = ASSET_LINK.search(value)
    if m:
        prose = " ".join(l.strip() for l in value.replace(m.group(0), "").splitlines() if l.strip())
        return render_asset(record, m.group(2), m.group(1), prose)
    lines = value.splitlines()
    rows = [l.strip() for l in lines if l.lstrip().startswith("|")]
    if rows:
        table = md("\n".join(rows))
        if "<table" not in table:
            record.problems.append("finding Result has a table markdown cannot parse")
            return f'<p class="missing">unparseable table: {esc(rows[0])}</p>', False
        cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
        check_width(record, cells, "inline Result table")
        wide = is_wide(cells)
        if wide:
            table = table.replace('<table class="data">', '<table class="data wide">')
        prose = " ".join(l.strip() for l in lines if l.strip() and not l.lstrip().startswith("|"))
        return f"<figure>{table}{caption(record, 'Table', '', md_inline(prose))}</figure>", wide
    record.problems.append("finding Result is not a figure or table")
    return f'<p class="missing">Result must be a figure or table, got: {esc(value)}</p>', False


def render_datasets(record: Record, block: str) -> str:
    """One card per dataset: its prose beside the first figure, further figures in
    pairs, then every statistics table at full width, each with a numbered caption."""
    out = []
    for n, (label, value) in enumerate(parse_kv(block).items(), 1):
        if not strip_placeholders(value):
            continue
        links = ASSET_LINK.findall(value)
        prose = value
        for full in ASSET_LINK.finditer(value):
            # a trailing link is the row's statistics table, shown below; one inside a
            # sentence stays as a link to its inlined rendering
            tail = value[full.end() :].strip()
            keep = tail and not tail.startswith("![") and not tail.startswith("[")
            prose = prose.replace(
                full.group(0),
                f"[{full.group(1)}]({full.group(2)})" if keep and not full.group(0).startswith("!") else "",
            )
        prose = " ".join(l.strip() for l in prose.splitlines() if l.strip())
        inline_srcs = {
            m.group(2)
            for m in ASSET_LINK.finditer(value)
            if (t := value[m.end() :].strip())
            and not t.startswith("![")
            and not t.startswith("[")
            and not m.group(0).startswith("!")
        }
        figs = [(lead, src) for lead, src in links if not src.lower().endswith(".csv") and src not in inline_srcs]
        tabs = [(lead, src) for lead, src in links if src.lower().endswith(".csv") and src not in inline_srcs]
        more = [(lead, src) for lead, src in links if src in inline_srcs]
        text = f'<div class="read">{md(cap(prose))}</div>'
        top = f'<div class="split">{text}{render_asset(record, figs[0][1], figs[0][0], "")[0]}</div>' if figs else text
        rest = (
            f'<div class="pair">{"".join(render_asset(record, src, lead, "")[0] for lead, src in figs[1:])}</div>'
            if len(figs) > 1
            else ""
        )
        tables = "".join(render_asset(record, src, lead, "")[0] for lead, src in tabs)
        fold = (
            (
                f'<details class="more"><summary>more tables and figures ({len(more)})</summary><div class="assets">'
                + "".join(render_asset(record, src, lead, "")[0] for lead, src in more)
                + "</div></details>"
            )
            if more
            else ""
        )
        out.append(
            f'<article class="ds" id="{record.slug}/dataset-{n}" data-title="{esc(plain(label))}"><h3>{md_inline(label)}</h3>{top}{rest}{tables}{fold}</article>'
        )
    return "".join(out)


def rows_html(pairs: list[tuple[str, str]]) -> str:
    """Label-value rows: the label as a small caption in a left column, the value beside it."""
    cells = "".join(f"<b>{esc(k)}</b><div>{v}</div>" for k, v in pairs if v)
    return f'<div class="rows">{cells}</div>' if cells else ""


def constraints(record: Record) -> dict[str, str]:
    """The Design rows of CONSTRAINT_ROWS, empty where a row is missing; a missing
    row is a warning, since the strip is what says whether the experiment can run."""
    design = parse_kv(record.sections.get("design", ""))
    out = {k: strip_placeholders(design.get(k, "")) for k in CONSTRAINT_ROWS}
    for k, v in out.items():
        if not v:
            record.problems.append(f"Design has no {k} row (constraints)")
    return out


def render_needs(record: Record) -> str:
    """One thin strip of what the experiment needs; a need that is not none or
    local is the one to notice, so it carries the highlight."""
    rows = constraints(record)
    if not any(rows.values()):
        return ""
    cells = "".join(
        f'<div class="{"hot" if needs_glance(v) else ""}"><span class="lbl">{esc(k)}</span>{md_inline(cap(v)) if v else "—"}</div>'
        for k, v in rows.items()
    )
    return f'<div class="needs"><div><span class="lbl">Needs</span></div>{cells}</div>'


def clause_rows(record: Record) -> list[dict]:
    """The Predictions clauses with their conditions split and the findings that read each."""
    predictions = parse_kv(record.sections.get("hypothesis", "")).get("Predictions", "")
    out = []
    for cid, claim, cond in clauses(predictions)[1]:
        m = CONDITIONS.match(cond.strip())
        yes, no, note = (m.group(1), m.group(2), m.group(3) or "") if m else (cond, "", "")
        names = re.compile(rf"(?<![\w\[]){cid}\b|\[{cid}\]")
        carriers = [
            f
            for f in record.findings
            if f.id and f.weight == "key" and names.search(f.kv.get("Reading", "") + f.kv.get("Summary", ""))
        ]
        out.append(
            {
                "id": cid,
                "claim": claim,
                "yes": yes.strip(),
                "no": no.strip().rstrip("."),
                "note": note.strip(),
                "findings": carriers,
            }
        )
    return out


def clause_outcome(f: Entry, cid: str) -> str:
    """What the finding says about the clause: the verdict word in the sentence that names it."""
    names = re.compile(rf"(?<![\w\[]){cid}\b|\[{cid}\]")
    for sentence in sentences(f.kv.get("Reading", "") + " " + f.kv.get("Summary", "")):
        if names.search(sentence):
            low = sentence.lower()
            if "inconclusive" in low:
                return "inconclusive"
            if "refut" in low:
                return "refuted"
            if "confirm" in low:
                return "confirmed"
    return ""


def clause_verdict(record: Record, row: dict) -> str:
    """The clause's own verdict: what its first reading finding says; `open` before any."""
    for f in row["findings"]:
        if v := clause_outcome(f, row["id"]):
            return v
    return "" if record.outcome else "open"


def verdict_badge(record: Record, row: dict) -> str:
    v = clause_verdict(record, row)
    if v == "open":
        return '<span class="badge b-running">open</span>'
    return f'<span class="badge b-{v}">{v}</span>' if v else '<span class="muted">not read</span>'


def render_clauses(record: Record) -> str:
    """The clause table across the page: claim, the conditions, and the finding that
    read it with its Summary. A record whose Predictions are prose gets that prose."""
    rows = clause_rows(record)
    predictions = parse_kv(record.sections.get("hypothesis", "")).get("Predictions", "")
    if not rows:
        if not strip_placeholders(predictions):
            return ""
        return f'<div class="pred"><span class="lbl">Predictions</span>{md(cap(predictions))}</div>'
    trs = []
    for r in rows:
        # one block per reading finding, so a clause read twice shows two rows
        result = (
            "".join(
                '<div class="reading">'
                + (f'<span class="badge b-{o}">{o}</span> ' if (o := clause_outcome(f, r["id"])) else "")
                + f'<a href="#{record.slug}/{f.id.lower()}"><span class="tag">{f.id}</span></a> {md_inline(cap(f.kv.get("Summary", "")))}'
                + "</div>"
                for f in r["findings"]
            )
            or '<span class="muted">not read yet</span>'
        )
        note = f' — <span class="muted">{md_inline(r["note"])}</span>' if r["note"] else ""
        trs.append(
            f'<tr id="{record.slug}/{r["id"].lower()}"><td><span class="tag">{r["id"]}</span></td><td>{md_inline(r["claim"])}</td>'
            f'<td class="muted">{md_inline(r["yes"])}</td><td class="muted">{md_inline(r["no"])}{note}</td><td class="result">{result}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="data wide clauses"><thead><tr><th>Clause</th><th>Claim</th><th>Confirmed if</th>'
        f'<th>Refuted if</th><th>Result</th></tr></thead><tbody>{"".join(trs)}</tbody></table></div>'
    )


def hypothesis_text(r: Record) -> str:
    """The hypothesis as stated: the Hypothesis section's Statement, else the TL;DR line without its status."""
    statement = strip_placeholders(parse_kv(r.sections.get("hypothesis", "")).get("Statement", ""))
    return statement or TLDR_STATUS.sub("", parse_kv(r.sections.get("tldr", "")).get("Hypothesis", ""))


def render_hypothesis(record: Record, block: str) -> str:
    """The Hypothesis rows; Predictions points at the clause table when the clauses
    are listed, and stays prose otherwise."""
    kv = parse_kv(block)
    pairs = []
    for k, v in kv.items():
        if not strip_placeholders(v):
            continue
        if k == "Outcome":
            continue
        if k == "Predictions" and (parts := clauses(v))[1]:
            before, _, after = parts
            lead = f"{md_inline(cap(before))} " if before else ""
            pairs.append(
                (
                    k,
                    f'<div class="muted">{lead}Shown as the clause table at the top of the page.{" " + md_inline(after) if after else ""}</div>',
                )
            )
        else:
            pairs.append((k, md(cap(v))))
    return rows_html(pairs)


def render_kv(block: str, skip: tuple[str, ...] = ()) -> str:
    kv = parse_kv(block)
    return rows_html([(k, md(cap(v))) for k, v in kv.items() if strip_placeholders(v) and k not in skip])


def more_assets(record: Record, f: Entry, result_src: str) -> str:
    """Tables and figures the Reading or Implication link beyond the Result, inlined
    in a fold at the card's foot so the Result keeps the card."""
    seen, items = {result_src}, []
    for key in ("Reading", "Implication"):
        for lead, src in ASSET_LINK.findall(f.kv.get(key, "")):
            if src not in seen:
                seen.add(src)
                items.append(render_asset(record, src, lead, "")[0])
    if not items:
        return ""
    return (
        f'<details class="more"><summary>more tables and figures ({len(items)})</summary>'
        f'<div class="assets">{"".join(items)}</div></details>'
    )


def render_finding(record: Record, f: Entry, cls: str = "key", extra: str = "") -> str:
    """One finding. A key finding is an open card: Summary, Reading and Implication
    beside a figure Result, or above a table Result with the prose in two columns
    under it. A supporting or minor finding is the same card folded to its title line."""
    kv = f.kv
    m = ASSET_LINK.search(kv.get("Result", ""))
    result_src = m.group(2) if m else ""
    result, _ = render_result(record, kv["Result"]) if kv.get("Result") else ("", False)
    kind = "table" if "<table" in result[:600] else ("figure" if result else "")

    def part(k: str) -> str:
        v = kv.get(k, "")
        if not strip_placeholders(v) or v.strip().lower() == "none":
            return ""
        return f'<div class="fp {k.lower()}"><span class="lbl">{k}</span>{md(cap(v))}</div>'

    summary, read = part("Summary"), part("Reading") + part("Implication")
    two = "read two" if part("Reading") and part("Implication") else "read"
    if kind == "figure":
        body = f'<div class="split"><div class="read">{summary}{read}</div>{result}</div>'
    elif kind == "table":
        body = f'{summary}{result}<div class="{two}">{read}</div>'
    else:
        body = f'{summary}<div class="{two}">{read}</div>'
    hs = sorted(
        set(
            re.findall(r"(?<![\w\[])(H\d+)\b|\[(H\d+)\]", kv.get("Reading", ""))
            and [a or b for a, b in re.findall(r"(?<![\w\[])(H\d+)\b|\[(H\d+)\]", kv.get("Reading", ""))]
        )
    )
    weight = " · ".join(x for x in (f.weight + (f" {f.target}" if f.target else ""), " ".join(hs)) if x)
    head = f'<h3><span class="tag">{f.id}</span>{md_inline(f.title)}<span class="wt">{esc(weight)}</span></h3>'
    attrs = (
        f'id="{record.slug}/{f.id.lower()}" data-title="{esc(f.id)} · {esc(re.sub("<.*?>", "", md_inline(f.title)))}"'
    )
    inner = body + footer(record, kv) + more_assets(record, f, result_src)
    if cls == "key":
        return f'<article class="res key" {attrs}>{head}{inner}{extra}</article>'
    return f'<details class="res {cls}" {attrs}><summary>{head}</summary><div class="body">{inner}</div></details>'


def render_findings(record: Record) -> str:
    """Key findings as open cards, each followed by its supporting findings folded to
    a line; minor findings and any supporting finding without a key target close."""
    out, placed = [], set()
    for f in record.findings:
        if f.weight != "key" or not f.id:
            continue
        subs = [x for x in record.findings if x.id and x.weight == "supporting" and x.target == f.id]
        placed.update(x.id for x in subs)
        out.append(render_finding(record, f, extra="".join(render_finding(record, x, "sub") for x in subs)))
    out += [
        render_finding(record, f, f.weight or "solo")
        for f in record.findings
        if f.id and f.weight != "key" and f.id not in placed
    ]
    return "".join(out)


def mark_refs(record: Record, html_: str, runs: bool = False) -> str:
    """Label every `code` span by what it is: an MLflow run id (linked with
    `mlflow:` in frontmatter, shown short), a commit (linked when the remote is
    known), or a file. In the Runs row any hex id is a run; elsewhere only a
    full 32-hex id is, and 7–12 hex is a commit."""
    base = str(record.meta.get("mlflow") or "").rstrip("/")

    def mark(m: re.Match) -> str:
        v = m.group(1)
        if re.fullmatch(r"[0-9a-f]{32}", v) or (runs and re.fullmatch(r"[0-9a-f]{8,32}", v)):
            chip = f'<code class="run" title="{v}">{v[:8]}</code>'
            return f'<a href="{esc(base)}/runs/{v}">{chip}</a>' if base else chip
        if re.fullmatch(r"[0-9a-f]{7,12}", v) and not v.isdigit():
            chip = f'<code class="commit">{v}</code>'
            return f'<a href="{REPO_URL}/commit/{v}">{chip}</a>' if REPO_URL else chip
        if "/" in v or FILE_EXT.search(v):
            return f'<code class="file">{v}</code>'
        return m.group(0)

    return re.sub(r"<code>([^<]+)</code>", mark, html_)


def footer(record: Record, kv: dict[str, str]) -> str:
    """Runs and the latest History line as two labelled rows."""
    rows = []
    if kv.get("Runs"):
        rows.append(f'<p><b>runs</b><span>{mark_refs(record, md_inline(kv["Runs"]), runs=True)}</span></p>')
    if kv.get("History"):
        last = kv["History"].splitlines()[-1].lstrip("- ").strip()
        rows.append(f"<p><b>history</b><span>{mark_refs(record, md_inline(last))}</span></p>")
    return f'<div class="src">{"".join(rows)}</div>' if rows else ""


def render_decisions(record: Record) -> str:
    """Each decision folded to its line; open, the Decision then Why and the rest as rows."""
    out = []
    for d in record.decisions:
        did, title, kv = d.id, d.title, d.kv
        pairs = [(k, md(cap(v))) for k, v in kv.items() if k != "History" and strip_placeholders(v)]
        if kv.get("History"):
            last = kv["History"].splitlines()[-1].lstrip("- ").strip()
            pairs.append(("History", mark_refs(record, md_inline(last))))
        body = rows_html(pairs)
        when = kv.get("History", "").splitlines()[-1].lstrip("- ").strip()[:10] if kv.get("History") else ""
        tag = f'<span class="tag">{did}</span>' if did else ""
        out.append(
            f'<details class="dec" id="{{slug}}/{did.lower()}" data-title="{esc(did)} · {esc(plain(title))}">'
            f'<summary><h3>{tag}{md_inline(title)}<span class="wt">{esc(when)}</span></h3></summary>'
            f'<div class="body">{body}</div></details>'
        )
    return "".join(out)


def render_method(record: Record) -> str:
    """Each step folded to its line with the code that carries it; open, what goes in
    beside what comes out, how the two connect, and the code and settings rows."""
    out = []
    for m in record.methods:
        mid, kv = m.id, m.kv
        pairs = [(k, md(cap(kv[k]))) for k in ("Input", "Output", "How") if strip_placeholders(kv.get(k, ""))]
        pairs += [
            (k, mark_refs(record, md_inline(kv[k])))
            for k in ("Code", "Settings")
            if strip_placeholders(kv.get(k, "")) and kv[k].strip().lower() != "none"
        ]
        body = rows_html(pairs)
        code = CODE_REF.search(kv.get("Code", ""))
        symbol = f'<span class="wt">{esc(code.group(0))}</span>' if code else ""
        tag = f'<span class="tag">{mid}</span>' if mid else ""
        out.append(
            f'<details class="met" id="{{slug}}/{mid.lower()}" '
            f'data-title="{esc(mid)} · {esc(re.sub("<.*?>", "", md_inline(m.title)))}">'
            f"<summary><h3>{tag}{md_inline(m.title)}{symbol}</h3></summary>"
            f'<div class="body">{body}</div></details>'
        )
    return "".join(out)


def render_steps(record: Record, code: str) -> str:
    """A Reproduce bash block as steps: each run of `#` comment lines titles the
    commands that follow; a group of bare `export` lines is the environment.
    The untitled group before the first comment is the environment when it
    carries an `export`. A number
    the author put in front of a title is dropped, since the list numbers
    itself. Commit ids in a title get the commit chip, the finding ids it names
    become citations; trailing comments are muted."""
    groups, title, cmds = [], [], []
    for line in code.splitlines():
        if line.startswith("#"):
            if cmds:
                groups.append((title, cmds))
                title, cmds = [], []
            title.append(line.lstrip("#").strip())
        elif line.strip():
            cmds.append(line.rstrip())
    if cmds:
        groups.append((title, cmds))
    out, ids = [], {e.id for e in record.findings + record.decisions if e.id}
    for title, cmds in groups:
        env = not title and any(c.lstrip().startswith("export ") for c in cmds)
        head = ""
        if title:
            text = re.sub(r"^(?:step\s*)?\d+[.:)]?\s+", "", " ".join(title), flags=re.I)
            text = re.sub(
                r"(?<!`)\b([0-9a-f]{7,12})\b(?!`)",
                lambda m: m.group(1) if m.group(1).isdigit() else f"`{m.group(1)}`",
                text,
            )
            text = re.sub(
                r"(?<![\[\w])([DF]\d+)\b(?!\])", lambda m: f"[{m.group(1)}]" if m.group(1) in ids else m.group(1), text
            )
            head = f'<div class="t">{mark_refs(record, md_inline(text))}</div>'
        body = "\n".join(re.sub(r"(\s)(#.*)$", r'\1<span class="c">\2</span>', esc(c)) for c in cmds)
        n = sum(1 for o in out if 'class="step">' in o) + 1
        mark = "env" if env else str(n)
        out.append(
            f'<li class="step{" env" if env else ""}"><span class="n">{mark}</span>{head}<pre><code>{body}</code></pre></li>'
        )
    return f'<ol class="steps">{"".join(out)}</ol>'


def render_reproduce(record: Record, block: str) -> str:
    """Prose stays prose; every fenced bash block becomes numbered steps."""
    parts = re.split(r"(?ms)^```(?:bash|sh|zsh)?\n(.*?)^```[ \t]*$", block)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(render_steps(record, part))
        elif part.strip():
            out.append(f'<div class="prose">{md(part)}</div>')
    return "".join(out)


def render_builds(record: Record) -> str:
    if not record.children:
        return ""
    items = "".join(
        f'<a href="#{c.slug}"><span class="t">{esc(c.title)} {badge(c)}</span>'
        f'<span class="m">{md_inline(c.key_lines()[0]) if c.key_lines() else ""}</span></a>'
        for c in record.children
    )
    return f'<div class="rel">{items}</div>'


def badge(r: Record) -> str:
    if r.outcome:
        return f'<span class="badge b-{r.outcome}">{r.outcome}</span>'
    return f'<span class="badge b-{r.status}">{r.status}</span>'


def pr_link(record: Record) -> str:
    pr = str(record.meta.get("pr") or "").strip()
    if not pr:
        return ""
    if re.match(r"https?://", pr):
        m = re.search(r"/(?:pull|merge_requests)/(\d+)", pr)
        return f'<a href="{esc(pr)}">PR{" #" + m.group(1) if m else ""}</a>'
    n = pr.lstrip("#")
    if n.isdigit() and REPO_URL and "github.com" in REPO_URL:
        return f'<a href="{esc(REPO_URL)}/pull/{n}">PR #{n}</a>'
    return esc(f"PR #{n}" if n.isdigit() else pr)


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<.*?>", "", restore_math(md_inline(text), source=True)))).strip()


def tip(head: str, body: str, label: str = "") -> str:
    """data-tip value: a plain heading line, then the body as inline HTML so code
    and emphasis render in the hover card. A label names the row the body is
    quoted from, for a card whose text does not say what it is on its own."""
    # an equation in a card is its LaTeX source: MathML carries quotes that would end the attribute
    inline = restore_math(re.sub(r"\s+", " ", md_inline(body)).strip(), source=True)
    if label:
        inline = f"<strong>{label}:</strong> {inline}"  # not <b>: the card styles its own <b> as the heading line
    return esc(f"{plain(head)}\n{inline}")


SKIP_TAGS = ("code", "a", "pre", "h1", "h2", "h3", "svg", "style", "script")
BLOCK_STARTS = ("<article", "<section", "<h2", '<div class="kv"', '<p class="lede"', '<ul class="lede"')


def sub_prose(html_: str, fn, per_block: bool = False) -> str:
    """Apply `fn(text, block)` to the prose runs of an HTML fragment, leaving text
    inside code, links, headings, figures and id tags alone. `block` counts the
    enclosing article/section/heading, for a caller that wants to act per block."""
    out, skip, block = [], [], 0
    for tok in re.split(r"(<[^>]+>)", html_):
        if tok.startswith("<"):
            m = re.match(r"</?([a-zA-Z0-9]+)", tok)
            name = m.group(1).lower() if m else ""
            if tok.startswith("</"):
                if skip and skip[-1] == name:
                    skip.pop()
            elif (name in SKIP_TAGS or 'class="tag"' in tok) and not tok.endswith("/>"):
                skip.append(name)
            if tok.startswith(BLOCK_STARTS):
                block += 1
            out.append(tok)
        else:
            out.append(tok if skip else fn(tok, block))
    return "".join(out)


def link_refs(record: Record, html_: str) -> str:
    """Bracketed `[D<n>]` / `[F<n>]` / `[M<n>]` / `[H<n>]` citations in prose become
    in-page links carrying a hover card — the decision's Decision line, the finding's
    Summary, the method step's Output, the clause's conditions. A bare `F1` stays
    text: it may be the metric. Text inside
    code, links, headings and id tags is left alone."""
    tips = {d.id: tip(f"{d.id} — {d.title}", d.kv.get("Decision", "")) for d in record.decisions if d.id}
    tips |= {f.id: tip(f"{f.id} — {f.title}", f.kv.get("Summary", "")) for f in record.findings if f.id}
    # a step's Output reads as a fragment out of context, so the card says which row it is
    tips |= {m.id: tip(f"{m.id} — {m.title}", cap(m.kv.get("Output", "")), "Output") for m in record.methods if m.id}
    predictions = parse_kv(record.sections.get("hypothesis", "")).get("Predictions", "")
    tips |= {cid: tip(f"{cid} — {claim}", cond) for cid, claim, cond in clauses(predictions)[1]}
    if not tips:
        return html_
    ref = re.compile(r"\[([DFMH]\d+)\]")

    def link(m: re.Match) -> str:
        rid = m.group(1)
        if rid not in tips:
            return m.group(0)
        return f'<a class="ref" href="#{record.slug}/{rid.lower()}" data-tip="{tips[rid]}">[{rid}]</a>'

    return sub_prose(html_, lambda text, _: ref.sub(link, text))


class Term:
    """One `- **Term** (alias, alias): definition` line of experiments/GLOSSARY.md."""

    LINE = re.compile(r"^- \*\*(.+?)\*\*(?:\s*\((.+?)\))?:\s*(.+)$")

    def __init__(self, name: str, aliases: list[str], definition: str, section: str):
        self.name, self.aliases, self.definition, self.section = name, aliases, definition, section
        self.slug = re.sub(r"[^\w]+", "-", name.lower()).strip("-")

    @property
    def forms(self) -> list[str]:
        return [self.name, *self.aliases]


def load_glossary() -> list[Term]:
    path = EXPERIMENTS / "GLOSSARY.md"
    if not path.is_file():
        return []
    terms, section = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif m := Term.LINE.match(line):
            terms.append(
                Term(
                    m.group(1).strip(),
                    [a.strip() for a in (m.group(2) or "").split(",") if a.strip()],
                    m.group(3).strip(),
                    section,
                )
            )
    return terms


def exact_case(form: str) -> bool:
    """Short or symbolic forms (F1, κ, q05, P@k) match as written; words match any case."""
    return len(form) < 4 or not form.isalpha()


def link_terms(html_: str, terms: list[Term]) -> str:
    """The first mention of a glossary term on a page becomes a link to the
    glossary page with its definition as a hover card; later mentions stay plain,
    so the prose is not stippled with markers. Longer forms win, so `Precision@k`
    is not read as `precision`."""
    if not terms:
        return html_
    by_form = {form.lower(): (form, t) for t in terms for form in t.forms}
    forms = sorted(by_form, key=len, reverse=True)
    pat = re.compile(r"(?<![\w@\-])(" + "|".join(re.escape(f) for f in forms) + r")(?![\w@\-])", re.I)
    seen: set[str] = set()

    def link(m: re.Match) -> str:
        hit = m.group(1)
        form, t = by_form[hit.lower()]
        if exact_case(form) and hit != form or t.slug in seen:
            return hit
        seen.add(t.slug)
        return f'<a class="term" href="#glossary/{t.slug}" data-tip="{tip(t.name, t.definition)}">{hit}</a>'

    return sub_prose(html_, lambda text, _: pat.sub(link, text))


ABBREV = re.compile(r"(?<![\w@/#.-])(?:[A-Z]{2,6}|[A-Za-z]+@\d*k|[a-z]\d{2,3}|[A-Z][a-z]?\d{1,2})(?![\w@/-])")
KNOWN_ABBREV = {
    "PR",
    "ID",
    "URL",
    "CSV",
    "JSON",
    "YAML",
    "API",
    "CLI",
    "TL",
    "DR",
    "OK",
    "TODO",
    "HTML",
    "SVG",
    "PNG",
    "MD",
    "USD",
    "EUR",
    "UTC",
    "CPU",
    "GPU",
    "RAM",
    "KB",
    "MB",
    "GB",
    "TB",
    "EV",
}


def undefined_terms(records: list[Record], terms: list[Term]) -> list[str]:
    """Abbreviation-shaped tokens and standard method names (METHOD_TERMS) in
    record prose that the glossary does not define."""
    known = {f.lower() for t in terms for f in t.forms} | {k.lower() for k in KNOWN_ABBREV}
    methods = re.compile(r"(?<![\w-])(" + "|".join(re.escape(m) for m in METHOD_TERMS) + r")(?![\w-])", re.I)
    hits: set[str] = set()
    for r in records:
        for key in ("hypothesis", "design", "datasets", "methods", "findings", "verdict"):
            text = re.sub(
                r"```.*?```|`[^`]*`|\[[DFMH]\d+\]|!?\[[^\]]*\]\([^)]*\)", " ", r.sections.get(key, ""), flags=re.S
            )
            text = "\n".join(l for l in MATH.sub(" ", text).splitlines() if not l.lstrip().startswith(("|", "###")))
            hits |= {w for w in ABBREV.findall(text) if w.lower() not in known and not re.fullmatch(r"[DFMH]\d+", w)}
            hits |= {w for w in methods.findall(text) if w.lower() not in known}
    return sorted({h.lower(): h for h in sorted(hits, reverse=True)}.values(), key=str.lower)


def render_glossary(terms: list[Term], used: dict[str, list[tuple[str, str]]] = {}) -> str:
    """One page: a section bar with a filter box, each `## ` section of GLOSSARY.md as
    a heading, its terms as rows that name the experiments using them."""
    if not terms:
        return ""
    sections: dict[str, int] = {}
    for t in terms:
        sections[t.section] = sections.get(t.section, 0) + 1
    sid = lambda name: re.sub(r"[^\w]+", "-", name.lower()).strip("-")
    bar = "".join(f'<a href="#glossary/{sid(k)}">{esc(k)} <i>{n}</i></a>' for k, n in sections.items())
    parts, section = [
        '<section class="page" id="glossary" data-title="Glossary"><p class="kicker">experiments/GLOSSARY.md</p><h1>Glossary</h1>'
        f'<div class="bar">{bar}<input class="filter" type="search" placeholder="filter terms" aria-label="filter terms" title="/ focuses, Esc clears"></div>'
    ], None
    for t in terms:
        if t.section != section:
            if section is not None:
                parts.append("</dl>")
            section = t.section
            parts.append(f'<h2 id="glossary/{sid(section)}">{esc(section)}</h2><dl class="gloss">')
        also = f' <span class="also">{esc(", ".join(t.aliases))}</span>' if t.aliases else ""
        links = ", ".join(f'<a href="#{h}">{esc(label)}</a>' for h, label in used.get(t.slug, []))
        where = f'<div class="used">used in {links}</div>' if links else ""
        q = esc(" ".join([t.name, *t.aliases, plain(t.definition)]).lower())
        parts.append(
            f'<div class="g" data-q="{q}"><dt id="glossary/{t.slug}">{esc(t.name)}{also}</dt><dd>{md_inline(cap(t.definition))}{where}</dd></div>'
        )
    parts.append("</dl></section>")
    return "".join(parts)


def relink(page: str, records: list[Record]) -> str:
    """Relative links out of records and the book: a record README becomes an
    in-page hash link (its `#f<n>` anchor too); anything else is unwrapped, since
    a self-contained file cannot resolve it. External links open in a new tab."""
    by_slug = {r.slug for r in records}

    anchors = {a for r in records for a in r.anchors}
    slugs = "|".join(re.escape(r.slug) for r in records)

    def sub(m: re.Match) -> str:
        href, text = m.group(1), m.group(2)
        t = re.match(r"(?:[^#]*/)?([^/#]+)/README\.md(?:#(f\d+)\b.*)?$", href)
        if t and t.group(1) in by_slug:
            return f'<a href="#{t.group(1)}{"/" + t.group(2) if t.group(2) else ""}">{text}</a>'
        a = re.match(r"(?:(" + slugs + r")/)?(?:tables|figures)/([^/]+)\.\w+$", href) if slugs else None
        if a:
            stem = re.sub(r"[^A-Za-z0-9_-]", "-", a.group(2))
            hits = [
                x
                for x in anchors
                if x.endswith("/asset/" + stem) and (not a.group(1) or x.startswith(a.group(1) + "/"))
            ]
            if hits:
                return f'<a class="ref" href="#{hits[0]}">{text}</a>'
        return text

    page = re.sub(r'<a href="(?!#|https?://|mailto:)([^"]*)">(.*?)</a>', sub, page, flags=re.S)
    return re.sub(r'<a href="(https?://[^"]*)"', r'<a href="\1" target="_blank" rel="noopener"', page)


def section_count(record: Record, key: str) -> int:
    return {
        "methods": len(record.methods),
        "decisions": len(record.decisions),
        "builds": len(record.children),
        "findings": len(record.key_findings()),
    }.get(key, 0)


def render_page(record: Record, parents: list[Record], terms: list[Term], progs: list[Programme] = ()) -> str:
    meta = [
        f'in <a href="#{g.page_id}" title="{esc(g.title)}">{esc(g.meta.get("short") or g.title)}</a>'
        for g in progs
        if record in g.members
    ]
    if parents:
        meta.append(
            "builds on "
            + ", ".join(
                f'<a href="#{p.slug}" title="{esc(p.title)}">{esc(p.meta.get("short") or p.title)}</a>' for p in parents
            )
        )
    if mlflow := str(record.meta.get("mlflow") or ""):
        exp = re.search(r"experiments/(\d+)", mlflow)
        meta.append(f'<a href="{esc(mlflow)}">MLflow{" #" + exp.group(1) if exp else ""}</a>')
    if pr := pr_link(record):
        meta.append(pr)
    if record.meta.get("concluded"):
        meta.append(f"concluded {esc(record.meta['concluded'])}")
    hyp = hypothesis_text(record)
    parts = [
        f'<p class="kicker">{esc(record.category)} · {esc(record.meta.get("branch", ""))}</p><h1>{esc(record.title)}</h1>',
        f'<div class="meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>' if meta else "",
    ]
    if hyp:
        parts.append(
            f'<div class="hyp"><div class="lab"><span class="lbl">Hypothesis</span></div>'
            f'<div class="v">{md_inline(cap(hyp))}</div><div class="out">{badge(record)}</div></div>'
        )
    parts.append(render_needs(record))
    parts.append(render_clauses(record))
    sections = []
    for key in PAGE_ORDER:
        if key == "builds":
            body, title = render_builds(record), "Builds on this"
        else:
            block = strip_placeholders(record.sections.get(key, ""))
            if not block:
                continue
            title = SECTION_TITLES[key]
            if key == "findings":
                body = render_findings(record)
                if record.status != "concluded":
                    title = "Findings so far"
            elif key == "hypothesis":
                body = render_hypothesis(record, block)
            elif key == "verdict":
                kv = parse_kv(block)
                pairs = [
                    (k, badge(record) if k == "Outcome" and record.outcome else md(cap(v)))
                    for k, v in kv.items()
                    if strip_placeholders(v)
                ]
                body = f'<div class="card">{rows_html(pairs)}</div>'
            elif key == "design":
                body = render_kv(block, CONSTRAINT_ROWS)
            elif key == "datasets":
                body = render_datasets(record, block)
            elif key == "methods":
                body = render_method(record).replace("{slug}", record.slug)
            elif key == "decisions":
                body = render_decisions(record).replace("{slug}", record.slug)
            elif key == "reproduce":
                body = render_reproduce(record, block)
            else:
                body = render_kv(block) or f'<div class="prose">{md(block)}</div>'
        if body:
            sections.append((key, title, body))
    bar = "".join(
        f'<a href="#{record.slug}/{key}">{esc(title)}{f" <i>{n}</i>" if (n := section_count(record, key)) else ""}</a>'
        for key, title, _ in sections
    )
    parts.append(
        f'<div class="bar">{bar}<span class="hint"><kbd>j</kbd><kbd>k</kbd> sections · <kbd>e</kbd> folds</span></div>'
    )
    parts += [f'<h2 id="{record.slug}/{key}">{title}</h2>{body}' for key, title, body in sections]
    return f'<section class="page" id="{record.slug}" data-title="{esc(record.meta.get("short") or record.title)}">{link_terms(link_refs(record, "".join(parts)), terms)}</section>'


class Programme:
    """One `experiments/programmes/<slug>.md`: a question several experiments answer
    together — its member records in story order, the experiments still planned, what
    is now believed and the threads a reader follows by topic rather than by experiment."""

    def __init__(self, path: Path, records: list[Record]):
        self.path = path
        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER.match(text)
        if not match:
            raise ValueError(f"{path}: missing YAML frontmatter")
        self.meta = yaml.safe_load(match.group(1)) or {}
        self.sections = Record._split(text[match.end() :])
        self.slug = self.meta.get("slug") or path.stem
        self.title = self.meta.get("title") or self.slug
        self.status = self.meta.get("status", "open")
        self.problems: list[str] = []
        by_dir = {r.dir.resolve(): r for r in records}
        self.members: list[Record] = []
        for m in self.meta.get("members") or []:
            r = by_dir.get((path.parent / m).resolve())
            if r is None:
                self.problems.append(f"member {m} is not a record")
            else:
                self.members.append(r)
        self.planned = [x for x in (self.meta.get("planned") or []) if isinstance(x, dict) and x.get("slug")]
        for key, name in PROGRAMME_SECTIONS.items():
            if not strip_placeholders(self.sections.get(key, "")):
                self.problems.append(f"no ## {name}")
        by_slug = {r.slug: r for r in self.members}
        for key in ("what-we-now-believe", "threads"):
            for slug, fid in XREF.findall(self.sections.get(key, "")):
                r = by_slug.get(slug)
                if r is None:
                    self.problems.append(f"[{slug}:{fid}] cites an experiment that is not a member")
                elif fid.startswith("H") and fid not in {c["id"] for c in clause_rows(r)}:
                    self.problems.append(f"[{slug}:{fid}] cites a clause {slug} does not have")
                elif fid.startswith("F") and fid not in {f.id for f in r.findings}:
                    self.problems.append(f"[{slug}:{fid}] cites a finding {slug} does not have")
        in_threads = set(XREF.findall(self.sections.get("threads", "")))
        for r in self.members:
            if rows := clause_rows(r):
                for c in rows:
                    if (r.slug, c["id"]) not in in_threads:
                        self.problems.append(f"clause {r.slug} {c['id']} sits in no thread")
            else:
                for f in r.key_findings():
                    if (r.slug, f.id) not in in_threads:
                        self.problems.append(f"key finding {r.slug} {f.id} sits in no thread")

    @property
    def page_id(self) -> str:
        return f"programme-{self.slug}"

    def finding(self, slug: str, fid: str) -> tuple[Record, Entry] | None:
        for r in self.members:
            if r.slug == slug:
                return next(((r, f) for f in r.findings if f.id == fid), None)
        return None

    def clause(self, slug: str, hid: str) -> tuple[Record, dict] | None:
        for r in self.members:
            if r.slug == slug:
                return next(((r, c) for c in clause_rows(r) if c["id"] == hid), None)
        return None


def link_xrefs(prog: Programme, html_: str) -> str:
    """`[<slug>:F<n>]` in programme prose becomes a link to that finding on its record's page,
    shown as `[F<n>]` with the finding's Summary as a hover card under the experiment's name."""

    def link(m: re.Match) -> str:
        if m.group(2).startswith("H"):
            hit = prog.clause(m.group(1), m.group(2))
            if hit is None:
                return m.group(0)
            r, c = hit
            return f'<a class="ref" href="#{r.slug}/{c["id"].lower()}" data-tip="{tip(f"{c[chr(105)+chr(100)]} — {c[chr(99)+chr(108)+chr(97)+chr(105)+chr(109)]}", clause_verdict(r, c), r.slug)}">[{c["id"]}]</a>'
        hit = prog.finding(m.group(1), m.group(2))
        if hit is None:
            return m.group(0)
        r, f = hit
        return f'<a class="ref" href="#{r.slug}/{f.id.lower()}" data-tip="{tip(f"{f.id} — {f.title}", f.kv.get("Summary", ""), r.slug)}">[{f.id}]</a>'

    return sub_prose(html_, lambda text, _: XREF.sub(link, text))


def found_line(r: Record) -> str:
    """One sentence of what the experiment found: its TL;DR Hypothesis line without the
    status prefix; before a verdict, the hypothesis it is testing."""
    if not r.outcome:
        return plain(cap(hypothesis_text(r)))
    hyp = TLDR_STATUS.sub("", parse_kv(r.sections.get("tldr", "")).get("Hypothesis", ""))
    return plain(cap(hyp))


def trigger_line(r: Record) -> str:
    """Why the experiment exists: its Hypothesis Builds-on row, as plain text."""
    return plain(parse_kv(r.sections.get("hypothesis", "")).get("Builds on", ""))


def render_map(prog: Programme) -> str:
    """The programme's experiments as layered columns, left to right by what each
    builds on: a node carries the outcome, the date, one line of what was found and
    its parents, each parent naming on hover the question that connected them."""
    nodes: list[dict] = []
    members = {r.slug for r in prog.members}
    for r in prog.members:
        parents = [p.name for p in r.builds_on if p.name in members]
        state = r.outcome or r.status
        when = str(r.meta.get("concluded") or "")
        nodes.append(
            {
                "slug": r.slug,
                "parents": parents,
                "cls": state,
                "href": f"#{r.slug}",
                "label": r.meta.get("short") or r.slug,
                "line": f"{state}{' · ' + when if when else ''}",
                "title": r.title,
                "found": found_line(r),
                "why": trigger_line(r),
            }
        )
    for x in prog.planned:
        line = "planned" + (f" · {x['ticket']}" if x.get("ticket") else "")
        nodes.append(
            {
                "slug": x["slug"],
                "parents": list(x.get("builds_on") or []),
                "cls": "planned",
                "href": "",
                "label": x.get("short") or x["slug"],
                "line": line,
                "title": x.get("title") or x["slug"],
                "found": x.get("title") or "",
                "why": "",
            }
        )
    known = {n["slug"]: n for n in nodes}
    depth: dict[str, int] = {}

    def d(slug: str, seen: tuple = ()) -> int:
        if slug in depth:
            return depth[slug]
        ps = [p for p in known[slug]["parents"] if p in known and p not in seen]
        depth[slug] = 1 + max((d(p, seen + (slug,)) for p in ps), default=-1)
        return depth[slug]

    for n in nodes:
        d(n["slug"])
    cols: dict[int, list[dict]] = {}
    for n in nodes:
        cols.setdefault(depth[n["slug"]], []).append(n)
    out = []
    for c in sorted(cols):
        items = []
        for n in cols[c]:
            parents = "".join(
                f'<div class="from" title="{esc(n["why"]) if n["why"] else ""}">← {esc(known[p]["label"]) if p in known else esc(p)}</div>'
                for p in n["parents"]
            )
            head = f'<div class="t">{esc(n["label"])}</div><div class="m">{esc(n["line"])}</div>'
            body = (
                f'<div class="f">{esc(n["found"][:180] + ("…" if len(n["found"]) > 180 else ""))}</div>'
                if n["found"]
                else ""
            )
            inner = f"{head}{body}{parents}"
            data = f'data-slug="{esc(n["slug"])}" data-parents="{esc(" ".join(p for p in n["parents"] if p in known))}"'
            items.append(
                f'<a class="node n-{esc(n["cls"])}" {data} href="{n["href"]}" title="{esc(n["title"])}">{inner}</a>'
                if n["href"]
                else f'<div class="node n-{esc(n["cls"])}" {data} title="{esc(n["title"])}">{inner}</div>'
            )
        out.append(f'<div class="col">{"".join(items)}</div>')
    return (
        f'<figure class="map"><div class="scroll"><div class="layers"><svg class="edges"></svg>{"".join(out)}</div></div>'
        "<figcaption>Left to right: what each experiment builds on. Colour is the outcome; a dashed box is planned; "
        "hover a parent for the question that connected them.</figcaption></figure>"
    )


def render_programme_constraints(prog: Programme) -> str:
    """One row per member: what each experiment needs beyond code, from its Design rows."""
    rows = [(r, constraints(r)) for r in prog.members]
    if not any(any(c.values()) for _, c in rows):
        return ""
    head = "".join(f"<th>{esc(k)}</th>" for k in CONSTRAINT_ROWS)
    body = "".join(
        f'<tr><td><a href="#{r.slug}" title="{esc(r.title)}">{esc(r.meta.get("short") or r.title)}</a></td>'
        + "".join(
            f'<td{" class=hot" if needs_glance(v) else ""}>{md_inline(cap(v)) if v else "—"}</td>' for v in c.values()
        )
        + "</tr>"
        for r, c in rows
    )
    return (
        f'<h2 id="{prog.page_id}/needs">Needs</h2><div class="scroll"><table class="data wide needs-table">'
        f"<thead><tr><th>Experiment</th>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def render_programme_table(prog: Programme) -> str:
    """Members and planned experiments as one table: title, category, outcome, date, what each builds on."""
    by_slug = {r.slug: r for r in prog.members}
    rows = []
    for r in prog.members:
        parents = (
            ", ".join(esc(by_slug[p.name].meta.get("short") or p.name) for p in r.builds_on if p.name in by_slug) or "—"
        )
        rows.append(
            f'<tr><td><a href="#{r.slug}" title="{esc(r.title)}">{esc(r.meta.get("short") or r.slug)}</a></td><td class="mono">{esc(r.category)}</td>'
            f'<td>{badge(r)}</td><td class="mono">{esc(str(r.meta.get("concluded") or "—"))}</td><td class="muted">{parents}</td></tr>'
        )
    for x in prog.planned:
        parents = (
            ", ".join(
                esc(by_slug[p].meta.get("short") or p) if p in by_slug else esc(p) for p in (x.get("builds_on") or [])
            )
            or "—"
        )
        rows.append(
            f'<tr class="planned"><td title="{esc(x.get("title") or "")}">{esc(x.get("short") or x.get("title") or x["slug"])}</td><td class="mono"></td>'
            f'<td><span class="badge b-draft">planned</span></td><td class="mono">{esc(x.get("ticket") or "—")}</td><td class="muted">{parents}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="data wide exp-table"><thead><tr><th>Experiment</th><th>Category</th><th>Outcome</th><th>Concluded</th><th>Builds on</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_threads(prog: Programme) -> str:
    """One card per thread; one aligned line per cited clause: experiment, id, claim, the
    clause's own verdict, then the findings that read it. A cited finding of a record
    without clauses stands on a line of its own."""
    out = []
    for i, (name, body) in enumerate(parse_kv(strip_placeholders(prog.sections.get("threads", ""))).items(), 1):
        sentence = re.sub(r"[\s—–,;-]+$", "", XREF.sub("", body))
        items = []
        for slug, cid in XREF.findall(body):
            hit = prog.clause(slug, cid) if cid.startswith("H") else prog.finding(slug, cid)
            if hit is None:
                items.append(
                    f'<div class="line"><span class="who">{esc(slug)}</span><span class="c"><span class="tag">{esc(cid)}</span> not found</span><span></span></div>'
                )
                continue
            r, x = hit
            who = f'<span class="who" title="{esc(r.title)}">{esc(r.meta.get("short") or r.slug)}</span>'
            if cid.startswith("H"):
                href = f"#{r.slug}/{cid.lower()}"
                evidence = "".join(
                    f'<div class="ev"><a href="#{r.slug}/{f.id.lower()}" data-tip="{tip(f"{f.id} — {f.title}", f.kv.get("Summary", ""), r.slug)}"><span class="tag">{f.id}</span>{md_inline(f.title)}</a></div>'
                    for f in x["findings"]
                )
                items.append(
                    f'<div class="line">{who}<span class="c"><a class="t" href="{href}"><span class="tag">{cid}</span>{md_inline(cap(x["claim"]))}</a>{evidence}</span>'
                    f'<span class="o">{verdict_badge(r, x)}</span></div>'
                )
            else:
                href = f"#{r.slug}/{x.id.lower()}"
                items.append(
                    f'<div class="line">{who}<span class="c"><a class="t" href="{href}" data-tip="{tip(f"{x.id} — {x.title}", x.kv.get("Summary", ""), r.slug)}">'
                    f'<span class="tag">{x.id}</span>{md_inline(x.title)}</a></span><span></span></div>'
                )
        out.append(
            f'<article class="thread" id="{prog.page_id}/t{i}" data-title="{esc(plain(name))}"><h3>{md_inline(name)}</h3>'
            f'<p>{md_inline(sentence)}</p>{"".join(items)}</article>'
        )
    return "".join(out)


def toc_findings(r: Record) -> str:
    fs = r.key_findings()
    if not fs:
        return f'<span class="muted">{esc(r.status)} — no findings yet</span>'
    return "".join(
        f'<div class="fl"><span class="tag">{f.id}</span><span>{md_inline(f.title)}</span></div>' for f in fs
    )


def short_link(r: Record) -> str:
    """A link to the record by its short title, the full title on hover."""
    return f'<a href="#{r.slug}" title="{esc(r.title)}">{esc(r.meta.get("short") or r.title)}</a>'


def toc_table(records: list[Record], by_slug: dict[str, Record], planned: list[dict] = ()) -> str:
    """One contents table: number, experiment with its category, date and parents, outcome, key findings."""
    rows = []
    for i, r in enumerate(records, 1):
        meta = [esc(r.category)]
        if r.meta.get("concluded"):
            meta.append(f"concluded {esc(str(r.meta['concluded']))}")
        if parents := [by_slug[p.name] for p in r.builds_on if p.name in by_slug]:
            meta.append("builds on " + ", ".join(short_link(p) for p in parents))
        rows.append(
            f'<tr><td class="mono">{i}</td><td><a href="#{r.slug}" title="{esc(r.title)}">{esc(r.meta.get("short") or r.title)}</a>'
            f'<div class="m">{" · ".join(meta)}</div></td><td>{badge(r)}</td><td>{toc_findings(r)}</td></tr>'
        )
    for x in planned:
        meta = [f"issue {esc(x['ticket'])}"] if x.get("ticket") else []
        if parents := [short_link(by_slug[p]) if p in by_slug else esc(p) for p in (x.get("builds_on") or [])]:
            meta.append("builds on " + ", ".join(parents))
        rows.append(
            f'<tr class="planned"><td class="mono">·</td><td><span class="t">{esc(x.get("short") or x["slug"].replace("-", " ").capitalize())}</span><div class="m">{" · ".join(meta)}</div></td>'
            f'<td><span class="badge b-draft">planned</span></td><td class="muted">{esc(cap(x.get("title") or ""))}</td></tr>'
        )
    return (
        '<div class="scroll"><table class="data wide toc"><thead><tr><th>#</th><th>Experiment</th><th>Outcome</th><th>Key findings</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_programme(prog: Programme, terms: list[Term]) -> str:
    pid = prog.page_id
    n = f"{len(prog.members)} experiments" + (f", {len(prog.planned)} planned" if prog.planned else "")
    kicker = f'programmes / {esc(prog.slug)} · <span class="badge b-{"running" if prog.status == "open" else "draft"}">{esc(prog.status)}</span> · {n}'
    parts = [
        f'<p class="kicker">{kicker}</p><h1>{esc(prog.title)}</h1>',
        f'<div class="book">{md(strip_placeholders(prog.sections.get("question", "")))}</div>',
        f'<div class="bar"><a href="#{pid}/map">Map</a><a href="#{pid}/believe">What we now believe</a><a href="#{pid}/threads">Threads</a>'
        f'<a href="#{pid}/open">Open</a><a href="#{pid}/needs">Needs</a><a href="#{pid}/experiments">Experiments</a>'
        '<span class="hint"><kbd>j</kbd><kbd>k</kbd> sections</span></div>',
        f'<h2 id="{pid}/map">Map</h2>{render_map(prog)}',
        f'<h2 id="{pid}/believe">What we now believe</h2><div class="believe">{md(strip_placeholders(prog.sections.get("what-we-now-believe", "")))}</div>',
        f'<h2 id="{pid}/threads">Threads</h2>{render_threads(prog)}',
    ]
    if block := strip_placeholders(prog.sections.get("open", "")):
        parts.append(f'<h2 id="{pid}/open">Open</h2><div class="card prose">{md(block)}</div>')
    parts.append(render_programme_constraints(prog))
    parts.append(f'<h2 id="{pid}/experiments">Experiments</h2>{render_programme_table(prog)}')
    page = f'<section class="page" id="{pid}" data-title="{esc(prog.title)}">{"".join(parts)}</section>'
    return link_terms(link_xrefs(prog, page), terms)


def render_overview(fams: list[list[tuple[Record, int]]], terms: list[Term], progs: list[Programme] = ()) -> str:
    """The contents page: counts, then every programme with its experiments, then experiments outside any programme."""
    book = EXPERIMENTS / "README.md"
    intro = md(book.read_text(encoding="utf-8")) if book.is_file() else ""
    # only the book's intro: its chapters and table repeat what the tables below show
    intro = re.sub(r"<h1>.*?</h1>", "", intro, count=1, flags=re.S)
    intro = intro.split("<h2>", 1)[0]
    intro = re.sub(r'<div class="scroll"><table class="data[^"]*">.*?</table></div>', "", intro, count=1, flags=re.S)
    records = [r for fam in fams for r, _ in fam]
    by_slug = {r.slug: r for r in records}
    loose = [r for r in records if r.slug not in {m.slug for g in progs for m in g.members}]
    first = "#overview/programmes" if progs else "#overview/experiments"
    counts = [
        ("Programmes", len(progs), "#overview/programmes"),
        ("Experiments", len(records), first),
        ("Glossary", len(terms), "#glossary"),
    ]
    nav = (
        '<div class="toc-nav">'
        + "".join(f'<a href="{h}">{t}<span class="n">{n}</span></a>' for t, n, h in counts if n)
        + "</div>"
    )
    parts = ['<h2 id="overview/programmes">Programmes</h2>'] if progs else []
    for g in progs:
        n = f"{len(g.members)} experiments" + (f", {len(g.planned)} planned" if g.planned else "")
        question = sentences(strip_placeholders(g.sections.get("question", "")))[:1]
        parts.append(
            f'<div class="card prog" id="overview/{esc(g.slug)}"><div class="head"><a class="t" href="#{g.page_id}" title="{esc(g.title)}">{esc(g.meta.get("short") or g.title)}</a>'
            f'<span class="badge b-{"running" if g.status == "open" else "draft"}">{esc(g.status)}</span><span class="m">{n}</span><a class="go" href="#{g.page_id}">Programme page →</a></div>'
            f'<p class="q">{md_inline(question[0]) if question else ""}</p>{toc_table(g.members, by_slug, g.planned)}</div>'
        )
    if loose:
        parts.append(
            f'<h2 id="overview/experiments">{"Not in a programme" if progs else "Experiments"}</h2>{toc_table(loose, by_slug)}'
        )
    return link_terms(
        f'<section class="page" id="overview" data-title="Overview"><p class="kicker">experiments/README.md</p><h1>Overview</h1>'
        f'<div class="book">{intro}</div>{nav}{"".join(parts)}</section>',
        terms,
    )


def render_sidebar(fams: list[list[tuple[Record, int]]], terms: list[Term], progs: list[Programme] = ()) -> str:
    """A flat list: every experiment by its short title in family order, the full
    title on hover, the category as a tag; no indentation, the map shows lineage."""
    out = [
        '<a class="item" href="#overview" data-page="overview">Overview</a>',
    ]
    if progs:
        out.append('<p class="h">Programmes</p>')
        out += [
            f'<a class="item" href="#{g.page_id}" data-page="{g.page_id}" title="{esc(g.title)}"><span class="dot {"d-open" if g.status == "open" else "d-done"}"></span>'
            f'<span class="lab">{esc(g.meta.get("short") or g.title)}</span><span class="sub">{len(g.members)}</span></a>'
            for g in progs
        ]
    out.append('<p class="h">Experiments</p>')
    for fam in fams:
        for r, _ in fam:
            dot = f"d-{r.outcome}" if r.outcome else ("d-done" if r.status == "concluded" else "d-open")
            out.append(
                f'<a class="item" href="#{r.slug}" data-page="{r.slug}" title="{esc(r.title)}">'
                f'<span class="dot {dot}"></span><span class="lab">{esc(r.meta.get("short") or r.title)}</span><span class="sub">{esc(r.category)}</span></a>'
            )
    if terms:
        out.append('<p class="h">Reference</p><a class="item" href="#glossary" data-page="glossary">Glossary</a>')
    out.append(
        '<div class="keys"><p class="h">Shortcuts</p>'
        + (
            "<div><kbd>?</kbd> glossary</div><div><kbd>/</kbd> filter the glossary</div><div><kbd>esc</kbd> back</div>"
            if terms
            else ""
        )
        + "<div><kbd>j</kbd><kbd>k</kbd> next / previous section</div><div><kbd>[</kbd><kbd>]</kbd> previous / next experiment</div><div><kbd>o</kbd> overview</div><div><kbd>e</kbd> expand / collapse folds</div>"
        + "<div><kbd>n</kbd> experiment list</div><div><kbd>t</kbd> theme</div></div>"
    )
    return "".join(out)


CSS = """
:root{--bg:#e6e9ee;--panel:#f7f8fa;--line:#c9cfd8;--fg:#1f252f;--muted:#5a6472;--accent:#3350c4;--accent-soft:#dbe1f6;--soft:#e8ebf0;--ok:#17643a;--ok-soft:#d7ecdf;--bad:#9c2820;--bad-soft:#f3dad7;--run:#7f4b00;--run-soft:#f2e2c3;--plate:#f0f2f5;--t-h1:32px;--t-h2:20px;--t-lead:17px;--t-h3:16px;--t-body:15px;--t-sm:13.5px;--t-mono:12px;--t-xs:11px;--t-micro:9px;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace;--sans:"Instrument Sans",system-ui,sans-serif}
:root[data-theme=dark]{--bg:#1c1c20;--panel:#292a30;--line:#464852;--fg:#dcdfe5;--muted:#9a9fac;--accent:#9aa8ff;--accent-soft:#2c3358;--soft:#35363e;--ok:#5ee39a;--ok-soft:#1b3b2c;--bad:#f98a8a;--bad-soft:#452524;--run:#f2c55c;--run-soft:#413513;--plate:#25262c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1c1c20;--panel:#292a30;--line:#464852;--fg:#dcdfe5;--muted:#9a9fac;--accent:#9aa8ff;--accent-soft:#2c3358;--soft:#35363e;--ok:#5ee39a;--ok-soft:#1b3b2c;--bad:#f98a8a;--bad-soft:#452524;--run:#f2c55c;--run-soft:#413513;--plate:#25262c}}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:var(--t-body);line-height:1.55}
a{color:var(--accent);text-decoration:none}a:hover{color:color-mix(in srgb,var(--accent),var(--fg) 25%)}
.app{display:grid;grid-template-rows:52px minmax(0,1fr);grid-template-columns:320px minmax(0,1fr) 280px;height:100vh}.app.nonav{grid-template-columns:minmax(0,1fr) 280px}.app.nonav .side{display:none}
.top{grid-column:1/-1;display:flex;align-items:center;gap:16px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.brand{font-weight:700}.crumb{display:flex;gap:8px;color:var(--muted);font-size:var(--t-sm);margin-left:8px}.crumb b{color:var(--fg);font-weight:600}
.theme{margin-left:auto}.nav,.theme{font:inherit;font-size:var(--t-sm);color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}
.side{background:var(--bg);border-right:1px solid var(--line);padding:16px 12px;overflow:auto;display:flex;flex-direction:column;gap:2px}
.side .h{font-size:var(--t-xs);font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:12px 10px 6px;margin:0}
.item{display:flex;align-items:flex-start;gap:10px;padding:6px 10px;border-radius:8px;font-size:var(--t-sm);line-height:1.4;color:var(--fg)}.item:hover{background:var(--panel);color:var(--fg)}.item.on{background:var(--accent-soft);color:var(--accent);font-weight:600}
.item.d1{margin-left:18px}.item.d2{margin-left:36px}.item.d3{margin-left:54px}
.dot{width:8px;height:8px;border-radius:50%;flex:none;margin-top:5px}.d-done{background:var(--fg)}.d-open{border:1.5px solid var(--fg)}.d-confirmed{background:var(--ok)}.d-refuted{background:var(--bad)}.d-inconclusive{background:var(--muted)}.item.on .dot{background:var(--accent);border-color:var(--accent)}.item.on .d-open{background:transparent}
.item .sub{margin-left:auto;padding-left:8px;margin-top:3px;font-family:var(--mono);font-size:var(--t-xs);color:var(--muted);flex:none}
.main{overflow:auto;padding:36px 56px 64px;background:var(--bg)}.page{max-width:1080px;margin:0 auto}.page{display:none}.page.on{display:block}
.kicker{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted);margin:0}
h1{font-size:var(--t-h1);font-weight:700;letter-spacing:-.02em;margin:8px 0 10px}
p.lede{font-size:var(--t-lead);color:var(--muted);margin:0 0 8px}ul.lede{font-size:var(--t-lead);margin:0 0 24px;padding-left:20px}ul.lede li{margin:0 0 8px}ul.lede a{color:inherit;font-weight:600;text-decoration:none}ul.lede a:hover{color:var(--accent)}ul.lede .s{display:block;font-size:var(--t-sm);color:var(--muted)}
.badge{display:inline-block;font-size:var(--t-xs);font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;vertical-align:middle}
.b-confirmed{background:var(--ok-soft);color:var(--ok)}.b-refuted{background:var(--bad-soft);color:var(--bad)}.b-inconclusive,.b-draft{background:var(--soft);color:var(--muted)}.b-running{background:var(--run-soft);color:var(--run)}
h2{font-size:var(--t-h2);font-weight:600;letter-spacing:-.01em;margin:64px 0 20px;scroll-margin-top:20px}
h3{font-size:var(--t-h3);font-weight:600;margin:0 0 8px}
.kv{display:flex;flex-direction:column;gap:26px}.kv h3{font-size:var(--t-mono);font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin:0 0 4px}.v p,.v ul{margin:0 0 6px}.v>:last-child{margin-bottom:0}
.res{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin:16px 0;display:grid;grid-template-columns:minmax(0,2fr) minmax(0,3fr);gap:10px 28px;scroll-margin-top:20px}
.res>h3{grid-column:1/-1;display:flex;gap:10px;align-items:center;font-size:var(--t-lead)}.tag{font-family:var(--mono);font-size:var(--t-mono);background:var(--soft);padding:1px 7px;border-radius:5px;font-weight:500}.wt{margin-left:auto;font-size:var(--t-mono);color:var(--muted);font-weight:500}
ol.clauses{list-style:none;margin:8px 0;padding:0}ol.clauses li{display:grid;grid-template-columns:auto 1fr;gap:0 10px;align-items:start;margin:0 0 8px;scroll-margin-top:20px}ol.clauses li .tag{margin-top:2px}ol.clauses li b{font-weight:600}
.fp{margin:0 0 10px}.fp .lbl{display:block;font-size:var(--t-xs);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}.fp.summary{font-size:var(--t-sm);background:var(--soft);padding:10px 12px;border-radius:8px;margin:0 0 12px}.fp p{margin:0 0 6px}.fp p:last-child{margin-bottom:0}.fp ul,.fp ol{margin:4px 0 0;padding-left:18px}.fp li{margin:0 0 6px}.fp li:last-child{margin-bottom:0}
.dec{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:12px 0;scroll-margin-top:20px}.dec h3{display:flex;gap:10px;align-items:center;font-size:var(--t-body);margin:0 0 8px}.dec .decision p{margin:0 0 6px}.dec details{border:0;padding:0;margin:6px 0 0}.dec summary{color:var(--muted)}.dec .src{margin:8px 0 0}
.met{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:12px 0;scroll-margin-top:20px}.met h3{display:flex;gap:10px;align-items:center;font-size:var(--t-lead);margin:0 0 12px}
.met .io{display:grid;grid-template-columns:1fr 1fr;gap:12px 28px;margin:0 0 14px}.met .io .fp{margin:0;font-size:var(--t-sm)}.met .how{font-size:var(--t-sm);line-height:1.6}.met .how p{margin:0 0 8px}.met .how>:last-child{margin-bottom:0}.met .src{margin:14px 0 0}
@media(max-width:700px){.met .io{grid-template-columns:minmax(0,1fr)}}
.math{font-size:1.05em}.math.block{display:block;margin:14px 0;padding:6px 0;line-height:1.2;overflow-x:auto;overflow-y:hidden;font-size:1.15em}math{font-family:"STIX Two Math","Latin Modern Math","Cambria Math",var(--sans)}
.res>.read,.res>figure{min-width:0}.res>.read{grid-column:1;grid-row:2;font-size:var(--t-sm);line-height:1.5}.res .read p{margin:0 0 8px}.res>figure{grid-column:2;grid-row:2;margin:0}.res>.src{grid-column:1/-1;margin:4px 0 0}.src{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted)}.src p{display:grid;grid-template-columns:58px 1fr;gap:0 6px;margin:0 0 3px}.src b{font-size:var(--t-xs);font-weight:600;letter-spacing:.06em;text-transform:uppercase;line-height:inherit;padding-top:1px}.src a code{color:var(--accent)}
.src code.run::before,.src code.commit::before,.src code.file::before{font-size:var(--t-micro);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:5px}.src code.run::before{content:"run"}.src code.commit::before{content:"commit"}.src code.file::before{content:"file"}
.res>details.support{grid-column:1/-1;border:0;padding:0;margin:2px 0 0}.res .support>summary{color:var(--muted);font-size:var(--t-mono)}
.res.sub{margin:10px 0 0;padding:14px 18px;background:transparent;border:0;border-left:2px solid var(--line);border-radius:0}.res.sub h3{font-size:var(--t-h3)}.res.minor{padding:12px 20px}.res.minor h3{font-size:var(--t-body)}
.res.wide{grid-template-columns:minmax(0,1fr)}.res.wide>.read{grid-column:1;font-size:var(--t-sm)}.res.wide>figure{grid-column:1;grid-row:3}
figure svg,figure img{width:100%;height:auto;display:block}.res>figure .fig{cursor:zoom-in}.res.wide>figure .fig{cursor:zoom-out}
figcaption{font-size:var(--t-mono);color:var(--muted);margin-top:8px;line-height:1.5}figcaption .lbl{color:var(--fg);font-weight:600}
.scroll{overflow-x:auto;max-width:100%;margin:6px 0 10px;scrollbar-width:thin;scrollbar-color:var(--line) transparent;padding-bottom:2px}.scroll.tall{overflow-y:auto;max-height:calc(20 * (1.45 * var(--t-sm) + 13px) + 2.6em)}.scroll.tall table.data thead th{position:sticky;top:0;background:var(--panel);z-index:1}.scroll.tall table.data th:first-child{z-index:2}
.scroll::-webkit-scrollbar{height:8px;width:8px}.scroll::-webkit-scrollbar-track{background:transparent}.scroll::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}.scroll.x{-webkit-mask-image:linear-gradient(to right,#000 calc(100% - 44px),transparent);mask-image:linear-gradient(to right,#000 calc(100% - 44px),transparent)}.scroll.y{-webkit-mask-image:linear-gradient(to bottom,#000 calc(100% - 36px),transparent);mask-image:linear-gradient(to bottom,#000 calc(100% - 36px),transparent)}.scroll.x.y{-webkit-mask-image:linear-gradient(to right,#000 calc(100% - 44px),transparent),linear-gradient(to bottom,#000 calc(100% - 36px),transparent);-webkit-mask-composite:source-in;mask-image:linear-gradient(to right,#000 calc(100% - 44px),transparent),linear-gradient(to bottom,#000 calc(100% - 36px),transparent);mask-composite:intersect}.sh{display:none;font-family:var(--mono);font-size:var(--t-xs);color:var(--muted);text-align:right;margin:-6px 0 8px}.sh.on{display:block}figure .scroll{margin:0}table.data{border-collapse:collapse;width:100%;font-size:var(--t-sm);line-height:1.45;font-variant-numeric:tabular-nums}table.data th{text-align:left;font-size:var(--t-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:bottom}table.data td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}table.data td.id{font-family:var(--mono);font-size:var(--t-mono);white-space:nowrap}table.data td:first-child,table.data th:first-child{position:sticky;left:0;background:inherit;font-weight:500}table.data.wide td:first-child,table.data.wide th:first-child{border-right:1px solid var(--line)}table.data tr{background:var(--panel)}.res.sub table.data tr,.prose table.data tr,.v table.data tr{background:var(--bg)}table.data th.n{text-align:right}table.data td.n{text-align:right;font-family:var(--mono);font-size:var(--t-mono);white-space:nowrap}table.data code{overflow-wrap:anywhere}
.prose h3,.v h3{font-size:var(--t-body);font-weight:600;margin:36px 0 10px}.prose h4,.v h4{font-size:var(--t-mono);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:24px 0 8px}.prose>:first-child{margin-top:0}.prose p,.prose ul,.prose ol{margin:0 0 10px}
.missing{color:var(--bad);font-family:var(--mono);font-size:var(--t-mono)}
.steps{list-style:none;padding:0;margin:8px 0 12px}.step{display:grid;grid-template-columns:30px minmax(0,1fr);gap:6px 12px;align-items:start;margin:0 0 18px}
.step .t{grid-column:2;font-size:var(--t-sm);font-weight:600;line-height:1.5;padding-top:3px}.step pre{grid-column:2;margin:0}.step .t code{font-size:.85em}
pre .c{color:#8a93a5}
a.ref{color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)}a.term{color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted)}a.term:hover{color:var(--accent);border-bottom-color:var(--accent)}dl.gloss{margin:0 0 28px}.gloss .g{display:grid;grid-template-columns:minmax(160px,260px) minmax(0,1fr);gap:0 24px;align-items:baseline;padding:10px 0;border-top:1px solid var(--line)}.gloss .g.off,#glossary h2.off{display:none}dl.gloss dt{font-weight:600;scroll-margin-top:20px}dl.gloss dt .also{display:block;font-weight:400;font-size:var(--t-mono);color:var(--muted);font-family:var(--mono);margin-top:2px}.gloss dd .used{margin-top:4px;font-size:var(--t-sm);color:var(--muted);font-family:var(--sans)}.gloss dd .used a{color:var(--muted)}.gloss dd .used a:hover{color:var(--accent)}dl.gloss dd{margin:0;color:var(--fg)}
.tip{position:fixed;z-index:9;max-width:400px;background:var(--fg);color:var(--bg);font-size:var(--t-mono);line-height:1.45;padding:8px 11px;border-radius:8px;white-space:pre-line;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.25)}.tip b{display:block;margin-bottom:2px}.tip code{background:color-mix(in srgb,var(--bg) 18%,transparent);color:inherit}
code{font-family:var(--mono);font-size:.88em;background:var(--soft);padding:1px 5px;border-radius:4px}.fp.summary code{background:color-mix(in srgb,var(--fg) 10%,transparent)}pre{background:#16181d;color:#e7e9ee;padding:14px 16px;border-radius:10px;font-family:var(--mono);font-size:var(--t-mono);line-height:1.55;overflow:auto}pre code{background:none;padding:0;color:inherit}
.rel{display:flex;flex-direction:column;gap:10px}.rel a{display:flex;flex-direction:column;gap:3px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;color:var(--fg)}.rel a:hover{border-color:var(--accent)}.rel .t{font-weight:600}.rel .m{color:var(--muted)}
details{border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin:8px 0}summary{cursor:pointer;font-family:var(--mono);font-size:var(--t-sm)}details .v{margin:8px 0 0}
.right{border-left:1px solid var(--line);padding:24px 16px;font-size:var(--t-sm);display:flex;flex-direction:column;min-height:0}.right .toc{flex:1;overflow:auto}.right .keys{border-top:1px solid var(--line);margin-top:16px;padding-top:14px}.keys div{display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:var(--muted);padding:3px 10px}kbd{font-family:var(--mono);font-size:var(--t-xs);line-height:1.6;min-width:18px;text-align:center;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--bg);color:var(--fg)}.right .h{font-size:var(--t-xs);font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;padding-left:10px}
.right a{display:block;color:var(--muted);padding:5px 10px;border-left:2px solid transparent;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.right a:hover{color:var(--fg)}.right a.sub{padding-left:22px;font-size:var(--t-mono)}.right a.on{color:var(--accent);border-left-color:var(--accent)}
.book{color:var(--muted);font-size:var(--t-h3)}.book h2{color:var(--fg)}
.map .layers{position:relative;display:flex;gap:60px;align-items:flex-start;width:max-content;padding:4px 2px}.map svg.edges{position:absolute;left:0;top:0;pointer-events:none;overflow:visible;z-index:0}.map .edge{fill:none;stroke:var(--muted);stroke-width:1.75;opacity:.85}.map .edge.planned{stroke-dasharray:5 4}.map .arrow{fill:none;stroke:var(--muted);stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}.map .dot{fill:var(--muted)}.map .node{position:relative;z-index:1}.map .col{display:flex;flex-direction:column;gap:12px;justify-content:center}.map .node{display:block;width:250px;background:var(--panel);border:1.5px solid var(--line);border-radius:10px;padding:10px 12px;color:var(--fg)}.map a.node:hover{border-color:var(--accent)}.map .n-confirmed{background:var(--ok-soft);border-color:var(--ok)}.map .n-refuted{background:var(--bad-soft);border-color:var(--bad)}.map .n-inconclusive{background:var(--soft);border-color:var(--muted)}.map .n-running{background:var(--run-soft);border-color:var(--run)}.map .n-planned{border-style:dashed;border-color:var(--muted)}.map .t{font-weight:600;font-size:var(--t-sm);font-family:var(--sans)}.map .m{font-family:var(--mono);font-size:10px;color:var(--muted);margin:2px 0 6px}.map .f{font-size:12.5px;line-height:1.45}.map .from{font-size:11.5px;color:var(--muted);margin-top:4px;cursor:help}.thread .line{display:grid;grid-template-columns:200px minmax(0,1fr) 110px;gap:0 16px;align-items:baseline;padding:10px 0;border-top:1px solid var(--line);font-size:var(--t-body)}.thread .line .c .tag{margin-right:8px}.thread .line .who{color:var(--muted);font-size:var(--t-sm);text-align:left}.thread .line a.t{color:inherit;font-weight:500}.thread .line a.t:hover{color:var(--accent)}.thread .line .o{text-align:right}.thread .ev{margin-top:5px;font-size:var(--t-sm);color:var(--muted)}.thread .ev+.ev{margin-top:3px}.thread .ev a{color:inherit}.thread .ev a:hover{color:var(--accent)}table.data td.mono{font-family:var(--mono);font-size:var(--t-mono)}tr.planned td{color:var(--muted)}.card.prose{font-size:var(--t-prose);line-height:1.6}.map svg{display:block;width:auto;max-width:none;font-family:var(--mono)}.map .n rect{fill:var(--panel);stroke:var(--line);stroke-width:1.2}.map .n-confirmed rect{fill:var(--ok-soft);stroke:var(--ok)}.map .n-refuted rect{fill:var(--bad-soft);stroke:var(--bad)}.map .n-inconclusive rect{fill:var(--soft);stroke:var(--muted)}.map .n-running rect{fill:var(--run-soft);stroke:var(--run)}.map .n-planned rect{stroke:var(--muted);stroke-dasharray:4 3}.map .l{font-size:13px;font-weight:600;fill:var(--fg);font-family:var(--sans)}.map .m{font-size:10px;fill:var(--muted)}.map .e{fill:none;stroke:var(--line);stroke-width:1.5}.map .e.planned{stroke-dasharray:4 3}.map a:hover rect{stroke:var(--accent)}
.thread{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:10px 0;scroll-margin-top:60px}.thread>h3{margin:0 0 2px}.thread>p{margin:0 0 10px;color:var(--muted);font-size:var(--t-sm)}.thread ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:10px}.thread li{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:2px 20px;align-items:baseline}.thread li>a{font-weight:600;color:inherit}.thread li>a:hover{color:var(--accent)}.thread .who{font-size:var(--t-sm);color:var(--muted);text-align:right}.thread .s{grid-column:1/-1;font-size:var(--t-sm);color:var(--muted)}.believe>ul{padding-left:20px}.believe li{margin:0 0 12px}
/* ---- layout D, type E: one column page, sticky section bar, prose in Plex, structure in Instrument ---- */
:root{--prose:"IBM Plex Sans",system-ui,sans-serif;--fig:#fbfcfd;--t-prose:15px}
body{font-family:var(--prose)}h1,h2,h3,.side,.top,.bar,.badge,.tag,.wt,.lbl,.kicker,.meta,th,figcaption,.keys,.hint,summary{font-family:var(--sans)}
.app{grid-template-columns:280px minmax(0,1fr)}.app.nonav{grid-template-columns:minmax(0,1fr)}.right{display:none}
.main{padding:28px 40px 64px}.page{max-width:1120px}
h1{font-size:26px;margin:6px 0 6px}h2{margin:52px 0 14px}.prose p,.v p{margin:0 0 12px}.theme,.nav{white-space:nowrap}h2{margin:44px 0 12px}
.item{align-items:flex-start;line-height:1.4}.item .dot{margin-top:6px}.item .lab{min-width:0}.item .sub{margin-top:3px}.item.d1,.item.d2,.item.d3{margin-left:0}
.side .keys{margin-top:auto;border-top:1px solid var(--line);padding-top:12px}.keys div{display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:var(--muted);padding:3px 10px;font-size:var(--t-xs)}
.lbl{display:block;font-size:var(--t-xs);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.muted{color:var(--muted)}figcaption .lbl{display:inline;margin:0;font-size:inherit;letter-spacing:0;text-transform:none;color:var(--fg)}
.step .n{grid-row:1/3;width:26px;height:26px;border-radius:50%;background:var(--accent);color:var(--bg);flex:none;font-family:var(--mono);font-size:var(--t-xs);font-weight:600;display:flex;align-items:center;justify-content:center;margin-top:1px}.step.env .n{background:var(--soft);color:var(--muted)}.step .t{font-family:var(--sans)}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:6px 0;font-size:var(--t-sm);color:var(--muted);margin:0 0 14px}.meta>span+span::before{content:"·";margin:0 12px;color:var(--line)}
.hyp{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px 16px;align-items:start;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:0 0 12px}
.hyp .lab{display:flex;flex-direction:column;gap:6px;align-items:flex-start;padding-top:4px}.hyp .lab .lbl{margin:0}.hyp .out{padding-top:4px}.hyp .v{font-size:var(--t-prose);line-height:1.6}.split{gap:24px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px}
.needs{display:grid;grid-template-columns:auto repeat(4,minmax(0,1fr));border:1px solid var(--line);border-radius:10px;background:var(--panel);font-size:var(--t-sm);margin:0 0 18px;overflow:hidden}
.needs>div{padding:8px 12px;border-right:1px solid var(--line);min-width:0}.needs>div:last-child{border-right:0}.needs .hot{background:var(--run-soft)}.needs .hot .lbl{color:var(--run)}
.pred{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:0 0 18px;font-size:var(--t-prose);line-height:1.6}.pred p{margin:0 0 6px}
table.clauses td.result{font-weight:500}table.clauses td.result .badge{margin-right:4px}table.clauses td.result .reading+.reading{margin-top:8px;padding-top:8px;border-top:1px solid var(--rule)}table.clauses td.muted{color:var(--muted)}td.hot{background:var(--run-soft)}
.bar{position:sticky;top:-28px;z-index:5;display:flex;flex-wrap:wrap;gap:2px;border-bottom:1px solid var(--line);background:var(--bg);margin:22px 0 0;font-size:var(--t-sm)}
.bar a{padding:8px 12px;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px}.bar a i{font-style:normal;font-family:var(--mono);font-size:var(--t-xs)}.bar a:hover{color:var(--fg)}.bar a.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.bar .hint{margin-left:auto;padding:8px 4px;font-family:var(--mono);font-size:var(--t-xs);color:var(--muted)}.bar .filter{margin:4px 0 4px auto;width:220px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--fg);padding:4px 10px;font:inherit;font-size:var(--t-sm)}.bar .filter:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}.hint kbd{margin:0 1px}
.rows{display:grid;grid-template-columns:130px minmax(0,1fr);gap:16px 16px;font-size:var(--t-prose);line-height:1.6}.rows>b{font-size:var(--t-xs);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding-top:5px;font-family:var(--sans)}.rows>div>p:first-child{margin-top:0}.rows>div>:last-child{margin-bottom:0}.rows p,.rows ul,.rows ol{margin:0 0 8px}
.res,.res.wide{display:block;padding:16px 22px}.res>h3,.res>summary>h3{display:flex;gap:10px;align-items:center;font-size:var(--t-lead);margin:0 0 12px}.res>summary>h3{margin:0}
.res .fp.summary{font-size:var(--t-prose);line-height:1.6;background:var(--soft);padding:10px 12px;border-radius:8px;margin:0 0 12px}
.res .fp{font-size:var(--t-prose);line-height:1.6}.res .fp+.fp{margin-top:16px}.res .fp p{margin:0 0 10px}.res .fp>:last-child{margin-bottom:0}.res .fp.summary+.read{margin-top:16px}.res>.read,.split>.read{display:block;grid-column:auto;grid-row:auto;font-size:var(--t-prose);line-height:1.6}
.split{display:grid;grid-template-columns:minmax(0,9fr) minmax(0,11fr);gap:20px;align-items:start}.split>figure{margin:0;grid-column:auto;grid-row:auto}
.read.two{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px 28px;margin-top:16px}.read.two .fp{min-width:0}
.res>figure{grid-column:auto;grid-row:auto;margin:0 0 6px}.res>.src{margin:10px 0 0}
.fig{background:var(--fig);border:1px solid var(--line);border-radius:4px;padding:10px 12px;cursor:auto}.fig svg,.fig img{width:100%;height:auto;display:block}
details.res,details.dec,details.met{border:1px solid var(--line);border-radius:12px;padding:0;margin:8px 0;background:var(--panel);scroll-margin-top:60px}
details.res>summary,details.dec>summary,details.met>summary{list-style:none;cursor:pointer;padding:12px 22px;display:flex;gap:10px;align-items:center}
details.res>summary::-webkit-details-marker,details.dec>summary::-webkit-details-marker,details.met>summary::-webkit-details-marker{display:none}
details.res>summary::before,details.dec>summary::before,details.met>summary::before{content:"▸";color:var(--muted);font-size:13px;flex:none}details[open]>summary::before{content:"▾"}
details.res>summary h3,details.dec>summary h3,details.met>summary h3{flex:1;min-width:0;font-size:var(--t-body);margin:0;display:flex;gap:10px;align-items:center}
details.res>.body,details.dec>.body,details.met>.body{padding:0 22px 16px}
.res.key>details.res{border:0;border-top:1px dashed var(--line);border-radius:0;background:transparent;margin:12px 0 0}.res.key>details.res>summary{padding:10px 0 0}.res.key>details.res>summary h3{font-size:var(--t-body)}.res.key>details.res>.body{padding:10px 0 0}.res.key>details.res .fp.summary{font-size:var(--t-sm)}.res.key>details.res .fp{font-size:var(--t-sm)}details.dec .rows,details.met .rows{margin-top:4px}details.met .rows>div code,details.dec .rows>div code{font-size:.85em}
details.met .io{display:grid;grid-template-columns:1fr 1fr;gap:12px 28px;margin:0 0 12px}details.met .fp{font-size:var(--t-sm);line-height:1.55}details.met .fp.how{font-size:var(--t-prose);line-height:1.6}details.met .src{margin:12px 22px 16px}
details.more{border:0;padding:0;margin:10px 0 0}details.more>summary{color:var(--muted);font-size:var(--t-mono)}.assets{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px}.assets>*{min-width:0;max-width:100%}.assets>figure:has(.scroll){flex:1 1 100%}.assets figure{margin:0;flex:1 1 100%}.assets figure:has(.fig){flex:1 1 calc(50% - 8px)}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:12px 0 0}.pair figure{margin:0}
.ds{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:12px 0;scroll-margin-top:60px}.ds>h3{margin:0 0 12px}.ds .read{font-size:var(--t-prose);line-height:1.6}.ds .read p{margin:0 0 8px}.ds figure{margin:14px 0 0}.ds .split>figure{margin:0}
.rel a .t{font-family:var(--sans)}
@media(max-width:1100px){.main{padding:24px 28px 56px}}
@media(max-width:800px){.app{grid-template-columns:minmax(0,1fr);grid-template-rows:52px minmax(0,1fr)}.side{position:fixed;top:52px;left:0;bottom:0;width:min(320px,85vw);z-index:8;box-shadow:0 0 24px rgba(0,0,0,.25)}.app.nonav .side{display:none}
.main{padding:16px 16px 48px}h1{font-size:22px}.top{padding:0 12px;gap:10px}.crumb{display:none}.bar{top:-16px}.bar .hint{display:none}
.needs{grid-template-columns:1fr 1fr}.needs>div{border-bottom:1px solid var(--line)}.split,.read.two,.pair,.rows,details.met .io,.thread li{grid-template-columns:minmax(0,1fr)}.rows{gap:4px 0}.rows>b{padding-top:8px}.res{padding:14px}.thread .who{text-align:left}}
#overview .book{font-size:var(--t-body);max-width:760px}.toc-nav{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 22px;font-family:var(--sans);font-size:var(--t-sm)}.toc-nav a{border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:5px 12px;color:var(--fg)}.toc-nav a:hover{color:var(--accent)}.toc-nav .n{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted);margin-left:6px}
.prog{margin:0 0 18px;scroll-margin-top:60px}.prog .head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;font-family:var(--sans)}.prog .head .t{font-size:18px;font-weight:600;color:var(--fg)}.prog .head .t:hover{color:var(--accent)}.prog .head .m{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted)}.prog .head .go{margin-left:auto;font-size:var(--t-sm)}.prog .q{margin:6px 0 0;color:var(--muted);font-size:var(--t-sm);max-width:820px}.prog .scroll{margin:14px 0 0}
table.toc td{padding:10px 12px}table.toc th{padding:8px 12px}table.toc td:nth-child(2){min-width:240px}table.toc td:nth-child(2) a,table.toc td .t{font-family:var(--sans);font-weight:600;color:var(--fg)}table.toc td:nth-child(2) a:hover{color:var(--accent)}table.toc td .m{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted);margin-top:3px}table.toc td .m a{font-weight:400;color:var(--accent)}table.toc .fl{display:flex;gap:8px;align-items:baseline}table.toc .fl+.fl{margin-top:3px}table.toc .fl .tag{flex:none}table.toc tr.planned{opacity:.75}table.toc td:last-child{width:52%}
@media print{.scroll.tall{max-height:none;overflow:visible}.scroll{-webkit-mask-image:none;mask-image:none}.sh{display:none}:root{--bg:#fff;--panel:#fff;--plate:#fff}.app{display:block;height:auto}.top,.side,.right{display:none}.main{overflow:visible;padding:0;background:#fff}.fig{border:0;padding:0;background:none}pre{background:#f4f5f7;color:#16181d;border:1px solid #d8dce3}pre .c{color:#6b7280}.page{display:block!important;max-width:none;break-before:page}h2{break-after:avoid}.res,.dec,.met,details,.rel a{break-inside:avoid}a{color:inherit}body{font-size:11pt}}
"""

JS = """
const pages=[...document.querySelectorAll('.page')],items=[...document.querySelectorAll('.side .item')],crumb=document.querySelector('.crumb'),main=document.querySelector('.main');
function route(){const [id,sub]=location.hash.slice(1).split('/');const page=document.getElementById(id)||pages[0];
 pages.forEach(p=>p.classList.toggle('on',p===page));items.forEach(i=>i.classList.toggle('on',i.dataset.page===page.id));
 const fam=[...document.querySelectorAll('.side .h')].reverse().find(h=>h.compareDocumentPosition(items.find(i=>i.dataset.page===page.id))&Node.DOCUMENT_POSITION_FOLLOWING);
 crumb.innerHTML=(page.id==='overview'?'':'<span>'+(fam?fam.textContent:'')+'</span><span>›</span>')+'<b>'+page.dataset.title+'</b>';
 const target=sub?document.getElementById(id+'/'+sub):null;for(let d=target&&target.closest('details');d;d=d.parentElement.closest('details'))d.open=true;
 if(target&&target.tagName==='DETAILS')target.open=true;
 (target||main).scrollIntoView?.({block:'start'});if(!target)main.scrollTop=0;spy();cut();edges();}
function edges(){for(const map of document.querySelectorAll('.page.on .map')){const layers=map.querySelector('.layers'),svg=map.querySelector('svg.edges');if(!layers||!svg)return;const mid='arrow-'+map.closest('.page').id;
 const box=layers.getBoundingClientRect();svg.setAttribute('width',layers.scrollWidth);svg.setAttribute('height',layers.scrollHeight);const nodes={};layers.querySelectorAll('.node').forEach(n=>nodes[n.dataset.slug]=n);let d='';
 for(const n of Object.values(nodes)){const r=n.getBoundingClientRect(),x2=r.left-box.left,y2=r.top-box.top+r.height/2,planned=n.classList.contains('n-planned');
  for(const p of (n.dataset.parents||'').split(' ').filter(Boolean)){const q=nodes[p];if(!q)continue;const s=q.getBoundingClientRect(),x1=s.right-box.left,y1=s.top-box.top+s.height/2,m=(x1+x2)/2;
   d+='<path class="edge'+(planned?' planned':'')+'" marker-start="url(#'+mid+'-dot)" marker-end="url(#'+mid+')" d="M'+x1+','+y1+' C'+m+','+y1+' '+m+','+y2+' '+(x2-1)+','+y2+'"/>';}}
 svg.innerHTML='<defs><marker id="'+mid+'" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path class="arrow" d="M2,1.5 L8,5 L2,8.5"/></marker><marker id="'+mid+'-dot" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="5" markerHeight="5"><circle class="dot" cx="3" cy="3" r="2.2"/></marker></defs>'+d;}}
addEventListener('resize',edges);addEventListener('load',edges);if(document.fonts)document.fonts.ready.then(edges);if(window.ResizeObserver)new ResizeObserver(()=>edges()).observe(main);
function spy(){const page=document.querySelector('.page.on');if(!page)return;const bar=page.querySelector('.bar');if(!bar)return;
 const top=main.getBoundingClientRect().top+90;const marks=[...page.querySelectorAll('h2')];let cur=marks[0];for(const m of marks){if(m.getBoundingClientRect().top<=top)cur=m;else break;}
 bar.querySelectorAll('a').forEach(a=>a.classList.toggle('on',cur&&a.getAttribute('href')==='#'+cur.id));}
const scrollers=[...document.querySelectorAll('.scroll')];
function cut(){for(const el of scrollers){const mx=el.scrollWidth-el.clientWidth,my=el.scrollHeight-el.clientHeight,x=mx>1&&el.scrollLeft<mx-1,y=my>1&&el.scrollTop<my-1;
 el.classList.toggle('x',x);el.classList.toggle('y',y);let h=el.nextElementSibling;if(!h||!h.classList.contains('sh')){h=document.createElement('div');h.className='sh';el.after(h);}
 const w=[];if(mx>1)w.push('\u2194 scrolls sideways');if(my>1)w.push('\u2195 scrolls down');h.textContent=w.join(' \u00b7 ');h.classList.toggle('on',w.length>0);}}
scrollers.forEach(el=>el.addEventListener('scroll',cut,{passive:true}));addEventListener('resize',cut);addEventListener('load',cut);if(document.fonts)document.fonts.ready.then(cut);
document.addEventListener('toggle',()=>cut(),true);if(window.ResizeObserver)new ResizeObserver(()=>cut()).observe(main);
addEventListener('hashchange',route);route();main.addEventListener('scroll',spy,{passive:true});
const root=document.documentElement,btn=document.querySelector('.theme');const modes=['system','light','dark'];
function apply(m){if(m==='system')root.removeAttribute('data-theme');else root.dataset.theme=m;btn.textContent='theme: '+m;}
let mode='system';try{mode=localStorage.getItem('report-theme')||'system'}catch(e){}apply(mode);
btn.onclick=()=>{mode=modes[(modes.indexOf(mode)+1)%3];apply(mode);try{localStorage.setItem('report-theme',mode)}catch(e){}};
addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
const tip=document.createElement('div');tip.className='tip';tip.hidden=true;document.body.appendChild(tip);
document.addEventListener('mouseover',e=>{const a=e.target.closest&&e.target.closest('a.ref, a.term');if(!a)return;const [head,...rest]=a.dataset.tip.split('\\n');
 const body=document.createElement('span');body.innerHTML=rest.join('\\n');tip.replaceChildren(Object.assign(document.createElement('b'),{textContent:head}),body);tip.hidden=false;
 const r=a.getBoundingClientRect();tip.style.left=Math.max(8,Math.min(r.left,innerWidth-tip.offsetWidth-12))+'px';
 tip.style.top=(r.bottom+8+tip.offsetHeight>innerHeight?r.top-tip.offsetHeight-8:r.bottom+8)+'px';});
document.addEventListener('mouseout',e=>{if(e.target.closest&&e.target.closest('a.ref, a.term'))tip.hidden=true;});
const app=document.querySelector('.app'),nav=document.querySelector('.nav');
function setNav(on){app.classList.toggle('nonav',!on);try{localStorage.setItem('report-nav',on?'1':'0')}catch(e){}}
let navOn=innerWidth>800;try{navOn=localStorage.getItem('report-nav')!=='0'&&innerWidth>800}catch(e){}setNav(navOn);nav.onclick=()=>setNav(app.classList.contains('nonav'));
items.forEach(i=>i.addEventListener('click',()=>{if(innerWidth<=800)setNav(false)}));
const gf=document.querySelector('#glossary .filter');if(gf){const gs=[...document.querySelectorAll('#glossary .g')],gh=[...document.querySelectorAll('#glossary h2')];
 gf.addEventListener('input',()=>{const q=gf.value.trim().toLowerCase();gs.forEach(g=>g.classList.toggle('off',!!q&&!g.dataset.q.includes(q)));
  gh.forEach(h=>{const dl=h.nextElementSibling;h.classList.toggle('off',!!q&&!(dl&&dl.querySelector('.g:not(.off)')));});});
 gf.addEventListener('keydown',e=>{if(e.key==='Escape'){gf.value='';gf.dispatchEvent(new Event('input'));gf.blur();e.stopPropagation();}});}
let back='';addEventListener('keydown',e=>{const altGr=e.getModifierState&&e.getModifierState('AltGraph');
 if(e.target.matches('input,textarea')||e.metaKey||(!altGr&&(e.ctrlKey||e.altKey)))return;
 const onGloss=location.hash.startsWith('#glossary'),gloss=document.getElementById('glossary'),page=document.querySelector('.page.on');
 const go=h=>{location.hash=h;e.preventDefault();};
 if(e.key==='?'&&gloss&&!onGloss){back=location.hash;go('#glossary');}
 else if((e.key==='?'||e.key==='Escape')&&onGloss)go(back||'#overview');
 else if(e.key==='['||e.key===']'||e.code==='BracketLeft'||e.code==='BracketRight'){
  const fwd=e.key===']'||(e.code==='BracketRight'&&e.key!=='[');
  const i=items.findIndex(x=>x.classList.contains('on')),n=i+(fwd?1:-1);if(items[n])go(items[n].getAttribute('href'));}
 else if(/^[jk]$/i.test(e.key)&&page){const hs=[...page.querySelectorAll('h2')];if(!hs.length)return;const top=main.getBoundingClientRect().top+90;
  let cur=-1;hs.forEach((h,i)=>{if(h.getBoundingClientRect().top<=top)cur=i;});const n=cur+(/^j$/i.test(e.key)?1:-1);if(hs[n])go('#'+hs[n].id);}
 else if(e.key==='/'&&onGloss&&gf){gf.focus();e.preventDefault();}
 else if(e.key==='o')go('#overview');
 else if(e.key==='e'&&page){const ds=[...page.querySelectorAll('details')],open=ds.some(d=>!d.open);ds.forEach(d=>d.open=open);}
 else if(e.key==='n')nav.onclick();
 else if(e.key==='t')btn.onclick();});
"""


def build(out: Path, warnings: bool = False) -> int:
    global REPO_URL
    REPO_URL = repo_url()
    records = load_records()
    terms = load_glossary()
    fams = families(records)
    progs = [Programme(p, records) for p in sorted((EXPERIMENTS / "programmes").glob("*.md"))]
    by_dir = {r.dir.resolve(): r for r in records}
    titled = [(g.page_id, g.meta.get("short") or g.title, render_programme(g, terms)) for g in progs] + [
        (
            r.slug,
            r.meta.get("short") or r.title,
            render_page(r, [by_dir[p] for p in r.builds_on if p in by_dir], terms, progs),
        )
        for fam in fams
        for r, _ in fam
    ]
    used: dict[str, list[tuple[str, str]]] = {}
    for pid, label, html_ in titled:
        for slug in set(re.findall(r'href="#glossary/([^"]+)"', html_)):
            used.setdefault(slug, []).append((pid, label))
    pages = [render_overview(fams, terms, progs)] + [h for _, _, h in titled] + [render_glossary(terms, used)]
    repo = EXPERIMENTS.parent.name
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(repo)} · Findings</title>"
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">'
        f'<style>{CSS}</style></head><body><div class="app">'
        f'<div class="top"><button class="nav" title="toggle experiment list">☰</button><span class="brand">{esc(repo)} · Findings</span><span class="crumb"></span><button class="theme">theme</button></div>'
        f'<nav class="side">{render_sidebar(fams, terms, progs)}</nav>'
        f'<main class="main">{"".join(pages)}</main></div>'
        f"<script>{JS}</script></body></html>"
    )
    out.write_text(restore_math(relink(page, records)), encoding="utf-8")
    found = [(r, p) for r in records for p in r.problems] + [(g, p) for g in progs for p in g.problems]
    problems = [(r, p) for r, p in found if not WARNING.search(p)]
    advice = [(r, p) for r, p in found if WARNING.search(p)]
    if MATH_HTML and latex_to_mathml is None:
        print(
            f"  {len(MATH_HTML)} equations left as plain text: latex2mathml is missing — "
            "run `uv run experiments/build_report.py`, or install it",
            file=sys.stderr,
        )
        problems.append(("math", "no converter"))
    print(
        f"{out}: {len(records)} records in {len(fams)} families, {len(progs)} programmes, {len(terms)} glossary terms, "
        f"{out.stat().st_size // 1024} KiB"
    )
    for r, p in problems:
        print(f"  {r.path.relative_to(EXPERIMENTS.parent)}: {p}", file=sys.stderr)
    counts: dict[Path, int] = {}
    for r, p in advice:
        counts[r.path] = counts.get(r.path, 0) + 1
        if warnings:
            print(f"  {r.path.relative_to(EXPERIMENTS.parent)}: warning: {p}", file=sys.stderr)
    for path, n in counts.items():
        print(
            f"  {path.relative_to(EXPERIMENTS.parent)}: {n} prose warnings"
            + ("" if warnings else " (--warnings lists them)"),
            file=sys.stderr,
        )
    if records and not terms:
        print(
            "  experiments/GLOSSARY.md missing: seed it from the report skill's references/glossary.md", file=sys.stderr
        )
        problems.append(("glossary", "missing"))
    if undefined := undefined_terms(records, terms):
        print(f"  glossary: abbreviations and methods without a definition: {', '.join(undefined)}", file=sys.stderr)
    return 1 if problems else 0


def selftest() -> int:
    """Build the fixture next to this script and check the rendering invariants."""
    global EXPERIMENTS
    fixture = Path(__file__).resolve().parent / "fixture" / "experiments"
    if not fixture.is_dir():
        print(f"no fixture at {fixture}; --selftest runs from the labflow skill copy", file=sys.stderr)
        return 2
    EXPERIMENTS = fixture
    out = fixture / "report.html"
    code = build(out)
    problems = Record(fixture / "data" / "alpha" / "README.md").problems
    records = load_records()
    html_ = out.read_text(encoding="utf-8")
    body = html_[html_.find("<body") : html_.find("<script>")]
    overview = body[body.find('id="overview"') : body.find('<section class="page"', body.find('id="overview"') + 1)]
    checks = {
        "gate fires on the broken F2 heading": code == 1,
        "no heading made from a #<ticket> line": len(re.findall(r"<h1>", body)) == 5,
        "findings first, then verdict, then the question": body.find('id="alpha/findings"')
        < body.find('id="alpha/verdict"')
        < body.find('id="alpha/hypothesis"'),
        "hypothesis card, needs strip, clause table and section bar": '<div class="hyp"><div class="lab"><span class="lbl">Hypothesis</span></div><div class="v">'
        in body
        and '<div class="out"><span class="badge b-inconclusive">' in body
        and '<div class="needs"><div><span class="lbl">Needs</span></div>' in body
        and 'class="hot"' in body
        and '<table class="data wide clauses">' in body
        and '<tr id="alpha/h1"><td><span class="tag">H1</span></td><td>something holds</td><td class="muted">a</td><td class="muted">b — <span class="muted">c.</span></td>'
        in body
        and '<div class="bar"><a href="#alpha/findings">Findings <i>2</i></a>' in body,
        "key finding: prose beside a figure result, supporting folded to a line": re.search(
            r'<article class="res key" id="alpha/f4"[^>]*>.*?<div class="split"><div class="read"><div class="fp summary">.*?<figure',
            body,
            re.S,
        )
        is not None
        and '<details class="res sub" id="alpha/f3"' in body
        and '<details class="res minor" id="alpha/f5"' in body,
        "one light figure on a plate, no dark variant": '<div class="fig"><svg' in body
        and "fig-dark" not in body
        and "fig-light" not in body,
        "short titles in the sidebar": '<span class="lab">Alpha gold</span>' in body
        and 'title="Alpha gold set"' in body,
        "programme needs table": '<table class="data wide needs-table">' in body,
        "prose lint is a warning, structure a problem": WARNING.search("F1 Summary is 29 words (cap 25)") is not None
        and WARNING.search("method step heading not `### M<n> — title`: M4 Broken heading no dash") is None,
        "dataset card: prose beside the figure, table full width under": re.search(
            r'<article class="ds" id="alpha/dataset-1"[^>]*><h3>Gold set</h3><div class="split"><div class="read">.*?<figure[^>]*><div class="fig">.*?</div><figure[^>]*><div class="scroll',
            body,
            re.S,
        )
        is not None,
        "programme page, sidebar entry, map and overview row": 'id="programme-gamma"' in body
        and '<a class="item" href="#programme-gamma" data-page="programme-gamma" title=' in body
        and 'class="node n-planned"' in body
        and 'class="node n-confirmed"' in body
        and '<div class="from" title=' in body
        and '<table class="data wide exp-table">' in body
        and '<h2 id="overview/programmes">Programmes</h2>' in overview
        and 'in <a href="#programme-gamma" title=' in body
        and 'href="#programme-gamma"' in overview,
        "programme citations linked with a hover card, thread lists the finding under its experiment": '<a class="ref" href="#alpha/f1" data-tip="F1 — First thing\n&lt;strong&gt;alpha:&lt;/strong&gt;'
        in body
        and '<article class="thread" id="programme-gamma/t1" data-title="Pool quality">' in body
        and "[alpha:F9]" in body
        and '<div class="line"><span class="who" title="Alpha gold set">' in body
        and '<span class="c"><a class="t" href="#alpha/h1"><span class="tag">H1</span>' in body
        and '<div class="ev"><a href="#alpha/f1"' in body,
        "programme gates fire": {
            "[alpha:F9] cites a finding alpha does not have",
            "[alpha:H9] cites a clause alpha does not have",
        }
        <= set(Programme(fixture / "programmes" / "gamma.md", records).problems),
        "no raw markdown table": not re.search(r">[^<]*\|[^<]*\|[^<]*\|", body),
        "every table styled and scrollable": "<table>" not in body,
        "no relative link survives": not re.search(r'href="(?!#|https?://)', body),
        "bullet-starting ticket ref kept in a list": "<li>#37, the post-hoc" in body,
        "sub-list under a label rendered as a list": "<li>audit <code>de</code>" in body,
        "float formatted": "0.6047</td>" in body and "0.6047136829414016" not in body,
        "svg ids prefixed": 'id="alpha-a-' in body and 'id="axes_1"' not in body,
        "PR link built from the remote or left as text": "PR #62" in body,
        "decision folded to its line, open as rows": "<b>Decision</b><div><p>Pool it" in body
        and '<details class="dec" id="alpha/d1"' in body
        and "<b>Why</b>" in body,
        "[D]/[F] citations linked with a hover card": '<a class="ref" href="#alpha/d1" data-tip="D1 — Pick the pool\npool it, so #33 proceeds.">[D1]</a>'
        in body
        and '<a class="ref" href="#alpha/f1"' in body
        and 'href="#alpha/d9"' not in body
        and "[D9]" in body,
        "bare F1 (the metric) and refs inside code or headings stay text": "macro F1 0.6047" in body
        and '<a class="ref" href="#beta/f1"' not in body[body.find('id="beta"') :]
        and "<code>[D1]</code>" in body
        and 'Pick the pool<span class="wt">' in body,
        "glossary page, sidebar entry and shortcut": '<a class="item" href="#glossary" data-page="glossary">Glossary</a>'
        in body
        and '<div class="keys"><p class="h">Shortcuts</p><div><kbd>?</kbd> glossary</div><div><kbd>/</kbd> filter the glossary</div>'
        in body
        and '<input class="filter" type="search"' in body
        and 'used in <a href="#alpha">' in body
        and "<kbd>]</kbd> previous / next experiment" in body
        and '<dt id="glossary/f1">F1 <span class="also">F1 score, macro F1</span></dt>' in body
        and '<h2 id="glossary/metrics">Metrics</h2>' in body,
        "metric linked to the glossary with its definition": 'threshold is 0.002 <a class="term" href="#glossary/f1" data-tip="F1\nharmonic mean of precision and recall; 1 is perfect.">F1</a>'
        in body
        and 'holds at <a class="term" href="#glossary/q05"' in body
        and "AUC is flat" in body
        and 'href="#glossary/auc"' not in body,
        "term linked once per page, never in code, headings or step titles": body[
            body.find('id="alpha"') : body.find('id="beta"')
        ].count('href="#glossary/precision"')
        == 1
        and "<code>[D1]</code>" in body
        and 'scores F1 0.605<span class="wt">' in body
        and 'class="t"><a class="ref" href="#alpha/f1"' in body,
        "undefined abbreviations and methods reported": undefined_terms(records, load_glossary())
        == ["AUC", "bootstrap"],
        "reproduce rendered as steps": '<ol class="steps"><li class="step env"><span class="n">env</span>' in body
        and '<li class="step"><span class="n">1</span><div class="t">' in body
        and 'class="c">#' in body
        and '<div class="t">3 ' not in body
        and 'class="t">tables/ and figures/ — every finding' in body,
        "spaced double dash renders as a dash, flags untouched": "pool — the members" in body
        and "<code>git log -- prep.py</code>" in body
        and "--dry-run" in body,
        "footer refs labelled, external links in a new tab": "<b>runs</b>" in body
        and '<a href="http://localhost:5000/#/experiments/7/runs/e826d111" target="_blank" rel="noopener"><code class="run" title="e826d111">e826d111</code></a>'
        in body
        and not re.search(r'<a href="https?://[^"]*">', body),
        "overview lists each record once": overview.count('<td><a href="#alpha"') == 1
        and '<table class="data wide toc">' in overview
        and "<h2>data</h2>" not in overview,
        "overview lists key finding titles": '<span class="tag">F1</span><span>First thing</span></div>' in overview,
        "row values start with a capital": "<p>Reading text; it confirms" in body
        and "<b>Decision</b><div><p>Pool it" in body,
        "reading and implication visible": '<div class="fp reading"><span class="lbl">Reading</span>' in body
        and "what it means" not in body,
        "caps linted": {
            "F5 is minor but has Implication",
            "D2 Why is 68 words (cap 60)",
            "F1 Summary is 29 words (cap 25)",
        }
        <= set(problems),
        "voice rules linted": {
            "F4 Reading cites [F1] with no words saying what it is: a reading that names nothing; see [F1].…",
            "D2 Why has a 68-word sentence (cap 35): word word word word word word word word word word …",
            "M1 Settings key `min_chars` carries no gloss: write `min_chars` (what it steers)",
            "F1 Reading carries 3 numbers beyond the Summary's (cap 2): the rest belong in the Result",
            "F6 is supporting but has no Reading: a number with no meaning is not a result",
            "TL;DR line is not `**F<n> — <title>.** <Summary>`: **F3** — held out, the harness agrees at κ 0.69.…",
        }
        <= set(problems)
        and any(p.startswith("F4 Reading is one paragraph of") for p in problems)
        and any(p.startswith("M3 How is one paragraph of") for p in problems)
        and any(p.startswith("Design Stages has a sentence with 3 semicolons") for p in problems)
        and not any(p.startswith(("F3 Reading", "M1 How", "Verdict Evidence")) for p in problems),
        "bold-lead reading blocks rendered as a list": '<div class="fp reading"><span class="lbl">Reading</span><ul>\n<li><strong>Holds on the hold-out.</strong>'
        in body,
        "fix-as-finding rejected": "F6 reads as a fix (defect, fixed, notebook): revise the F<n> it corrects instead"
        in problems,
        "supporting folded under its key finding": re.search(
            r'<article class="res key" id="alpha/f1"(?:(?!</article>).)*<details class="res sub" id="alpha/f3"',
            body,
            re.S,
        )
        is not None,
        "method step rendered with its rows, footer and rail entry": '<details class="met" id="alpha/m1" data-title="M1 · The candidate pool each language contributes">'
        in body
        and "<b>Input</b>" in body
        and "<b>Output</b>" in body
        and "<b>Code</b>" in body
        and '<code class="file">prep.py::pool_candidates</code>' in body
        and "<b>Settings</b>" in body,
        "equations rendered as MathML, none left as source": '<span class="math block"><math' in body
        and "<mfrac>" in body
        and '<span class="math"><math' in body
        and "$$" not in body
        and "mathx0x" not in body,
        "[M<n>] citation linked with a hover card, no MathML inside a card": '<a class="ref" href="#alpha/m2"' in body
        and not re.search(r'data-tip="[^"]*<math', body)
        and "mathx" not in body,
        "method gates fire": {
            "M2 has no Output: a step reads cold or not at all",
            "M3 Code names vanished, which prep.py does not define",
            "method step heading not `### M<n> — title`: M4 Broken heading no dash",
        }
        <= set(problems),
        "concluded record without Method flagged": "concluded record has no ## Methods"
        in Record(fixture / "methods" / "beta" / "README.md").problems,
        "long table scrolls under a sticky header, short one does not": '<div class="scroll tall"><table class="data tall">'
        in body
        and body.count('class="scroll tall"') == 1,
        "prose link to an inlined table reaches its figure": 'id="alpha/asset/t"' in body
        and '<a class="ref" href="#alpha/asset/t">Every row</a>' in body,
        "datasets section rendered with its table and figure": '<h2 id="alpha/datasets">Datasets</h2><article class="ds" id="alpha/dataset-1" data-title="Gold set"><h3>Gold set</h3>'
        in body
        and body.count("<figcaption>") >= 2
        and 'id="alpha-a-' in body[body.find('id="alpha/datasets"') : body.find('id="alpha/methods"')],
        "prose outside Design rows flagged": "Design has content outside `- **Label**:` rows: #### Notes" in problems,
        "ticket ref in a Summary flagged": "F1 Summary cites a ticket (#32): findings read cold" in problems,
        "key finding without a clause citation flagged": "F4 is key but its Reading cites no clause as [H<n>]"
        in problems,
        "clauses rendered as an id-marked list, [H<n>] linked with a hover card": '<a class="ref" href="#alpha/h1" data-tip="H1 — something holds\nconfirmed if a; refuted if b — c.">[H1]</a>'
        in body,
        "missing Discussion flagged": "concluded record has no Verdict Discussion"
        in Record(fixture / "methods" / "beta" / "README.md").problems,
    }
    for name, ok in checks.items():
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
    out.unlink()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EXPERIMENTS / "report.html")
    ap.add_argument("--selftest", action="store_true", help="build the bundled fixture and check it")
    ap.add_argument(
        "--warnings", action="store_true", help="list every prose warning instead of counting them per record"
    )
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else build(args.out, args.warnings))
