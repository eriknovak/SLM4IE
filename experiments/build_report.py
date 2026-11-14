#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.6", "pyyaml>=6.0", "latex2mathml>=3.77", "jinja2>=3.1"]
# ///
"""Render every labflow experiment record into one self-contained HTML report.

Reads the book (experiments/README.md) and every record
(experiments/<category>/<slug>/README.md) and writes experiments/report.html:
an app-style page with the rail (labflow_report/rail.py: the Overview,
Research, Reference and Glossary lists and the top bar), one report page per
record, programme, topic and entry, and the overview and glossary. Figures and `tables/*.csv`
are inlined, so the file needs no data source. Requires `markdown` and
`pyyaml`.

    uv run experiments/build_report.py [--out experiments/report.html]

With `experiments/reference/` present the report gains the reference: one
page per topic (`reference/<topic>/README.md`) and per entry
(`reference/<topic>/<entry>.md`), the ideas read from the tracker (issues
labelled `idea`, cached in `reference/ideas.yaml`; `--offline` reads the
cache only), and the links between entries and the records that name them
under `varies` and `uses`. Without the folder the report is unchanged.

`$…$` and `$$…$$` in a record become MathML, so equations need no script or
font at read time. Under plain `python` that needs `latex2mathml` installed
beside `markdown` and `pyyaml`; `uv run` fetches all three itself.

Exit code 1 on a structural hole, a part a page cannot be built from: a
`### F<n>`, `### M<n>` or `### R<n>` heading the parser cannot read, a
`<folder>/<slug>/README.md` outside the category folders (data, methods,
validation), a topic folder without README.md, a missing GLOSSARY.md or one
none of whose term lines parse, and an equation with latex2mathml missing.
Every writing rule lives in the checker and its specs
(`labflow_report/specs/*.yaml`, `python -m labflow_report check`, or
`--check` here), never in this script: the build prints the count of the
checker's breaches at each document's stage and exits 0 on them, since the
stage gates are the hard stop. Abbreviations and standard methods named in
record prose but not in the glossary are listed, never fail the build.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
import yaml

try:
    from latex2mathml.converter import convert as latex_to_mathml
except ModuleNotFoundError:  # only an equation needs it; build() fails loudly if one appears
    latex_to_mathml = None

EXPERIMENTS = Path(__file__).resolve().parent
# the spec-led builder ships beside this script; a lone script reads the skill's copy, as report-guard does
SKILL_REFERENCES = Path.home() / ".claude" / "skills" / "labflow" / "skills" / "report" / "references"
BUILDER_HOME = next((f for f in (EXPERIMENTS, SKILL_REFERENCES) if (f / "labflow_report").is_dir()), None)
if BUILDER_HOME is None:
    raise ImportError(
        f"no labflow_report/ beside {Path(__file__).name} or in {SKILL_REFERENCES}: "
        "run the report skill's labflow_report/ship.py on this experiments folder"
    )
if str(BUILDER_HOME) not in sys.path:
    sys.path.insert(0, str(BUILDER_HOME))
from labflow_report import builder as spec_builder  # noqa: E402
from labflow_report import landing_page, rail  # noqa: E402
from labflow_report.check import SPEC_FOLDER  # noqa: E402
from labflow_report.checks.glossary import Term  # noqa: E402
from labflow_report.checks.glossary import terms as glossary_terms  # noqa: E402
from labflow_report.document import read_document  # noqa: E402
from labflow_report.spec import load_specs  # noqa: E402

REPO_URL = None  # set in build()
GITHUB = ""  # owner/name of a GitHub origin, set in build()
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
# `#32` at line or bullet start is a ticket ref, not a heading; escape it for markdown
HASH_REF = re.compile(r"(?m)^([ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?)#(?=\d)")
CLAUSE_ID = r"[HCQ]\d+"
ROUND_ID = r"R\d+"
# a README.md one folder down that the record spec's path does not match
OUTSIDE_RECORD_FOLDERS = "not in a category folder (data, methods, validation), so it has no page"
# the glossary rule that flags a list line under a heading that is no term line
GLOSSARY_LINE_RULE = "GLO-005"
# a `### F<n>`, `### M<n>` or `### R<n>` heading the page reads: its id, a dash, a title
UNREAD_HEADINGS = {
    "findings": ("F", "finding heading not `### F<n> — title · weight`"),
    "methods": ("M", "method step heading not `### M<n> — title`"),
    "rounds": ("R", "round heading not `### R<n> — what it tests`"),
}
TABLE_ROWS = 20  # a table longer than this scrolls vertically under a sticky header (the CSS hard-codes the same 20)
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


MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\w$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])", re.S)
MATH_TOKEN = re.compile(r"mathx(\d+)x")
MATH_HTML: list[str] = []
MATH_SRC: list[str] = []


MTEXT = re.compile(r"<mtext([^>]*)>(.*?)</mtext>", re.S)


def escape_mtext(mathml: str) -> str:
    """latex2mathml copies `\\texttt{<q>}` into `<mtext>` unescaped, which opens a
    stray element that swallows the rest of the report; escape the text."""
    return MTEXT.sub(lambda m: f"<mtext{m.group(1)}>{esc(html.unescape(m.group(2)))}</mtext>", mathml)


def stash_math(m: re.Match) -> str:
    """One `$…$` or `$$…$$` becomes a plain token, so markdown, the reference
    linker and the glossary linker all pass over it; build() puts the MathML
    back once every substitution has run."""
    latex, block = (m.group(1), True) if m.group(1) is not None else (m.group(2), False)
    latex = latex.strip()
    if latex_to_mathml is None:
        rendered = f"<code>{esc(latex)}</code>"
    else:
        rendered = escape_mtext(latex_to_mathml(latex, display="block" if block else "inline"))
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


def md_inline(text: str) -> str:
    out = md(text).strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S) if out.count("<p>") == 1 else out


GITHUB_ICON = (
    '<svg class="gh" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 '
    "7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-"
    ".52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
    "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
    "1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73"
    '.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>'
)
# Inline flask glyph in the report's muted ink: a light grey that reads on light and dark tabs.
FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%238a93a3'%3E"
    "%3Cpath d='M8 2h8a1 1 0 0 1 0 2h-1v5.2l5.3 9.3A2 2 0 0 1 18.6 22H5.4a2 2 0 0 1-1.7-3.5L9 9.2V4H8a1 1 0 0 1 0-2z"
    "m3 2v5.8L8.1 15h7.8L13 9.8V4h-2z'/%3E%3C/svg%3E"
)


def git_head() -> str | None:
    """Short hash of the commit the report is built from; None outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(EXPERIMENTS), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


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


class Record:
    """One record README as the page builder reads it: its frontmatter, its
    sections and the `### F<n>`, `### M<n>` or `### R<n>` headings the page
    cannot read, its holes. Writing rules are the record checker's."""

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
        self.builds_on = [(self.dir / p).resolve() for p in (self.meta.get("builds_on") or [])]
        self.children: list[Record] = []
        self.problems = [
            f"{message}: {head}"
            for key, (kind, message) in UNREAD_HEADINGS.items()
            for head in unread_headings(strip_placeholders(self.sections.get(key, "")), kind)
        ]

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


def unread_headings(block: str, kind: str) -> list[str]:
    """The `### ` headings of an entries section that open with no `<kind><n> —`."""
    heading = re.compile(rf"{kind}\d+\s*[—–-]\s*\S")
    heads = [chunk.partition("\n")[0].strip() for chunk in re.split(r"^### ", block, flags=re.M)[1:]]
    return [head for head in heads if not heading.match(head)]


def strip_placeholders(text: str) -> str:
    return "" if re.fullmatch(r"\s*<.*>\s*", text, flags=re.S) else text


def split_readmes() -> tuple[list[Path], list[Path]]:
    """The `<folder>/<slug>/README.md` files outside reference/: those the record
    spec's path matches, as the models are read, and the rest, which get no page."""
    spec = load_specs(SPEC_FOLDER)["record"]
    matched = {f.resolve() for f in (EXPERIMENTS / spec.root if spec.root else EXPERIMENTS).glob(spec.path)}
    readmes = [p for p in sorted(EXPERIMENTS.glob("*/*/README.md")) if p.parent.parent.name != REFERENCE]
    return [p for p in readmes if p.resolve() in matched], [p for p in readmes if p.resolve() not in matched]


def load_records() -> list[Record]:
    records = [Record(p) for p in split_readmes()[0]]
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
    seen: set = set()  # a record with parents in two families renders once, under the first
    return [walk(r, 0, seen) for r in roots]


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


SKIP_TAGS = ("code", "a", "pre", "h1", "h2", "h3", "th", "svg", "style", "script")
VOID_TAGS = ("br", "img", "hr", "wbr", "input", "meta", "link", "col", "source")


# a tag, badge, chip or said-in-words gap is a label, never prose to link
NOLINK_CLASS = re.compile(r'class="(?:[^"]* )?(?:tag|badge|cc|ct|sha|rid|when|byhand|missing)[ "]')


def sub_prose(html_: str, fn) -> str:
    """Apply `fn(text)` to the prose runs of an HTML fragment, leaving text
    inside code, links, headings, table heads, figures, id tags and any element
    marked `data-nolink` alone."""
    out, skip = [], []
    for tok in re.split(r"(<[^>]+>)", html_):
        if tok.startswith("<"):
            m = re.match(r"</?([a-zA-Z0-9]+)", tok)
            name = m.group(1).lower() if m else ""
            if tok.startswith("</"):
                if skip and skip[-1] == name:
                    skip.pop()
            elif (
                (skip or name in SKIP_TAGS or NOLINK_CLASS.search(tok) or "data-nolink" in tok)
                and not tok.endswith("/>")
                and name not in VOID_TAGS
                and not tok.startswith("<!")
            ):
                skip.append(name)
            out.append(tok)
        else:
            out.append(tok if skip else fn(tok))
    return "".join(out)


def load_glossary() -> list[Term]:
    """The term lines of experiments/GLOSSARY.md, read as the glossary checks read them."""
    path = EXPERIMENTS / "GLOSSARY.md"
    if not path.is_file():
        return []
    return glossary_terms(read_document(path, "glossary", EXPERIMENTS))


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
        for key in ("hypothesis", "rounds", "design", "datasets", "methods", "findings", "verdict"):
            text = re.sub(
                rf"```.*?```|`[^`]*`|\[(?:[DFM]\d+|{CLAUSE_ID})\]|!?\[[^\]]*\]\([^)]*\)",
                " ",
                r.sections.get(key, ""),
                flags=re.S,
            )
            text = "\n".join(l for l in MATH.sub(" ", text).splitlines() if not l.lstrip().startswith(("|", "###")))
            hits |= {
                w
                for w in ABBREV.findall(text)
                if w.lower() not in known and not re.fullmatch(rf"[DFM]\d+|{CLAUSE_ID}|{ROUND_ID}", w)
            }
            hits |= {w for w in methods.findall(text) if w.lower() not in known}
    return sorted({h.lower(): h for h in sorted(hits, reverse=True)}.values(), key=str.lower)


def relink(page: str, records: list[Record]) -> str:
    """Relative links out of records and the book: a record README becomes an
    in-page hash link (its `#f<n>` anchor too); anything else is unwrapped, since
    a self-contained file cannot resolve it. External links open in a new tab."""
    by_slug = {r.slug for r in records}

    def sub(m: re.Match) -> str:
        href, text = m.group(1), m.group(2)
        t = re.match(r"(?:[^#]*/)?([^/#]+)/README\.md(?:#(f\d+)\b.*)?$", href)
        if t and t.group(1) in by_slug:
            return f'<a href="#{t.group(1)}{"/" + t.group(2) if t.group(2) else ""}">{text}</a>'
        return text

    page = re.sub(r'<a href="(?!#|https?://|mailto:)([^"]*)">(.*?)</a>', sub, page, flags=re.S)
    # any attribute order; a link that already names its target keeps it
    return re.sub(r'<a\b((?![^>]*\btarget=)[^>]*\bhref="https?://[^"]*"[^>]*)>', r'<a\1 target="_blank" rel="noopener">', page)


# ---- reference: topics, entries and ideas (experiments/reference/<topic>/<entry>.md) ----
REFERENCE = "reference"
IDEAS_ONLINE = True  # --offline and the self-test read the cache only


class RefEntry:
    """One `experiments/reference/<topic>/<entry>.md`, as entry links read it;
    labflow_report/entry_page.py draws its page."""

    def __init__(self, path: Path):
        self.path = path
        match = FRONTMATTER.match(path.read_text(encoding="utf-8"))
        if not match:
            raise ValueError(f"{path}: missing YAML frontmatter")
        meta = yaml.safe_load(match.group(1)) or {}
        self.key = f"{path.parent.name}/{path.stem}"
        self.page_id = f"ref-{path.parent.name}-{path.stem}"
        self.title = str(meta.get("title") or path.stem)
        self.kind = str(meta.get("kind") or "")
        self.summary = str(meta.get("summary") or "").strip()


class Topic:
    """One `experiments/reference/<topic>/`: its README.md states the question and
    names the Facts labels compared; every other `.md` in the folder is an entry."""

    def __init__(self, folder: Path):
        self.path = folder / "README.md"
        # the topic's page is drawn from its README, so a topic without one has none
        self.problems = [] if self.path.is_file() else ["topic has no README.md"]
        # an entry name is kebab-case, as the entry spec reads it, so README.md is no entry
        self.entries = [RefEntry(p) for p in sorted(folder.glob("[a-z0-9]*.md"))]


class Ideas:
    """Every idea, read from the tracker and kept in `reference/ideas.yaml` so a
    build without tracker access shows the same ones; the ideas checks
    (IDE-001..003) read the file."""

    def __init__(self, folder: Path):
        self.path = folder / "ideas.yaml"
        self.source = ""
        issues = self.fetch() if IDEAS_ONLINE else None
        if issues is not None:
            self.source = "tracker"
            issues.sort(key=lambda x: x.get("number") or 0)
            text = yaml.safe_dump(issues, sort_keys=False, allow_unicode=True, width=100)
            if (issues or self.path.is_file()) and (
                not self.path.is_file() or self.path.read_text(encoding="utf-8") != text
            ):
                self.path.write_text(text, encoding="utf-8")
        elif self.path.is_file():
            self.source = "cache"
            issues = yaml.safe_load(self.path.read_text(encoding="utf-8")) or []
        self.items = [x for x in issues or [] if isinstance(x, dict)]

    @staticmethod
    def fetch() -> list[dict] | None:
        try:
            out = subprocess.run(
                ["gh", "issue", "list", "--label", "idea", "--state", "all", "--limit", "500"]
                + ["--json", "number,title,state,url,body"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                cwd=EXPERIMENTS,
            ).stdout
            return [x for x in json.loads(out) if isinstance(x, dict)]
        except (OSError, subprocess.SubprocessError, ValueError):
            return None


class Reference:
    """Topics, their entries and the ideas."""

    def __init__(self):
        folder = EXPERIMENTS / REFERENCE
        self.topics = [Topic(d) for d in sorted(folder.iterdir()) if d.is_dir()] if folder.is_dir() else []
        self.topics = [t for t in self.topics if t.entries or t.path.is_file()]
        self.entries = [e for t in self.topics for e in t.entries]
        self.ideas = Ideas(folder) if self.topics else None

    def __bool__(self) -> bool:
        return bool(self.topics)


def entry_link(e: RefEntry, text: str = "") -> str:
    return (
        f'<a class="term" href="#{e.page_id}" data-tip="{tip(e.title, e.summary or e.kind)}">{esc(text or e.title)}</a>'
    )


def link_entries(html_: str, ref: Reference, skip: str = "") -> str:
    """The first mention of an entry's title on a page becomes a link to the entry
    with its summary as a hover card."""
    by_title = {e.title.lower(): e for e in ref.entries if e.key != skip and len(e.title) > 3}
    if not by_title:
        return html_
    pat = re.compile(
        r"(?<![\w@\-])(" + "|".join(re.escape(t) for t in sorted(by_title, key=len, reverse=True)) + r")(?![\w@\-])",
        re.I,
    )
    seen: set[str] = set()

    def link(m: re.Match) -> str:
        e = by_title[m.group(1).lower()]
        if e.key in seen:
            return m.group(1)
        seen.add(e.key)
        return entry_link(e, m.group(1))

    return sub_prose(html_, lambda text: pat.sub(link, text))


REF_CSS = """
/* ---- reference: the topic and entry pages style themselves (templates/topic.css, entry.css) ---- */
.small{font-size:var(--t-sm)}
.lgd .cm{color:var(--muted);font-style:italic;margin-left:18px}.lgd .cm code{font-style:normal;font-size:11px}
p.muted{margin:0 0 4px}
"""
REF_FONT = "&family=STIX+Two+Text:ital,wght@0,400;0,600;1,400"


# the figure view: one dialog the script moves a figure's drawing and caption into
FIGURE_VIEW = (
    '<dialog class="figv" id="figview" role="dialog" aria-modal="true" aria-labelledby="figview-cap">'
    '<button class="figx" type="button" aria-label="Close the full-screen figure" title="close (Esc)">×</button>'
    '<figure class="figvf"><div class="figvp"></div><figcaption id="figview-cap"></figcaption></figure></dialog>'
)


CSS = """
:root{--bg:#e6e9ee;--panel:#f7f8fa;--line:#c9cfd8;--fg:#1f252f;--muted:#5a6472;--accent:#3350c4;--accent-soft:#dbe1f6;--soft:#e8ebf0;--ok:#17643a;--ok-soft:#d7ecdf;--bad:#9c2820;--bad-soft:#f3dad7;--run:#7f4b00;--run-soft:#f2e2c3;--plate:#f0f2f5;--t-h1:32px;--t-h2:20px;--t-lead:17px;--t-h3:16px;--t-body:15px;--t-sm:13.5px;--t-mono:12px;--t-xs:11px;--t-micro:9px;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace;--sans:"Instrument Sans",system-ui,sans-serif}
:root[data-theme=dark]{--bg:#1c1c20;--panel:#292a30;--line:#464852;--fg:#dcdfe5;--muted:#9a9fac;--accent:#9aa8ff;--accent-soft:#2c3358;--soft:#35363e;--ok:#5ee39a;--ok-soft:#1b3b2c;--bad:#f98a8a;--bad-soft:#452524;--run:#f2c55c;--run-soft:#413513;--plate:#25262c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1c1c20;--panel:#292a30;--line:#464852;--fg:#dcdfe5;--muted:#9a9fac;--accent:#9aa8ff;--accent-soft:#2c3358;--soft:#35363e;--ok:#5ee39a;--ok-soft:#1b3b2c;--bad:#f98a8a;--bad-soft:#452524;--run:#f2c55c;--run-soft:#413513;--plate:#25262c}}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:var(--t-body);line-height:1.55}
a{color:var(--accent);text-decoration:none}a:hover{color:color-mix(in srgb,var(--accent),var(--fg) 25%)}
.app{--rail:345px;display:grid;grid-template-rows:52px minmax(0,1fr);grid-template-columns:var(--rail) minmax(0,1fr);height:100vh}
.top{grid-column:1/-1;display:flex;align-items:center;gap:16px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.top .lhs{display:flex;align-items:center;gap:16px;flex:none;width:calc(var(--rail) - 21px);min-width:0;padding-right:14px}.brand{font-weight:700;color:var(--fg);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}a.brand:hover{color:var(--accent)}.top .rpt{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted);flex:none}.gh{width:18px;height:18px;fill:currentColor;flex:none}.crumb{display:flex;gap:8px;color:var(--muted);font-size:var(--t-sm);min-width:0;overflow:hidden;white-space:nowrap}.crumb>*{flex:none}.crumb b{flex:0 1 auto;overflow:hidden;text-overflow:ellipsis}.crumb b{color:var(--fg);font-weight:600}.crumb a{color:inherit;text-decoration:none}.crumb a:hover{color:var(--accent)}
.foot{max-width:1120px;margin:48px auto 0;padding-top:16px;border-top:1px solid var(--line);font-size:var(--t-sm);color:var(--muted);display:flex;flex-wrap:wrap;gap:6px;align-items:center}.foot a{display:inline-flex;align-items:center;gap:5px}.foot .gh{width:14px;height:14px}
.legend{margin-left:auto}.nav,.theme,.legend{font:inherit;font-size:var(--t-sm);color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}
.side{--vp-ink:#fff;--spec:#a3215f;border-right:1px solid var(--line);overflow:auto;display:flex;flex-direction:column;font-family:var(--sans);color:var(--fg);background:linear-gradient(to right,var(--plate) 0 64px,var(--line) 64px 65px,var(--bg) 65px)}
:root[data-theme=dark] .side{--vp-ink:#1c1c20;--spec:#ff8cc6}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .side{--vp-ink:#1c1c20;--spec:#ff8cc6}}
.chev{color:var(--muted);font-size:12px;font-weight:400;line-height:1.5;letter-spacing:0;flex:none}
details>summary .chev::before,.hd.shut>.chev::before{content:"▸"}details[open]>summary .chev::before,.hd>.chev::before{content:"▾"}
.main{overflow:auto;padding:36px 56px 64px;background:var(--bg)}.page{max-width:1080px;margin:0 auto}.page{display:none}.page.on{display:block}
.kicker{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted);margin:0}
h1{font-size:var(--t-h1);font-weight:700;letter-spacing:-.02em;margin:8px 0 10px}
p.lede{font-size:var(--t-lead);color:var(--muted);margin:0 0 8px}ul.lede{font-size:var(--t-lead);margin:0 0 24px;padding-left:20px}ul.lede li{margin:0 0 8px}ul.lede a{color:inherit;font-weight:600;text-decoration:none}ul.lede a:hover{color:var(--accent)}ul.lede .s{display:block;font-size:var(--t-sm);color:var(--muted)}
.badge{display:inline-block;font-size:var(--t-xs);font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:5px;vertical-align:middle}
.b-confirmed,.b-better,.b-adopted,.b-answered{background:var(--ok-soft);color:var(--ok)}.b-refuted,.b-worse,.b-rejected,.b-unanswered{background:var(--bad-soft);color:var(--bad)}.b-inconclusive,.b-equivalent,.b-undecided,.b-partial,.b-draft,.b-abandoned,.b-dropped{background:var(--soft);color:var(--muted)}.b-running{background:var(--run-soft);color:var(--run)}.b-superseded{background:var(--soft);color:var(--muted);text-decoration:line-through}
table.clauses .role{display:inline-block;margin-left:6px;font-size:var(--t-xs);text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}table.clauses tr.diagnostic td,table.clauses tr.sup td{color:var(--muted)}table.clauses tr.diagnostic td:nth-child(2),table.clauses tr.sup td:nth-child(2){font-weight:500}
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
.math{font-size:1.05em}.math.block{display:block;margin:14px 0;padding:6px 0;line-height:1.2;overflow-x:auto;overflow-y:hidden;font-size:1.15em}math{font-family:"STIX Two Math","Latin Modern Math","Cambria Math","Noto Sans Math",math,var(--sans)}math .gk{font-style:italic}
.res>.read,.res>figure{min-width:0}.res>.read{grid-column:1;grid-row:2;font-size:var(--t-sm);line-height:1.5}.res .read p{margin:0 0 8px}.res>figure{grid-column:2;grid-row:2;margin:0}.res>.src{grid-column:1/-1;margin:4px 0 0}.src{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted)}.src p{display:grid;grid-template-columns:58px 1fr;gap:0 6px;margin:0 0 3px}.src b{font-size:var(--t-xs);font-weight:600;letter-spacing:.06em;text-transform:uppercase;line-height:inherit;padding-top:1px}.src a code{color:var(--accent)}
.src code.run::before,.src code.commit::before,.src code.file::before{font-size:var(--t-micro);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:5px}.src code.run::before{content:"run"}.src code.commit::before{content:"commit"}.src code.file::before{content:"file"}
.res>details.support{grid-column:1/-1;border:0;padding:0;margin:2px 0 0}.res .support>summary{color:var(--muted);font-size:var(--t-mono)}
.res.sub{margin:10px 0 0;padding:14px 18px;background:transparent;border:0;border-left:2px solid var(--line);border-radius:0}.res.sub h3{font-size:var(--t-h3)}.res.minor{padding:12px 20px}.res.minor h3{font-size:var(--t-body)}
.res.wide{grid-template-columns:minmax(0,1fr)}.res.wide>.read{grid-column:1;font-size:var(--t-sm)}.res.wide>figure{grid-column:1;grid-row:3}
figure{margin-inline:0}figure svg,figure img{width:100%;height:auto;display:block}.res>figure .fig{cursor:zoom-in}.res.wide>figure .fig{cursor:zoom-out}
figcaption{font-size:var(--t-mono);color:var(--muted);margin-top:8px;line-height:1.5}figcaption .lbl{color:var(--fg);font-weight:600}
.scroll{overflow-x:auto;max-width:100%;margin:6px 0 10px;scrollbar-width:thin;scrollbar-color:var(--line) transparent;padding-bottom:2px}.scroll.tall{overflow-y:auto;max-height:calc(20 * (1.45 * var(--t-sm) + 13px) + 2.6em)}.scroll.tall table.data thead th{position:sticky;top:0;background:var(--panel);z-index:1}.scroll.tall table.data th:first-child{z-index:2}
.scroll::-webkit-scrollbar{height:8px;width:8px}.scroll::-webkit-scrollbar-track{background:transparent}.scroll::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}.sh{display:none;font-family:var(--mono);font-size:var(--t-xs);color:var(--muted);text-align:right;margin:6px 0 8px}.sh.on{display:block}figure .scroll{margin:0}table.data{border-collapse:collapse;width:100%;font-size:var(--t-sm);line-height:1.45;font-variant-numeric:tabular-nums}table.data th{text-align:left;font-size:var(--t-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:bottom}table.data td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}table.data td.id{font-family:var(--mono);font-size:var(--t-mono);white-space:nowrap}table.data td:first-child,table.data th:first-child{position:sticky;left:0;background:inherit;font-weight:500}table.data.wide td:first-child,table.data.wide th:first-child{border-right:1px solid var(--line)}table.data tr{background:var(--panel)}.res.sub table.data tr,.prose table.data tr,.v table.data tr{background:var(--bg)}table.data th.n{text-align:right}table.data td.n{text-align:right;font-family:var(--mono);font-size:var(--t-mono);white-space:nowrap}table.data code{overflow-wrap:anywhere}
.prose h3,.v h3{font-size:var(--t-body);font-weight:600;margin:36px 0 10px}.prose h4,.v h4{font-size:var(--t-mono);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:24px 0 8px}.prose>:first-child{margin-top:0}.prose p,.prose ul,.prose ol{margin:0 0 10px}
.missing{color:var(--bad);font-family:var(--mono);font-size:var(--t-mono)}
.steps{list-style:none;padding:0;margin:8px 0 12px}.step{display:grid;grid-template-columns:38px minmax(0,1fr);gap:6px 12px;align-items:start;margin:0 0 18px}
.step .t{grid-column:2;font-size:var(--t-sm);font-weight:600;line-height:1.5;padding-top:3px}.step pre{grid-column:2;margin:0}.step .t code{font-size:.85em}
pre .c{color:#8a93a5}
a.ref{color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)}a.term{color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted)}a.term:hover{color:var(--accent);border-bottom-color:var(--accent)}
.tip{position:fixed;z-index:9;max-width:400px;background:var(--fg);color:var(--bg);font-size:var(--t-mono);line-height:1.45;padding:8px 11px;border-radius:8px;white-space:pre-line;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.25)}.tip b{display:block;margin-bottom:2px}.tip code{background:color-mix(in srgb,var(--bg) 18%,transparent);color:inherit}
code{font-family:var(--mono);font-size:.88em;background:var(--soft);padding:1px 5px;border-radius:4px}.fp.summary code{background:color-mix(in srgb,var(--fg) 10%,transparent)}pre{background:#16181d;color:#e7e9ee;padding:14px 16px;border-radius:10px;font-family:var(--mono);font-size:var(--t-mono);line-height:1.55;overflow:auto}pre code{background:none;padding:0;color:inherit}
.rel{display:flex;flex-direction:column;gap:10px}.rel a{display:flex;flex-direction:column;gap:3px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;color:var(--fg)}.rel a:hover{border-color:var(--accent)}.rel .t{font-weight:600}.rel .m{color:var(--muted)}
details{border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin:8px 0}summary{cursor:pointer;font-family:var(--mono);font-size:var(--t-sm)}details .v{margin:8px 0 0}
.right{border-left:1px solid var(--line);padding:24px 16px;font-size:var(--t-sm);display:flex;flex-direction:column;min-height:0}.right .toc{flex:1;overflow:auto}.right .keys{border-top:1px solid var(--line);margin-top:16px;padding-top:14px}.keys div{display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:var(--muted);padding:3px 10px}kbd{font-family:var(--mono);font-size:var(--t-xs);line-height:1.6;min-width:18px;text-align:center;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--bg);color:var(--fg)}.right .h{font-size:var(--t-xs);font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;padding-left:10px}
.right a{display:block;color:var(--muted);padding:5px 10px;border-left:2px solid transparent;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.right a:hover{color:var(--fg)}.right a.sub{padding-left:22px;font-size:var(--t-mono)}.right a.on{color:var(--accent);border-left-color:var(--accent)}
table.data td.mono{font-family:var(--mono);font-size:var(--t-mono)}tr.planned td{color:var(--muted)}
/* ---- layout D, type E: one column page, sticky section bar, prose in Plex, structure in Instrument ---- */
:root{--prose:"IBM Plex Sans",system-ui,sans-serif;--fig:#fbfcfd;--t-prose:15px}
body{font-family:var(--prose)}h1,h2,h3,.side,.top,.bar,.badge,.tag,.wt,.lbl,.kicker,.meta,th,figcaption,.keys,.hint,summary{font-family:var(--sans)}
.right{display:none}
.main{--main-top:28px;padding:var(--main-top) 40px 64px}.page,.foot{max-width:1120px}
h1{font-size:26px;margin:6px 0 6px}h2{margin:52px 0 14px}.prose p,.v p{margin:0 0 12px}.theme,.nav,.legend{white-space:nowrap}h2{margin:44px 0 12px}
.lbl{display:block;font-size:var(--t-xs);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.muted{color:var(--muted)}figcaption .lbl{display:inline;margin:0;font-size:inherit;letter-spacing:0;text-transform:none;color:var(--fg)}
.step .n{grid-row:1/3;align-self:start;justify-self:start;min-width:38px;box-sizing:border-box;text-align:center;background:var(--accent);color:var(--bg);font-family:var(--mono);font-size:var(--t-mono);font-weight:500;padding:1px 7px;border-radius:5px;white-space:nowrap;margin-top:3px}.step.env .n{background:var(--soft);color:var(--muted)}.step .t{font-family:var(--sans)}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:6px 0;font-size:var(--t-sm);color:var(--muted);margin:0 0 14px}.meta>span+span::before{content:"·";margin:0 12px;color:var(--line)}
.hyp{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px 16px;align-items:start;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:0 0 12px}
.hyp .lab{display:flex;flex-direction:column;gap:6px;align-items:flex-start;padding-top:4px}.hyp .lab .lbl{margin:0}.hyp .out{padding-top:4px}.hyp .v{font-size:var(--t-prose);line-height:1.6}.split{gap:24px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px}
.needs{display:grid;grid-template-columns:auto repeat(4,minmax(0,1fr));border:1px solid var(--line);border-radius:10px;background:var(--panel);font-size:var(--t-sm);margin:0 0 18px;overflow:hidden}
.needs>div{padding:8px 12px;border-right:1px solid var(--line);min-width:0}.needs>div:last-child{border-right:0}.needs .hot{background:var(--run-soft)}.needs .hot .lbl{color:var(--run)}
.pred{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:0 0 18px;font-size:var(--t-prose);line-height:1.6}.pred p{margin:0 0 6px}
table.clauses .kw{font-size:var(--t-xs);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-right:4px}table.clauses th .kind{font-weight:400;color:var(--muted);text-transform:none;margin-left:6px}table.clauses td.result{font-weight:500}table.clauses td.result .badge{margin-right:4px}table.clauses td.result .reading+.reading{margin-top:8px;padding-top:8px;border-top:1px solid var(--rule)}table.clauses td.muted{color:var(--muted)}td.hot{background:var(--run-soft)}
/* sticky clamps at the scroller content box: undo .main padding to sit flush under .top */
.bar{position:sticky;top:calc(-1 * var(--main-top));z-index:5;display:flex;flex-wrap:wrap;gap:2px;border-bottom:1px solid var(--line);background:var(--bg);margin:22px 0 0;font-size:var(--t-sm)}
.bar a{padding:8px 12px;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px}.bar a i{font-style:normal;font-family:var(--mono);font-size:var(--t-xs)}.bar a:hover{color:var(--fg)}.bar a.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.bar .hint{margin-left:auto;padding:8px 4px;font-family:var(--mono);font-size:var(--t-xs);color:var(--muted)}.bar .filter{margin:4px 0 4px auto;width:220px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--fg);padding:4px 10px;font:inherit;font-size:var(--t-sm)}.bar .filter:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}.hint kbd{margin:0 1px}
.rows{display:grid;grid-template-columns:130px minmax(0,1fr);gap:16px 16px;font-size:var(--t-prose);line-height:1.6}.rows>b{font-size:var(--t-xs);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding-top:5px;font-family:var(--sans)}.rows>div>p:first-child{margin-top:0}.rows>div>:last-child{margin-bottom:0}.rows p,.rows ul,.rows ol{margin:0 0 8px}
.res,.res.wide{display:block;padding:16px 22px}.res>h3,.res>summary>h3{display:flex;gap:10px;align-items:center;font-size:var(--t-lead);margin:0 0 12px}.res>summary>h3{margin:0}
.res .fp.summary{font-size:var(--t-prose);line-height:1.6;background:var(--soft);padding:10px 12px;border-radius:8px;margin:0 0 12px}
.res .fp{font-size:var(--t-prose);line-height:1.6}.res .fp+.fp{margin-top:16px}.res .fp p{margin:0 0 10px}.res .fp>:last-child{margin-bottom:0}.res .fp.summary+.read{margin-top:16px}.res>.read,.split>.read{display:block;grid-column:auto;grid-row:auto;font-size:var(--t-prose);line-height:1.6}
.split{display:grid;grid-template-columns:minmax(0,9fr) minmax(0,11fr);gap:20px;align-items:start}.split>figure{margin:0;grid-column:auto;grid-row:auto}
.read.two{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px 28px;margin-top:16px}.read.two .fp{min-width:0}
.res>figure{grid-column:auto;grid-row:auto;margin:0 0 6px}.res>.src,details.res>.body>.src{margin:22px 0 0}
.fig{background:var(--fig);border:1px solid var(--line);border-radius:4px;padding:10px 12px;cursor:auto}.fig svg,.fig img{width:100%;height:auto;display:block}
details.res,details.dec,details.met{border:1px solid var(--line);border-radius:12px;padding:0;margin:8px 0;background:var(--panel);scroll-margin-top:60px}
details.res>summary,details.dec>summary,details.met>summary{list-style:none;cursor:pointer;padding:12px 22px;display:flex;gap:10px;align-items:baseline}
details.res>summary::-webkit-details-marker,details.dec>summary::-webkit-details-marker,details.met>summary::-webkit-details-marker{display:none}
details.res>summary::before,details.dec>summary::before,details.met>summary::before{content:"▸";color:var(--muted);font-size:18px;line-height:1;flex:none}details[open]>summary::before{content:"▾"}
details.res>summary h3,details.dec>summary h3,details.met>summary h3{flex:1;min-width:0;font-size:var(--t-body);margin:0;display:flex;gap:10px;align-items:center}
details.res>.body,details.dec>.body,details.met>.body{padding:0 22px 16px}
.res.key>details.res{border:0;border-top:1px dashed var(--line);border-radius:0;background:transparent;margin:12px 0 0}.res.key>details.res>summary{padding:10px 0 0}.res.key>details.res>summary h3{font-size:var(--t-body)}.res.key>details.res>.body{padding:10px 0 0}.res.key>details.res .fp.summary{font-size:var(--t-sm)}.res.key>details.res .fp{font-size:var(--t-sm)}details.dec .rows,details.met .rows{margin-top:4px}details.met .rows>div code,details.dec .rows>div code{font-size:.85em}
details.met .io{display:grid;grid-template-columns:1fr 1fr;gap:12px 28px;margin:0 0 12px}details.met .fp{font-size:var(--t-sm);line-height:1.55}details.met .fp.how{font-size:var(--t-prose);line-height:1.6}details.met .src{margin:22px 22px 16px}
details.more{border:0;padding:0;margin:10px 0 0}details.more>summary{color:var(--muted);font-size:var(--t-mono)}.assets{display:flex;flex-wrap:wrap;gap:16px;margin-top:10px}.assets>*{min-width:0;max-width:100%}.assets>figure:has(.scroll){flex:1 1 100%}.assets figure{margin:0;flex:1 1 100%}.assets figure:has(.fig){flex:1 1 calc(50% - 8px)}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:12px 0 0}.pair figure{margin:0}
.ds{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:12px 0;scroll-margin-top:60px}.ds>h3{margin:0 0 12px}.ds .read{font-size:var(--t-prose);line-height:1.6}.ds .read p{margin:0 0 8px}.ds figure{margin:14px 0 0}.ds .split>figure{margin:0}
.rel a .t{font-family:var(--sans)}
@media(max-width:1100px){.main{--main-top:24px;padding:var(--main-top) 28px 56px}}
@media(min-width:801px){.top .nav{display:none}}
@media(max-width:800px){.app{grid-template-columns:minmax(0,1fr);grid-template-rows:52px minmax(0,1fr)}.side{position:fixed;top:52px;left:0;bottom:0;width:min(345px,90vw);z-index:8;box-shadow:0 0 24px rgba(0,0,0,.25)}.app.nonav .side{display:none}
.main{--main-top:16px;padding:var(--main-top) 16px 48px}h1{font-size:22px}.top{padding:0 12px;gap:10px}.top .lhs{width:auto;gap:10px;flex:0 1 auto;min-width:0;padding-right:0}.top .legend,.top .theme{flex:none}.top .legend{margin-left:auto}.crumb{display:none}.bar .hint{display:none}
.needs{grid-template-columns:1fr 1fr}.needs>div{border-bottom:1px solid var(--line)}.split,.read.two,.pair,.rows,details.met .io{grid-template-columns:minmax(0,1fr)}.rows{gap:4px 0}.rows>b{padding-top:8px}.res{padding:14px}}
/* ---- responsive: tokens wrap, blocks follow their column, type scales on wide screens ---- */
/* 1. long tokens wrap instead of pushing the page wider */
code{overflow-wrap:anywhere}
.res>h3,details.met>summary h3,details.res>summary h3{flex-wrap:wrap}
.wt{min-width:0;text-align:right;overflow-wrap:anywhere}
details.dec>summary h3{align-items:baseline}details.dec>summary h3 .wt{flex:none;white-space:nowrap}
/* 2. wide screens: content hugs the sidebar, grows a little, type scales */
.page{margin:0}.foot{margin-left:0}
@media(min-width:1700px){
 :root{--t-h1:36px;--t-h2:22px;--t-lead:19px;--t-h3:17px;--t-body:16.5px;--t-sm:15px;--t-mono:13px;--t-xs:12px;--t-prose:16.5px}
 .app{--rail:365px}
 .main{--main-top:40px;padding:var(--main-top) 72px 80px}.page,.foot{max-width:1320px}
}
@media(min-width:2300px){
 :root{--t-h1:42px;--t-h2:25px;--t-lead:21px;--t-h3:19px;--t-body:18.5px;--t-sm:17px;--t-mono:14.5px;--t-xs:13px;--t-prose:18.5px}
 .app{--rail:400px;grid-template-rows:60px minmax(0,1fr)}
 .main{--main-top:48px;padding:var(--main-top) 96px 96px}.page,.foot{max-width:1540px}
}
/* 3. laptops: a narrower sidebar, and blocks stack by the width they actually get */
@media(max-width:1200px) and (min-width:801px){.app{--rail:300px}}
.main{container-type:inline-size}
@container (max-width:860px){
 .res{grid-template-columns:minmax(0,1fr)}.res>.read,.res>figure{grid-column:1;grid-row:auto}
 .read.two{grid-template-columns:minmax(0,1fr)}
 details.met>summary h3 .wt{flex-basis:100%;margin-left:0;text-align:left}
 .needs{grid-template-columns:repeat(2,minmax(0,1fr))}.needs>div:first-child{grid-column:1/-1;border-right:0;border-bottom:1px solid var(--line);padding:6px 12px}
 .needs>div{border-bottom:1px solid var(--line)}.needs>div:nth-child(2n+1){border-right:0}
 /* the clause table becomes one card per clause: tag and claim as the head, the three columns as labelled rows */
 .scroll:has(>table.clauses){overflow:visible;-webkit-mask-image:none;mask-image:none}
 table.clauses,table.clauses tbody{display:block}table.clauses thead{display:none}
 table.clauses tr{display:grid;grid-template-columns:auto minmax(0,1fr);gap:2px 12px;align-items:baseline;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px 12px;margin:0 0 10px;scroll-margin-top:20px}
 table.clauses td,table.data.wide.clauses td:first-child{display:block;border:0;padding:0;position:static;background:none;font-weight:inherit}
 table.clauses td:nth-child(2){font-weight:600}
 table.clauses td:nth-child(n+3){grid-column:1/-1}table.clauses td:nth-child(n+3):empty{display:none}
 table.clauses td:nth-child(n+3)::before{content:attr(data-l);display:block;font-size:var(--t-xs);font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:8px 0 2px}
 table.clauses td.result{font-weight:500}
 /* an idea's own table: one card per entry — name and relation on the head line, touches / change / effect under it */
 table.data.wide.inner td:first-child{grid-column:1/-1;font-weight:600;font-family:var(--sans);display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
}
/* 4. phones: the section bar is one scrollable row, boxes stack to one column */
@media(max-width:800px){
 .bar{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;-webkit-mask-image:linear-gradient(to right,#000 calc(100% - 32px),transparent);mask-image:linear-gradient(to right,#000 calc(100% - 32px),transparent)}
 .bar::-webkit-scrollbar{display:none}.bar a{white-space:nowrap;flex:none}.bar .filter{flex:none;width:150px;margin-left:8px}
 .hyp{grid-template-columns:minmax(0,1fr);gap:8px}.hyp .lab,.hyp .out{padding:0}
 /* finding and method heads flow as text: tag, title and weight on one line, a method's code path under its title */
 .res>h3,details.res>summary h3,details.met>summary h3{display:block;line-height:1.45}.res>h3 .tag,details.res>summary h3 .tag,details.met>summary h3 .tag{margin-right:8px;vertical-align:1px}
 .res>h3 .wt,details.res>summary h3 .wt{display:inline;margin-left:8px;text-align:left;font-size:var(--t-xs);white-space:nowrap}details.met>summary h3 .wt{display:block;margin:2px 0 0;text-align:left;font-size:var(--t-xs)}
}
@media(max-width:560px){
 .needs{grid-template-columns:minmax(0,1fr)}.needs>div{border-right:0}
 .meta>span+span::before{margin:0 8px}
}
/* the legend: a dialog over every page; a full-screen sheet on a phone */
.lgd{padding:0;border:1px solid var(--line);border-radius:12px;background:var(--panel);color:var(--fg);width:min(780px,calc(100vw - 48px));max-width:none;max-height:min(86vh,940px);box-shadow:0 18px 48px rgba(0,0,0,.28)}
.lgd[open]{display:flex;flex-direction:column}
.lgd::backdrop{background:rgba(12,14,20,.5)}
.lgd .lgh{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px 12px 22px;border-bottom:1px solid var(--line);flex:none}
.lgd .lgh h2{margin:0;font-size:var(--t-h2)}
.lgd .lgx{font:inherit;font-size:20px;line-height:1;color:var(--muted);background:none;border:1px solid transparent;border-radius:8px;padding:4px 10px;cursor:pointer}
.lgd .lgx:hover,.lgd .lgx:focus-visible{color:var(--fg);border-color:var(--line)}
.lgd .page.lgb{display:block;max-width:none;margin:0;overflow:auto;overscroll-behavior:contain;padding:4px 22px 24px;background:var(--panel)}
.lgd h3{font-size:var(--t-h3);margin:22px 0 10px}
.lgd dl{display:grid;grid-template-columns:minmax(0,250px) minmax(0,1fr);gap:10px 18px;align-items:center;margin:12px 0 0}
.lgd .lgr{display:contents}
.lgd dt{display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0}
.lgd dd{margin:0;font-size:var(--t-sm);color:var(--muted)}
.lgd .grp{font-size:var(--t-xs);color:var(--muted);font-family:var(--mono);white-space:nowrap}
.lgd .lgw{width:100%}.lgd .lgw td{font-size:var(--t-sm);vertical-align:middle}.lgd .lgw .lgfam th{text-transform:uppercase;letter-spacing:.06em;font-size:var(--t-xs);color:var(--muted);padding-top:12px}
.lgd .lgs{display:block;width:120px}.lgd .sh{margin:0}.lgd .rneeds{margin:0;font-size:var(--t-sm)}.lgd .runs{margin:0}.lgd .keynum{font-size:22px}.lgd .lgn{margin:8px 0 0}
@media(max-width:600px){.lgd{width:100vw;height:100dvh;max-height:100dvh;margin:0;border:0;border-radius:0}.lgd .page.lgb{padding:4px 16px 24px}.lgd dl{grid-template-columns:minmax(0,1fr);gap:4px}.lgd dd{margin-bottom:10px}.lgd .lgw,.lgd .lgw tbody{display:block}.lgd .lgw thead{display:none}.lgd .lgw tr{display:grid;grid-template-columns:32px max-content max-content;gap:0 10px;align-items:center}.lgd .page.lgb .lgw td,.lgd .page.lgb .lgw th{display:block;border:0;padding:6px 0}.lgd .lgw tr{border-bottom:1px solid var(--line);padding:4px 0}.lgd .lgw td:last-child,.lgd .lgw .lgfam th{grid-column:1/-1}}
.lgd .nd{width:7px;height:7px;border-radius:50%;background:var(--spec);flex:none;margin-top:6px}
@media print{.lgd{display:none!important}}
/* the figure view: a figure scaled to the screen over a darkened page */
:root{--scrim:rgba(16,18,24,.82)}:root[data-theme=dark]{--scrim:rgba(3,3,5,.94)}@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--scrim:rgba(3,3,5,.94)}}
:root.js .fig.zoom{cursor:zoom-in;position:relative}.fzb{display:none}:root.js .fig.zoom>.fzb{position:absolute;top:8px;right:8px;display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;padding:0;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--muted);cursor:zoom-in}:root.js .fig.zoom>.fzb:hover,:root.js .fig.zoom>.fzb:focus-visible{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}.fzb svg{width:16px;height:16px;fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}:root.js .figv .fzb{display:none}@media print{:root.js .fig.zoom>.fzb{display:none}}:root.js .fig.zoom:focus-visible{outline:2px solid var(--accent);outline-offset:2px}@media(hover:hover){:root.js .fig.zoom:hover{border-color:var(--accent)}}
.figlock,.figlock body,.figlock .main{overflow:hidden}
.figv{position:fixed;inset:0;width:100%;height:100%;max-width:none;max-height:none;margin:0;padding:0;border:0;background:var(--scrim);color:var(--fg);overscroll-behavior:contain}
.figv[open]{display:flex;align-items:center;justify-content:center}
.figv::backdrop{background:transparent}
.figv .figx{position:absolute;top:12px;right:12px;font:inherit;font-size:22px;line-height:1;color:#f2f4f8;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.3);border-radius:8px;padding:6px 12px;cursor:pointer}
.figv .figx:hover,.figv .figx:focus-visible{background:rgba(255,255,255,.22);outline:none;border-color:#f2f4f8}
.figv .figvf{display:flex;flex-direction:column;gap:10px;margin:0;max-width:95vw;max-height:calc(100dvh - 72px);overflow:auto;background:var(--panel);border-radius:10px;padding:12px;box-shadow:0 18px 48px rgba(0,0,0,.4)}
.figv .figvp{background:var(--fig);border-radius:8px;padding:12px}
.figv .figvp>svg,.figv .figvp>img{display:block;max-width:none;margin:0 auto}
.figv figcaption{order:0;width:0;min-width:100%;margin:0;font-family:var(--prose);font-size:14px;line-height:1.55;color:var(--fg)}
@media print{.figv{display:none!important}}
@media print{.scroll.tall{max-height:none;overflow:visible}.sh{display:none}:root{--bg:#fff;--panel:#fff;--plate:#fff}.app{display:block;height:auto}.top,.side,.right,.foot{display:none}.main{overflow:visible;padding:0;background:#fff}.fig{border:0;padding:0;background:none}pre{background:#f4f5f7;color:#16181d;border:1px solid #d8dce3}pre .c{color:#6b7280}.page{display:block!important;max-width:none;break-before:page}h2{break-after:avoid}.res,.dec,.met,details,.rel a{break-inside:avoid}a{color:inherit}body{font-size:11pt}}
"""

JS = """
const pages=[...document.querySelectorAll('.main>.page')],crumb=document.querySelector('.crumb'),main=document.querySelector('.main');
const tabs=[...document.querySelectorAll('.rail .rb')],panels=[...document.querySelectorAll('.rail .rpanel')],links=[...document.querySelectorAll('.rail .rpanel a')];
const heads=[...document.querySelectorAll('.rail .hd[data-group]')],exps=[...document.querySelectorAll('.rail [data-exp]')],TAB_ORDER=['research','reference','terms','overview'];
const pageOf=a=>(a.getAttribute('href')||'#').slice(1).split('/')[0],el=(t,x,h)=>Object.assign(document.createElement(t),{textContent:x},h?{href:h}:{});
let panel=panels[0];
function showTab(tab){panel=panels.find(p=>p.dataset.tab===tab)||panels[0];if(!panel)return;
 panels.forEach(p=>p.classList.toggle('off',p!==panel));tabs.forEach(b=>b.classList.toggle('on',b.dataset.tab===panel.dataset.tab));filter(panel);}
// a tab opens its landing page; a tab whose page the report lacks only shows its list
tabs.forEach(b=>b.addEventListener('click',()=>{const id=b.dataset.landing;if(!pages.some(p=>p.id===id))showTab(b.dataset.tab);
 else if(location.hash==='#'+id)route();else location.hash=id;}));
function setGroup(h,open){h.classList.toggle('shut',!open);
 document.querySelectorAll('.rail [data-in="'+h.dataset.group+'"]').forEach(i=>i.classList.toggle('shut',!open));}
heads.forEach(h=>{setGroup(h,!h.classList.contains('shut'));h.addEventListener('click',e=>{if(h.tagName==='A'&&!e.target.closest('.hc'))return;e.preventDefault();setGroup(h,h.classList.contains('shut'));});});
function filter(p){const input=p.querySelector('.rfilter input'),q=input?input.value.trim().toLowerCase():'';let head=null,hits=0;
 const close=()=>{if(head){head.classList.toggle('off',!!q&&!hits&&!head.textContent.toLowerCase().includes(q));head.classList.toggle('hit',!!q&&hits>0);}};
 for(const x of p.querySelectorAll('.hd,.item')){if(x.classList.contains('hd')){close();head=x;hits=0;continue;}
  const hit=!q||(x.textContent+' '+(x.title||'')).toLowerCase().includes(q);x.classList.toggle('off',!hit);x.classList.toggle('hit',!!q&&hit);if(q&&hit)hits++;}
 close();}
for(const p of panels){const input=p.querySelector('.rfilter input');input.addEventListener('input',()=>filter(p));
 input.addEventListener('keydown',e=>{if(e.key==='Escape'){input.value='';filter(p);input.blur();e.stopPropagation();}});}
function groups(page){for(const h of heads){const ins=[...document.querySelectorAll('.rail [data-in="'+h.dataset.group+'"]')];
  const here=pageOf(h)===page.id||ins.some(i=>pageOf(i)===page.id);
  if(h.closest('.rpanel[data-tab="reference"]'))setGroup(h,here);else if(here)setGroup(h,true);}}
function activate(page,hash){links.forEach(a=>a.classList.remove('on'));
 const mine=p=>[...p.querySelectorAll('a')].filter(a=>pageOf(a)===page.id);
 showTab(TAB_ORDER.find(t=>{const p=panels.find(x=>x.dataset.tab===t);return p&&mine(p).length;})||'overview');
 const own=panel?mine(panel):[],a=own.find(a=>a.getAttribute('href')==='#'+hash)||own[0],lead=own[0];
 if(a){a.classList.add('on');a.scrollIntoView?.({block:'nearest'});}
 let head=null;if(lead&&!lead.classList.contains('hd'))for(let x=lead.previousElementSibling;x;x=x.previousElementSibling)if(x.classList.contains('hd')){head=x;break;}
 const label=lead&&lead.querySelector('.lab,.hl'),name=label?label.textContent:page.dataset.title,ttl=panel?panel.querySelector('.ttl').textContent:'';
 // a tab titled as its page (Glossary) reads once, not Glossary › Glossary
 const parts=page.id==='overview'||!panel||ttl===name?[]:[el('span',ttl),el('span','\u203a')];
 if(head){const t=head.querySelector('.hl').textContent;parts.push(head.tagName==='A'?el('a',t,head.getAttribute('href')):el('span',t),el('span','\u203a'));}
 crumb.replaceChildren(...parts,el('b',name));}
document.documentElement.classList.add('js');
function showX(tab){for(const t of tab.parentElement.querySelectorAll('.xtab')){const on=t===tab;t.setAttribute('aria-selected',on);t.tabIndex=on?0:-1;document.getElementById(t.getAttribute('aria-controls')).classList.toggle('on',on);}}
document.addEventListener('click',e=>{const t=e.target.closest&&e.target.closest('.xtab');if(t&&!t.closest('.lgd'))showX(t);});
document.addEventListener('keydown',e=>{const t=e.target.closest&&e.target.closest('.xtab');if(!t||t.closest('.lgd')||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
 const ts=[...t.parentElement.querySelectorAll('.xtab')],i=ts.indexOf(t),n=e.key==='Home'?0:e.key==='End'?ts.length-1:(i+(e.key==='ArrowRight'?1:-1)+ts.length)%ts.length;
 showX(ts[n]);ts[n].focus();e.preventDefault();e.stopPropagation();});
function route(){const hash=location.hash.slice(1),[id,sub]=hash.split('/');const page=document.getElementById(id)||pages[0];
 pages.forEach(p=>p.classList.toggle('on',p===page));groups(page);activate(page,hash);
 const target=sub?document.getElementById(id+'/'+sub):null;for(let d=target&&target.closest('details');d;d=d.parentElement.closest('details'))d.open=true;
 if(target&&target.tagName==='DETAILS')target.open=true;
 const xp=(document.getElementById(hash)||target)?.closest?.('.xpanel');if(xp)showX(document.getElementById(xp.getAttribute('aria-labelledby')));
 below();(target||main).scrollIntoView?.({block:'start'});if(!target)main.scrollTop=0;spy();cut();}
function below(){const bar=document.querySelector('.page.on .bar');if(bar)main.style.setProperty('--bar-h',bar.offsetHeight+'px');
 return (bar?Math.max(bar.getBoundingClientRect().bottom,main.getBoundingClientRect().top):main.getBoundingClientRect().top+70)+24;}
function spy(){const page=document.querySelector('.page.on');if(!page)return;const bar=page.querySelector('.bar:not(.kinds)');if(!bar)return;
 const top=below(),links=[...bar.querySelectorAll('a')],marks=links.map(a=>document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
 let cur=marks[0];for(const m of marks){if(m.getBoundingClientRect().top<=top)cur=m;else break;}
 const end=main.scrollTop+main.clientHeight>=main.scrollHeight-2,hashed=marks.find(m=>'#'+m.id===location.hash);if(end&&hashed&&hashed.getBoundingClientRect().top>top)cur=hashed;
 links.forEach(a=>a.classList.toggle('on',cur&&a.getAttribute('href')==='#'+cur.id));}
const scrollers=[...document.querySelectorAll('.scroll,.scrollx,.alg,.math.block')];
function cut(){for(const el of scrollers){const mx=el.scrollWidth-el.clientWidth,my=el.scrollHeight-el.clientHeight,x=mx>1&&el.scrollLeft<mx-1,y=my>1&&el.scrollTop<my-1;
 let hint=el.previousElementSibling;if(!hint||!hint.classList.contains('sh')){hint=document.createElement('div');hint.className='sh';el.before(hint);}
 const h=(el.scrollLeft>1?'\u2190':'')+(x?'\u2192':''),v=(el.scrollTop>1?'\u2191':'')+(y?'\u2193':'');
 const w=[];if(mx>1)w.push(h+' scrolls sideways');if(my>1)w.push(v+' scrolls '+(y?'down':'up'));hint.textContent=w.join(' \u00b7 ');hint.classList.toggle('on',w.length>0);hint.classList.toggle('x',mx>1);}}
scrollers.forEach(el=>el.addEventListener('scroll',cut,{passive:true}));addEventListener('resize',cut);addEventListener('load',cut);if(document.fonts)document.fonts.ready.then(cut);
document.addEventListener('toggle',()=>cut(),true);if(window.ResizeObserver)new ResizeObserver(()=>{cut();below();}).observe(main);addEventListener('resize',below);
// the Research landing page: show one outcome or state, order each programme's records
for(const lp of document.querySelectorAll('.page[data-page=research]')){
 const btns=[...lp.querySelectorAll('[data-show]')],pick=lp.querySelector('select[name=show]'),order=lp.querySelector('select[name=order]');
 // a repo with no experiment draws no filter
 if(!pick)continue;
 function show(w){btns.forEach(b=>b.setAttribute('aria-pressed',b.dataset.show===w));pick.value=w;
  for(const g of lp.querySelectorAll('.ln-group')){const rows=[...g.querySelectorAll('tr[data-word]')];if(!rows.length)continue;let n=0;
   for(const r of rows){const hit=w==='all'||r.dataset.word===w;r.hidden=!hit;if(hit)n++;}
   const none=g.querySelector('.ln-none[data-none]');none.hidden=n>0;none.textContent=none.dataset.none.replace('{}',w);g.querySelector('.ln-box').hidden=!n;}}
 // records by place or concluded date; the undated after the dated, the planned last
 function sort(o){for(const tb of lp.querySelectorAll('.ln-t tbody')){const rows=[...tb.rows],key=r=>[r.classList.contains('plan')?1:0,o!=='members'&&!r.dataset.date?1:0];
  rows.sort((a,b)=>{const ka=key(a),kb=key(b);if(ka[0]!==kb[0]||ka[1]!==kb[1])return ka[0]-kb[0]||ka[1]-kb[1];
   if(o!=='members'&&a.dataset.date!==b.dataset.date)return (a.dataset.date<b.dataset.date?-1:1)*(o==='new'?-1:1);return a.dataset.pos-b.dataset.pos;});tb.append(...rows);}}
 btns.forEach(b=>b.addEventListener('click',()=>show(b.dataset.show)));pick.addEventListener('change',()=>show(pick.value));order.addEventListener('change',()=>sort(order.value));}
addEventListener('hashchange',route);route();main.addEventListener('scroll',spy,{passive:true});
const root=document.documentElement,btn=document.querySelector('.theme');const modes=['system','light','dark'];
function apply(m){if(m==='system')root.removeAttribute('data-theme');else root.dataset.theme=m;btn.textContent='theme: '+m;}
let mode='system';try{mode=localStorage.getItem('report-theme')||'system'}catch(e){}apply(mode);
btn.onclick=()=>{mode=modes[(modes.indexOf(mode)+1)%3];apply(mode);try{localStorage.setItem('report-theme',mode)}catch(e){}};
addEventListener('beforeprint',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
const tip=document.createElement('div');tip.className='tip';tip.hidden=true;document.body.appendChild(tip);
document.addEventListener('mouseover',e=>{const a=e.target.closest&&e.target.closest('a.ref, a.term, .tipped');if(!a)return;const [head,...rest]=a.dataset.tip.split('\\n');
 const body=document.createElement('span');body.innerHTML=rest.join('\\n');tip.replaceChildren(Object.assign(document.createElement('b'),{textContent:head}),body);tip.hidden=false;
 const r=a.getBoundingClientRect();tip.style.left=Math.max(8,Math.min(r.left,innerWidth-tip.offsetWidth-12))+'px';
 tip.style.top=(r.bottom+8+tip.offsetHeight>innerHeight?r.top-tip.offsetHeight-8:r.bottom+8)+'px';});
document.addEventListener('mouseout',e=>{if(e.target.closest&&e.target.closest('a.ref, a.term, .tipped'))tip.hidden=true;});
const app=document.querySelector('.app'),nav=document.querySelector('.nav');
function setNav(on){app.classList.toggle('nonav',!on);try{localStorage.setItem('report-nav',on?'1':'0')}catch(e){}}
let navOn=innerWidth>800;try{navOn=localStorage.getItem('report-nav')!=='0'&&innerWidth>800}catch(e){}setNav(navOn);nav.onclick=()=>setNav(app.classList.contains('nonav'));
links.forEach(a=>a.addEventListener('click',e=>{if(innerWidth<=800&&!e.target.closest('.hc'))setNav(false)}));
const gl=document.getElementById('glossary'),gf=gl&&gl.querySelector('.filter');if(gf){const gs=[...gl.querySelectorAll('.g')],lg=[...gl.querySelectorAll('.lgrp')],kinds=[...gl.querySelectorAll('.kinds a')],az=[...gl.querySelectorAll('.az a')];let kind='';
 function glossShow(){const q=gf.value.trim().toLowerCase();gs.forEach(g=>g.classList.toggle('off',(!!kind&&g.dataset.kind!==kind)||(!!q&&!g.dataset.q.includes(q))));
  lg.forEach(x=>x.classList.toggle('off',!x.querySelector('.g:not(.off)')));kinds.forEach(a=>a.classList.toggle('on',a.dataset.kind===kind));gl.classList.toggle('bykind',!!kind);
  az.forEach(a=>{const h=document.getElementById(a.getAttribute('href').slice(1));a.classList.toggle('off',!h||h.parentElement.classList.contains('off'));});glossFit();}
 function glossFit(){gs.forEach(g=>{const d=g.querySelector('.d'),line=parseFloat(getComputedStyle(d).lineHeight)||20;
  // hidden text is at least half a line more; inline code padding adds only a pixel or two
  g.classList.toggle('long',g.classList.contains('open')||d.scrollHeight-d.clientHeight>line/2);});}
 function glossKind(){const hash=location.hash.slice(1),[id,sub]=hash.split('/');if(id!=='glossary')return;
  // a kind button picks its kind; a letter keeps it; any other glossary link shows all, so its term is visible
  if(!sub||!sub.startsWith('letter-')){const a=kinds.find(a=>a.id===hash);kind=a?a.dataset.kind:'';}glossShow();}
 gf.addEventListener('input',glossShow);addEventListener('hashchange',glossKind);addEventListener('resize',glossFit);addEventListener('load',glossFit);if(document.fonts)document.fonts.ready.then(glossFit);glossKind();
 gl.addEventListener('click',e=>{const b=e.target.closest('.more');if(!b)return;const g=b.closest('.g'),open=!g.classList.contains('open');g.classList.toggle('open',open);b.setAttribute('aria-expanded',open);b.textContent=open?'less':'more';});
 gf.addEventListener('keydown',e=>{if(e.key==='Escape'){gf.value='';gf.dispatchEvent(new Event('input'));gf.blur();e.stopPropagation();}});}
const lgd=document.getElementById('legend'),lgOpen=document.querySelector('.top .legend');let lgFrom=null;
function legend(open){if(open&&!lgd.open){lgFrom=document.activeElement;lgd.showModal();lgd.querySelector('.lgx').focus();cut();}else if(!open&&lgd.open)lgd.close();}
lgd.addEventListener('close',()=>{lgFrom?.focus?.();lgFrom=null;});
lgd.addEventListener('click',e=>{if(e.target===lgd||e.target.closest('.lgx'))legend(false);else if(e.target.closest('.lgb a'))e.preventDefault();});
lgOpen.onclick=()=>legend(true);
const fv=document.getElementById('figview'),fvPlate=fv.querySelector('.figvp'),fvCap=document.getElementById('figview-cap');
const FIG_VARS=['--fig','--fig-ink','--fig-muted','--fig-line','--fig-soft','--fig-paper','--fig-series-dark','--fig-tint'];
let fvFrom=null,fvFromCap=null,fvRatio=1,fvStyle=null;
function fvFit(){const d=fvPlate.firstElementChild;if(!d)return;const room=Math.min(innerWidth*.95,innerWidth-16)-48,tall=Math.min(innerHeight*.85,innerHeight-72);
 for(let i=0;i<2;i++){const w=Math.max(80,Math.min(room,(tall-fvCap.offsetHeight-58)*fvRatio));d.style.width=w+'px';d.style.height=w/fvRatio+'px';}}
function figOpen(f){if(fv.open||lgd.open)return;const d=f.firstElementChild,r=d&&d.getBoundingClientRect(),cs=getComputedStyle(f);
 fvRatio=r&&r.height?r.width/r.height:1;fvFrom=f;fvFromCap=f.parentElement.querySelector(':scope>figcaption');fvStyle=d&&d.getAttribute('style');
 // the figure keeps its theme ink outside its page; its slot keeps its height
 FIG_VARS.forEach(v=>{const x=cs.getPropertyValue(v).trim();if(x)fv.style.setProperty(v,x);else fv.style.removeProperty(v);});
 f.style.minHeight=f.offsetHeight+'px';if(fvFromCap)fvFromCap.style.minHeight=fvFromCap.offsetHeight+'px';
 fvPlate.append(...fvFrom.childNodes);if(fvFromCap)fvCap.append(...fvFromCap.childNodes);
 tip.hidden=true;root.classList.add('figlock');fv.showModal();fvFit();fv.querySelector('.figx').focus();}
function fvRestore(){if(!fvFrom)return;const d=fvPlate.firstElementChild;if(d){if(fvStyle===null)d.removeAttribute('style');else d.setAttribute('style',fvStyle);}
 fvFrom.append(...fvPlate.childNodes);if(fvFromCap){fvFromCap.append(...fvCap.childNodes);fvFromCap.style.minHeight='';}
 fvFrom.style.minHeight='';root.classList.remove('figlock');fvFrom.focus({preventScroll:true});fvFrom=fvFromCap=null;}
fv.addEventListener('close',fvRestore);
fv.addEventListener('click',e=>{if(e.target.closest('.figx,a')||!e.target.closest('.figvf'))fv.close();});
document.addEventListener('click',e=>{const f=e.target.closest&&e.target.closest('.fig.zoom');if(f&&!e.target.closest('a'))figOpen(f);});
document.addEventListener('keydown',e=>{if(e.target.matches&&e.target.matches('.fig.zoom')&&(e.key==='Enter'||e.key===' ')){figOpen(e.target);e.preventDefault();}});
addEventListener('resize',()=>{if(fv.open)fvFit();});addEventListener('beforeprint',()=>{if(fv.open){fv.close();fvRestore();}});
let tapped=false;document.addEventListener('pointerdown',e=>{tapped=e.pointerType!=='mouse';},true);
function pmHot(card){const w=card.closest('[data-pmap]'),k=card.dataset.k,on=new Set([k,...card.dataset.links.split(' ').filter(Boolean)]);w.dataset.hot=k;
 w.querySelectorAll('[data-k]').forEach(c=>{const x=c.dataset.k;c.classList.toggle('pm-hot',x===k);c.classList.toggle('pm-link',x!==k&&on.has(x));c.classList.toggle('pm-dim',!on.has(x));});
 w.querySelectorAll('[data-e]').forEach(x=>{const [s,d]=x.dataset.e.split(' '),mine=s===k||d===k;if(x.classList.contains('po'))x.classList.toggle('pm-show',mine);else{x.classList.toggle('pm-dim',!mine);x.classList.toggle('pm-thick',mine);}});}
function pmCold(w){if(!w||!w.dataset.hot)return;delete w.dataset.hot;w.querySelectorAll('.pm-hot,.pm-link,.pm-dim,.pm-show,.pm-thick').forEach(x=>x.classList.remove('pm-hot','pm-link','pm-dim','pm-show','pm-thick'));}
document.addEventListener('mouseover',e=>{const c=!tapped&&e.target.closest&&e.target.closest('[data-pmap] [data-k]');if(c&&c.closest('[data-pmap]').dataset.hot!==c.dataset.k)pmHot(c);});
document.addEventListener('mouseout',e=>{const w=!tapped&&e.target.closest&&e.target.closest('[data-pmap]'),to=e.relatedTarget;if(w&&!(to&&to.closest&&to.closest('[data-pmap] [data-k]')))pmCold(w);});
document.addEventListener('focusin',e=>{const c=!tapped&&e.target.closest&&e.target.closest('[data-pmap] [data-k]');if(c)pmHot(c);});
document.addEventListener('focusout',e=>{const w=!tapped&&e.target.closest&&e.target.closest('[data-pmap]');if(w)pmCold(w);});
document.addEventListener('click',e=>{const w=tapped&&e.target.closest&&e.target.closest('[data-pmap]');if(!w)return;const c=e.target.closest('[data-k]');
 if(!c)pmCold(w);else if(w.dataset.hot!==c.dataset.k){e.preventDefault();pmHot(c);}});
const pmaps=[...document.querySelectorAll('[data-pmap]')],pmTimers=new Map();let pmDrag=null;
const pmCols=port=>port.dataset.cols.split(' ').map(Number);
function pmView(w){const port=w.querySelector('.pport'),head=w.querySelector('.pmhead');if(!port||!head)return;
 const max=port.scrollWidth-port.clientWidth,over=port.clientWidth>0&&max>1;head.hidden=!over;if(!over)return;
 const x=port.scrollLeft,view=port.clientWidth,input=head.querySelector('input'),svg=head.querySelector('svg');
 const width=svg.viewBox.baseVal.width,unit=width/port.scrollWidth,inset=width/svg.getBoundingClientRect().width;
 head.querySelectorAll('.pmm-frame,.pmm-halo').forEach(f=>{f.setAttribute('x',x*unit+inset);f.setAttribute('width',Math.max(0,view*unit-2*inset));});input.max=max;input.value=x;
 // a column is in view while its middle is; with none, the one under the middle of the view
 const cols=pmCols(port),mid=c=>c+Number(port.dataset.colW)/2,seen=cols.map((c,i)=>mid(c)>=x&&mid(c)<=x+view?i+1:0).filter(Boolean);
 const a=seen.length?seen[0]:Math.max(1,cols.filter(c=>c<=x+view/2).length),b=seen.length?seen[seen.length-1]:a,n=cols.length;
 input.setAttribute('aria-valuetext',(a===b?'Column '+a:'Columns '+a+' to '+b)+' of '+n+' in view');
 head.querySelector('.pmhint').replaceChildren((x>1?'←':'')+(x<max-1?'→':'')+' scrolls sideways',document.createElement('br'),el('b',(a===b?'column '+a:'columns '+a+'–'+b)+' of '+n),' in view');}
function pmTo(w,x){const port=w.querySelector('.pport');port.scrollLeft=Math.max(0,Math.min(x,port.scrollWidth-port.clientWidth));pmView(w);}
const pmAll=()=>pmaps.forEach(pmView);
pmaps.forEach(w=>w.querySelector('.pport')?.addEventListener('scroll',()=>pmView(w),{passive:true}));
addEventListener('resize',pmAll);addEventListener('load',pmAll);if(document.fonts)document.fonts.ready.then(pmAll);if(window.ResizeObserver)new ResizeObserver(pmAll).observe(main);
document.addEventListener('keydown',e=>{const input=e.target.closest&&e.target.closest('.pmm input');if(!input||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key))return;
 const w=input.closest('[data-pmap]'),port=w.querySelector('.pport'),x=port.scrollLeft,cols=pmCols(port),starts=cols.map(c=>c-cols[0]),fwd=e.key==='ArrowRight'||e.key==='ArrowUp';
 pmTo(w,e.key==='Home'?0:e.key==='End'?Infinity:fwd?(starts.find(s=>s>x+1)??Infinity):(starts.filter(s=>s<x-1).pop()??0));e.preventDefault();});
document.addEventListener('input',e=>{const input=e.target.closest&&e.target.closest('.pmm input');if(input&&!pmDrag)pmTo(input.closest('[data-pmap]'),Number(input.value));});
// a click centres the frame on its point; a drag from inside the frame keeps where it was grabbed
function pmPoint(e){const {w,input,grab}=pmDrag,port=w.querySelector('.pport'),r=input.getBoundingClientRect();pmTo(w,(e.clientX-r.left)/r.width*port.scrollWidth-grab);}
document.addEventListener('pointerdown',e=>{const input=e.target.closest&&e.target.closest('.pmm input');if(!input||e.button)return;
 e.preventDefault();input.focus({preventScroll:true});input.setPointerCapture?.(e.pointerId);
 const w=input.closest('[data-pmap]'),port=w.querySelector('.pport'),r=input.getBoundingClientRect(),at=(e.clientX-r.left)/r.width*port.scrollWidth;
 const inside=at>=port.scrollLeft&&at<=port.scrollLeft+port.clientWidth;pmDrag={w,input,inside,moved:false,grab:inside?at-port.scrollLeft:port.clientWidth/2};if(!inside)pmPoint(e);});
document.addEventListener('pointermove',e=>{if(pmDrag){pmDrag.moved=true;pmPoint(e);}});
document.addEventListener('pointerup',e=>{if(pmDrag&&pmDrag.inside&&!pmDrag.moved){pmDrag.grab=pmDrag.w.querySelector('.pport').clientWidth/2;pmPoint(e);}pmDrag=null;});
document.addEventListener('pointercancel',()=>{pmDrag=null;});
// the ring holds 1.2 s, then fades over 0.4 s
function pmFlash(w,parts){const off=xs=>xs.forEach(x=>x.classList.remove('pm-flash','pm-fade')),last=pmTimers.get(w);if(last){clearTimeout(last.timer);off(last.parts);}
 parts.forEach(x=>x.classList.add('pm-flash'));
 const timer=setTimeout(()=>{parts.forEach(x=>x.classList.replace('pm-flash','pm-fade'));pmTimers.set(w,{parts,timer:setTimeout(()=>off(parts),400)});},1200);pmTimers.set(w,{parts,timer});}
function pmJump(card,ring=true){const w=card.closest('[data-pmap]'),k=CSS.escape(card.dataset.k),port=w.querySelector('.pport');
 // the script's ring is the only mark: drop the focus the browser gives an anchor's target
 if(document.activeElement===card)card.blur();
 if(!card.offsetParent){const t=w.querySelector('.ptree [data-k="'+k+'"]');if(t){t.scrollIntoView?.({block:'center'});if(ring)pmFlash(w,[t]);}return;}
 pmTo(w,card.offsetLeft+card.offsetWidth/2-port.clientWidth/2);
 // the map's box under the section bar; the card in the middle of the window when the map is taller than that
 const top=below(),box=w.querySelector('.pmap').getBoundingClientRect(),r=card.getBoundingClientRect(),room=main.getBoundingClientRect().bottom;
 let dy=box.top-top;if(r.bottom-dy>room-16)dy=(r.top+r.bottom)/2-(top+room)/2;main.scrollTop+=dy;
 if(ring)pmFlash(w,[card,w.querySelector('.pmm [data-k="'+k+'"]')].filter(Boolean));}
function pmHash(ring=true){const t=document.getElementById(location.hash.slice(1));if(t&&t.matches('[data-pmap] .pnode'))pmJump(t,ring);}
addEventListener('hashchange',()=>{pmAll();pmHash();});
document.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('a.pnum');if(a&&a.getAttribute('href')===location.hash){e.preventDefault();pmHash();}});
pmAll();pmHash();
// a direct open: the browser's own scroll to the anchor and late fonts move the map after this runs
addEventListener('load',()=>requestAnimationFrame(()=>{pmAll();pmHash(false);}));if(document.fonts)document.fonts.ready.then(()=>{pmAll();pmHash(false);});
let back='';addEventListener('keydown',e=>{const altGr=e.getModifierState&&e.getModifierState('AltGraph');
 if(e.target.matches('input,textarea')||e.metaKey||(!altGr&&(e.ctrlKey||e.altKey)))return;
 if(fv.open)return;if(lgd.open){if(e.key==='?'){legend(false);e.preventDefault();}return;}
 const onGloss=location.hash.startsWith('#glossary'),gloss=document.getElementById('glossary'),page=document.querySelector('.page.on');
 const go=h=>{location.hash=h;e.preventDefault();};
 if(e.key==='?'){legend(true);e.preventDefault();}
 else if(e.key==='g'&&gloss&&!onGloss){back=location.hash;go('#glossary');}
 else if((e.key==='g'||e.key==='Escape')&&onGloss)go(back||'#overview');
 else if(e.key==='['||e.key===']'||e.code==='BracketLeft'||e.code==='BracketRight'){
  const fwd=e.key===']'||(e.code==='BracketRight'&&e.key!=='[');let n=exps.findIndex(x=>pageOf(x)===page.id);
  n=n>=0?n+(fwd?1:-1):(fwd?0:exps.length-1);if(exps[n])go(exps[n].getAttribute('href'));}
 else if(/^[jk]$/i.test(e.key)&&page){const hs=[...page.querySelectorAll('h2')];if(!hs.length)return;const top=below();
  let cur=-1;hs.forEach((h,i)=>{if(h.getBoundingClientRect().top<=top)cur=i;});const n=cur+(/^j$/i.test(e.key)?1:-1);if(hs[n])go('#'+hs[n].id);}
 else if(e.key==='/'){e.preventDefault();if(onGloss&&gf)gf.focus();else{if(app.classList.contains('nonav'))setNav(true);panel?.querySelector('.rfilter input')?.focus();}}
 else if(e.key==='o')go('#overview');
 else if(e.key==='e'&&page){const ds=[...page.querySelectorAll('details')],open=ds.some(d=>!d.open);ds.forEach(d=>d.open=open);page.classList.toggle('allx',open);}
 else if(e.key==='n')nav.onclick();
 else if(e.key==='t')btn.onclick();});
"""


MATHML = re.compile(r'<span class="math[^"]*"><math.*?</math></span>', re.S)


def link_ported(page: str, ref: Reference, skip: str = "") -> str:
    """A page from the new builder with reference entries linked on their first
    mention, but `skip`, the entry the page is about.
    Equations are set aside first, so an entry name never matches inside one."""
    equations = MATHML.findall(page)
    page = MATHML.sub(lambda m: f"mathy{equations.index(m.group(0))}y", page)
    if ref:
        page = link_entries(page, ref, skip)
    return re.sub(r"mathy(\d+)y", lambda m: equations[int(m.group(1))], page)


def build(out: Path) -> int:
    global REPO_URL, GITHUB
    REPO_URL = repo_url()
    GITHUB = spec_builder.github_repo(REPO_URL)
    records = load_records()
    terms = load_glossary()
    fams = families(records)
    progs = sorted((EXPERIMENTS / "programmes").glob("*.md"))
    ref = Reference()
    site = spec_builder.read_site(EXPERIMENTS, load_specs(SPEC_FOLDER), GITHUB)
    models = {m.document.path.resolve(): m for m in site.models}

    def record_page(r: Record) -> str:
        # load_records keeps only the files the record spec's path matches, so each has a model
        return link_ported(spec_builder.render_page(models[r.path.resolve()], site), ref)

    def entry_page(e: RefEntry) -> str:
        # every entry the topic lists matches the entry spec's path, so it has a model
        page = spec_builder.render_page(models[e.path.resolve()], site)
        return link_ported(page, ref, e.key)

    # every programme matches the programme spec's path, so it has a model
    research = [spec_builder.render_page(models[g.resolve()], site) for g in progs]
    research += [record_page(r) for fam in fams for r, _ in fam]
    reference = []
    for t in ref.topics:
        # a topic without README.md has no model and no page; its problem fails the build
        if topic := models.get(t.path.resolve()):
            reference.append(link_ported(spec_builder.render_page(topic, site), ref))
        reference += [entry_page(e) for e in t.entries]
    glossary = next(iter(site.of_kind("glossary")), None)
    # the book, experiments/README.md, is the overview's model; a repo without one still gets the overview
    book = next(iter(site.of_kind("book")), None)
    overview = spec_builder.render_page(book, site) if book else spec_builder.overview_page.render_without_book(site)
    # each landing page opens the pages of its tab
    running = rail.running_slugs(EXPERIMENTS)
    pages = [overview, landing_page.render_research(site, running), *research]
    pages += [landing_page.render_reference(site), *reference]
    pages.append(spec_builder.render_page(glossary, site) if glossary else "")
    repo = rail.brand_name(site, REPO_URL)
    foot = [f"Built {datetime.date.today().isoformat()}"]
    if sha := git_head():
        code = f"<code>{esc(sha)}</code>"
        foot[0] += f' at <a href="{esc(REPO_URL)}/commit/{esc(sha)}">{code}</a>' if REPO_URL else f" at {code}"
    if REPO_URL:
        foot.append(f'<a href="{esc(REPO_URL)}">{GITHUB_ICON}GitHub</a>')
    foot = " · ".join(foot)
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(repo)}</title>"
        f'<link rel="icon" href="{FAVICON}">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+Math'
        f'{REF_FONT if ref else ""}&display=swap">'
        f'<style>{CSS}{REF_CSS if ref else ""}{spec_builder.page_styles()}</style></head><body><div class="app">'
        f"{rail.render_top(repo, REPO_URL)}"
        f'<nav class="side">{rail.render_rail(site, running)}</nav>'
        f'<main class="main">{"".join(pages)}<footer class="foot">{foot}</footer></main></div>{spec_builder.render_legend()}{FIGURE_VIEW}'
        f"<script>{JS}</script></body></html>"
    )
    out.write_text(restore_math(relink(page, records)), encoding="utf-8")
    problems = [(str(p.relative_to(EXPERIMENTS.parent)), OUTSIDE_RECORD_FOLDERS) for p in split_readmes()[1]]
    problems += [(x, p) for x in records + ref.topics for p in x.problems]
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
    if ref:
        ideas = f"{len(ref.ideas.items)} ideas from the {ref.ideas.source}" if ref.ideas.source else "no ideas loaded"
        print(f"  reference: {len(ref.topics)} topics, {len(ref.entries)} entries, {ideas}")
    for r, p in problems:
        where = r if isinstance(r, str) else r.path.relative_to(EXPERIMENTS.parent)
        print(f"  {where}: {p}", file=sys.stderr)
    # the writing rules are the checker's: count its breaches, never fail on them
    if breached := [m for m in site.models if m.breaches]:
        total = sum(len(m.breaches) for m in breached)
        print(
            f"  checker: {total} breaches in {len(breached)} documents (`build_report.py --check` lists them)",
            file=sys.stderr,
        )
    # an empty glossary draws its empty page; one whose lines all fail to parse is a hole
    unparsed = [b for b in glossary.breaches if b.rule == GLOSSARY_LINE_RULE] if glossary else []
    if records and glossary is None:
        print(
            "  experiments/GLOSSARY.md missing: seed it from the report skill's references/glossary.md", file=sys.stderr
        )
        problems.append(("glossary", "missing"))
    elif not terms and unparsed:
        print(
            f"  experiments/GLOSSARY.md: no term line parses, first at line {unparsed[0].line}: {unparsed[0].message}"
            " (`python -m labflow_report check` lists every one)",
            file=sys.stderr,
        )
        problems.append(("glossary", "no term line parses"))
    if undefined := undefined_terms(records, terms):
        print(f"  glossary: abbreviations and methods without a definition: {', '.join(undefined)}", file=sys.stderr)
    return 1 if problems else 0


def selftest() -> int:
    """Build the fixture next to this script and check the rendering invariants."""
    global EXPERIMENTS, IDEAS_ONLINE
    IDEAS_ONLINE = False
    fixture = Path(__file__).resolve().parent / "fixture" / "experiments"
    if not fixture.is_dir():
        print(f"no fixture at {fixture}; --selftest runs from the labflow skill copy", file=sys.stderr)
        return 2
    # the same fixture without its reference folder: the report a repo without one gets
    with tempfile.TemporaryDirectory() as tmp:
        EXPERIMENTS = Path(tmp) / "experiments"
        shutil.copytree(fixture, EXPERIMENTS, ignore=shutil.ignore_patterns(REFERENCE, "report.html"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            build(EXPERIMENTS / "report.html")
        bare = (EXPERIMENTS / "report.html").read_text(encoding="utf-8")
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
        "every external link opens in a new tab, whatever its attribute order": not re.search(
            r'<a\b(?![^>]*\btarget=)[^>]*\bhref="https?://', body
        )
        and 'target="_blank"' in relink('<a class="url" href="https://example.org">example.org</a>', []),
        "no heading made from a #<ticket> line": len(re.findall(r"<h1\b", body)) == 12,
        "answer card first, then the clause trace, then findings": body.find('id="alpha/answer"')
        < body.find('id="alpha/clauses"')
        < body.find('id="alpha/findings"'),
        "answer card: question, verdict line, clause table, needs line and section bar": '<p class="chap qlabel">Question</p>'
        in body
        and '<div class="row vline"><span class="vp inc lg"' in body
        and '<span class="badge b-inconclusive">inconclusive</span>' in body
        and '<p class="small rneeds"><b>Needs</b>' in body
        and 'class="hot"' in body
        and '<a class="cc ok" href="#alpha/h1">' in body
        and '<nav class="bar" aria-label="Sections"><a class="on" href="#alpha/findings">Findings <i>5</i></a>' in body,
        "key finding cards, supporting and minor findings as open small cards": '<div id="alpha/f4" class="box keycard">'
        in body
        and '<details id="alpha/f3" class="mc fcard"><summary>' in body
        and '<details id="alpha/f5" class="mc fcard"><summary>' in body,
        "one light figure on a plate, no dark variant": '<div class="fig zoom" role="button" tabindex="0" aria-label="Open Figure 1 full screen"><svg' in body
        and "fig-dark" not in body
        and "fig-light" not in body,
        "short titles in the sidebar": '<span class="lab">Alpha gold</span>' in body
        and 'title="Alpha gold set"' in body,
        "programme needs table": '<table class="data pneeds">' in body,
        "dataset row opening its own fold, which holds the tables and the figure": '<div class="input"><span class="chev"></span><span class="nm">Gold set</span>'
        in body
        and re.search(r'<details class="dfold" id="alpha/dataset-1"(?: open)?>.*?<figure[^>]*><div class="scroll.*?<figure[^>]*><div class="fig zoom"', body, re.S)
        is not None,
        "programme page, sidebar entry, map and overview row": 'id="programme-gamma"' in body
        and '<a class="hd" href="#programme-gamma" data-group="programme-gamma" title=' in body
        and '<div class="pnode planned"' in body
        and '<a class="pnode member" href="#alpha"' in body
        and '<path class="pe pl"' in body
        and '<table class="data pexp">' in body
        and 'id="overview/programmes"' in overview
        and 'href="#programme-gamma"' in overview,
        "programme citations as chips with a hover card, the threads matrix holds each clause under its experiment": '<a class="ct f ref" href="#alpha/f1" data-tip="F1 — First thing\n&lt;strong&gt;Alpha gold:&lt;/strong&gt;'
        in body
        and ">1·F1</a>" in body
        and "[alpha:F9]" in body
        and '<table class="data pthreads">' in body
        and '<td><div class="cells"><a class="cc ok" href="#alpha/h1">' in body,
        "no raw markdown table": not re.search(r">[^<]*\|[^<]*\|[^<]*\|", body),
        "every table styled and scrollable": "<table>" not in body,
        "no relative link survives": not re.search(r'href="(?!#|https?://)', body),
        "line-starting ticket ref kept in its sentence": re.search(r"<p>Pool it, so\s(?:#33|<a [^>]*>#33</a>) proceeds\.</p>", body) is not None,
        "Next items listed under What it opened": '<p><span>audit <code>de</code> and <code>fr</code> by hand</span>' in body,
        "float formatted": "0.6047</td>" in body and "0.6047136829414016" not in body,
        "svg ids prefixed": 'id="alpha-a-' in body and 'id="axes_1"' not in body,
        "PR link built from the remote or left as text": "PR #62" in body,
        "decision folded to its line, open as rows": "<b>Decision</b><div><p>Pool it" in body
        and '<details class="fold" id="alpha/d1">' in body
        and "<b>Why</b>" in body,
        "[D]/[F] citations as chips with a hover card, an unknown one without brackets": '<a class="ct d ref" href="#alpha/d1" data-tip="D1 — Pick the pool\npool it, so #33 proceeds.">D1</a>'
        in body
        and '<a class="ct f ref" href="#alpha/f1"' in body
        and 'href="#alpha/d9"' not in body
        and "(D9, D15)" in body,
        "a bare F1 is the metric, never a finding chip": re.search(r'<a class="ct f[^"]*"[^>]*>F1</a> 0\.605', body) is None,
        "glossary page, rail glossary tab and keys": '<p class="ttl">Glossary</p>' in body
        and '<a class="hd" href="#glossary" data-group="t-glossary"' in body
        and '<a class="item nest" href="#glossary/metrics" title="Metrics" data-in="t-glossary">' in body
        and '<a class="item nest" href="#glossary/group-statistics" title="Statistics" data-in="t-glossary">' in body
        and '<a class="item nest" href="#glossary/other" title="Other" data-in="t-glossary">' in body
        and '<div class="rkeys pkeys"><span><kbd>?</kbd> legend</span><span><kbd>g</kbd> glossary</span><span><kbd>[</kbd><kbd>]</kbd> experiments</span>'
        '<span><kbd>t</kbd> theme</span></div>' in body
        and '<input class="filter" type="search"' in body
        and '<dt id="glossary/f1">F1 <span class="grp">Metrics</span> <span class="also">F1 score, macro F1</span></dt>' in body
        and '<a id="glossary/metrics" href="#glossary/metrics" data-kind="Metrics">Metrics <i>' in body,
        "glossary terms in page text stay plain text": 'href="#glossary/f1"' not in body
        and 'href="#glossary/precision"' not in body
        and "Precision holds at q05 (0.1, 0.2, 0.3) and AUC is flat" in body
        and '<div class="t"><a class="ct f ref" href="#alpha/f1"' in body,
        "undefined abbreviations and methods reported": undefined_terms(records, load_glossary())
        == ["AUC", "bootstrap"],
        "reproduce rendered as steps": '<ol class="steps"><li class="step env"><span class="n">env</span>' in body
        and '<li class="step"><span class="n">1</span><div class="t">' in body
        and 'class="c">#' in body
        and '<div class="t">3 ' not in body
        and 'class="t">tables/ and figures/ — every finding' in body,
        "spaced double dash renders as a dash, flags untouched": "pool — the members" in body
        and '<code class="file">git log -- prep.py</code>' in body
        and "--dry-run" in body,
        "runs line: a short hex is a commit, a local MLflow is text, external links open in a new tab": '<span class="lb">Runs</span><span class="missing">no run</span><span class="muted">· analysis</span><span class="sha">e826d11</span>'
        in body
        and 'title="http://localhost:5000/#/experiments/7 — opens only where that server runs">MLflow · local</span>' in body
        and 'href="http://localhost' not in body
        and not re.search(r'<a href="https?://[^"]*">', body),
        "overview ledger lists each record once": overview.count('<a href="#alpha" title="Alpha gold set"><b class="nm">') == 1
        and '<table class="pt ledger">' in overview
        and "<h2>data</h2>" not in overview,
        "overview ledger names the finding that settled a clause": '<span class="muted">— First thing <a class="ct f ref" href="#alpha/f1"'
        in overview,
        "row values start with a capital": '<p class="small muted">Reading text; it confirms' in body
        and "<b>Decision</b><div><p>Pool it" in body,
        "reading and implication visible": '<div class="small so"><b>So</b>' in body
        and "what it means" not in body,
        "bold-lead reading blocks as paragraphs, the third what could overturn it, folded in a small card": '<p class="small muted">Holds on the hold-out. '
        in body
        and '<b>Could overturn it</b> A ' in body,
        "supporting folded under its key finding": re.search(
            r'<div id="alpha/f1" class="box keycard">(?:(?!class="box keycard").)*id="alpha/f3" class="mc fcard"',
            body,
            re.S,
        )
        is not None,
        "method step folded to its line, open as in → out, how, code and settings": '<details class="fold" id="alpha/m1">'
        in body
        and '<p class="k">Input</p>' in body
        and '<p class="k">Output</p>' in body
        and "<b>Code</b>" in body
        and '<code class="file">prep.py::<wbr>pool_<wbr>candidates</code>' in body
        and "<b>Settings</b>" in body,
        "equations rendered as MathML, none left as source": '<span class="math block"><math' in body
        and "<mfrac>" in body
        and '<span class="math"><math' in body
        and "$$" not in body
        and "mathx0x" not in body,
        "[M<n>] citation linked with a hover card, no MathML inside a card": '<a class="ct m ref" href="#alpha/m2"' in body
        and not re.search(r'data-tip="[^"]*<math', body)
        and "mathx" not in body,
        "headings the page cannot read are holes, and the only ones": problems
        == [
            "finding heading not `### F<n> — title · weight`: F2 Broken heading no dash",
            "method step heading not `### M<n> — title`: M4 Broken heading no dash",
        ],
        "long table scrolls under a sticky header, short one does not": '<div class="scroll tall"><table class="data tall">'
        in body
        and body.count('class="scroll tall"') == 1,
        "prose link to an inlined table reaches its figure": 'id="alpha/asset/t"' in body
        and '<a class="tlink" href="#alpha/asset/t"><span class="tic">▦</span>Every row</a>' in body,
        "datasets section rendered with its table and figure": '<h2 class="chap" id="alpha/datasets"' in body
        and body.count("<figcaption>") >= 2
        and 'id="alpha-a-' in body[body.find('id="alpha/datasets"') : body.find('id="alpha/question"')],
        "comparison and question clauses: their keywords on the condition, outcome chips": '<a class="cc eq" href="#beta/c1">'
        in body
        and 'C1<span class="ow">equivalent</span>' in body
        and '<div class="cond">better if the interval' in body
        and '<div class="cond">read on the expected-best curve' in body
        and '<span class="vp ok" title="answered">' in body
        and '<span class="badge b-adopted">adopted</span>' in body,
        "programme links a comparison clause with its verdict, and the thread holds it": 'href="#beta/c1" data-tip="C1 — refined parameters against production'
        in body,
        "clause trace rows carry their id, [H<n>] a chip with a hover card": '<a class="ct h ref" href="#alpha/h1" data-tip="H1 — something holds\nconfirmed if a; refuted if b — c.">H1</a>'
        in body
        and '<tr id="alpha/h1">' in body,
        "no reference folder, no reference in the report": "Topics" not in bare
        and bare.count('class="rb"') == 3
        and 'class="rpanel off" data-tab="reference"' not in bare
        and "ref-cutting" not in bare
        and 'class="algrow"' not in bare
        and "STIX+Two+Text" not in bare
        and "varies " not in bare[bare.find("<body") :],
        "rail: four tabs, each opening its landing page, each list headed by a link to it": body.count('class="rb"') == 4
        and '<button type="button" class="rb" data-tab="terms" data-landing="glossary">' in body
        and '<a class="hd" href="#research"><span class="hl">Research</span></a>' in body
        and '<a class="hd" href="#reference"><span class="hl">Reference</span></a>' in body
        and 'href="#reference"' not in bare,
        "rail: topics closed with their entries' status squares": '<a class="hd shut" href="#topic-cutting" data-group="topic-cutting" title="Cutting methods">' in body
        and 'data-in="topic-cutting"><span class="sq chk"></span><span class="lab">Base cut</span>' in body
        and 'data-in="topic-cutting"><span class="sq dft"></span><span class="lab">Test split</span>' in body,
        "rail: experiments under their programme with outcome mark, new dot and planned mark": re.search(
            r'<a class="hd" href="#programme-gamma"[^>]*>.*?<a class="item nest" href="#alpha" title="Alpha gold set" '
            r'data-in="programme-gamma" data-exp=""><span class="vp inc" title="inconclusive"><svg.*?</span>'
            r'<span class="lab">Alpha gold</span><span class="nd" title="new result"></span></a>.*?'
            r'<span class="vp plan" title="planned"></span><span class="lab">delta</span>',
            body,
            re.S,
        )
        is not None
        and '<span class="hl">Changed since 08-25</span>' in body,
        "top bar: brand, report word, crumb, legend and theme": re.search(
            r'<div class="top"><div class="lhs"><button class="nav"[^>]*>☰</button><(?:span|a) class="brand"[^>]*>[^<]+</(?:span|a)>'
            r'<span class="rpt">report</span></div><span class="crumb"></span><button class="legend"[^>]*>Legend</button><button class="theme"',
            body,
        )
        is not None,
        "overview shelf counts topics and entries": "1 topic, 4 entries" in overview
        and '<h2 id="overview/topics">Reference</h2>' in overview,
        "topic README is not read as a record": len(records) == 2,
        "entry status line: square and word, predecessor by title and square, topic, a role from a couple": '<p class="row small status"><span class="sq pap"></span>from paper <span class="muted">·</span> variant of <a class="term" href="#ref-cutting-base-cut"'
        in body
        and '<span class="sq chk" title="checked against code"></span>Base cut</a> <span class="muted">·</span> in <a href="#topic-cutting">Cutting methods</a>' in body
        and '<span class="muted">·</span> the training sample of <a class="term" href="#ref-cutting-base-cut"' in body,
        "facts show the short value, the note on hover, a compare label not written": '<tr><td class="muted">Chunk</td><td><span class="note" title="the run of characters between two marks">marked piece</span></td></tr>'
        in body
        and '<tr><td class="muted">Base unit</td><td><span class="missing">not recorded</span></td></tr>' in body,
        "sources: one paper per line with its role on hover, code files on one line by name": '<div class="sl" title="the method">Neural translation of rare words, <a href="https://arxiv.org/abs/1508.07909" target="_blank" rel="noopener">arXiv:1508.07909</a>, §3.2</div>'
        in body
        and '<div class="sl code"><code title="experiments/data/alpha/prep.py">prep.py</code> · <code title="experiments/data/alpha/prep.py:1">prep.py:1</code></div>' in body
        and '<div class="pvc small"><b>Paper vs code</b> — no difference found.</div>' in body,
        "paper notes folded under the description, the In short row as digest": '<span class="t">What the paper adds</span><span class="dg2">two boundary scores; the cut changes one line.</span>'
        in body,
        "algorithms as folds, each closed, keywords bold, depth kept": '<details class="fold algf" id="ref-cutting-base-cut/alg-1">'
        in body
        and '<details class="fold algf" id="ref-cutting-base-cut/alg-2">' in body
        and '<li class="i1"><span class="no">4</span><span class="tx"><b>if</b> the pair is new <b>then</b></span><span class="rf"></span></li><li class="i2"><span class="no">5</span><span class="tx">record it</span>' in body
        and "<b>for all</b> merge in" in body
        and "\\State" not in body,
        "algorithm digest: open idea, library lines, changed lines counted": '<span class="dg2 one" title="changed by the open idea #7">changed by the open idea <a class="iss" href="#topic-cutting/idea-7">'
        in body
        and '<span class="cnt">2 lines · 1 changed</span>' in body
        and "line 3 runs inside the library" in body,
        "algorithm head names the code file once, each line its range, a note under its line": '<span class="cf">lines in <code title="experiments/data/alpha/prep.py">prep.py</code></span>'
        in body
        and '<span class="rf"><span title="experiments/data/alpha/prep.py, line 1">1</span></span>' in body
        and '<span class="nt"><span class="tri">▷</span><span>ties by pair</span></span>' in body,
        "reading beside its algorithm": '<div class="reading"><p class="k">Reading Algorithm 1</p><div class="walk"><b>1</b><span>Words are counted once.</span><b>2–7</b>'
        in body,
        "difference: step table headed by the entries' titles, variant table under it": '<thead><tr><th>Step</th><th>Base cut</th><th>Morph cut</th><th>Where in the algorithms</th></tr></thead>'
        in body
        and '>Difference from Base cut</h2>' in body
        and '<a class="ct i" href="#ref-cutting-morph-cut/alg-1">Alg 1 · 1</a>' in body
        and re.search(r"<th[^>]*>Base cut</th><th[^>]*>Morph cut</th><th[^>]*>Why</th>", body) is not None,
        "worked example: input line, tables in place, a link to an entry file reaches its page": '<p class="small muted lead"><code>lowest</code> five times'
        in body
        and '<figure id="ref-cutting-morph-cut/asset/morph-cut-merges"><figcaption class="small muted">Merges on a toy word</figcaption>' in body
        and 'It reuses <a href="#ref-cutting-base-cut">the baseline</a>' in body,
        "definition box: equation, symbols, the rule sentence": '<dl class="syms"><dt>' in body
        and "Each round joins the pair that scores highest." in body,
        "limits: the question and the idea issue": '<p class="small q">Does a boundary mark keep endings apart at the same budget?</p>'
        in body,
        "related on one line from both sides, with the role": '<span class="muted">variant</span> <a class="term" href="#ref-cutting-morph-cut"'
        in body
        and '<span class="muted">coupled from</span>' in body
        and '<span class="muted">as training sample</span>' in body,
        "draft entry shows its expected sections with empty states": '<p class="lede"><span class="missing">Not written yet.</span></p>'
        in body
        and "No experiment uses Test split yet." in body
        and "None written yet." in body,
        "topic page: meta line, question, section bar, a variant nested under its predecessor": '<section class="page" id="topic-cutting" data-page="topic" data-title="Cutting methods">'
        in body
        and "<span>4 entries</span><span>2 checked against code</span><span>1 open idea</span>" in body
        and '<nav class="bar" aria-label="Sections"><a class="on" href="#topic-cutting/map">Map</a>' in body
        and '<ul><li><div class="tn"><a class="node" href="#ref-cutting-morph-cut">' in body
        and '<div class="from"><b>Changes:</b> chunks are marked pieces, not whole words.</div>' in body
        and '<a class="node n-planned" href="#ref-cutting-test-split">' in body,
        "topic compare: one tab per label, groups on the short value, notes and missing facts": '<button type="button" class="fsw xtab" role="tab" id="topic-cutting/fact-1-tab"'
        in body
        and '<div class="gtop"><b>Characters</b><span class="gc">2</span></div>' in body
        and '<p class="small muted gn"><b>Base cut</b>: the word as the corpus spells it</p>' in body
        and '<p class="small muted nw">Not written yet: ' in body,
        "topic ideas from the cache: a card per open idea, entries by relation, tested in": '<div class="idea" id="topic-cutting/idea-7">'
        in body
        and '<span class="muted">already in</span><span class="chips"><a class="term" href="#ref-cutting-base-cut"' in body
        and '<span class="muted">not assessed</span>' in body
        and '<span class="muted">tested in</span> <a class="cc pill" href="#alpha"' in body
        and 'id="topic-cutting/idea-8"' not in body,
        "topic used by and entries table: records by pill, missing cells in words": '<div class="row small ub"><a class="cc pill" href="#alpha"'
        in body
        and '<table class="data ents">' in body
        and '<span class="missing">no code cited</span>' in body
        and '<a class="iss" href="#topic-cutting/idea-7">' in body,
        "record names the entries it varies and uses by title and status square": '<span class="muted">varies</span> <a class="term" href="#ref-cutting-morph-cut" data-tip="Morph cut'
        in body
        and '<span class="sq pap" title="from paper"></span>Morph cut</a>' in body
        and '<span class="muted">uses</span> <a class="term" href="#ref-cutting-base-cut"' in body,
        "evidence generated per experiment, downstream apart from direct": re.search(
            r'id="ref-cutting-morph-cut/evidence"[^>]*>Evidence</h2><h3 class="ev">Direct</h3>.*?<h3 class="ev">Downstream</h3>.*?<a class="ct f ref" href="#alpha/f1"[^>]*>F1</a> First thing</td><td><a class="cc pill" href="#alpha".*?<td><a class="term" href="#ref-cutting-test-split"',
            body,
            re.S,
        )
        is not None
        and '<div class="used small">No experiment varies Base cut yet. It is used in <a class="cc pill" href="#alpha"' in body
        and '<tr><td class="muted">Varied in</td><td><a class="cc pill" href="#alpha"' in body,
        "entry mention in a record linked once with a hover card": body[
            body.find('id="alpha"') : body.find('id="beta"')
        ].count('<a class="term" href="#ref-cutting-test-split" data-tip="Test split')
        == 1,
        "clause roles, standards and rounds tagged in the trace, a measured clause read by its own keywords": '<div class="small muted">diagnostic · R1</div>'
        in body
        and '<div class="small muted">measured · R2</div>' in body
        and '<div class="cond">read on the budget table</div>' in body,
        "rounds listed in the question and setup fold with how each closed": '<li id="beta/r2"><b>R2</b> a ten-trial budget' in body
        and "closed 2026-09-08 conclude" in body,
    }
    for name, ok in checks.items():
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
    out.unlink()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=EXPERIMENTS / "report.html")
    ap.add_argument("--selftest", action="store_true", help="build the bundled fixture and check it")
    ap.add_argument("--offline", action="store_true", help="read ideas from the cache, never from the tracker")
    ap.add_argument(
        "--check",
        nargs="*",
        type=Path,
        metavar="PATH",
        help="check records against their spec instead of building, as `python -m labflow_report check` does "
        "(default: every document a spec declares)",
    )
    args = ap.parse_args()
    if args.check is not None:
        from labflow_report.__main__ import main as check_main

        sys.exit(check_main(["check", *map(str, args.check)]))
    IDEAS_ONLINE = not args.offline
    sys.exit(selftest() if args.selftest else build(args.out))
