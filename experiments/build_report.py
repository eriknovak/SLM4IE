"""Render every labflow experiment record into one self-contained HTML report.

Reads the book (experiments/README.md) and every record
(experiments/<category>/<slug>/README.md) and writes experiments/report.html:
an app-style page with a sidebar of experiments grouped by family (a root
record plus everything whose `builds_on` chain reaches it), one report page
per record, and a rail of that page's sections. Figures and `tables/*.csv`
are inlined, so the file needs no data source. Requires `markdown` and
`pyyaml`.

    python experiments/build_report.py [--out experiments/report.html]

Exit code 1 if a record has a hole: a missing figure or table, a finding
whose Result is neither, a `### F<n>` heading the parser cannot read, a
concluded record without Hypothesis, Design, Verdict or Reproduce, or prose
over the record template's word caps (CAPS below), prose outside the rows of
Design or Verdict, a finding titled as a fix (FIX_WORDS — a fix revises the
entry it corrects, it is not an entry), a supporting finding naming no key
finding, a key finding whose Reading names no prediction, a Summary citing a
ticket or carrying more than a pair of numbers, or a concluded record whose
Verdict has no Discussion.
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

EXPERIMENTS = Path(__file__).resolve().parent
REPO_URL = None  # set in build()
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
KV = re.compile(r"^- \*\*(.+?)\*\*:\s*(.*)$")
# `#32` at line or bullet start is a ticket ref, not a heading; escape it for markdown
HASH_REF = re.compile(r"(?m)^([ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?)#(?=\d)")
REQUIRED_WHEN_CONCLUDED = {
    "hypothesis": "Hypothesis",
    "design": "Design",
    "verdict": "Verdict",
    "reproduce": "Reproduce",
}
SECTION_TITLES = {
    "hypothesis": "Question and hypothesis",
    "design": "Setup",
    "findings": "Results",
    "verdict": "Verdict",
    "decisions": "Decisions",
    "reproduce": "Reproduce",
}
PAGE_ORDER = ["hypothesis", "design", "findings", "verdict", "builds", "decisions", "reproduce"]
# word caps from the record template; Alternatives is per bullet, Design per row
CAPS = {
    "Summary": 25,
    "Reading": 240,
    "Implication": 50,
    "Decision": 60,
    "Why": 60,
    "Alternatives": 25,
    "Design": 120,
    "Discussion": 200,
}
SUMMARY_NUMBERS = 2  # a Summary carries its one number, at most a pair
TICKET = re.compile(r"(?<![\w`])#\d+\b")
NUMBER = re.compile(r"(?<![#\w.])\d[\d,.]*%?")
PREDICTION = re.compile(r"\b(predict\w*|confirm\w*|refute\w*)\b", re.I)
MINOR_LABELS = {"Summary", "Runs", "Result", "History"}
DECISION_LABELS = {"Decision", "Why", "Alternatives", "History"}
# a finding titled as a fix is a History line on the entry it corrects, not an entry
FIX_WORDS = re.compile(r"\b(bug|bugs|defect|defects|fix|fixed|fixes|rewrite|rewritten|typo|notebook)\b", re.I)


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
            else f"{f:.4f}".rstrip("0").rstrip(".")
            if len(c.split(".")[1]) > 4
            else c
        )
    return cell


def md(text: str) -> str:
    """Markdown → HTML; every table gets the report's table styling and a scroll wrapper.
    A spaced `--` in prose renders as a dash; code is left alone."""
    text = "".join(
        p if i % 2 else re.sub(r"(?<=\s)--(?=\s)", "—", p)
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
    return out.replace("<table>", '<div class="scroll"><table class="data">').replace("</table>", "</table></div>")


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


def words(text: str) -> int:
    """Prose word count: table rows, code fences and link targets do not count."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
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
        hyp = parse_kv(self.sections.get("hypothesis", ""))
        outcome = re.sub(r"<.*?>", "", hyp.get("Outcome", "")).strip().split()[:1]
        self.outcome = outcome[0].lower() if outcome and outcome[0].lower() != "open" else ""
        if self.status == "concluded":
            for key, name in REQUIRED_WHEN_CONCLUDED.items():
                if not strip_placeholders(self.sections.get(key, "")):
                    self.problems.append(f"concluded record has no ## {name}")
        self.findings = parse_entries(strip_placeholders(self.sections.get("findings", "")), "F")
        self.decisions = parse_entries(strip_placeholders(self.sections.get("decisions", "")), "D")
        for f in self.findings:
            if not f.id:
                self.problems.append(f"finding heading not `### F<n> — title · weight`: {f.title}")
        self.lint()

    def lint(self) -> None:
        """Enforce the record template's caps so the report stays scannable."""

        def over(eid: str, label: str, text: str, cap: int) -> None:
            if (n := words(text)) > cap:
                self.problems.append(f"{eid} {label} is {n} words (cap {cap})")

        keys = {f.id for f in self.findings if f.weight == "key"}
        for f in self.findings:
            fid, kv = f.id, f.kv
            if not fid:
                continue
            for label in ("Summary", "Reading", "Implication"):
                over(fid, label, kv.get(label, ""), CAPS[label])
                if refs := TICKET.findall(kv.get(label, "")):
                    self.problems.append(f"{fid} {label} cites a ticket ({', '.join(refs)}): findings read cold")
            if len(nums := NUMBER.findall(kv.get("Summary", ""))) > SUMMARY_NUMBERS:
                self.problems.append(
                    f"{fid} Summary carries {len(nums)} numbers (cap {SUMMARY_NUMBERS}): the rest belong in the Result"
                )
            if f.weight == "minor" and (extra := sorted(set(kv) - MINOR_LABELS)):
                self.problems.append(f"{fid} is minor but has {', '.join(extra)}")
            if f.weight in ("key", "supporting") and not strip_placeholders(kv.get("Reading", "")):
                self.problems.append(f"{fid} is {f.weight} but has no Reading")
            if f.weight == "key" and kv.get("Reading") and not PREDICTION.search(kv["Reading"]):
                self.problems.append(f"{fid} is key but its Reading names no prediction it confirms or refutes")
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
        for key, name in (("design", "Design"), ("verdict", "Verdict")):
            block = strip_placeholders(self.sections.get(key, ""))
            for line in outside_rows(block):
                self.problems.append(f"{name} has content outside `- **Label**:` rows: {line[:60]}")
            for label, value in parse_kv(block).items():
                over(name, label, value, CAPS.get(label, CAPS["Design"]))
        if self.status == "concluded" and not strip_placeholders(
            parse_kv(self.sections.get("verdict", "")).get("Discussion", "")
        ):
            self.problems.append("concluded record has no Verdict Discussion")

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
        return f'<table class="data{wide}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
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
        print(
            f"  {record.path.relative_to(EXPERIMENTS.parent)}: {what} has {len(rows[0])} columns; "
            "it will scroll — consider transposing or splitting it",
            file=sys.stderr,
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
    return f"<figcaption>{label}{' — ' + body if body else ''}</figcaption>"


def render_result(record: Record, value: str) -> tuple[str, bool]:
    """A finding's Result: a figure or table from figures/ or tables/, or an inline
    markdown table; prose around either becomes the caption. Returns (html, wide)."""
    # only an inlinable asset counts as the Result; a tables/*.md link is a pointer, not the result
    m = re.search(r"!?\[([^\]]*)\]\(((?:figures|tables)/[^)]+\.(?:csv|svg|png|jpe?g|gif|webp))\)", value, re.I)
    if m:
        inlined = inline_asset(record, m.group(2))
        if inlined is None:
            return f'<p class="missing">missing: {esc(m.group(2))}</p>', False
        wide = 'class="data wide"' in inlined
        prose = " ".join(l.strip() for l in value.replace(m.group(0), "").splitlines() if l.strip())
        kind = "Table" if inlined.startswith("<table") else "Figure"
        cap_html = caption(record, kind, esc(m.group(1)), md_inline(prose))
        if inlined.startswith("<table"):
            inlined = f'<div class="scroll">{inlined}</div>'
        elif not wide:
            src = Path(m.group(2))
            dark = src.with_name(f"{src.stem}.dark{src.suffix}")
            variants = f'<div class="fig-light">{inlined}</div>'
            if (record.dir / dark).is_file():
                variants += f'<div class="fig-dark">{inline_asset(record, str(dark))}</div>'
            inlined = f'<div class="fig">{variants}</div>'
        return f"<figure>{inlined}{cap_html}</figure>", wide
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


def render_kv(block: str) -> str:
    kv = parse_kv(block)
    rows = "".join(
        f'<div><h3>{esc(k)}</h3><div class="v">{md(cap(v))}</div></div>' for k, v in kv.items() if strip_placeholders(v)
    )
    return f'<div class="kv">{rows}</div>' if rows else ""


def render_finding(record: Record, f: Entry, cls: str = "", extra: str | Callable[[], str] = "") -> str:
    """One box: highlighted Summary, then Reading and Implication, the Result
    beside them; `extra` (the supporting fold) goes after the footer."""
    kv = f.kv
    result, wide = render_result(record, kv["Result"]) if kv.get("Result") else ("", False)

    def part(k: str) -> str:
        v = kv.get(k, "")
        if not strip_placeholders(v) or v.strip().lower() == "none":
            return ""
        return f'<div class="fp {k.lower()}"><span class="lbl">{k}</span>{md(cap(v))}</div>'

    reading = part("Summary") + part("Reading") + part("Implication")
    meta = footer(record, kv)
    weight = f.weight + (f" {f.target}" if f.target else "")
    classes = " ".join(c for c in ("res", cls, "wide" if wide or not result else "") if c)
    return (
        f'<article class="{classes}" id="{record.slug}/{f.id.lower()}" data-title="{esc(f.id)} · {esc(re.sub("<.*?>", "", md_inline(f.title)))}">'
        f'<h3><span class="tag">{f.id}</span>{md_inline(f.title)}<span class="wt">{esc(weight)}</span></h3>'
        f'<div class="read">{reading}</div>{result}{meta}{extra() if callable(extra) else extra}</article>'
    )


def render_findings(record: Record) -> str:
    """Key findings as boxes, each with its supporting findings folded beneath it;
    minor findings and any supporting finding without a key target follow."""
    out, placed = [], set()
    for f in record.findings:
        if f.weight != "key" or not f.id:
            continue
        subs = [s for s in record.findings if s.id and s.weight == "supporting" and s.target == f.id]
        placed.update(s.id for s in subs)

        def fold(subs: list[Entry] = subs) -> str:
            if not subs:
                return ""
            return (
                f'<details class="support"><summary>supporting evidence ({len(subs)})</summary>'
                f"{''.join(render_finding(record, s, 'sub') for s in subs)}</details>"
            )

        out.append(render_finding(record, f, "key", fold))
    out += [
        render_finding(record, f, f.weight or "solo")
        for f in record.findings
        if f.id and f.weight != "key" and f.id not in placed
    ]
    return "".join(out)


FILE_EXT = re.compile(r"\.(ya?ml|py|csv|json|svg|png|md|txt|toml)$")


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
        rows.append(f"<p><b>runs</b><span>{mark_refs(record, md_inline(kv['Runs']), runs=True)}</span></p>")
    if kv.get("History"):
        last = kv["History"].splitlines()[-1].lstrip("- ").strip()
        rows.append(f"<p><b>history</b><span>{mark_refs(record, md_inline(last))}</span></p>")
    return f'<div class="src">{"".join(rows)}</div>' if rows else ""


def render_decisions(record: Record) -> str:
    """Decision line open; Why, Alternatives and the rest fold; latest History as footer."""
    out = []
    for d in record.decisions:
        did, title, kv = d.id, d.title, d.kv
        decision = md(cap(kv["Decision"])) if kv.get("Decision") else ""
        rest = "".join(
            f'<div class="v"><b>{esc(k)}.</b> {md_inline(cap(v))}</div>'
            for k, v in kv.items()
            if k not in ("Decision", "History")
        )
        fold = f"<details><summary>why and alternatives</summary>{rest}</details>" if rest else ""
        foot = footer(record, {"History": kv.get("History", "")})
        tag = f'<span class="tag">{did}</span>' if did else ""
        out.append(
            f'<article class="dec" id="{{slug}}/{did.lower()}"><h3>{tag}{md_inline(title)}</h3>'
            f'<div class="decision">{decision}</div>{fold}{foot}</article>'
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
        out.append(f'<li class="step{" env" if env else ""}">{head}<pre><code>{body}</code></pre></li>')
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
        return f'<a href="{esc(pr)}">PR</a>'
    n = pr.lstrip("#")
    if n.isdigit() and REPO_URL and "github.com" in REPO_URL:
        return f'<a href="{esc(REPO_URL)}/pull/{n}">PR #{n}</a>'
    return esc(f"PR #{n}" if n.isdigit() else pr)


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<.*?>", "", md_inline(text)))).strip()


def tip(head: str, body: str) -> str:
    """data-tip value: a plain heading line, then the body as inline HTML so code
    and emphasis render in the hover card."""
    inline = re.sub(r"\s+", " ", md_inline(body)).strip()
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
    """Bracketed `[D<n>]` / `[F<n>]` citations in prose become in-page links carrying
    a hover card — the decision's Decision line, the finding's Summary. A bare
    `F1` stays text: it may be the metric. Text inside code, links, headings and
    id tags is left alone."""
    tips = {d.id: tip(f"{d.id} — {d.title}", d.kv.get("Decision", "")) for d in record.decisions if d.id}
    tips |= {f.id: tip(f"{f.id} — {f.title}", f.kv.get("Summary", "")) for f in record.findings if f.id}
    if not tips:
        return html_
    ref = re.compile(r"\[([DF]\d+)\]")

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
    """Abbreviation-shaped tokens in record prose that the glossary does not define."""
    known = {f.lower() for t in terms for f in t.forms} | {k.lower() for k in KNOWN_ABBREV}
    hits: set[str] = set()
    for r in records:
        for key in ("hypothesis", "design", "findings", "verdict"):
            text = re.sub(
                r"```.*?```|`[^`]*`|\[[DF]\d+\]|!?\[[^\]]*\]\([^)]*\)", " ", r.sections.get(key, ""), flags=re.S
            )
            text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(("|", "###")))
            hits |= {w for w in ABBREV.findall(text) if w.lower() not in known and not re.fullmatch(r"[DF]\d+", w)}
    return sorted(hits)


def render_glossary(terms: list[Term]) -> str:
    """One page: each `## ` section of GLOSSARY.md as a heading, its terms as a list."""
    if not terms:
        return ""
    parts, section = (
        [
            '<section class="page" id="glossary" data-title="Glossary"><p class="kicker">experiments/GLOSSARY.md</p><h1>Glossary</h1>'
        ],
        None,
    )
    for t in terms:
        if t.section != section:
            if section is not None:
                parts.append("</dl>")
            section = t.section
            sid = re.sub(r"[^\w]+", "-", section.lower()).strip("-")
            parts.append(f'<h2 id="glossary/{sid}">{esc(section)}</h2><dl class="gloss">')
        also = f' <span class="also">{esc(", ".join(t.aliases))}</span>' if t.aliases else ""
        parts.append(f'<dt id="glossary/{t.slug}">{esc(t.name)}{also}</dt><dd>{md_inline(t.definition)}</dd>')
    parts.append("</dl></section>")
    return "".join(parts)


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
    return re.sub(r'<a href="(https?://[^"]*)"', r'<a href="\1" target="_blank" rel="noopener"', page)


def render_page(record: Record, parents: list[Record], terms: list[Term]) -> str:
    kicker = [f"{esc(record.category)} / {esc(record.slug)}", esc(record.meta.get("branch", "")), badge(record)]
    if record.meta.get("concluded"):
        kicker.append(f"concluded {esc(record.meta['concluded'])}")
    kicker.append(pr_link(record))
    for p in parents:
        kicker.append(f'builds on <a href="#{p.slug}">{esc(p.title)}</a>')
    lede = "".join(
        f'<li><a href="#{record.slug}/{f.id.lower()}"><span class="tag">{f.id}</span> {md_inline(f.title)}</a>'
        f'<span class="s">{md_inline(cap(f.kv.get("Summary", "")))}</span></li>'
        for f in record.key_findings()
    ) or "".join(f"<li>{md_inline(l)}</li>" for l in record.key_lines())
    hyp_status = parse_kv(record.sections.get("tldr", "")).get("Hypothesis", "")
    parts = [f'<p class="kicker">{" · ".join(k for k in kicker if k)}</p><h1>{esc(record.title)}</h1>']
    if hyp_status:
        parts.append(f'<p class="lede">{md_inline(cap(hyp_status))}</p>')
    if lede:
        parts.append(f'<ul class="lede">{lede}</ul>')
    for key in PAGE_ORDER:
        if key == "builds":
            body = render_builds(record)
            title = "Builds on this"
        else:
            block = strip_placeholders(record.sections.get(key, ""))
            if not block:
                continue
            title = SECTION_TITLES[key]
            if key == "findings":
                body = render_findings(record)
                if record.status != "concluded":
                    title = "Results so far"
            elif key == "decisions":
                body = render_decisions(record).replace("{slug}", record.slug)
            elif key == "reproduce":
                body = render_reproduce(record, block)
            else:
                body = render_kv(block) or f'<div class="prose">{md(block)}</div>'
        if not body:
            continue
        parts.append(f'<h2 id="{record.slug}/{key}">{title}</h2>{body}')
    return f'<section class="page" id="{record.slug}" data-title="{esc(record.title)}">{link_terms(link_refs(record, "".join(parts)), terms)}</section>'


def render_overview(fams: list[list[tuple[Record, int]]], terms: list[Term]) -> str:
    book = EXPERIMENTS / "README.md"
    intro = md(book.read_text(encoding="utf-8")) if book.is_file() else ""
    # only the book's intro: its chapters and table repeat what the family lists below show
    intro = re.sub(r"<h1>.*?</h1>", "", intro, count=1, flags=re.S)
    intro = intro.split("<h2>", 1)[0]
    intro = re.sub(r'<div class="scroll"><table class="data[^"]*">.*?</table></div>', "", intro, count=1, flags=re.S)
    rows = []
    for fam in fams:
        rows.append(f'<h2 id="overview/{fam[0][0].slug}">{esc(fam[0][0].title)}</h2><div class="list">')
        for r, depth in fam:
            when = r.meta.get("concluded") or r.status
            lines = "".join(
                f'<li><span class="tag">{f.id}</span> {md_inline(f.title)}</li>' for f in r.key_findings()
            ) or "".join(f"<li>{md_inline(l)}</li>" for l in r.key_lines())
            rows.append(
                f'<a class="row d{min(depth, 3)}" href="#{r.slug}"><span class="who"><span class="t">{esc(r.title)}</span>'
                f'<span class="m">{badge(r)}<span>{esc(when)}</span></span></span><ul>{lines}</ul></a>'
            )
        rows.append("</div>")
    return link_terms(
        f'<section class="page" id="overview" data-title="Overview"><p class="kicker">experiments/README.md</p><h1>Findings</h1><div class="book">{intro}</div>{"".join(rows)}</section>',
        terms,
    )


def render_sidebar(fams: list[list[tuple[Record, int]]], terms: list[Term]) -> str:
    out = ['<a class="item" href="#overview" data-page="overview">Overview</a>']
    for fam in fams:
        out.append(f'<p class="h">{esc(fam[0][0].title)}</p>')
        for r, depth in fam:
            dot = f"d-{r.outcome}" if r.outcome else ("d-done" if r.status == "concluded" else "d-open")
            out.append(
                f'<a class="item d{min(depth, 3)}" href="#{r.slug}" data-page="{r.slug}">'
                f'<span class="dot {dot}"></span>{esc(r.title)}<span class="sub">{esc(r.category)}</span></a>'
            )
    if terms:
        out.append('<p class="h">Reference</p><a class="item" href="#glossary" data-page="glossary">Glossary</a>')
    return "".join(out)


CSS = """
:root{--bg:#dde1e7;--panel:#e8ebf0;--line:#c8ced7;--fg:#1f252f;--muted:#5a6472;--accent:#3350c4;--accent-soft:#d8def4;--soft:#d5dae2;--ok:#1e7a45;--ok-soft:#d5eadc;--bad:#a52c24;--bad-soft:#f0dad8;--run:#9a5b00;--run-soft:#efe1c4;--plate:#eef0f4;--t-h1:32px;--t-h2:20px;--t-lead:17px;--t-h3:16px;--t-body:15px;--t-sm:13.5px;--t-mono:12px;--t-xs:11px;--t-micro:9px;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace;--sans:"Instrument Sans",system-ui,sans-serif}
:root[data-theme=dark]{--bg:#121418;--panel:#191c22;--line:#2a2f38;--fg:#e6e8ee;--muted:#8b93a3;--accent:#8b9cff;--accent-soft:#232a4d;--soft:#262b35;--ok:#4ade80;--ok-soft:#153524;--bad:#f87171;--bad-soft:#3b1a18;--run:#fbbf24;--run-soft:#3a2d0c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121418;--panel:#191c22;--line:#2a2f38;--fg:#e6e8ee;--muted:#8b93a3;--accent:#8b9cff;--accent-soft:#232a4d;--soft:#262b35;--ok:#4ade80;--ok-soft:#153524;--bad:#f87171;--bad-soft:#3b1a18;--run:#fbbf24;--run-soft:#3a2d0c}}
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
.fp{margin:0 0 10px}.fp .lbl{display:block;font-size:var(--t-xs);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}.fp.summary{font-size:var(--t-sm);background:var(--soft);padding:10px 12px;border-radius:8px;margin:0 0 12px}.fp p{margin:0 0 6px}.fp p:last-child{margin-bottom:0}
.dec{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:12px 0;scroll-margin-top:20px}.dec h3{display:flex;gap:10px;align-items:center;font-size:var(--t-body);margin:0 0 8px}.dec .decision p{margin:0 0 6px}.dec details{border:0;padding:0;margin:6px 0 0}.dec summary{color:var(--muted)}.dec .src{margin:8px 0 0}
.res>.read,.res>figure{min-width:0}.res>.read{grid-column:1;grid-row:2;font-size:var(--t-sm);line-height:1.5}.res .read p{margin:0 0 8px}.res>figure{grid-column:2;grid-row:2;margin:0}.res>.src{grid-column:1/-1;margin:4px 0 0}.src{font-family:var(--mono);font-size:var(--t-mono);color:var(--muted)}.src p{display:grid;grid-template-columns:58px 1fr;gap:0 6px;margin:0 0 3px}.src b{font-size:var(--t-xs);font-weight:600;letter-spacing:.06em;text-transform:uppercase;line-height:inherit;padding-top:1px}.src a code{color:var(--accent)}
.src code.run::before,.src code.commit::before,.src code.file::before{font-size:var(--t-micro);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:5px}.src code.run::before{content:"run"}.src code.commit::before{content:"commit"}.src code.file::before{content:"file"}
.res>details.support{grid-column:1/-1;border:0;padding:0;margin:2px 0 0}.res .support>summary{color:var(--muted);font-size:var(--t-mono)}
.res.sub{margin:10px 0 0;padding:14px 18px;background:transparent;border:0;border-left:2px solid var(--line);border-radius:0}.res.sub h3{font-size:var(--t-h3)}.res.minor{padding:12px 20px}.res.minor h3{font-size:var(--t-body)}
.res.wide{grid-template-columns:minmax(0,1fr)}.res.wide>.read{grid-column:1;font-size:var(--t-sm)}.res.wide>figure{grid-column:1;grid-row:3}
figure svg,figure img{width:100%;height:auto;display:block}.res>figure .fig{cursor:zoom-in}.res.wide>figure .fig{cursor:zoom-out}
/* analysis.py writes both variants, so each theme gets colours picked for its own
   ground rather than a hue-rotated flip. A figure that has only the light file
   keeps its own colours on a plate; a raster carries a background, so multiply
   drops it against either surface. */
.fig-dark{display:none}.fig-light img{mix-blend-mode:multiply}.fig-dark img{mix-blend-mode:screen}
:root[data-theme=dark] .fig-light:not(:only-child){display:none}:root[data-theme=dark] .fig-dark{display:block}:root[data-theme=dark] .fig-light:only-child{background:var(--plate);border-radius:10px;padding:12px 14px}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .fig-light:not(:only-child){display:none}:root:not([data-theme=light]) .fig-dark{display:block}:root:not([data-theme=light]) .fig-light:only-child{background:var(--plate);border-radius:10px;padding:12px 14px}}
figcaption{font-size:var(--t-mono);color:var(--muted);margin-top:8px;line-height:1.5}figcaption .lbl{color:var(--fg);font-weight:600}
.scroll{overflow-x:auto;max-width:100%;margin:6px 0 10px;scrollbar-width:thin;scrollbar-color:var(--line) transparent;padding-bottom:2px}.scroll::-webkit-scrollbar{height:8px}.scroll::-webkit-scrollbar-track{background:transparent}.scroll::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}.scroll.cut{-webkit-mask-image:linear-gradient(to right,#000 calc(100% - 44px),transparent);mask-image:linear-gradient(to right,#000 calc(100% - 44px),transparent)}figure .scroll{margin:0}table.data{border-collapse:collapse;width:100%;font-size:var(--t-sm);line-height:1.45;font-variant-numeric:tabular-nums}table.data th{text-align:left;font-size:var(--t-xs);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:bottom}table.data td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}table.data td.id{font-family:var(--mono);font-size:var(--t-mono);white-space:nowrap}table.data td:first-child,table.data th:first-child{position:sticky;left:0;background:inherit;font-weight:500}table.data.wide td:first-child,table.data.wide th:first-child{border-right:1px solid var(--line)}table.data tr{background:var(--panel)}.res.sub table.data tr,.prose table.data tr,.v table.data tr{background:var(--bg)}table.data th.n{text-align:right}table.data td.n{text-align:right;font-family:var(--mono);font-size:var(--t-mono);white-space:nowrap}table.data code{overflow-wrap:anywhere}
.prose h3,.v h3{font-size:var(--t-body);font-weight:600;margin:36px 0 10px}.prose h4,.v h4{font-size:var(--t-mono);font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:24px 0 8px}.prose>:first-child{margin-top:0}.prose p,.prose ul,.prose ol{margin:0 0 10px}
.missing{color:var(--bad);font-family:var(--mono);font-size:var(--t-mono)}
.steps{list-style:none;padding:0;margin:8px 0 12px;counter-reset:step}.step{display:grid;grid-template-columns:30px minmax(0,1fr);gap:6px 12px;align-items:start;margin:0 0 18px}
.step::before{grid-row:1/3;width:26px;height:26px;border-radius:50%;background:var(--soft);color:var(--muted);font-family:var(--mono);font-size:var(--t-xs);font-weight:600;display:flex;align-items:center;justify-content:center;content:"env";margin-top:1px}
.step:not(.env)::before{counter-increment:step;content:counter(step);background:var(--accent-soft);color:var(--accent)}
.step .t{grid-column:2;font-size:var(--t-sm);font-weight:600;line-height:1.5;padding-top:3px}.step pre{grid-column:2;margin:0}.step .t code{font-size:.85em}
pre .c{color:#8a93a5}
a.ref{color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)}a.term{color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted)}a.term:hover{color:var(--accent);border-bottom-color:var(--accent)}dl.gloss{margin:0 0 24px;display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:8px 24px;align-items:baseline}dl.gloss dt{font-weight:600;scroll-margin-top:20px}dl.gloss dt .also{display:block;font-weight:400;font-size:var(--t-mono);color:var(--muted);font-family:var(--mono)}dl.gloss dd{margin:0;color:var(--fg)}
.tip{position:fixed;z-index:9;max-width:400px;background:var(--fg);color:var(--bg);font-size:var(--t-mono);line-height:1.45;padding:8px 11px;border-radius:8px;white-space:pre-line;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.25)}.tip b{display:block;margin-bottom:2px}.tip code{background:color-mix(in srgb,var(--bg) 18%,transparent);color:inherit}
code{font-family:var(--mono);font-size:.88em;background:var(--soft);padding:1px 5px;border-radius:4px}.fp.summary code{background:color-mix(in srgb,var(--fg) 10%,transparent)}pre{background:#16181d;color:#e7e9ee;padding:14px 16px;border-radius:10px;font-family:var(--mono);font-size:var(--t-mono);line-height:1.55;overflow:auto}pre code{background:none;padding:0;color:inherit}
.rel{display:flex;flex-direction:column;gap:10px}.rel a{display:flex;flex-direction:column;gap:3px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;color:var(--fg)}.rel a:hover{border-color:var(--accent)}.rel .t{font-weight:600}.rel .m{color:var(--muted)}
details{border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin:8px 0}summary{cursor:pointer;font-family:var(--mono);font-size:var(--t-sm)}details .v{margin:8px 0 0}
.right{border-left:1px solid var(--line);padding:24px 16px;font-size:var(--t-sm);display:flex;flex-direction:column;min-height:0}.right .toc{flex:1;overflow:auto}.right .keys{border-top:1px solid var(--line);margin-top:16px;padding-top:14px}.keys div{display:flex;gap:8px;align-items:center;color:var(--muted);padding:3px 10px}kbd{font-family:var(--mono);font-size:var(--t-xs);line-height:1.6;min-width:18px;text-align:center;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--bg);color:var(--fg)}.right .h{font-size:var(--t-xs);font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;padding-left:10px}
.right a{display:block;color:var(--muted);padding:5px 10px;border-left:2px solid transparent;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.right a:hover{color:var(--fg)}.right a.sub{padding-left:22px;font-size:var(--t-mono)}.right a.on{color:var(--accent);border-left-color:var(--accent)}
.book{color:var(--muted);font-size:var(--t-h3)}.book h2{color:var(--fg)}.list{display:flex;flex-direction:column}.row{display:grid;grid-template-columns:minmax(200px,1fr) minmax(0,3fr);gap:24px;padding:12px 0;border-bottom:1px solid var(--line);color:var(--fg);align-items:baseline}.row:hover .t{color:var(--accent)}.row .who{display:flex;flex-direction:column;gap:4px}.row .t{font-weight:600}.row .m{font-size:var(--t-mono);color:var(--muted);font-family:var(--mono);white-space:nowrap}.row ul{margin:0;padding-left:16px;font-size:var(--t-sm)}.row.d1 .who{padding-left:18px}.row.d2 .who{padding-left:36px}.row.d3 .who{padding-left:54px}
@media(max-width:900px){.app{grid-template-columns:1fr;grid-template-rows:52px auto minmax(0,1fr)}.side{max-height:200px;border-right:0;border-bottom:1px solid var(--line)}.right{display:none}.main{padding:24px}}
@media print{:root{--bg:#fff;--panel:#fff;--plate:#fff}.app{display:block;height:auto}.top,.side,.right{display:none}.main{overflow:visible;padding:0;background:#fff}.fig-dark{display:none!important}.fig-light{display:block!important;background:none!important;padding:0!important}pre{background:#f4f5f7;color:#16181d;border:1px solid #d8dce3}pre .c{color:#6b7280}.page{display:block!important;max-width:none;break-before:page}h2{break-after:avoid}.res,details,.rel a{break-inside:avoid}a{color:inherit}body{font-size:11pt}}
"""

JS = """
const pages=[...document.querySelectorAll('.page')],items=[...document.querySelectorAll('.side .item')],rail=document.querySelector('.right .toc'),crumb=document.querySelector('.crumb');
function route(){const [id,sub]=location.hash.slice(1).split('/');const page=document.getElementById(id)||pages[0];
 pages.forEach(p=>p.classList.toggle('on',p===page));items.forEach(i=>i.classList.toggle('on',i.dataset.page===page.id));
 const fam=[...document.querySelectorAll('.side .h')].reverse().find(h=>h.compareDocumentPosition(items.find(i=>i.dataset.page===page.id))&Node.DOCUMENT_POSITION_FOLLOWING);
 crumb.innerHTML=(page.id==='overview'?'':'<span>'+(fam?fam.textContent:'')+'</span><span>›</span>')+'<b>'+page.dataset.title+'</b>';
 rail.innerHTML='<p class="h">On this page</p>'+[...page.querySelectorAll('h2, .res:not(.sub)')].map(h=>h.tagName==='H2'?'<a href="#'+h.id+'">'+h.textContent+'</a>':'<a class="sub" href="#'+h.id+'" title="'+h.dataset.title+'">'+h.dataset.title+'</a>').join('');
 const target=sub?document.getElementById(id+'/'+sub):null;for(let d=target&&target.closest('details');d;d=d.parentElement.closest('details'))d.open=true;(target||document.querySelector('.main')).scrollIntoView?.({block:'start'});if(!target)document.querySelector('.main').scrollTop=0;spy();cut();}
function spy(){const page=document.querySelector('.page.on');if(!page)return;const top=document.querySelector('.main').getBoundingClientRect().top+80;
 const marks=[...page.querySelectorAll('h2, .res:not(.sub)')];let cur=marks[0];for(const m of marks){if(m.getBoundingClientRect().top<=top)cur=m;else break;}
 rail.querySelectorAll('a').forEach(a=>a.classList.toggle('on',cur&&a.getAttribute('href')==='#'+cur.id));}
const scrollers=[...document.querySelectorAll('.scroll')];
function cut(){for(const el of scrollers){const max=el.scrollWidth-el.clientWidth;el.classList.toggle('cut',max>1&&el.scrollLeft<max-1);}}
scrollers.forEach(el=>el.addEventListener('scroll',cut,{passive:true}));addEventListener('resize',cut);
addEventListener('hashchange',route);route();document.querySelector('.main').addEventListener('scroll',spy,{passive:true});
document.querySelectorAll('.res .fig').forEach(f=>f.onclick=()=>f.closest('.res').classList.toggle('wide'));
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
let navOn=true;try{navOn=localStorage.getItem('report-nav')!=='0'}catch(e){}setNav(navOn);nav.onclick=()=>setNav(app.classList.contains('nonav'));
let back='';addEventListener('keydown',e=>{if(e.target.matches('input,textarea')||e.metaKey||e.ctrlKey||e.altKey)return;
 const onGloss=location.hash.startsWith('#glossary'),gloss=document.getElementById('glossary'),page=document.querySelector('.page.on');
 const go=h=>{location.hash=h;e.preventDefault();};
 if(e.key==='?'&&gloss&&!onGloss){back=location.hash;go('#glossary');}
 else if((e.key==='?'||e.key==='Escape')&&onGloss)go(back||'#overview');
 else if(e.key==='['||e.key===']'){const i=items.findIndex(x=>x.classList.contains('on')),n=i+(e.key===']'?1:-1);if(items[n])go(items[n].getAttribute('href'));}
 else if(e.key==='o')go('#overview');
 else if(e.key==='e'&&page){const ds=[...page.querySelectorAll('details')],open=ds.some(d=>!d.open);ds.forEach(d=>d.open=open);}
 else if(e.key==='n')nav.onclick();
 else if(e.key==='t')btn.onclick();});
"""


def build(out: Path) -> int:
    global REPO_URL
    REPO_URL = repo_url()
    records = load_records()
    terms = load_glossary()
    fams = families(records)
    by_dir = {r.dir.resolve(): r for r in records}
    pages = (
        [render_overview(fams, terms)]
        + [render_page(r, [by_dir[p] for p in r.builds_on if p in by_dir], terms) for fam in fams for r, _ in fam]
        + [render_glossary(terms)]
    )
    repo = EXPERIMENTS.parent.name
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(repo)} · Findings</title>"
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">'
        f'<style>{CSS}</style></head><body><div class="app">'
        f'<div class="top"><button class="nav" title="toggle experiment list">☰</button><span class="brand">{esc(repo)} · Findings</span><span class="crumb"></span><button class="theme">theme</button></div>'
        f'<nav class="side">{render_sidebar(fams, terms)}</nav>'
        f'<main class="main">{"".join(pages)}</main>'
        '<aside class="right"><div class="toc"></div>'
        + '<div class="keys"><p class="h">Shortcuts</p>'
        + ("<div><kbd>?</kbd> glossary</div><div><kbd>esc</kbd> back</div>" if terms else "")
        + "<div><kbd>[</kbd><kbd>]</kbd> previous / next experiment</div><div><kbd>o</kbd> overview</div><div><kbd>e</kbd> expand / collapse folds</div>"
        + "<div><kbd>n</kbd> experiment list</div><div><kbd>t</kbd> theme</div></div>"
        + "</aside></div>"
        f"<script>{JS}</script></body></html>"
    )
    out.write_text(relink(page, records), encoding="utf-8")
    problems = [(r, p) for r in records for p in r.problems]
    print(
        f"{out}: {len(records)} records in {len(fams)} families, {len(terms)} glossary terms, {out.stat().st_size // 1024} KiB"
    )
    for r, p in problems:
        print(f"  {r.path.relative_to(EXPERIMENTS.parent)}: {p}", file=sys.stderr)
    if records and not terms:
        print(
            "  experiments/GLOSSARY.md missing: seed it from the report skill's references/glossary.md", file=sys.stderr
        )
        problems.append(("glossary", "missing"))
    if undefined := undefined_terms(records, terms):
        print(f"  glossary: abbreviations without a definition: {', '.join(undefined)}", file=sys.stderr)
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
        "no heading made from a #<ticket> line": len(re.findall(r"<h1>", body)) == 4,
        "no raw markdown table": not re.search(r">[^<]*\|[^<]*\|[^<]*\|", body),
        "every table styled and scrollable": "<table>" not in body,
        "no relative link survives": not re.search(r'href="(?!#|https?://)', body),
        "bullet-starting ticket ref kept in a list": "<li>#37, the post-hoc" in body,
        "sub-list under a label rendered as a list": "<li>audit <code>de</code>" in body,
        "float formatted": "0.6047</td>" in body and "0.6047136829414016" not in body,
        "svg ids prefixed": 'id="alpha-a-' in body and 'id="axes_1"' not in body,
        "PR link built from the remote or left as text": "PR #62" in body,
        "decision open, rest folded": '<div class="decision"><p>Pool it' in body and "why and alternatives" in body,
        "[D]/[F] citations linked with a hover card": '<a class="ref" href="#alpha/d1" data-tip="D1 — Pick the pool\npool it, so #33 proceeds.">[D1]</a>'
        in body
        and '<a class="ref" href="#alpha/f1"' in body
        and 'href="#alpha/d9"' not in body
        and "[D9]" in body,
        "bare F1 (the metric) and refs inside code or headings stay text": "macro F1 0.6047" in body
        and '<a class="ref" href="#beta/f1"' not in body
        and "<code>[D1]</code>" in body
        and "Pick the pool</h3>" in body,
        "glossary page, sidebar entry and shortcut": '<a class="item" href="#glossary" data-page="glossary">Glossary</a>'
        in body
        and '<div class="keys"><p class="h">Shortcuts</p><div><kbd>?</kbd> glossary</div>' in body
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
        "undefined abbreviations reported": undefined_terms(records, load_glossary()) == ["AUC"],
        "reproduce rendered as steps": '<ol class="steps"><li class="step env">' in body
        and '<li class="step"><div class="t">' in body
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
        "overview lists each record once": overview.count('class="row d0" href="#alpha"') == 1
        and "<h2>data</h2>" not in overview,
        "lede and overview list key finding titles": '<span class="tag">F1</span> First thing</a>' in body
        and '<span class="tag">F1</span> First thing</li>' in overview,
        "row values start with a capital": "<p>Inconclusive — see" in body
        and "<p>Reading text; it confirms" in body
        and '<div class="decision"><p>Pool it' in body,
        "reading and implication visible": '<div class="fp reading"><span class="lbl">Reading</span>' in body
        and "what it means" not in body,
        "caps linted": {"F5 is minor but has Reading", "D2 Why is 68 words (cap 60)", "F1 Summary is 29 words (cap 25)"}
        <= set(problems),
        "fix-as-finding rejected": "F6 reads as a fix (defect, fixed, notebook): revise the F<n> it corrects instead"
        in problems,
        "supporting folded under its key finding": re.search(
            r'id="alpha/f1".*?<summary>supporting evidence \(1\)</summary><article class="res sub[^"]*" id="alpha/f3"',
            body,
            re.S,
        )
        is not None,
        "prose outside Design rows flagged": "Design has content outside `- **Label**:` rows: #### Notes" in problems,
        "ticket ref in a Summary flagged": "F1 Summary cites a ticket (#32): findings read cold" in problems,
        "key finding without a prediction flagged": "F4 is key but its Reading names no prediction it confirms or refutes"
        in problems,
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
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else build(args.out))
