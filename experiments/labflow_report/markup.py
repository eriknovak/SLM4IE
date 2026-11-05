"""Record prose and files as HTML: markdown, equations, and inlined figures and tables.

The page views hand every piece of record prose through `markdown_html` or
`inline_html`, and every figure or table file through `inline_asset`, so the
new builder never reaches into `build_report.py`. Equations become MathML
with `latex2mathml`; without it they stay as their LaTeX source in code.
"""

from __future__ import annotations

import base64
import csv
import html
import re
from pathlib import Path

import markdown

# ===============================
# Check dependencies
# ===============================

try:
    from latex2mathml.converter import convert as latex_to_mathml
except ModuleNotFoundError:
    latex_to_mathml = None

# ===============================
# Defaults
# ===============================

MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\w$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])", re.S)
MATH_TOKEN = re.compile(r"mathx(\d+)x")
CODE_SPANS = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)
# a `#12` at a line's start is a ticket, not a heading
HASH_REF = re.compile(r"(?m)^([ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?)#(?=\d)")
SPACED_DASH = re.compile(r"(?<=\s)--(?=\s)")
NUMERIC = re.compile(r"^[-+−]?[\d,.]+%?$|^—$")
FLOAT = re.compile(r"^[-+]?\d+\.\d+$")
HEX_ID = re.compile(r"^[0-9a-f]{16,}$")
# a table longer than this scrolls under a sticky header
TABLE_ROWS = 20
# a table with more columns, or a cell longer than this, takes the full width
WIDE_COLUMNS = 3
WIDE_CELL = 60
# decimals a long float cell keeps
DECIMALS = 4
# a whole number this long reads with thousands separators
GROUPED_DIGITS = 5
WHOLE = re.compile(r"^-?\d+$")
# words a column header writes in their own case, and units it puts in brackets
HEADER_WORDS = {
    "f1": "F1",
    "f2": "F2",
    "cpu": "CPU",
    "gpu": "GPU",
    "ci": "CI",
    "url": "URL",
    "mlflow": "MLflow",
    "kpi": "KPI",
    "llm": "LLM",
    "api": "API",
    "auc": "AUC",
    "mae": "MAE",
    "rmse": "RMSE",
    "bpe": "BPE",
    "ner": "NER",
}
HEADER_UNITS = {"s": "s", "ms": "ms", "min": "min", "gb": "GB", "mb": "MB", "pct": "%"}
# a word with a digit, an inner capital or a hyphen and a digit is a name
NAME_WORD = re.compile(r"\d|^[a-z]+[A-Z]")
CLAUSE_WORD = re.compile(r"^[hcq]\d+$", re.I)

ASSET_LINK = re.compile(
    r"!?\[([^\]]*)\]\(((?:figures|tables)/[^)]+\.(?:csv|svg|png|jpe?g|gif|webp))\)",
    re.I,
)
IMAGE_TYPES = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
}
XML_PROLOGUE = re.compile(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>")
XML_COMMENT = re.compile(r"<!--.*?-->", re.S)
# matplotlib paints the figure and each axes white; the page gives the ground
FIGURE_PATCH = re.compile(
    r'(<g id="(?:figure|axes)_\d+">\s*<g id="patch_\d+">\s*<path[^>]*?fill: )#ffffff'
)
SENTENCE = re.compile(r"(.+?[.!?])(?=\s|$)", re.S)
TAG = re.compile(r"<[^>]+>")
MTEXT = re.compile(r"<mtext([^>]*)>(.*?)</mtext>", re.S)

# ===============================
# Main functions
# ===============================


def markdown_html(text: str) -> str:
    """Renders record markdown, with tables styled, equations as MathML and `--` as a dash.

    Args:
        text: The markdown, as a record row or section holds it.

    Returns:
        The HTML; every table is a `table.data` in a `div.scroll`.
    """
    if not text or not text.strip():
        return ""

    # set equations aside so markdown never reads their underscores
    equations: list[str] = []
    parts = CODE_SPANS.split(text)
    text = "".join(
        part
        if i % 2
        else MATH.sub(lambda m: _stash_math(m, equations), SPACED_DASH.sub("—", part))
        for i, part in enumerate(parts)
    )
    text = HASH_REF.sub(r"\1\\#", text)

    # markdown needs a blank line where a table or list starts or ends after prose
    text = re.sub(r"(?m)^(?![ \t]*\|)(\S.*)\n(?=[ \t]*\|)", r"\1\n\n", text)
    text = re.sub(r"(?m)^([ \t]*\|.*)\n(?=[ \t]*[^|\s])", r"\1\n\n", text)
    text = re.sub(
        r"(?m)^(?![ \t]*(?:[-*] |\d+\. ))(\S.*)\n(?=[ \t]*(?:[-*] |\d+\. ))",
        r"\1\n\n",
        text,
    )
    out = markdown.markdown(text, extensions=["tables", "fenced_code"])

    # style the tables and put the equations back
    out = re.sub(
        r"<td([^>]*)>([^<]*)</td>",
        lambda m: (
            f'<td class="n"{m.group(1)}>{number(m.group(2))}</td>'
            if NUMERIC.match(m.group(2).strip())
            else m.group(0)
        ),
        out,
    )
    out = re.sub(r"<table>(.*?)</table>", _wrap_table, out, flags=re.S)
    return MATH_TOKEN.sub(lambda m: equations[int(m.group(1))], out)


def inline_html(text: str) -> str:
    """`markdown_html` without the paragraph around a single paragraph."""
    out = markdown_html(text).strip()
    if out.count("<p>") == 1 and out.startswith("<p>") and out.endswith("</p>"):
        return out[3:-4]
    return out


def plain_text(html_: str) -> str:
    """The text of an HTML fragment, its whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(TAG.sub("", html_))).strip()


def capitalize(text: str) -> str:
    """A row value as a sentence: its first lowercase letter upper-cased.

    A first word that is a name (`gpt-oss-20b`, `mmBERT`, `e5-small`) keeps
    its case.
    """
    stripped = text.lstrip()
    first = stripped.split(" ", 1)[0] if stripped else ""
    if NAME_WORD.search(first):
        return text
    if stripped and stripped[0].islower():
        return stripped[0].upper() + stripped[1:]
    return text


def first_sentence(text: str) -> str:
    """The first sentence of a text, for a slot that quotes a row in one line."""
    match = SENTENCE.match(text.strip())
    return (match.group(1) if match else text).strip()


def column_header(key: str) -> str:
    """A table column's name as a reader reads it.

    `MACRO_F1` reads `Macro F1`, `p95_8_cpu_threads_s` reads
    `P95 8 CPU threads (s)`; a name the author wrote with spaces keeps its
    case and only gains a capital.
    """
    text = key.strip()
    if " " in text and "_" not in text and not text.isupper():
        return capitalize(text)
    words = [w for w in re.split(r"[_\s]+", text) if w]
    shown = []
    for i, word in enumerate(words):
        low = word.lower()
        if i == len(words) - 1 and i and low in HEADER_UNITS:
            shown.append(f"({HEADER_UNITS[low]})")
        elif low in HEADER_WORDS:
            shown.append(HEADER_WORDS[low])
        elif CLAUSE_WORD.match(word):
            shown.append(word.upper())
        else:
            shown.append(low if text.isupper() or word.islower() else word)
    return capitalize(" ".join(shown))


def number(cell: str) -> str:
    """A numeric cell for display: a long float to four decimals, `750.0` to `750`, `5246773` to `5,246,773`."""
    value = cell.strip()
    if WHOLE.match(value) and len(value.lstrip("-")) >= GROUPED_DIGITS:
        return f"{int(value):,}"
    if not FLOAT.match(value):
        return cell
    parsed = float(value)
    if parsed == int(parsed) and value.endswith(".0"):
        return str(int(parsed))
    if len(value.split(".")[1]) > DECIMALS:
        return f"{parsed:.{DECIMALS}f}".rstrip("0").rstrip(".")
    return value


def to_float(value) -> float | None:
    """A cell as a number, thousands separators allowed; None when it is not one."""
    try:
        return float(str(value).replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def read_table(path: Path) -> list[dict[str, str]]:
    """A CSV file as one dict per row, keyed by its header; empty when the file is absent."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(file)
        ]


def inline_asset(folder: Path, src: str, prefix: str) -> tuple[str, str] | None:
    """One figure or table file as HTML that needs nothing at read time.

    Args:
        folder: The record's folder, which `src` is relative to.
        src: `figures/<name>.svg`, an image, or `tables/<name>.csv`.
        prefix: Put before every id inside an SVG, so figures never collide.

    Returns:
        `("Figure", html)` or `("Table", html)`, or None when the file is
        absent or of a kind the page cannot inline.
    """
    target = folder / src
    if not target.is_file():
        return None
    suffix = target.suffix.lower()

    # a figure: an SVG made transparent with unique ids, or an image as data
    if suffix == ".svg":
        svg = XML_PROLOGUE.sub("", target.read_text(encoding="utf-8")).strip()
        svg = XML_COMMENT.sub("", FIGURE_PATCH.sub(r"\1none", svg))
        return "Figure", _unique_ids(svg, prefix)
    if suffix in IMAGE_TYPES:
        data = base64.b64encode(target.read_bytes()).decode()
        image = f'<img src="data:image/{IMAGE_TYPES[suffix]};base64,{data}" alt="{html.escape(src)}">'
        return "Figure", image
    if suffix != ".csv":
        return None

    # a table: numeric columns right-aligned, long ids in mono
    with target.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if not rows:
        return "Table", '<table class="data"></table>'
    numeric = [
        any(NUMERIC.match(r[i].strip()) for r in rows[1:] if i < len(r))
        and all(
            not r[i].strip() or NUMERIC.match(r[i].strip())
            for r in rows[1:]
            if i < len(r)
        )
        for i in range(len(rows[0]))
    ]

    def cell_class(i: int, cell: str = "") -> str:
        if i < len(numeric) and numeric[i]:
            return ' class="n"'
        return ' class="id"' if HEX_ID.match(cell.strip()) else ""

    head = "".join(
        f"<th{cell_class(i)}>{html.escape(column_header(c))}</th>"
        for i, c in enumerate(rows[0])
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{cell_class(i, c)}>{html.escape(number(c))}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows[1:]
    )
    wide = len(rows[0]) > WIDE_COLUMNS or any(
        len(c) > WIDE_CELL for r in rows for c in r
    )
    tall = len(rows) - 1 > TABLE_ROWS
    classes = "data" + (" wide" if wide else "") + (" tall" if tall else "")
    table = f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return "Table", f'<div class="scroll{" tall" if tall else ""}">{table}</div>'


# ===============================
# Helper functions
# ===============================


def _stash_math(match: re.Match, equations: list[str]) -> str:
    latex, block = (
        (match.group(1), True)
        if match.group(1) is not None
        else (match.group(2), False)
    )
    latex = latex.strip()
    if latex_to_mathml is None:
        rendered = f"<code>{html.escape(latex)}</code>"
    else:
        mathml = latex_to_mathml(latex, display="block" if block else "inline")
        # latex2mathml copies `\texttt{<q>}` into <mtext> unescaped
        mathml = MTEXT.sub(
            lambda m: f"<mtext{m.group(1)}>{html.escape(html.unescape(m.group(2)))}</mtext>",
            mathml,
        )
        rendered = f'<span class="math{" block" if block else ""}">{mathml}</span>'
    equations.append(rendered)
    return f"mathx{len(equations) - 1}x"


def _wrap_table(match: re.Match) -> str:
    tall = " tall" if match.group(0).count("<tr>") - 1 > TABLE_ROWS else ""
    return f'<div class="scroll{tall}"><table class="data{tall}">{match.group(1)}</table></div>'


def _unique_ids(svg: str, prefix: str) -> str:
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', svg)
    return re.sub(
        r'(href="#|url\(#)([^")]+)',
        lambda m: m.group(1) + prefix + m.group(2) if m.group(2) in ids else m.group(0),
        svg,
    )
