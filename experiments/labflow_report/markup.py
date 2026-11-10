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

from .rules import FENCED_GUIDANCE, GUIDANCE, is_template_guidance

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
# column widths follow content: a short or numeric column stays on one line, a
# text column gets room for its longest cell in a few lines, between the bounds
SHORT_CELL = 32
SHORT_LINK = 80
TEXT_MIN_CHARACTERS = 20
# a text column's longest cell spread over about this many lines
TEXT_LINES = 2.5
TEXT_MAX_CHARACTERS = 44
# rem per character of a cell (15 px sans) and of a header (12 px bold sans)
CELL_REM = 0.47
HEADER_REM = 0.45
# a cell's padding, and the widest table that sits beside a figure unwrapped
CELL_PADDING_REM = 1.25
SIDE_TABLE_REM = 26
# an empty cell: a muted dash a screen reader names
EMPTY_CELL = '<span class="nil" title="no value" aria-label="no value">–</span>'
# a URL in a cell links out under its host and path, cut to this many characters
URL = re.compile(r"https?://[^\s;,<>\"']+")
SHORT_URL = 32
LIST_BREAK = re.compile(r"(?<=[A-Za-z)])([,;/])(?=[A-Za-z(])")
ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL = re.compile(r"<(t[hd])([^>]*)>(.*?)</t[hd]>", re.S)
# the first lower-case Greek letter, α
GREEK_LOWER = 0x3B1
# decimals a long float cell keeps
DECIMALS = 4
# a whole number this long reads with thousands separators
GROUPED_DIGITS = 4
# a year, seed or date column, or one that is only an id, keeps its digits ungrouped
UNGROUPED_HEADER = re.compile(
    r"(?<![a-z])(?:year|seed|date)s?(?![a-z])|^(?:\w+[_ ])?id$", re.I
)
# a table's boolean cells read as words
BOOLEAN_WORDS = {"True": "yes", "False": "no", "true": "yes", "false": "no"}
WHOLE = re.compile(r"^-?\d+$")
PASS_HEADER = re.compile(r"\b(?:pass(?:es)?|clears?|fits?|meets?|holds?)\b", re.I)
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
    "rope": "RoPE",
    "v1": "V1",
    "v2": "V2",
    "rmse": "RMSE",
    "bpe": "BPE",
    "ner": "NER",
    "usd": "USD",
    "eur": "EUR",
    "gpt": "GPT",
    "bert": "BERT",
    "roberta": "RoBERTa",
    "deberta": "DeBERTa",
    "mdeberta": "mDeBERTa",
    "mmbert": "mmBERT",
    "xlm": "XLM",
    "bge": "BGE",
    "opus": "Opus",
    "sonnet": "Sonnet",
    "haiku": "Haiku",
    "holm": "Holm",
}
HEADER_UNITS = {
    "s": "s",
    "ms": "ms",
    "min": "min",
    "gb": "GB",
    "gib": "GiB",
    "mb": "MB",
    "pct": "%",
    "relative": "relative",
}
# a last word that is no unit after this word: `max_min` is a range
NOT_UNIT_AFTER = {"min": "max"}
# a percent level inside a name, `f1_95pct_low` reads `F1 95% low`
PERCENT_WORD = re.compile(r"^(\d+)pct$", re.I)
# word pairs a header writes with a hyphen
HEADER_PAIRS = {("p", "value"): "p-value"}
# a pair that qualifies what follows it takes a comma: `p-value, uncorrected`
QUALIFIED_PAIRS = {("p", "value")}
# word pairs that close a header as a bracketed range: `(max − min)`
HEADER_RANGES = {("max", "min"): "(max − min)"}
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
# the neutral colours a figure draws its ink with, each read from a theme
# variable with the light colour as its fallback; series colours stay as drawn
FIGURE_INK = {
    "#1f252f": "--fig-ink",
    "#262626": "--fig-ink",
    "#000000": "--fig-ink",
    "#06102b": "--fig-ink",
    "#0b1f44": "--fig-ink",
    "#5a6472": "--fig-muted",
    "#c8ced7": "--fig-line",
    "#dde3e8": "--fig-line",
    "#cccccc": "--fig-line",
    "#e8ebf0": "--fig-soft",
    "#efefef": "--fig-soft",
    "#ffffff": "--fig-paper",
    # the near-black series turn near-white on the dark panel
    "#2b2f3a": "--fig-series-dark",
    "#2c3e50": "--fig-series-dark",
    "#34495e": "--fig-series-dark",
}
# matplotlib leaves a label in its default black without any fill
UNFILLED_INK = "fill: var(--fig-ink, #000000)"
INK_COLOUR = re.compile(r"(fill|stroke): ?(#[0-9a-fA-F]{6})\b")
# a colour as a presentation attribute, as graphviz writes it: `fill="#aec7e8"`, `stroke="black"`
INK_ATTRIBUTE = re.compile(r'\s(fill|stroke)="([^"]+)"')
# the colour names diagram tools write for neutral ink and light fills
NAMED_COLOURS = {
    "black": "#000000",
    "white": "#ffffff",
    "gray": "#808080",
    "grey": "#808080",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "lightblue": "#add8e6",
}
HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{6}")
# a shape fill lighter than this is mixed toward the panel by --fig-tint, so light ink reads on it
LIGHT_FILL = 0.5
TAG_STYLE = re.compile(r'\sstyle="([^"]*)"')
SVG_TAG = re.compile(r"<(\w+)\b[^>]*>")
SVG_IMAGE = re.compile(r"<image\b[^>]*>")
SVG_ROOT = re.compile(r"<svg\b[^>]*>")
VIEW_BOX = re.compile(
    r'\sviewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)\s*"'
)
# a width or height in one unit; a percentage says nothing of the shape
SVG_LENGTH = r'\s{}="([\d.]+)\s*(px|pt|in|cm|mm|)"'
# CSS pixels per unit of an SVG length, 96 to the inch
CSS_PIXELS = {"": 1.0, "px": 1.0, "pt": 96 / 72, "in": 96.0, "cm": 96 / 2.54, "mm": 96 / 25.4}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_FRAMES = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# matplotlib draws a heatmap's cells as an image flipped by this transform
IMAGE_FLIP = re.compile(r"scale\(1 -1\) translate\(0 (-?[\d.]+)\)")
SENTENCE = re.compile(r"(.+?[.!?])(?=\s|$)", re.S)
TAG = re.compile(r"<[^>]+>")
# a lone Greek letter in <mi>, which a browser maps to the math italic block
# that most fonts lack; set as the plain letter it shows on any machine
GREEK_MI = re.compile(r"<mi>&#x0*(3[9A-F][0-9A-F]);</mi>", re.I)
MTEXT = re.compile(r"<mtext([^>]*)>(.*?)</mtext>", re.S)
# a command latex2mathml leaves as its source, such as `\arg` or `\softmax`
RAW_COMMAND_MI = re.compile(r"<mi>\\([A-Za-z]+)</mi>")

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
        else _escape_placeholders(
            MATH.sub(lambda m: _stash_math(m, equations), SPACED_DASH.sub("—", part))
        )
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


def unguided(text: str) -> str:
    """A text without the template's `<…>` guidance; "" when nothing else is left."""
    if not text:
        return ""

    # a placeholder in code is the author's own, except the template's in a fenced block
    def leftover(match: re.Match, fenced: bool) -> str:
        inner = match.group(1)
        is_guidance = (
            inner in FENCED_GUIDANCE if fenced else is_template_guidance(inner)
        )
        return "" if is_guidance else match.group(0)

    kept = []
    for i, part in enumerate(CODE_SPANS.split(text)):
        fenced = part.startswith("```")
        if i % 2 and not fenced:
            kept.append(part)
        else:
            kept.append(GUIDANCE.sub(lambda m, f=fenced: leftover(m, f), part))
    stripped = "".join(kept)
    return stripped.strip() if re.search(r"\w", stripped) else ""


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
    `P95 8 CPU threads (s)`, `f1_95pct_low` reads `F1 95% low`,
    `p_value_uncorrected` reads `P-value, uncorrected` and
    `seed_spread_max_min` reads `Seed spread (max − min)`; a name the author
    wrote with spaces keeps its case and only gains a capital.
    """
    text = key.strip()
    if " " in text and "_" not in text and not text.isupper():
        return capitalize(text)
    words = [w for w in re.split(r"[_\s]+", text) if w]
    shown, i = [], 0
    while i < len(words):
        word, low = words[i], words[i].lower()
        pair = (low, words[i + 1].lower()) if i + 1 < len(words) else None
        is_unit = (
            i == len(words) - 1
            and i
            and low in HEADER_UNITS
            and words[i - 1].lower() != NOT_UNIT_AFTER.get(low)
        )
        if pair in HEADER_RANGES and i and i + 2 == len(words):
            shown.append(HEADER_RANGES[pair])
            i += 1
        elif pair in HEADER_PAIRS:
            is_qualified = pair in QUALIFIED_PAIRS and i == 0 and i + 2 < len(words)
            shown.append(HEADER_PAIRS[pair] + ("," if is_qualified else ""))
            i += 1
        elif is_unit:
            shown.append(f"({HEADER_UNITS[low]})")
        elif low in HEADER_WORDS:
            shown.append(HEADER_WORDS[low])
        elif PERCENT_WORD.match(word):
            shown.append(PERCENT_WORD.sub(r"\1%", word))
        elif CLAUSE_WORD.match(word):
            shown.append(word.upper())
        else:
            shown.append(low if text.isupper() or word.islower() else word)
        i += 1
    return capitalize(" ".join(shown))


def number(cell: str) -> str:
    """A numeric cell for display: a long float to four decimals, `750.0` to `750`, `5246773.5` to `5,246,773.5`."""
    value = cell.strip()
    if WHOLE.match(value):
        return _grouped(value)
    if not FLOAT.match(value):
        return cell
    parsed = float(value)
    if parsed == int(parsed) and value.endswith(".0"):
        return _grouped(str(int(parsed)))
    if len(value.split(".")[1]) > DECIMALS:
        value = f"{parsed:.{DECIMALS}f}".rstrip("0").rstrip(".")
    return _grouped(value)


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


def inline_asset(
    folder: Path, src: str, prefix: str, headers: dict[str, str] | None = None
) -> tuple[str, str] | None:
    """One figure or table file as HTML that needs nothing at read time.

    Args:
        folder: The record's folder, which `src` is relative to.
        src: `figures/<name>.svg`, an image, or `tables/<name>.csv`.
        prefix: Put before every id inside an SVG, so figures never collide.
        headers: A shown name by the CSV name it replaces: the whole column
            name, or a name standing as whole words anywhere in it
            (`morph_cut` in `word_in_morph_cut`, the rest through
            `column_header`); any other column keeps `column_header`.

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
        return "Figure", _unique_ids(_theme_ink(svg), prefix)
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

    # a year or seed column keeps its digits ungrouped
    plain = [bool(UNGROUPED_HEADER.search(c)) for c in rows[0]]

    # a pass or fit column's yes and no read in the outcome colours
    passes = [bool(PASS_HEADER.search(column_header(c))) for c in rows[0]]

    def cell(i: int, value: str) -> str:
        text = value.strip()
        if not text:
            return EMPTY_CELL
        word = BOOLEAN_WORDS.get(text, text if text in ("yes", "no") else "")
        if word and i < len(passes) and passes[i]:
            return f'<span class="pass-{word}">{word}</span>'
        if text in BOOLEAN_WORDS:
            return BOOLEAN_WORDS[text]
        if i < len(numeric) and numeric[i]:
            return html.escape(value if i < len(plain) and plain[i] else number(value))
        # a URL links out; a list written without spaces (`bg,cs,da`) may break after each separator
        parts = URL.split(value)
        urls = URL.findall(value)
        return "".join(
            LIST_BREAK.sub(r"\1<wbr>", html.escape(part))
            + (_url_link(urls[i]) if i < len(urls) else "")
            for i, part in enumerate(parts)
        )

    shown = headers or {}
    head = "".join(
        f"<th{cell_class(i)}>{html.escape(_shown_header(c, shown))}</th>"
        for i, c in enumerate(rows[0])
    )
    body = "".join(
        "<tr>"
        + "".join(f"<td{cell_class(i, c)}>{cell(i, c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows[1:]
    )
    sized, natural = _size_columns(
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
    )
    tall = len(rows) - 1 > TABLE_ROWS
    classes = (
        "data"
        + (" wide" if natural > SIDE_TABLE_REM else "")
        + (" tall" if tall else "")
    )
    table = f'<table class="{classes}">{sized}</table>'
    return "Table", f'<div class="scroll{" tall" if tall else ""}">{table}</div>'


def figure_ratio(path: Path) -> float | None:
    """A figure's width over its height, read from the file without drawing it.

    Args:
        path: An SVG (its viewBox, else its width and height), PNG, GIF or JPEG.

    Returns:
        Width divided by height, or None when the file does not say.
    """
    if not path.is_file():
        return None
    data = path.read_bytes()

    # an SVG: the viewBox, else width and height in one unit
    if path.suffix.lower() == ".svg":
        root = SVG_ROOT.search(data.decode("utf-8", errors="replace"))
        if root is None:
            return None
        if box := VIEW_BOX.search(root.group(0)):
            width, height = float(box.group(1)), float(box.group(2))
        else:
            width = re.search(SVG_LENGTH.format("width"), root.group(0))
            height = re.search(SVG_LENGTH.format("height"), root.group(0))
            if not (width and height) or width.group(2) != height.group(2):
                return None
            width, height = float(width.group(1)), float(height.group(1))
        return width / height if height else None

    # an image: the size its header gives
    size = _image_size(data)
    return size[0] / size[1] if size and size[1] else None


def figure_width(path: Path) -> float | None:
    """A figure's own width in CSS pixels, 96 to the inch, read without drawing it.

    Args:
        path: An SVG (its width in any unit, else its viewBox), PNG, GIF or JPEG.

    Returns:
        The width, or None when the file does not say.
    """
    if not path.is_file():
        return None
    data = path.read_bytes()
    if path.suffix.lower() != ".svg":
        size = _image_size(data)
        return float(size[0]) if size else None
    root = SVG_ROOT.search(data.decode("utf-8", errors="replace"))
    if root is None:
        return None
    if width := re.search(SVG_LENGTH.format("width"), root.group(0)):
        return float(width.group(1)) * CSS_PIXELS[width.group(2)]
    box = VIEW_BOX.search(root.group(0))
    return float(box.group(1)) if box else None


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
            lambda m: (
                f"<mtext{m.group(1)}>{html.escape(html.unescape(m.group(2)))}</mtext>"
            ),
            mathml,
        )
        mathml = GREEK_MI.sub(_greek_upright, mathml)
        # an operator name it does not know is set upright, without its backslash
        mathml = RAW_COMMAND_MI.sub(r'<mi mathvariant="normal">\1</mi>', mathml)
        rendered = f'<span class="math{" block" if block else ""}">{mathml}</span>'
    equations.append(rendered)
    return f"mathx{len(equations) - 1}x"


def _grouped(value: str) -> str:
    """A plain number with thousands separators once its whole part has four digits."""
    sign, rest = (value[0], value[1:]) if value[:1] in "-+" else ("", value)
    whole, dot, fraction = rest.partition(".")
    if len(whole) < GROUPED_DIGITS:
        return value
    return f"{sign}{int(whole):,}{dot}{fraction}"


def _escape_placeholders(text: str) -> str:
    """A `<repo-root>` or `<key>` as text, so the browser never reads it as a tag."""
    return GUIDANCE.sub(
        lambda m: (
            f"&lt;{m.group(1)}&gt;" if is_template_guidance(m.group(1)) else m.group(0)
        ),
        text,
    )


def _greek_upright(match: re.Match) -> str:
    """A lone Greek `<mi>` as its plain letter: lower case slanted by the page, upper case upright."""
    point = int(match.group(1), 16)
    slant = ' class="gk"' if point >= GREEK_LOWER else ""
    return f'<mi mathvariant="normal"{slant}>&#x{point:04X};</mi>'


def _url_link(url: str) -> str:
    """A URL as a link labelled by its host and path, cut to `SHORT_URL` characters."""
    shown = re.sub(r"^https?://(?:www\.)?", "", url).rstrip("/")
    if len(shown) > SHORT_URL:
        shown = shown[:SHORT_URL].rstrip("/.-_") + "…"
    full = html.escape(url)
    return f'<a class="url" href="{full}" title="{full}">{html.escape(shown)}</a>'


def _shown_header(column: str, headers: dict[str, str]) -> str:
    """A column's shown name: each `headers` name standing as whole words in it
    (`morph_cut` in `word_in_morph_cut`), the rest through `column_header`."""
    if column in headers:
        return headers[column]
    if not headers:
        return column_header(column)
    names = "|".join(re.escape(key) for key in sorted(headers, key=len, reverse=True))
    parts = re.split(rf"(?:(?<=_)|^)({names})(?=_|$)", column)
    shown = []
    for i, part in enumerate(parts):
        if i % 2:
            shown.append(headers[part])
            continue
        if not (part := part.strip("_")):
            continue
        text = column_header(part)
        # lower an ordinary capitalised word after a title, keep an acronym (`CI`, `F1`)
        if shown and text[1:2].islower():
            text = text[0].lower() + text[1:]
        shown.append(text)
    return " ".join(shown)


def _wrap_table(match: re.Match) -> str:
    tall = " tall" if match.group(0).count("<tr>") - 1 > TABLE_ROWS else ""
    rows, natural = _size_columns(match.group(1))
    wide = " wide" if natural > SIDE_TABLE_REM else ""
    return f'<div class="scroll{tall}"><table class="data{wide}{tall}">{rows}</table></div>'


def _size_columns(table: str) -> tuple[str, float]:
    """A table's columns sized by content: short ones kept on one line, text ones given a minimum width.

    A header wraps to at most two lines; a text column's minimum holds its
    longest cell in about `TEXT_LINES` lines, between `TEXT_MIN_CHARACTERS` and
    `TEXT_MAX_CHARACTERS`.

    Returns:
        The table's rows with the widths written in, and the width in rem the
        table needs to show without wrapping a word.
    """
    rows = [CELL.findall(row) for row in ROW.findall(table)]
    if not rows:
        return table, 0.0
    head, body = rows[0], rows[1:]

    # measure each column
    styles, nowrap, natural = [], [], 0.0
    for i, (_, _, header) in enumerate(head):
        cells = [plain_text(r[i][2]) for r in body if i < len(r)]
        numeric = any('class="n"' in r[i][1] for r in body if i < len(r))
        longest = max((len(c) for c in cells), default=0)
        linked = cells and all("/" in c and " " not in c for c in cells if c)
        short = numeric or longest <= (SHORT_LINK if linked else SHORT_CELL)
        title = plain_text(header)
        header_characters = _two_line_width(title)
        text_characters = (
            0
            if short
            else min(
                TEXT_MAX_CHARACTERS,
                max(TEXT_MIN_CHARACTERS, int(longest / TEXT_LINES + 0.5)),
            )
        )
        # a cell's min-width holds its padding too
        width = max(header_characters * HEADER_REM, text_characters * CELL_REM)
        styles.append(
            f' style="min-width:{width + CELL_PADDING_REM:.1f}rem"' if width else ""
        )
        nowrap.append(short and not numeric)
        natural += max(width, longest * CELL_REM if short else 0) + CELL_PADDING_REM

    # write the widths on the header and the one-line class on short cells
    def sized_row(row: str, header: bool) -> str:
        cells = iter(range(len(head)))

        def sized(match: re.Match) -> str:
            tag, attrs, inner = match.groups()
            i = next(cells, len(head))
            if i >= len(head):
                return match.group(0)
            if header:
                attrs += styles[i]
            elif nowrap[i]:
                attrs = (
                    attrs.replace('class="', 'class="nw ')
                    if 'class="' in attrs
                    else attrs + ' class="nw"'
                )
            return f"<{tag}{attrs}>{inner}</{tag}>"

        return CELL.sub(sized, row)

    found = iter(range(len(rows)))
    sized_table = ROW.sub(
        lambda m: f"<tr>{sized_row(m.group(1), next(found) == 0)}</tr>", table
    )
    return sized_table, natural


def _two_line_width(title: str) -> int:
    """The characters a header needs to fit on at most two lines, split at its best space."""
    words = title.split()
    if len(words) < 2:
        return len(title)
    return min(
        max(len(" ".join(words[:i])), len(" ".join(words[i:])))
        for i in range(1, len(words))
    )


def _unique_ids(svg: str, prefix: str) -> str:
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', svg)
    return re.sub(
        r'(href="#|url\(#)([^")]+)',
        lambda m: m.group(1) + prefix + m.group(2) if m.group(2) in ids else m.group(0),
        svg,
    )


def _theme_ink(svg: str) -> str:
    """The figure's neutral ink read from theme variables, so it reads on a dark panel.

    A colour is read from a `style` declaration or a `fill`/`stroke`
    attribute, by hex value or a name in NAMED_COLOURS. A label inside a
    heatmap's cells keeps its colour, which was picked against the cell under
    it, and white text stays white. A light shape fill is mixed toward the
    panel by `--fig-tint` (unset, 100%: as drawn), so a page that sets it
    keeps light ink readable on a diagram's light boxes.
    """
    fields = [box for tag in SVG_IMAGE.findall(svg) if (box := _image_box(tag))]

    def recolour(tag: re.Match) -> str:
        is_text = tag.group(1) == "text"
        if is_text:
            x, y = _number(tag.group(0), "x"), _number(tag.group(0), "y")
            if (
                x is not None
                and y is not None
                and any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in fields)
            ):
                return tag.group(0)
            if "fill" not in tag.group(0):
                return _add_style(tag.group(0), UNFILLED_INK)

        def swap(colour: re.Match) -> str:
            themed = _themed_colour(colour.group(1), colour.group(2), is_text)
            return f"{colour.group(1)}: {themed}" if themed else colour.group(0)

        # a style declaration outranks the attribute, so an attribute colour gains one
        found = INK_COLOUR.sub(swap, tag.group(0))
        style = TAG_STYLE.search(found)
        for name, value in INK_ATTRIBUTE.findall(found):
            if style and re.search(rf"(^|;)\s*{name}\s*:", style.group(1)):
                continue
            if themed := _themed_colour(name, value, is_text):
                found = _add_style(found, f"{name}: {themed}")
                style = TAG_STYLE.search(found)
        return found

    return SVG_TAG.sub(recolour, svg)


def _themed_colour(name: str, value: str, is_text: bool) -> str:
    """A colour as the theme draws it, or "" to keep it as drawn."""
    value = NAMED_COLOURS.get(value.strip().lower(), value.strip().lower())
    if not HEX_COLOUR.fullmatch(value) or (is_text and value == "#ffffff"):
        return ""
    if value in FIGURE_INK:
        return f"var({FIGURE_INK[value]}, {value})"
    if name == "fill" and not is_text and _luminance(value) > LIGHT_FILL:
        return f"color-mix(in srgb, {value} var(--fig-tint, 100%), var(--fig-paper, #ffffff))"
    return ""


def _luminance(value: str) -> float:
    # the relative luminance of a `#rrggbb` colour, 0 black to 1 white
    channels = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _add_style(tag: str, declaration: str) -> str:
    """The SVG start tag with one more CSS declaration in its style attribute."""
    if style := TAG_STYLE.search(tag):
        rule = style.group(1).rstrip("; ")
        joined = f"{rule}; {declaration}" if rule else declaration
        return f"{tag[: style.start(1)]}{joined}{tag[style.end(1) :]}"
    head, _, rest = tag.partition(" ")
    return (
        f'{head} style="{declaration}" {rest}'
        if rest
        else f'{tag[:-1]} style="{declaration}">'
    )


def _image_box(tag: str) -> tuple[float, float, float, float] | None:
    """The page box `(x0, y0, x1, y1)` an SVG `<image>` covers, or None without its size."""
    x, y, width, height = (_number(tag, key) for key in ("x", "y", "width", "height"))
    if None in (x, y, width, height):
        return None
    if flip := IMAGE_FLIP.search(tag):
        top = -float(flip.group(1))
        return x, top - y - height, x + width, top - y
    return x, y, x + width, y + height


def _number(tag: str, key: str) -> float | None:
    """A numeric attribute of an SVG tag, or None when the tag has none."""
    match = re.search(rf'\s{key}="(-?[\d.]+)"', tag)
    return float(match.group(1)) if match else None


def _image_size(data: bytes) -> tuple[int, int] | None:
    # a PNG, GIF or JPEG's width and height in pixels, from its header
    if data.startswith(PNG_SIGNATURE) and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """A JPEG's `(width, height)` from its first frame header, or None without one."""
    at = 2
    while at + 9 <= len(data) and data[at] == 0xFF:
        marker = data[at + 1]
        length = int.from_bytes(data[at + 2 : at + 4], "big")
        if marker in JPEG_FRAMES:
            height = int.from_bytes(data[at + 5 : at + 7], "big")
            width = int.from_bytes(data[at + 7 : at + 9], "big")
            return width, height
        at += 2 + length
    return None
