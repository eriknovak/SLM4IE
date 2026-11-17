"""Table rules: the `TAB-nnn` checks of `labflow/references/tables.md`, standard library only.

`save_table` writes a CSV only when it breaks no rule; `check_table` runs the
same rules on a table already written. `create_assets.py` imports both from
here and re-exports them, so an `analysis.py` keeps `from create_assets import
save_table`. The report checker (REC-160) imports this file alone, with
pyyaml and no plotting library, so it must import nothing beyond the
standard library.
"""

import csv
import math
import numbers
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

# ===============================
# Defaults
# ===============================

CLAUSE = re.compile(r"^[HCQ]\d+$")

# a figure or result table is `<stem>-<quantity>-by-<dimension>`, plain lowercase words
RESULT_STEM = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-by-[a-z0-9]+(?:-[a-z0-9]+)*$")
TABLES_FOLDER = "tables"
# an entry table sits in experiments/reference/<topic>/tables/, named for an entry
REFERENCE_FOLDER = "reference"
ENTRY_STEM = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
VARIANT_SEPARATOR = "-vs-"
ENTRY_KINDS = ("entry", "entry-variant")

TABLE_RULES: dict[str, str] = {
    "TAB-001": "file name",
    "TAB-010": "column names",
    "TAB-015": "run ids",
    "TAB-020": "bare numbers",
    "TAB-023": "missing values",
    "TAB-032": "row width",
    "TAB-040": "clause readings columns",
    "TAB-041": "clause ids",
    "TAB-042": "reading numbers",
    "TAB-043": "outcome words",
    "TAB-044": "one row per reading",
    "TAB-050": "arm first",
    "TAB-051": "verdict lead columns",
    "TAB-052": "arm roles",
    "TAB-053": "adoption",
    "TAB-054": "verdict clause",
    "TAB-060": "statistics columns",
    "TAB-061": "item count",
    "TAB-062": "summary columns",
    "TAB-063": "summary values",
    "TAB-064": "fields columns",
    "TAB-065": "field types",
    "TAB-066": "field values",
    "TAB-070": "entry table name",
    "TAB-071": "variant table columns",
    "TAB-072": "one row per form",
}

# the standard tables of tables.md, by the name a skill writes them under
STANDARD_TABLES: dict[str, re.Pattern] = {
    "clause-readings": re.compile(r"^clause-readings$"),
    "verdict-by-arm": re.compile(r"^verdict-by-arm$"),
    "dataset-statistics": re.compile(r"^dataset-[a-z0-9]+(?:-[a-z0-9]+)*-statistics$"),
    "dataset-summary": re.compile(r"^dataset-[a-z0-9]+(?:-[a-z0-9]+)*-summary$"),
    "dataset-fields": re.compile(r"^dataset-[a-z0-9]+(?:-[a-z0-9]+)*-fields$"),
    "scoping": re.compile(r"^scoping-[a-z0-9]+(?:-[a-z0-9]+)*$"),
}
ARM_TABLE_SUFFIX = "-by-arm"
CLAUSE_READINGS_COLUMNS = [
    "clause",
    "qualifier",
    "value",
    "ci_low",
    "ci_high",
    "bar",
    "margin",
    "outcome",
]
READING_NUMBERS = ("value", "ci_low", "ci_high", "bar", "margin")
VERDICT_LEAD_COLUMNS = ["arm", "clause", "role", "adopted"]
ARM_ROLES = ("arm", "baseline")
ADOPTED_WORDS = ("yes", "no")
SUMMARY_COLUMNS = ["statistic", "value", "note"]
FIELDS_COLUMNS = ["field", "type", "distinct", "values"]
FIELD_TYPES = ("categorical", "numeric", "text")
FIELD_VALUES_MAX = 5
FIELD_VALUES_SEPARATOR = "; "

# the outcome words a reading may carry, by clause letter; mirrors
# labflow_report/checks/record.py OUTCOME_WORDS
OUTCOME_WORDS: dict[str, tuple[str, ...]] = {
    "H": ("confirmed", "refuted", "inconclusive"),
    "C": ("better", "worse", "equivalent", "undecided"),
    "Q": ("answered", "partial", "unanswered"),
}
MEASURED_WORDS = ("answered", "unanswered")

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
RUN_ID = re.compile(r"^[0-9a-f]{32}$")
INTEGER = re.compile(r"^[-+]?\d+$")
PLAIN_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
DECORATED_NUMBER = re.compile(r"^[-+]?(?:[$€£]\s?[\d.,]+|\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?:\d+\.?\d*|\.\d+)\s?%)$")
MISSING_MARKERS = frozenset({"nan", "NaN", "None", "null", "NULL", "N/A", "n/a", "-", "—"})

# ===============================
# Main functions
# ===============================


class Breach(NamedTuple):
    """One broken rule: its `FIG-nnn` or `TAB-nnn` id and what was measured."""

    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.rule} {self.message}"


class TableCheckError(ValueError):
    """Raised when a table breaks a rule; the table is not written.

    Attributes:
        path (Path): The file that was not written.
        breaches (tuple[Breach, ...]): The rules broken, in rule order.
    """

    def __init__(self, path: Path, breaches: list[Breach]) -> None:
        self.path = path
        self.breaches = tuple(breaches)
        lines = "\n".join(f"  {breach}" for breach in self.breaches)
        super().__init__(f"table {path} not written, {len(self.breaches)} breaches (tables.md):\n{lines}")


def save_table(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Checks and writes one CSV table under the `TAB-nnn` rules of tables.md.

    A missing value (`None` or NaN) is written as an empty cell and a float
    keeps its full precision, so the formats of tables.md hold by construction.

    Examples:
        >>> save_table([{"arm": "ours", "f1": 0.81}], "tables/f1-by-arm.csv")

    Args:
        rows: One mapping per row, column name to value.
        path: Destination of the table, `tables/<name>.csv`.
        columns: The column order; the first row's keys when not given.

    Raises:
        ValueError: If `path` is not a `.csv`, there are no columns, or a row
            holds a column outside `columns`.
        TableCheckError: If the table breaks a rule; nothing is written.
    """
    target = Path(path)
    if target.suffix != ".csv":
        raise ValueError(f"table path must end in .csv: {target}")
    header = list(columns if columns is not None else (rows[0] if rows else ()))
    if not header:
        raise ValueError(f"table {target} has no columns: pass rows or columns")
    for line, row in enumerate(rows, start=2):
        extra = [key for key in row if key not in header]
        if extra:
            raise ValueError(f"table {target} line {line} holds columns {extra} outside {header}")

    # format, check and write the cells
    cells = [[_cell(row.get(column)) for column in header] for row in rows]
    breaches = _table_breaches(target, header, cells)
    if breaches:
        raise TableCheckError(target, breaches)
    with target.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows([header, *cells])


def check_table(path: str | Path) -> list[Breach]:
    """Runs the `TAB-nnn` rules of tables.md on a table already written.

    Args:
        path: The CSV file.

    Returns:
        The breaches found, in rule order; empty for a table that passes.
    """
    target = Path(path)
    with target.open(encoding="utf-8", newline="") as handle:
        lines = list(csv.reader(handle))
    if not lines:
        return [Breach("TAB-032", "the table has no header row")]
    return _table_breaches(target, lines[0], lines[1:])


# ===============================
# Table check helpers
# ===============================


class _Table(NamedTuple):
    """One table as its rules read it: its name stem, header and rows of text cells."""

    stem: str
    header: list[str]
    rows: list[dict[str, str]]
    widths: list[int]

    def column(self, name: str) -> list[tuple[int, str]]:
        """The cells of one column with their file line, empty when it is absent."""
        return [(line, row.get(name, "")) for line, row in _lines(self.rows)]


def _lines(rows: list) -> Iterable[tuple[int, object]]:
    """Pairs each data row with its line in the file, the header being line 1."""
    return enumerate(rows, start=2)


def _cell(value: object) -> str:
    """The CSV text of one value: empty for a missing one, full precision for a float."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return "" if math.isnan(value) else repr(float(value))
    return str(value)


def _table_breaches(target: Path, header: list[str], cells: list[list[str]]) -> list[Breach]:
    """Runs every table rule that applies to the file's name on its cells."""
    table = _Table(
        target.stem,
        header,
        [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in cells],
        [len(row) for row in cells],
    )
    # an entry table is named for its entry, so TAB-070 stands in for TAB-001
    kind = _entry_table_kind(target) or _table_kind(table.stem)
    if kind in ENTRY_KINDS:
        name_check = ("TAB-070", lambda _: _entry_table_misnamed(target))
    else:
        name_check = ("TAB-001", lambda _: _table_misnamed(target, kind))
    checks = [
        name_check,
        ("TAB-010", _unsnaked_columns),
        ("TAB-015", _short_run_ids),
        ("TAB-020", _decorated_numbers),
        ("TAB-023", _missing_markers),
        ("TAB-032", _ragged_rows),
    ]
    if table.stem.endswith(ARM_TABLE_SUFFIX) and kind not in ENTRY_KINDS:
        checks.append(("TAB-050", _arm_not_first))
    breaches = [Breach(rule, message) for rule, check in checks for message in check(table)]

    # a standard table's own rules run once its column rule, the first, passes
    for index, (rule, check) in enumerate(KIND_CHECKS.get(kind, [])):
        found = [Breach(rule, message) for message in check(table)]
        breaches += found
        if found and index == 0:
            break
    return sorted(breaches, key=lambda breach: breach.rule)


def _table_kind(stem: str) -> str | None:
    """The standard table a name is, `result` for a result table, `None` for neither."""
    for kind, pattern in STANDARD_TABLES.items():
        if pattern.match(stem):
            return kind
    return "result" if RESULT_STEM.match(stem) else None


def _entry_table_kind(target: Path) -> str | None:
    """`entry-variant` or `entry` for a table in a reference topic's tables/, else `None`."""
    folder = target.resolve().parent
    if folder.name != TABLES_FOLDER or folder.parent.parent.name != REFERENCE_FOLDER:
        return None
    return "entry-variant" if VARIANT_SEPARATOR in target.stem else "entry"


def _is_plain_number(text: str) -> bool:
    return bool(PLAIN_NUMBER.match(text))


def _numeric_columns(table: _Table) -> list[str]:
    """The columns whose filled cells are all numbers, with at least one filled."""
    numeric = []
    for name in table.header:
        filled = [cell for _, cell in table.column(name) if cell]
        if filled and all(_is_plain_number(cell) for cell in filled):
            numeric.append(name)
    return numeric


def _repeats(values: list[object]) -> list[object]:
    return [value for value, count in Counter(values).items() if count > 1]


def _columns_differ(table: _Table, expected: list[str], what: str) -> list[str]:
    if table.header == expected:
        return []
    return [f"{what} has columns {table.header}, not {expected}"]


def _table_misnamed(target: Path, kind: str | None) -> list[str]:
    messages = []
    if target.parent.name != TABLES_FOLDER:
        messages.append(f"the table sits in {target.parent.name or '.'}/, not {TABLES_FOLDER}/")
    if kind is None:
        messages.append(
            f"name {target.name!r} is neither a standard table nor <stem>-<quantity>-by-<dimension>.csv "
            "in lowercase hyphenated words"
        )
    return messages


def _entry_table_misnamed(target: Path) -> list[str]:
    topic = target.resolve().parent.parent
    entries = [path.stem for path in topic.glob("*.md") if path.name != "README.md"]
    if ENTRY_STEM.match(target.stem) and any(target.stem.startswith(f"{entry}-") for entry in entries):
        return []
    message = (
        f"name {target.name!r} is not <entry>-<what>.csv in lowercase hyphenated words, "
        f"<entry> an entry of {topic.name}/"
    )
    return [message]


def _unsnaked_columns(table: _Table) -> list[str]:
    return [f"column {name!r} is not lowercase snake case" for name in table.header if not SNAKE_CASE.match(name)]


def _short_run_ids(table: _Table) -> list[str]:
    return [
        f"line {line}: run_id {cell!r} is not a full 32-character MLflow id"
        for line, cell in table.column("run_id")
        if cell and not RUN_ID.match(cell)
    ]


def _first_cell_matching(table: _Table, matches: Callable[[str], bool]) -> list[tuple[str, int, str]]:
    """Per column, the first cell `matches` accepts, with its line."""
    found = []
    for name in table.header:
        hit = next(((line, cell) for line, cell in table.column(name) if matches(cell)), None)
        if hit:
            found.append((name, *hit))
    return found


def _decorated_numbers(table: _Table) -> list[str]:
    return [
        f"line {line}: {cell!r} in column {name!r} is a number with a separator or unit sign; "
        "write it bare and put the unit in the column name"
        for name, line, cell in _first_cell_matching(table, lambda cell: bool(DECORATED_NUMBER.match(cell.strip())))
    ]


def _missing_markers(table: _Table) -> list[str]:
    return [
        f"line {line}: {cell!r} in column {name!r} marks a missing value; leave the cell empty"
        for name, line, cell in _first_cell_matching(table, lambda cell: cell.strip() in MISSING_MARKERS)
    ]


def _ragged_rows(table: _Table) -> list[str]:
    return [
        f"line {line}: {width} cells for {len(table.header)} columns"
        for line, width in _lines(table.widths)
        if width != len(table.header)
    ]


def _clause_columns(table: _Table) -> list[str]:
    return _columns_differ(table, CLAUSE_READINGS_COLUMNS, "clause-readings.csv")


def _clause_ids(table: _Table) -> list[str]:
    return [
        f"line {line}: clause {cell!r} is not an id such as H1, C2 or Q3"
        for line, cell in table.column("clause")
        if not CLAUSE.match(cell)
    ]


def _reading_numbers(table: _Table) -> list[str]:
    messages = []
    for line, row in _lines(table.rows):
        words = [
            f"{name} {row.get(name)!r}" for name in READING_NUMBERS if row.get(name) and not _is_plain_number(row[name])
        ]
        if words:
            messages.append(f"line {line}: {', '.join(words)} not a number")
            continue
        low, value, high = (row.get(name, "") for name in ("ci_low", "value", "ci_high"))
        if low and value and high and not float(low) <= float(value) <= float(high):
            messages.append(f"line {line}: value {value} lies outside its interval {low} to {high}")
    return messages


def _outcome_words(table: _Table) -> list[str]:
    messages = []
    for line, row in _lines(table.rows):
        clause, outcome = row.get("clause", ""), row.get("outcome", "")
        if not CLAUSE.match(clause):
            continue
        words = OUTCOME_WORDS[clause[0]]
        if outcome not in words + MEASURED_WORDS:
            messages.append(
                f"line {line}: outcome {outcome!r} is not a {clause[0]} clause's word "
                f"({', '.join(words)}) nor a measured clause's ({', '.join(MEASURED_WORDS)})"
            )
    return messages


def _repeated_readings(table: _Table) -> list[str]:
    readings = [(row.get("clause", ""), row.get("qualifier", "")) for row in table.rows]
    return [
        f"clause {clause} has two rows for qualifier {qualifier!r}; one row per reading"
        for clause, qualifier in _repeats(readings)
    ]


def _arm_not_first(table: _Table) -> list[str]:
    if not table.header or table.header[0] != "arm":
        return [f"the first column is {table.header[0] if table.header else None!r}, not arm"]
    arms = [cell for _, cell in table.column("arm")]
    messages = [f"line {line}: the arm is empty" for line, cell in table.column("arm") if not cell]
    messages += [f"arm {arm!r} has more than one row" for arm in _repeats(arms) if arm]
    return messages


def _verdict_lead(table: _Table) -> list[str]:
    if table.header[: len(VERDICT_LEAD_COLUMNS)] == VERDICT_LEAD_COLUMNS:
        return []
    return [f"verdict-by-arm.csv opens with {table.header[: len(VERDICT_LEAD_COLUMNS)]}, not {VERDICT_LEAD_COLUMNS}"]


def _arm_roles(table: _Table) -> list[str]:
    roles = table.column("role")
    messages = [
        f"line {line}: role {cell!r} is neither {' nor '.join(ARM_ROLES)}"
        for line, cell in roles
        if cell not in ARM_ROLES
    ]
    baselines = sum(cell == "baseline" for _, cell in roles)
    if baselines != 1:
        messages.append(f"{baselines} baseline rows; the table has exactly one")
    return messages


def _adoption(table: _Table) -> list[str]:
    messages = [
        f"line {line}: adopted {row.get('adopted')!r} is neither {' nor '.join(ADOPTED_WORDS)}"
        for line, row in _lines(table.rows)
        if row.get("adopted") not in ADOPTED_WORDS
    ]
    adopted = [(line, row) for line, row in _lines(table.rows) if row.get("adopted") == "yes"]
    if len(adopted) > 1:
        messages.append(f"{len(adopted)} arms adopted; at most one")
    messages += [
        f"line {line}: the baseline row is adopted; adopt an arm or none"
        for line, row in adopted
        if row.get("role") == "baseline"
    ]
    return messages


def _verdict_clause(table: _Table) -> list[str]:
    return [
        f"line {line}: clause {cell!r} is not an id such as C1; leave it empty when no clause reads the arm"
        for line, cell in table.column("clause")
        if cell and not CLAUSE.match(cell)
    ]


def _statistics_columns(table: _Table) -> list[str]:
    numeric = _numeric_columns(table)
    if not numeric:
        return ["no count column: the table counts nothing"]
    if table.header[0] in numeric:
        return [f"the first column {table.header[0]!r} holds numbers; it names the unit"]
    first = table.header.index(numeric[0])
    return [
        f"text column {name!r} follows the count column {numeric[0]!r}; category columns come first"
        for name in table.header[first:]
        if name not in numeric
    ]


def _item_count(table: _Table) -> list[str]:
    numeric = _numeric_columns(table)
    return [
        f"line {line}: the item count {numeric[0]} {cell!r} is not an integer"
        for line, cell in table.column(numeric[0])
        if not INTEGER.match(cell)
    ]


def _summary_columns(table: _Table) -> list[str]:
    return _columns_differ(table, SUMMARY_COLUMNS, "a dataset summary")


def _summary_values(table: _Table) -> list[str]:
    values = table.column("value")
    messages = [] if len(values) >= 2 else [f"{len(values)} rows; row 1 is the item count, row 2 the main grouping"]
    messages += [
        f"line {line}: value {cell!r} is not a plain number" for line, cell in values if not _is_plain_number(cell)
    ]
    if values and _is_plain_number(values[0][1]) and not INTEGER.match(values[0][1]):
        messages.append(f"line 2: the item count {values[0][1]!r} is not an integer")
    return messages


def _fields_columns(table: _Table) -> list[str]:
    messages = _columns_differ(table, FIELDS_COLUMNS, "a dataset fields table")
    fields = [cell for _, cell in table.column("field")]
    return messages + [f"field {field!r} has more than one row" for field in _repeats(fields)]


def _field_types(table: _Table) -> list[str]:
    messages = []
    for line, row in _lines(table.rows):
        if row.get("type") not in FIELD_TYPES:
            messages.append(f"line {line}: type {row.get('type')!r} is not {', '.join(FIELD_TYPES)}")
        if row.get("distinct") and not INTEGER.match(row["distinct"]):
            messages.append(f"line {line}: distinct {row['distinct']!r} is not an integer")
    return messages


def _field_values(table: _Table) -> list[str]:
    messages = []
    for line, row in _lines(table.rows):
        values = [value for value in row.get("values", "").split(FIELD_VALUES_SEPARATOR.strip()) if value.strip()]
        if row.get("type") == "categorical" and not 1 <= len(values) <= FIELD_VALUES_MAX:
            messages.append(
                f"line {line}: a categorical field lists {len(values)} values; "
                f"list 1 to {FIELD_VALUES_MAX}, joined by {FIELD_VALUES_SEPARATOR!r}"
            )
        elif row.get("type") in ("numeric", "text") and values:
            messages.append(f"line {line}: a {row['type']} field lists values; leave them empty")
    return messages


def _variant_columns(table: _Table) -> list[str]:
    entry, predecessor = table.stem.split(VARIANT_SEPARATOR, 1)
    expected = ["form", _snake(predecessor), _snake(entry), "why"]
    return _columns_differ(table, expected, f"{table.stem}.csv")


def _variant_rows(table: _Table) -> list[str]:
    forms = table.column("form")
    messages = [f"line {line}: empty form" for line, form in forms if not form]
    messages += [f"form {form!r} has more than one row" for form in _repeats([form for _, form in forms if form])]
    messages += [
        f"line {line}: no why; say why the two cut the form as they do"
        for line, why in table.column("why")
        if not why.strip()
    ]
    return messages


def _snake(name: str) -> str:
    return name.replace("-", "_")


# the rules each standard table adds to the general ones, its column rule first
KIND_CHECKS: dict[str | None, list[tuple[str, Callable[[_Table], list[str]]]]] = {
    "clause-readings": [
        ("TAB-040", _clause_columns),
        ("TAB-041", _clause_ids),
        ("TAB-042", _reading_numbers),
        ("TAB-043", _outcome_words),
        ("TAB-044", _repeated_readings),
    ],
    "verdict-by-arm": [
        ("TAB-051", _verdict_lead),
        ("TAB-052", _arm_roles),
        ("TAB-053", _adoption),
        ("TAB-054", _verdict_clause),
    ],
    "dataset-statistics": [
        ("TAB-060", _statistics_columns),
        ("TAB-061", _item_count),
    ],
    "dataset-summary": [
        ("TAB-062", _summary_columns),
        ("TAB-063", _summary_values),
    ],
    "dataset-fields": [
        ("TAB-064", _fields_columns),
        ("TAB-065", _field_types),
        ("TAB-066", _field_values),
    ],
    "entry-variant": [
        ("TAB-071", _variant_columns),
        ("TAB-072", _variant_rows),
    ],
}
