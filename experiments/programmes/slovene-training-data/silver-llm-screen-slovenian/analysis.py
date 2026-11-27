r"""Table the real passages and the writing briefs of silver-llm-screen-slovenian.

`data` reads the twenty real passages of round 1 and the twenty writing briefs
paired with them, and writes, for each of the two datasets, the summary, the
fields and the statistics tables behind its row in the record:

- passages: the count and the sets, the fields of one passage, and per set its
  domain, whether it is native, its passage count and its words per passage;
- briefs: the count and the domains, the fields of one brief, and per domain
  and label the briefs that require the label.

Every table goes through `save_table` from the shared `create_assets.py`, which
checks the rules of labflow's `tables.md` before writing.

Run it with:

    uv run --group analysis python \
        experiments/programmes/slovene-training-data/silver-llm-screen-slovenian/analysis.py data
"""

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from slm4ie.utils.config import load_yaml

# find the shared report helpers in experiments/tools/, or in experiments/ itself
EXPERIMENTS = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
TOOLS = EXPERIMENTS / "tools"
sys.path.insert(0, str(TOOLS if (TOOLS / "create_assets.py").is_file() else EXPERIMENTS))
from create_assets import save_table  # noqa: E402  — path set above

# ===============================
# Logging
# ===============================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

SLUG = "silver-llm-screen-slovenian"
HERE = Path(__file__).resolve().parent
TABLES_DIR = HERE / "tables"
PASSAGES_PATH = EXPERIMENTS.parent / "data" / "experiments" / SLUG / "final" / "passages-round1" / "passages.jsonl"
BRIEFS_PATH = HERE / "configs" / "briefs-passages.yaml"
MAX_LISTED_VALUES = 5

# ===============================
# Helper functions
# ===============================


def read_passages(path: Path) -> list[dict[str, Any]]:
    """Read the real passages.

    Args:
        path: The `passages.jsonl` file.

    Returns:
        One mapping per passage.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"passages file {path} is missing")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def categorical_row(field: str, values: list[Any]) -> dict[str, Any]:
    """Build the fields-table row of a categorical field.

    Args:
        field: Field name.
        values: The field's value on every item; empty values are left out.

    Returns:
        A `field`, `type`, `distinct`, `values` row listing the most frequent values.
    """
    counts = Counter(str(v) for v in values if v not in ("", None))
    listed = [value for value, _ in counts.most_common(MAX_LISTED_VALUES)]
    return {"field": field, "type": "categorical", "distinct": len(counts), "values": "; ".join(listed)}


def text_row(field: str, values: list[Any]) -> dict[str, Any]:
    """Build the fields-table row of a text field.

    Args:
        field: Field name.
        values: The field's value on every item.

    Returns:
        A `field`, `type`, `distinct`, `values` row.
    """
    return {"field": field, "type": "text", "distinct": len(set(values)), "values": None}


def numeric_row(field: str) -> dict[str, Any]:
    """Build the fields-table row of a numeric field.

    Args:
        field: Field name.

    Returns:
        A `field`, `type`, `distinct`, `values` row.
    """
    return {"field": field, "type": "numeric", "distinct": None, "values": None}


def passages_tables(passages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the three tables of the real passages.

    Args:
        passages: The real passages.

    Returns:
        Table kind -> rows.
    """
    by_set: dict[str, list[dict[str, Any]]] = {}
    for passage in passages:
        by_set.setdefault(passage["set"], []).append(passage)
    ordered = sorted(by_set.items(), key=lambda item: -len(item[1]))

    # summary tiles
    words = [p["words"] for p in passages]
    summary = [
        {"statistic": "passages", "value": len(passages), "note": "real passages written by people"},
        {"statistic": "sets", "value": len(by_set), "note": "; ".join(name for name, _ in ordered)},
        {"statistic": "documents", "value": len({p["doc_id"] for p in passages}), "note": "source documents"},
        {"statistic": "mean_words", "value": statistics.mean(words), "note": "words per passage"},
    ]

    # fields of one passage
    fields = [
        categorical_row("set", [p["set"] for p in passages]),
        categorical_row("domain", [p["domain"] for p in passages]),
        categorical_row("native", [p["native"] for p in passages]),
        text_row("passage_id", [p["passage_id"] for p in passages]),
        text_row("doc_id", [p["doc_id"] for p in passages]),
        categorical_row("group", [p["group"] for p in passages]),
        text_row("text", [p["text"] for p in passages]),
        numeric_row("words"),
    ]

    # per set statistics
    stats = [
        {
            "set": name,
            "domain": items[0]["domain"],
            "native": str(items[0]["native"]).lower(),
            "passages": len(items),
            "mean_words": statistics.mean(p["words"] for p in items),
            "min_words": min(p["words"] for p in items),
            "max_words": max(p["words"] for p in items),
        }
        for name, items in ordered
    ]
    return {"summary": summary, "fields": fields, "statistics": stats}


def briefs_tables(briefs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the three tables of the writing briefs.

    Args:
        briefs: The brief specs.

    Returns:
        Table kind -> rows.
    """
    domains = Counter(b["domain"] for b in briefs)

    # summary tiles
    summary = [
        {"statistic": "briefs", "value": len(briefs), "note": "one brief per real passage"},
        {"statistic": "domains", "value": len(domains), "note": "; ".join(name for name, _ in domains.most_common())},
        {"statistic": "mean_words", "value": statistics.mean(b["words"] for b in briefs), "note": "target words"},
    ]

    # fields of one brief
    fields = [
        text_row("brief_id", [b["brief_id"] for b in briefs]),
        text_row("passage_id", [b["passage_id"] for b in briefs]),
        categorical_row("domain", [b["domain"] for b in briefs]),
        text_row("text_type", [b["text_type"] for b in briefs]),
        text_row("topic", [b["topic"] for b in briefs]),
        numeric_row("words"),
        categorical_row("labels", [label for b in briefs for label in b["labels"]]),
    ]

    # briefs requiring each label, largest first within a domain
    stats: list[dict[str, Any]] = []
    for domain, _ in domains.most_common():
        counts = Counter(label for b in briefs if b["domain"] == domain for label in b["labels"])
        stats += [{"domain": domain, "label": label, "briefs": count} for label, count in counts.most_common()]
    return {"summary": summary, "fields": fields, "statistics": stats}


def write_tables(dataset: str, tables: dict[str, list[dict[str, Any]]]) -> None:
    """Write a dataset's tables through `save_table`.

    Args:
        dataset: The dataset name in `dataset-<name>-<kind>`.
        tables: Kind (`summary`, `fields`, `statistics`) -> rows.

    Raises:
        TableCheckError: If a table breaks a rule of tables.md; nothing is written.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for kind, rows in tables.items():
        columns = list(dict.fromkeys(key for row in rows for key in row))
        save_table(rows, TABLES_DIR / f"dataset-{dataset}-{kind}.csv", columns=columns)
        logger.info("wrote dataset-%s-%s.csv (%d rows)", dataset, kind, len(rows))


# ===============================
# Main
# ===============================


def build_data() -> None:
    """Write the six dataset tables of the passages and the briefs."""
    write_tables("passages", passages_tables(read_passages(PASSAGES_PATH)))
    write_tables("briefs", briefs_tables(load_yaml(BRIEFS_PATH)["briefs"]))


def get_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Table the passages and briefs of silver-llm-screen-slovenian")
    parser.add_argument("command", choices=["data"], help="data: write the dataset tables")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the chosen command.

    Args:
        args: Parsed command-line arguments.
    """
    if args.command == "data":
        build_data()


if __name__ == "__main__":
    main(get_args())
