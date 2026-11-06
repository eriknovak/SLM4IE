"""Table the four Slovene NER test sets the benchmark scores models on.

`data` reads the validation and test splits of the four NER task registry
entries the record's `## Datasets` names (D9) and writes, per test set, the
summary, the fields and the statistics tables behind its row: documents and
characters per split, and spans, their share and their mean length per split
and label.

Run it with:

    uv run --group data python experiments/validation/ie-benchmark-slovenian/analysis.py data
"""

import argparse
import csv
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# ===============================
# Defaults
# ===============================

HERE = Path(__file__).resolve().parent
TABLES_DIR = HERE / "tables"
TASKS_DIR = HERE.parents[2] / "data" / "tasks"
SPLITS = ("val", "test")

# table name -> task registry entry, in the order the record lists them
TEST_SETS = {
    "ssj500k": "ner/ssj500k",
    "suk": "ner/suk",
    "janes-tag": "ner/janes_tag",
    "slavicner": "ner/slavicner",
}


# ===============================
# Helper functions
# ===============================


def read_split(entry: str, split: str) -> list[dict[str, Any]]:
    """Read one split of a task registry entry.

    Args:
        entry: Task registry key, e.g. `ner/suk`.
        split: Split name.

    Returns:
        The split's examples, each with `id`, `text` and `spans`.

    Raises:
        FileNotFoundError: If the split was not built.
    """
    path = TASKS_DIR / entry / f"{split}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"split {path} is missing; build it with `prepare_datasets.py tasks {entry}`")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def count_nested(spans: list[dict[str, Any]]) -> int:
    """Count spans that start inside an earlier span.

    Args:
        spans: One document's spans.

    Returns:
        The number of spans starting before an earlier span has ended.
    """
    nested = 0
    furthest_end = -1
    for start, end in sorted((s["start"], s["end"]) for s in spans):
        nested += start < furthest_end
        furthest_end = max(furthest_end, end)
    return nested


def summary_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the summary table of one test set.

    Args:
        splits: Split name -> examples.

    Returns:
        `statistic`, `value`, `note` rows.
    """
    rows: list[dict[str, Any]] = []
    for split, examples in splits.items():
        lengths = [len(e["text"]) for e in examples]
        spans = [s for e in examples for s in e["spans"]]
        rows += [
            {"statistic": f"{split}_documents", "value": len(examples), "note": "hash split of the held-out documents"},
            {"statistic": f"{split}_spans", "value": len(spans), "note": "gold entity mentions"},
            {"statistic": f"{split}_median_document_chars", "value": statistics.median_low(lengths), "note": ""},
            {"statistic": f"{split}_max_document_chars", "value": max(lengths), "note": "longest prompt input"},
            {
                "statistic": f"{split}_documents_without_spans",
                "value": sum(1 for e in examples if not e["spans"]),
                "note": "a model scores only false positives here",
            },
            {
                "statistic": f"{split}_nested_spans",
                "value": sum(count_nested(e["spans"]) for e in examples),
                "note": "spans starting inside another span",
            },
        ]
    return rows


def fields_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the fields table of one test set.

    Args:
        splits: Split name -> examples.

    Returns:
        `field`, `type`, `distinct`, `values` rows.
    """
    examples = [e for split in splits.values() for e in split]
    labels = sorted({s["label"] for e in examples for s in e["spans"]})
    return [
        {"field": "id", "type": "string", "distinct": len({e["id"] for e in examples}), "values": "<source>:<doc_id>"},
        {"field": "text", "type": "string", "distinct": len({e["text"] for e in examples}), "values": "document text"},
        {"field": "spans.start", "type": "integer", "distinct": "", "values": "character offset"},
        {"field": "spans.end", "type": "integer", "distinct": "", "values": "character offset, exclusive"},
        {"field": "spans.label", "type": "string", "distinct": len(labels), "values": " ".join(labels)},
    ]


def statistics_rows(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build the per split and label statistics table of one test set.

    Args:
        splits: Split name -> examples.

    Returns:
        One row per split and label, most frequent label first within a split.
    """
    rows: list[dict[str, Any]] = []
    for split, examples in splits.items():
        spans = [s for e in examples for s in e["spans"]]
        counts = Counter(s["label"] for s in spans)
        for label, count in counts.most_common():
            lengths = [s["end"] - s["start"] for s in spans if s["label"] == label]
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "spans": count,
                    "share_pct": 100 * count / len(spans),
                    "mean_span_chars": statistics.mean(lengths),
                    "documents_with_label": sum(1 for e in examples if any(s["label"] == label for s in e["spans"])),
                    "documents": len(examples),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as a CSV with the first row's keys as header.

    Args:
        path: Destination file.
        rows: Rows sharing one key set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ===============================
# Main function
# ===============================


def build_data() -> None:
    """Write the summary, fields and statistics tables of every test set."""
    for name, entry in TEST_SETS.items():
        splits = {split: read_split(entry, split) for split in SPLITS}
        write_csv(TABLES_DIR / f"dataset-{name}-summary.csv", summary_rows(splits))
        write_csv(TABLES_DIR / f"dataset-{name}-fields.csv", fields_rows(splits))
        write_csv(TABLES_DIR / f"dataset-{name}-statistics.csv", statistics_rows(splits))


def get_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Table the Slovene NER test sets of ie-benchmark-slovenian")
    parser.add_argument("command", choices=["data"], help="data: write the dataset statistics tables")
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
