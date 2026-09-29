"""Turn the judged sample into the record's tables.

Every stage of the curation pipeline makes one decision per document: keep it
or drop it. This script scores those decisions against the judge's labels and
writes the result to `tables/`, so the record cites a file rather than a number
somebody typed.

Two quantities carry the argument. Drop precision is the share of a stage's
dropped documents that are genuinely bad, which is what the stage is for.
Residual bad rate is the share of its kept documents that are bad, which is
what it failed to catch. Both are reported at the two bars D8 fixed before any
of this was computed: bad means coherence 2 or less, garbage or boilerplate, or
adult-or-spam, and the stricter bar moves coherence to 3 or less.

The sample is stratified — 40 documents per source, stage and decision — so a
rate is a statement about that cell, and the pooled rows weight every source
equally rather than by how much text it contributes. Corpus-weighted totals
need the per-stage funnel, which is step 11.

Run it with:

    uv run python experiments/data/curation-quality-slovenian/analysis.py
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Where this experiment's derived data lives, relative to the repository root.
DATA_ROOT = Path("data/experiments/data/curation-quality-slovenian")

#: Where the committed tables go, beside this script.
TABLES_DIR = Path(__file__).resolve().parent / "tables"

#: The stages whose decisions are scored against the judge, in pipeline order.
STAGES: Tuple[str, ...] = ("language", "spam", "quality", "repetition", "exact_dedup", "sentence_dedup")

#: Text shapes that are bad whatever the coherence score says.
BAD_TEXT_TYPES = frozenset({"garbage", "boilerplate"})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file written by the sampler or the judge.

    Args:
        path: File to read.

    Returns:
        One dict per non-empty line, in file order.
    """
    # split("\n"), not splitlines(): document text carries line separators such
    # as U+2028, which splitlines() would honour and so tear a JSON row in half.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def is_bad(verdict: Dict[str, Any], strict: bool = False) -> bool:
    """Decide whether a judged document falls below the bar (D8).

    Args:
        verdict: One judge verdict, carrying `coherence`, `text_type` and
            `adult_or_spam`.
        strict: Use the stricter bar, which also counts coherence 3.

    Returns:
        True when the document counts as bad text.
    """
    floor = 3 if strict else 2
    return verdict["coherence"] <= floor or verdict["text_type"] in BAD_TEXT_TYPES or bool(verdict["adult_or_spam"])


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Return a Wilson score interval for a share.

    The Wilson interval is used rather than the textbook normal one because
    several cells sit near 0 or 1, where the normal interval runs past the
    ends of the scale.

    Args:
        successes: Count of documents with the property.
        total: Count of documents in the cell.
        z: Standard-normal quantile; the default is the 95% interval.

    Returns:
        The interval's lower and upper bounds, or `(0.0, 0.0)` when the cell
        is empty.
    """
    if total == 0:
        return 0.0, 0.0
    share = successes / total
    denominator = 1 + z**2 / total
    centre = (share + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _cells(
    sample: Iterable[Dict[str, Any]], verdicts: Dict[str, Dict[str, Any]]
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Group judged documents by the source, stage and decision they came from.

    A document can be drawn into several cells — kept by one stage and dropped
    by the next — and counts once in each, which is what makes a cell's rate a
    statement about that stage's decision rather than about the document.

    Args:
        sample: Sample rows, each carrying `dataset` and its `cells`.
        verdicts: Judge verdicts by document id.

    Returns:
        Verdicts keyed by source, stage and decision.
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sample:
        verdict = verdicts.get(row["id"])
        if verdict is None:
            continue
        for cell in row["cells"]:
            grouped[(row["dataset"], cell["stage"], cell["decision"])].append(verdict)
    return grouped


def _rate(verdicts: List[Dict[str, Any]], strict: bool) -> Tuple[int, int, float, float, float]:
    """Score one cell at one bar.

    Args:
        verdicts: The cell's judge verdicts.
        strict: Use the stricter bar.

    Returns:
        Bad count, total count, share, and the share's interval bounds.
    """
    bad = sum(1 for verdict in verdicts if is_bad(verdict, strict))
    total = len(verdicts)
    low, high = wilson(bad, total)
    return bad, total, (bad / total if total else 0.0), low, high


def stage_rows(grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]], by_source: bool) -> List[Dict[str, Any]]:
    """Score every stage's decisions, pooled or per source.

    Args:
        grouped: Verdicts keyed by source, stage and decision.
        by_source: Report one row per source and stage instead of per stage.

    Returns:
        Rows ready for `write_csv`, in pipeline order.
    """
    sources = sorted({key[0] for key in grouped})
    rows: List[Dict[str, Any]] = []
    for stage in STAGES:
        for source in sources if by_source else [None]:
            dropped = [
                v
                for (src, stg, dec), vs in grouped.items()
                if stg == stage and dec == "dropped" and (source is None or src == source)
                for v in vs
            ]
            kept = [
                v
                for (src, stg, dec), vs in grouped.items()
                if stg == stage and dec == "kept" and (source is None or src == source)
                for v in vs
            ]
            if not dropped and not kept:
                continue
            row: Dict[str, Any] = {"stage": stage}
            if by_source:
                row["source"] = source
            for name, verdicts in (("drop_precision", dropped), ("residual_bad_rate", kept)):
                for bar, strict in (("", False), ("_strict", True)):
                    bad, total, share, low, high = _rate(verdicts, strict)
                    # A cell the sample never reached has no rate, and writing 0
                    # there would read as "this stage drops only good text".
                    empty = total == 0
                    row[f"{name}{bar}"] = "" if empty else round(share, 4)
                    if not bar:
                        row[f"{name}_n"] = total
                    row[f"{name}{bar}_low"] = "" if empty else round(low, 4)
                    row[f"{name}{bar}_high"] = "" if empty else round(high, 4)
                    row[f"{name}{bar}_bad"] = bad
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows to a CSV, creating the directory if needed.

    Args:
        path: File to write.
        rows: Rows sharing one set of keys.

    Returns:
        The path written.

    Raises:
        ValueError: If there are no rows to write.
    """
    if not rows:
        raise ValueError(f"nothing to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_stage_table(rows: List[Dict[str, Any]]) -> None:
    """Print the pooled per-stage table the way the record quotes it.

    Args:
        rows: Pooled rows from `stage_rows`.
    """
    print(f"{'stage':15s}{'drops':>7s}{'drop precision':>26s}{'kept':>7s}{'residual bad':>26s}")
    for row in rows:
        drop = f"{row['drop_precision']:.1%} [{row['drop_precision_low']:.1%}-{row['drop_precision_high']:.1%}]"
        keep = (
            f"{row['residual_bad_rate']:.1%} [{row['residual_bad_rate_low']:.1%}-{row['residual_bad_rate_high']:.1%}]"
        )
        strict = f"{row['drop_precision_strict']:.1%}"
        print(
            f"{row['stage']:15s}{row['drop_precision_n']:7d}{drop:>20s}{strict:>6s}"
            f"{row['residual_bad_rate_n']:7d}{keep:>20s}{row['residual_bad_rate_strict']:>6.1%}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Score the curation pipeline's decisions against the judge.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="This experiment's derived-data folder.")
    parser.add_argument("--verdicts", default="verdicts-full-sonnet.jsonl", help="Judge verdicts under interim/.")
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Where the CSV tables are written.")
    return parser.parse_args(argv)


def main() -> None:
    """Write the stage-metric tables and print the pooled one."""
    args = parse_args()
    sample = read_jsonl(args.data_root / "interim" / "sample.jsonl")
    verdicts = {row["id"]: row for row in read_jsonl(args.data_root / "interim" / args.verdicts)}
    print(f"{len(verdicts)} verdicts over {len(sample)} sampled documents")

    grouped = _cells(sample, verdicts)
    pooled = stage_rows(grouped, by_source=False)
    per_source = stage_rows(grouped, by_source=True)
    print_stage_table(pooled)
    for path in (
        write_csv(args.tables / "stage-decisions.csv", pooled),
        write_csv(args.tables / "source-stage-decisions.csv", per_source),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
