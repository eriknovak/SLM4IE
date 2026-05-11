r"""Report drop volume per curation stage from a `--debug` scratch folder.

After `scripts/data/curate.py --all --debug`, the scratch folder
(`<output_dir>/_dedup/` by default) contains one
`<stage>_dropped/<dataset>/<rank>.jsonl.gz` tree per filter/dedup
stage that ran with an `exclusion_writer`. This script walks those
trees and prints a per-stage table of `(n_dropped, total_words,
mean_words_per_doc)` so you can decide whether each pass earns its
keep on the next round of pipeline cuts.

The currently produced drop folders (see `slm4ie.data.curate.pipeline`)
are: `lang_dropped`, `quality_dropped`, `repetition_dropped`,
`exact_dropped`, `sentence_dropped`. Any other `*_dropped/` folder
is also picked up automatically.

For a corpus-wide percentage, pass `--input-dir` pointing at the
upstream `<output_dir>/datatrove` folder used by curate. The input
total is computed once and used as the denominator for every stage's
`% docs` and `% words` column.

Examples:
    # Pure stage table — no input total
    uv run python scripts/data/analyze_dedup_drops.py \\
        /vault/data/SLM4IE/final/_dedup

    # With percentages relative to the upstream corpus
    uv run python scripts/data/analyze_dedup_drops.py \\
        /vault/data/SLM4IE/final/_dedup \\
        --input-dir /vault/data/SLM4IE/processed/datatrove

    # Machine-readable JSON for downstream tooling
    uv run python scripts/data/analyze_dedup_drops.py \\
        /vault/data/SLM4IE/final/_dedup --json
"""

import argparse
import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageReport:
    """Aggregate metrics for one `*_dropped/` folder.

    Attributes:
        name: Stage label (folder basename with `_dropped` stripped).
        n_docs: Number of dropped documents.
        n_words: Sum of whitespace-split word counts across dropped docs.
        mean_words: `n_words / n_docs`, or 0 when `n_docs == 0`.
    """

    name: str
    n_docs: int
    n_words: int

    @property
    def mean_words(self) -> float:
        """Return `n_words / n_docs`, or `0.0` when no docs were dropped."""
        return (self.n_words / self.n_docs) if self.n_docs else 0.0


def _iter_jsonl_gz(folder: Path) -> Iterable[dict]:
    """Yield JSON records from every `*.jsonl.gz` file under *folder*.

    Args:
        folder: Folder to walk recursively. Missing folders yield
            nothing — they are not an error, because not every
            curation run produces every drop stage.

    Yields:
        One dict per JSONL line. Lines that fail to parse are
        skipped with a warning.
    """
    if not folder.is_dir():
        return
    for shard in sorted(folder.glob("**/*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line in %s", shard)


def _measure_folder(folder: Path, label: str) -> StageReport:
    """Walk a JSONL folder and return its `(name, n_docs, n_words)` triple.

    Args:
        folder: A `*_dropped/` or input-shard root.
        label: Display name for the resulting report.

    Returns:
        A `StageReport` with whitespace-split word counts. Missing
        folders return a zero-doc report rather than raising.
    """
    n_docs = 0
    n_words = 0
    for record in _iter_jsonl_gz(folder):
        text = record.get("text") or ""
        n_docs += 1
        n_words += len(text.split())
    return StageReport(name=label, n_docs=n_docs, n_words=n_words)


def _discover_drop_folders(scratch: Path) -> List[Path]:
    """Return every `*_dropped/` folder under *scratch* in lexical order.

    Args:
        scratch: The `--debug` scratch root (e.g. `<output_dir>/_dedup`).

    Returns:
        Sorted folder paths. Pipeline order is best preserved by the
        existing folder naming (`lang_dropped`, `quality_dropped`,
        `repetition_dropped`, `exact_dropped`, `sentence_dropped`); we
        re-order them explicitly so the report reads top-to-bottom in
        execution order.
    """
    canonical = [
        "lang_dropped",
        "quality_dropped",
        "repetition_dropped",
        "exact_dropped",
        "sentence_dropped",
    ]
    found = {p.name: p for p in scratch.iterdir() if p.is_dir() and p.name.endswith("_dropped")}
    ordered: List[Path] = []
    for name in canonical:
        if name in found:
            ordered.append(found.pop(name))
    ordered.extend(sorted(found.values(), key=lambda p: p.name))
    return ordered


def collect_reports(scratch: Path, input_dir: Optional[Path]) -> Dict[str, Any]:
    """Walk *scratch* and produce per-stage drop reports plus optional totals.

    Args:
        scratch: The `--debug` scratch root. Must exist.
        input_dir: Optional upstream input folder (the
            `<output_dir>/datatrove` fed to curate). When set, the
            return dict gets an `input` entry used for percentage
            denominators.

    Returns:
        Dict shaped as:
        `{"scratch": str, "input": Optional[dict], "stages": List[dict]}`.

    Raises:
        FileNotFoundError: If *scratch* is missing.
    """
    if not scratch.is_dir():
        raise FileNotFoundError(f"Scratch folder not found: {scratch}")

    stages = [_measure_folder(folder, folder.name.removesuffix("_dropped")) for folder in _discover_drop_folders(scratch)]
    input_report: Optional[StageReport] = None
    if input_dir is not None:
        input_report = _measure_folder(input_dir, "input")
        if input_report.n_docs == 0:
            logger.warning("input folder %s contained no documents", input_dir)

    return {
        "scratch": str(scratch),
        "input": (
            {"n_docs": input_report.n_docs, "n_words": input_report.n_words}
            if input_report is not None
            else None
        ),
        "stages": [
            {
                "name": s.name,
                "n_docs": s.n_docs,
                "n_words": s.n_words,
                "mean_words": s.mean_words,
            }
            for s in stages
        ],
    }


def format_table(report: Dict[str, Any]) -> str:
    """Render the result of `collect_reports` as a fixed-width text table.

    Args:
        report: The dict returned by `collect_reports`.

    Returns:
        Multiline string with a header line, one row per drop stage,
        and (when `report["input"]` is set) `% docs` / `% words`
        columns relative to the upstream input total.
    """
    input_info = report.get("input")
    stages = report["stages"]  # type: ignore[assignment]

    has_pct = input_info is not None
    columns = ["stage", "n_docs", "n_words", "mean_words"]
    if has_pct:
        columns.extend(["% docs", "% words"])

    rows: List[List[str]] = []
    input_docs = (input_info or {}).get("n_docs", 0) if has_pct else 0
    input_words = (input_info or {}).get("n_words", 0) if has_pct else 0
    for s in stages:
        row = [
            str(s["name"]),
            f"{s['n_docs']:,}",
            f"{s['n_words']:,}",
            f"{s['mean_words']:,.1f}",
        ]
        if has_pct:
            doc_pct = (100.0 * s["n_docs"] / input_docs) if input_docs else 0.0
            word_pct = (100.0 * s["n_words"] / input_words) if input_words else 0.0
            row.extend([f"{doc_pct:.2f}", f"{word_pct:.2f}"])
        rows.append(row)

    if has_pct and input_info is not None:
        rows.append(
            [
                "(input)",
                f"{input_info['n_docs']:,}",
                f"{input_info['n_words']:,}",
                "",
                "",
                "",
            ]
        )

    widths = [max(len(r[i]) for r in [columns] + rows) for i in range(len(columns))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    header = fmt.format(*columns)
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(fmt.format(*r) for r in rows)
    return "\n".join([header, sep, body])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace with `scratch`, `input_dir`, and `json_output`
        attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Report drop volume per curation stage from a `--debug` "
            "scratch folder produced by scripts/data/curate.py."
        )
    )
    parser.add_argument(
        "scratch",
        type=Path,
        help=(
            "The `--debug` scratch root (typically <output_dir>/_dedup, or "
            "the path passed via --debug-dir to curate.py)."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Upstream input folder (the <output_dir>/datatrove fed to "
            "curate.py). When set, the report adds `% docs` and `% words` "
            "columns relative to this total."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit JSON to stdout instead of the fixed-width text table.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the `analyze_dedup_drops.py` CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    report = collect_reports(args.scratch, args.input_dir)
    if args.json_output:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_table(report))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
