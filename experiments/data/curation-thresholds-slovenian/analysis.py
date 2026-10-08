"""Turn the curated corpus and the judge verdicts into the record's tables.

The first pass restates the earlier audit's per-source baseline from the shared
corpus rebuilt at main's settings (D5). The audit's build counted five sources
twice and kept two sources that are now skipped as copies of others, so its
funnel cannot serve as the starting point for tuning. Every count here comes
from the stage sentinels and the statistics stage; the one stage without a
per-source sentinel, exact dedup, is counted once from its output and cached
under `interim/`.

The second pass scores each content filter on judged documents it dropped:
the share that is the filter's own target (D1) and, for the spam, quality and
repetition filters, the share that is bad text at the lenient bar. Rates are
read pooled over sources and per source, and called against the record's
bars (D2). By default it reads the earlier audit's sample and verdicts, the
evidence thresholds are tuned on (D3).

Run it with:

    uv run --group corpus --group analysis python experiments/data/curation-thresholds-slovenian/analysis.py
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the shared helper resolves

from slm4ie.data.curate.inspect.profile import count_source_documents  # noqa: E402
from slm4ie.utils.stats import wilson  # noqa: E402

#: Where this experiment's derived data lives, relative to the repository root.
DATA_ROOT = Path("data/experiments/data/curation-thresholds-slovenian")

#: Where the committed tables go, beside this script.
TABLES_DIR = Path(__file__).resolve().parent / "tables"

#: Where the committed figures go, beside this script.
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

#: The earlier audit's tables, read for its build's per-source funnel.
AUDIT_TABLES = Path(__file__).resolve().parents[1] / "curation-quality-slovenian" / "tables"

#: The stage folders that keep one sentinel per source, in pipeline order.
SCOPED_STAGES: Tuple[str, ...] = ("00_convert", "01_language", "02_spam", "03_quality", "04_repetition")

#: The corpus-wide stage whose per-source output has to be counted.
COUNTED_STAGE = "05_exact_dedup"

#: The stages that drop documents, named as in the record.
DROP_STAGES: Tuple[str, ...] = ("language", "spam", "quality", "repetition", "exact_dedup", "sentence_dedup")

#: Subword tokens per word until the project tokenizer exists (GLOSSARY: estimated tokens).
TOKENS_PER_WORD = 2

#: KPI 3, estimated tokens the whole corpus must exceed.
KPI3_TOKENS = 5_000_000_000

#: KPI 4, estimated tokens each domain must exceed.
KPI4_TOKENS = 500_000

#: The earlier audit's derived data: its judged sample and frozen verdicts.
AUDIT_DATA = Path("data/experiments/data/curation-quality-slovenian")

#: Text shapes that are bad whatever the coherence score says.
BAD_TEXT_TYPES = frozenset({"garbage", "boilerplate"})

#: What each content filter is built to remove, as a test on one verdict (D1).
TARGETS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "language": lambda verdict: verdict.get("language", "sl") != "sl",
    "spam": lambda verdict: bool(verdict["adult_or_spam"]),
    "quality": lambda verdict: verdict["coherence"] <= 2 or verdict["text_type"] == "garbage",
    "repetition": lambda verdict: verdict["text_type"] in BAD_TEXT_TYPES,
}

#: Filters also scored on bad text; whether text is Slovene says nothing of its quality (D1).
BAD_TEXT_STAGES = frozenset({"spam", "quality", "repetition"})

#: Judged drops a source needs before its own rate is read (D2).
MIN_SOURCE_DROPS = 20

#: A rate at or above this holds (D2).
CONFIRM_RATE = 0.6

#: A rate below this fails; between the two bars it is open (D2).
REFUTE_RATE = 0.5


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write rows as a CSV with the first row's keys as the header.

    Args:
        path: Destination file.
        rows: Rows sharing one set of keys.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV written by this script or the earlier audit's.

    Args:
        path: File to read.

    Returns:
        One dict per row.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_access(extract_config: Path) -> Dict[str, str]:
    """Read whether each source may be redistributed.

    Args:
        extract_config: `configs/data/extract.yaml`.

    Returns:
        `open` or `gated` per source.
    """
    catalog = yaml.safe_load(extract_config.read_text(encoding="utf-8")) or {}
    entries = catalog.get("datasets", catalog)
    return {key: (entry or {}).get("access", "open") for key, entry in entries.items() if isinstance(entry, dict)}


def scoped_counts(pretrain_dir: Path) -> Tuple[Dict[str, Dict[str, int]], Set[str]]:
    """Read each source's document count after every per-source stage.

    A stage must read exactly what the stage before it wrote for the same
    source. The earlier audit's build broke that for five sources, because
    stale output files sat beside fresh ones, and its counts were inflated.

    Args:
        pretrain_dir: The curation `output_dir`.

    Returns:
        Documents out per source and stage folder, and the commits the
        sentinels were built at.

    Raises:
        ValueError: If a stage read a different count than its predecessor wrote.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    commits: Set[str] = set()
    for previous, stage in zip((None, *SCOPED_STAGES), SCOPED_STAGES):
        for path in sorted((pretrain_dir / stage).glob("*/.complete")):
            sentinel = json.loads(path.read_text(encoding="utf-8"))
            source = path.parent.name
            if previous and sentinel["records_in"] != counts[source].get(previous):
                raise ValueError(
                    f"{stage}/{source} read {sentinel['records_in']} documents, "
                    f"{previous} wrote {counts[source].get(previous)}"
                )
            counts[source][stage] = sentinel["records_out"]
            commits.add((sentinel.get("info") or {}).get("git_commit", "unknown"))
    return counts, commits


def counted_stage(pretrain_dir: Path, cache: Path, workers: int) -> Dict[str, int]:
    """Count each source's documents after exact dedup, once.

    Exact dedup runs over the whole corpus and keeps no per-source sentinel,
    so its output is counted. The count reads every shard, so it is cached and
    reused while the stage's sentinel is unchanged.

    Args:
        pretrain_dir: The curation `output_dir`.
        cache: JSON file holding the counts and the sentinel they were taken at.
        workers: Shards counted at once.

    Returns:
        Documents per source after exact dedup.
    """
    sentinel = json.loads((pretrain_dir / COUNTED_STAGE / ".complete").read_text(encoding="utf-8"))
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("document_digest") == sentinel["document_digest"]:
            return cached["counts"]
    print(f"counting {COUNTED_STAGE} per source; this reads every shard")
    counts = count_source_documents(pretrain_dir / COUNTED_STAGE, workers=workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"document_digest": sentinel["document_digest"], "counts": counts}, indent=1))
    return counts


def funnel_rows(pretrain_dir: Path, exact_counts: Dict[str, int], access: Dict[str, str]) -> List[Dict[str, Any]]:
    """Report how many documents each source holds after every stage.

    Args:
        pretrain_dir: The curation `output_dir`.
        exact_counts: Documents per source after exact dedup.
        access: `open` or `gated` per source.

    Returns:
        One row per source, least retained first, with a totals row last.

    Raises:
        FileNotFoundError: If the statistics stage has not run.
    """
    counts, _ = scoped_counts(pretrain_dir)
    stats_dir = pretrain_dir / "07_statistics" / "per_dataset"
    if not stats_dir.is_dir():
        raise FileNotFoundError(f"no corpus statistics at {stats_dir}; run the statistics stage first")
    final = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in stats_dir.glob("*.json")}

    rows: List[Dict[str, Any]] = []
    for source in sorted(final):
        stages = counts[source]
        row: Dict[str, Any] = {
            "source": source,
            "domain": final[source]["domain"],
            "access": access.get(source, "open"),
        }
        row.update({stage[3:]: stages[stage] for stage in SCOPED_STAGES})
        row["exact_dedup"] = exact_counts[source]
        row["final"] = final[source]["doc_count"]
        row["final_words"] = final[source]["word_count"]
        row["tokens_estimated"] = TOKENS_PER_WORD * final[source]["word_count"]
        row["retained"] = round(row["final"] / row["convert"], 4)
        rows.append(row)
    rows.sort(key=lambda row: row["retained"])

    total: Dict[str, Any] = {"source": "TOTAL", "domain": "", "access": ""}
    for key in (*(stage[3:] for stage in SCOPED_STAGES), "exact_dedup", "final", "final_words", "tokens_estimated"):
        total[key] = sum(row[key] for row in rows)
    total["retained"] = round(total["final"] / total["convert"], 4)
    rows.append(total)
    return rows


def loss_rows(funnel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split each source's converted documents by the stage that dropped them.

    Args:
        funnel: Rows from `funnel_rows`.

    Returns:
        One row per source, in the funnel's order, with one share per stage and
        the share kept, summing to one.
    """
    rows: List[Dict[str, Any]] = []
    for row in funnel:
        if row["source"] == "TOTAL":
            continue
        loss: Dict[str, Any] = {"source": row["source"]}
        previous = row["convert"]
        for stage in DROP_STAGES:
            current = row["final"] if stage == "sentence_dedup" else row[stage]
            loss[stage] = round((previous - current) / row["convert"], 4)
            previous = current
        loss["kept"] = row["retained"]
        rows.append(loss)
    return rows


def build_comparison_rows(funnel: List[Dict[str, Any]], audit_funnel: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Set each source's retention in the rebuilt corpus beside the audit's build.

    A source the audit's build doubled kept its true converted count, so its
    retention there is still final over converted; what moved is how much of it
    the later stages let through.

    Args:
        funnel: Rows from `funnel_rows`.
        audit_funnel: The earlier audit's `source-funnel.csv`.

    Returns:
        One row per source in either build, the largest change first; a source
        missing from one build has blanks there.
    """
    rebuilt = {row["source"]: row for row in funnel if row["source"] != "TOTAL"}
    audit = {row["source"]: row for row in audit_funnel if row["source"] != "TOTAL"}
    rows: List[Dict[str, Any]] = []
    for source in sorted(set(rebuilt) | set(audit)):
        before, after = audit.get(source), rebuilt.get(source)
        retained_audit = float(before["retained"]) if before else ""
        retained_rebuild = after["retained"] if after else ""
        rows.append(
            {
                "source": source,
                "domain": (after or before)["domain"],
                "doubled_in_audit": bool(before) and before["duplicated_input"] == "True",
                "retained_audit": retained_audit,
                "retained_rebuild": retained_rebuild,
                "change": round(retained_rebuild - retained_audit, 4) if before and after else "",
                "words_audit": int(before["final_words"]) if before else "",
                "words_rebuild": after["final_words"] if after else "",
            }
        )
    rows.sort(key=lambda row: -abs(row["change"]) if row["change"] != "" else 1)
    return rows


def domain_token_rows(funnel: List[Dict[str, Any]], audit_funnel: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Total the finished corpus per domain in estimated tokens, against the KPIs.

    Domains are the labels each source declares (D4). Each domain is held to
    KPI 4 and the whole corpus to KPI 3.

    Args:
        funnel: Rows from `funnel_rows`.
        audit_funnel: The earlier audit's `source-funnel.csv`, for its totals.

    Returns:
        One row per domain, fewest tokens first, then the corpus total.
    """
    tokens: Dict[str, int] = defaultdict(int)
    documents: Dict[str, int] = defaultdict(int)
    sources: Dict[str, int] = defaultdict(int)
    audit_tokens: Dict[str, int] = defaultdict(int)
    for row in funnel:
        if row["source"] == "TOTAL":
            continue
        tokens[row["domain"]] += row["tokens_estimated"]
        documents[row["domain"]] += row["final"]
        sources[row["domain"]] += 1
    for row in audit_funnel:
        if row["source"] != "TOTAL":
            audit_tokens[row["domain"]] += TOKENS_PER_WORD * int(row["final_words"])

    total = sum(tokens.values())
    rows = [
        {
            "domain": domain,
            "sources": sources[domain],
            "documents": documents[domain],
            "tokens_estimated": tokens[domain],
            "tokens_estimated_audit": audit_tokens.get(domain, ""),
            "token_share": round(tokens[domain] / total, 4),
            "kpi_threshold": KPI4_TOKENS,
            "clears_kpi": tokens[domain] > KPI4_TOKENS,
        }
        for domain in sorted(tokens, key=tokens.get)
    ]
    rows.append(
        {
            "domain": "TOTAL",
            "sources": sum(sources.values()),
            "documents": sum(documents.values()),
            "tokens_estimated": total,
            "tokens_estimated_audit": sum(audit_tokens.values()),
            "token_share": 1.0,
            "kpi_threshold": KPI3_TOKENS,
            "clears_kpi": total > KPI3_TOKENS,
        }
    )
    return rows


def corpus_dataset_rows(funnel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Describe every source of the baseline corpus, for the record's Datasets section.

    Args:
        funnel: Rows from `funnel_rows`.

    Returns:
        One row per source with its domain, kind, access and sizes before and
        after curation.
    """
    return [
        {
            "source": row["source"],
            "domain": row["domain"],
            "kind": "raw web crawl" if row["domain"] == "web" else "curated",
            "access": row["access"],
            "converted": row["convert"],
            "finished": row["final"],
            "finished_words": row["final_words"],
        }
        for row in sorted(funnel, key=lambda row: row["source"])
        if row["source"] != "TOTAL"
    ]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file written by the sampler or the judge.

    Args:
        path: File to read.

    Returns:
        One dict per non-empty line, in file order.
    """
    # split("\n"), not splitlines(): document text carries U+2028, which
    # splitlines() would honour and so tear a JSON row in half.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def is_bad_text(verdict: Dict[str, Any]) -> bool:
    """Decide whether a judged document is bad text at the lenient bar.

    Args:
        verdict: One judge verdict, carrying `coherence`, `text_type` and
            `adult_or_spam`.

    Returns:
        True for coherence 2 or lower, garbage, boilerplate, or adult or spam.
    """
    return verdict["coherence"] <= 2 or verdict["text_type"] in BAD_TEXT_TYPES or bool(verdict["adult_or_spam"])


def judged_drops(
    sample: Iterable[Dict[str, Any]], verdicts: Dict[str, Dict[str, Any]], sources: Set[str]
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Collect the verdicts on documents each content filter dropped, per source.

    Args:
        sample: Sample rows, each carrying `id`, `dataset` and its `cells`
            (`stage`, `decision`).
        verdicts: Judge verdicts by document id.
        sources: The selected sources; a sampled source outside them is skipped.

    Returns:
        Verdicts keyed by stage and source.
    """
    drops: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sample:
        verdict = verdicts.get(row["id"])
        if verdict is None or row["dataset"] not in sources:
            continue
        for cell in row["cells"]:
            if cell["decision"] == "dropped" and cell["stage"] in TARGETS:
                drops[(cell["stage"], row["dataset"])].append(verdict)
    return drops


def call_rate(share: float) -> str:
    """Call a rate against the record's bars (D2).

    Args:
        share: The rate.

    Returns:
        `holds` at or above the confirm bar, `fails` below the refute bar,
        `open` between them.
    """
    if share >= CONFIRM_RATE:
        return "holds"
    return "fails" if share < REFUTE_RATE else "open"


def precision_rows(drops: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Score each content filter's drops on its target and on bad text.

    Args:
        drops: Verdicts keyed by stage and source, from `judged_drops`.

    Returns:
        Per filter, a pooled row then one row per source, in pipeline order.
        A source with fewer than `MIN_SOURCE_DROPS` judged drops is listed
        with `read` false and no call; the language filter has no bad-text
        columns.
    """
    rows: List[Dict[str, Any]] = []
    for stage, is_target in TARGETS.items():
        per_source = sorted((source, verdicts) for (stg, source), verdicts in drops.items() if stg == stage)
        pooled = [verdict for _, verdicts in per_source for verdict in verdicts]
        for source, verdicts in [("pooled", pooled), *per_source]:
            read = source == "pooled" or len(verdicts) >= MIN_SOURCE_DROPS
            row: Dict[str, Any] = {"stage": stage, "source": source, "drops_judged": len(verdicts), "read": read}
            bad_test = is_bad_text if stage in BAD_TEXT_STAGES else None
            for name, test in (("target", is_target), ("bad_text", bad_test)):
                if test is None or not verdicts:
                    row.update({f"{name}_{key}": "" for key in ("hits", "precision", "low", "high", "call")})
                    continue
                hits = sum(1 for verdict in verdicts if test(verdict))
                low, high = wilson(hits, len(verdicts))
                row[f"{name}_hits"] = hits
                row[f"{name}_precision"] = round(hits / len(verdicts), 4)
                row[f"{name}_low"] = round(low, 4)
                row[f"{name}_high"] = round(high, 4)
                row[f"{name}_call"] = call_rate(hits / len(verdicts)) if read else ""
            rows.append(row)
    return rows


def draw_figures(figures_dir: Path, losses: List[Dict[str, Any]], comparison: List[Dict[str, Any]]) -> List[Path]:
    """Draw the record's figures from the tables `main` has already built.

    Args:
        figures_dir: Where the SVGs go.
        losses: Rows from `loss_rows`.
        comparison: Rows from `build_comparison_rows`.

    Returns:
        The paths written.
    """
    import matplotlib.pyplot as plt
    from datachart.charts import BarChart, DumbbellChart
    from datachart.constants import BAR_MODE, FIG_SIZE, LEGEND_LOCATION, ORIENTATION

    # fixed clip-path ids, so a rerun on unchanged tables leaves the committed SVG untouched
    plt.rcParams["svg.hashsalt"] = "curation-thresholds-slovenian"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # long legend labels would squeeze the plot beside it, so the legend goes below
    def below(ncols: int) -> Dict[str, Any]:
        return {"show_legend": True, "legend": {"location": LEGEND_LOCATION.OUTSIDE_BOTTOM, "ncols": ncols}}

    figures: Dict[str, Any] = {}
    parts = (*DROP_STAGES, "kept")
    figures["curate-losses-by-source"] = BarChart(
        [[{"label": row["source"], "y": row[part]} for row in reversed(losses)] for part in parts],
        title="Document losses per source",
        subtitle=[f"dropped by {stage.replace('_', ' ')}" for stage in DROP_STAGES] + ["kept"],
        xlabel="share of converted documents",
        ylabel="source",
        xmax=1.0,
        bar_mode=BAR_MODE.STACK,
        orientation=ORIENTATION.HORIZONTAL,
        # the palette holds six colours, so the seventh part gets a neutral one
        style=[None] * len(DROP_STAGES) + [{"plot_bar_color": "#b8b8b8"}],
        figsize=FIG_SIZE.FULL_TALL,
        **below(3),
    )

    both = [row for row in comparison if row["change"] != ""]
    both.sort(key=lambda row: row["retained_rebuild"])
    figures["curate-retention-by-build-and-source"] = DumbbellChart(
        [
            {
                "label": row["source"] + (" *" if row["doubled_in_audit"] else ""),
                "start": row["retained_audit"],
                "end": row["retained_rebuild"],
            }
            for row in both
        ],
        title="Retention per source in two builds",
        start_name="earlier audit's build",
        end_name="rebuild at main's settings",
        xlabel="share of converted documents kept",
        ylabel="source (* doubled in the audit's build)",
        xmin=0.0,
        xmax=1.0,
        figsize=FIG_SIZE.FULL_TALL,
        **below(2),
    )

    written = []
    for name, figure in figures.items():
        path = figures_dir / f"{name}.svg"
        save_figure(figure, path)
        written.append(path)
    return written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse; the process's own when None.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Write the curation-thresholds record's tables and figures.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="This experiment's derived-data folder.")
    parser.add_argument("--pretrain-dir", type=Path, default=Path("data/pretrain"), help="The baseline corpus.")
    parser.add_argument("--extract-config", type=Path, default=Path("configs/data/extract.yaml"))
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Where the CSV tables are written.")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR, help="Where the SVG figures are written.")
    parser.add_argument("--workers", type=int, default=30, help="Shards counted at once for exact dedup.")
    parser.add_argument("--sample", type=Path, default=AUDIT_DATA / "interim" / "sample.jsonl", help="Judged sample.")
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=AUDIT_DATA / "final" / "judge-verdicts-sample.jsonl",
        help="Judge verdicts on that sample.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Write the record's tables and figures and print the headline numbers."""
    args = parse_args()
    _, commits = scoped_counts(args.pretrain_dir)
    print(f"baseline corpus built at {', '.join(sorted(commits))}")

    exact_counts = counted_stage(args.pretrain_dir, args.data_root / "interim" / "stage-counts.json", args.workers)
    funnel = funnel_rows(args.pretrain_dir, exact_counts, read_access(args.extract_config))
    audit_funnel = read_csv(AUDIT_TABLES / "source-funnel.csv")
    losses = loss_rows(funnel)
    comparison = build_comparison_rows(funnel, audit_funnel)
    domains = domain_token_rows(funnel, audit_funnel)

    for row in domains:
        print(f"{row['domain']:14s}{row['tokens_estimated']:>16,d} tokens  clears KPI: {row['clears_kpi']}")

    verdicts = {verdict["id"]: verdict for verdict in read_jsonl(args.verdicts)}
    sources = {row["source"] for row in funnel if row["source"] != "TOTAL"}
    precision = precision_rows(judged_drops(read_jsonl(args.sample), verdicts, sources))
    for row in precision:
        if row["source"] == "pooled":
            print(
                f"{row['stage']:12s} target {row['target_precision']} ({row['target_call']})"
                f"  bad text {row['bad_text_precision']} ({row['bad_text_call']})  n={row['drops_judged']}"
            )
    written = [
        write_csv(args.tables / "curate-documents-by-source.csv", funnel),
        write_csv(args.tables / "curate-losses-by-source.csv", losses),
        write_csv(args.tables / "curate-retention-by-build-and-source.csv", comparison),
        write_csv(args.tables / "curate-tokens-by-domain.csv", domains),
        write_csv(args.tables / "dataset-corpus-statistics.csv", corpus_dataset_rows(funnel)),
        write_csv(args.tables / "audit-precision-by-filter-and-source.csv", precision),
    ]
    written += draw_figures(args.figures, losses, comparison)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
