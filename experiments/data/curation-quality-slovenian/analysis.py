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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from slm4ie.data.curate.profile import iter_stage_sentinels
from slm4ie.data.judge import interleave

#: Where this experiment's derived data lives, relative to the repository root.
DATA_ROOT = Path("data/experiments/data/curation-quality-slovenian")

#: Where the committed tables go, beside this script.
TABLES_DIR = Path(__file__).resolve().parent / "tables"

#: Where the committed figures go, beside this script.
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

#: The stages whose decisions are scored against the judge, in pipeline order.
STAGES: Tuple[str, ...] = ("language", "spam", "quality", "repetition", "exact_dedup", "sentence_dedup")

#: How a pair can come out once both orders are read, in the order reported.
PAIRWISE_CALLS: Tuple[str, ...] = ("kept", "dropped", "tie", "orders_differ")

#: Stages that remove duplicates rather than bad text (D9).
DEDUP_STAGES = frozenset({"exact_dedup", "sentence_dedup"})

#: Text shapes that are bad whatever the coherence score says.
BAD_TEXT_TYPES = frozenset({"garbage", "boilerplate"})


#: Every datatrove executor the curation run launches, in pipeline order. The
#: scoped stages launch one per config bucket but all of them log into the same
#: folder, so only the bucket that ran last leaves its timings (issue #10).
LOG_STEPS: Tuple[Tuple[str, str], ...] = (
    ("language", ""),
    ("spam", ""),
    ("quality", ""),
    ("repetition", ""),
    ("exact_dedup", "1_sig"),
    ("exact_dedup", "2_find"),
    ("exact_dedup", "3_filter"),
    ("sentence_dedup", "1_sig"),
    ("sentence_dedup", "2_find"),
    ("sentence_dedup", "3_filter"),
    ("statistics", "1_map"),
    ("statistics", "2_reduce"),
)

#: Stages that run once over the whole corpus rather than once per bucket.
CORPUS_STAGES = frozenset({"exact_dedup", "sentence_dedup", "statistics"})

#: The curated corpus, as folders, from conversion to the finished text.
FUNNEL_STAGES: Tuple[str, ...] = (
    "00_convert",
    "01_language",
    "02_spam",
    "03_quality",
    "04_repetition",
)


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


def is_bad(verdict: Dict[str, Any], strict: bool = False, stage: Optional[str] = None) -> bool:
    """Decide whether a judged document falls below the bar (D8).

    At the language filter a document is also bad when the judge does not call
    it Slovene, since removing other languages is that filter's whole purpose;
    the bar would otherwise count its correct drops of well-written English as
    mistakes.

    Args:
        verdict: One judge verdict, carrying `coherence`, `text_type`,
            `adult_or_spam` and `language`.
        strict: Use the stricter bar, which also counts coherence 3.
        stage: The stage whose decision is scored, or None for no stage.

    Returns:
        True when the document counts as bad text for that stage.
    """
    floor = 3 if strict else 2
    if stage == "language" and verdict.get("language", "sl") != "sl":
        return True
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


def _rate(verdicts: List[Dict[str, Any]], strict: bool, stage: str) -> Tuple[int, int, float, float, float]:
    """Score one cell at one bar.

    Args:
        verdicts: The cell's judge verdicts.
        strict: Use the stricter bar.
        stage: The stage whose decisions the cell holds.

    Returns:
        Bad count, total count, share, and the share's interval bounds.
    """
    bad = sum(1 for verdict in verdicts if is_bad(verdict, strict, stage))
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
                    bad, total, share, low, high = _rate(verdicts, strict, stage)
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


def lost_text_rows(unmatched: List[Dict[str, Any]], verdicts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score what the dedup stages lost, rather than what they dropped.

    A dedup drop whose twin survives costs the corpus nothing, so only the
    unmatched drops are worth reading — and the question about them is the
    ordinary one: was the text any good? These rows answer it from the
    verdicts the rubric pass already produced, so nothing is judged twice.

    Args:
        unmatched: Rows written by the `duplication` subcommand, each carrying
            `id`, `stage`, `dataset`, `window_coverage` and `sharers`.
        verdicts: Judge verdicts by document id.

    Returns:
        One row per dedup stage, in the order the stages appear.
    """
    rows: List[Dict[str, Any]] = []
    for stage in (name for name in STAGES if any(row["stage"] == name for row in unmatched)):
        lost = [row for row in unmatched if row["stage"] == stage]
        judged = [verdicts[row["id"]] for row in lost if row["id"] in verdicts]
        good = sum(1 for verdict in judged if not is_bad(verdict))
        low, high = wilson(good, len(judged))
        coverages = sorted(row["window_coverage"] for row in lost)
        rows.append(
            {
                "stage": stage,
                "lost": len(lost),
                "judged": len(judged),
                "good_text": good,
                "good_text_share": round(good / len(judged), 4) if judged else "",
                "good_text_low": round(low, 4),
                "good_text_high": round(high, 4),
                "median_window_coverage": round(coverages[len(coverages) // 2], 4) if coverages else "",
                "shares_no_window": sum(1 for row in lost if not row["sharers"]),
            }
        )
    return rows


def funnel_rows(pretrain_dir: Path, extra_counts: Optional[Dict[str, Dict[str, int]]] = None) -> List[Dict[str, Any]]:
    """Report how many documents each source keeps at every stage.

    The scoped stages record their own counts in a sentinel beside their
    output, and the finished corpus is described by the statistics stage, so
    the funnel needs no corpus read. A source whose language stage emitted more
    than it read is flagged: a filter cannot do that, and the cause is stale
    shards from an earlier run (issue #9), which inflates every later count for
    that source.

    Args:
        pretrain_dir: The curation `output_dir`.
        extra_counts: Per-source counts for stages that keep no per-source
            sentinel, keyed by stage folder then source. The dedup stages run
            over the whole corpus, so theirs have to be counted.

    Returns:
        One row per source, ordered by how much of it survived, with a totals
        row last.

    Raises:
        FileNotFoundError: If the statistics stage has not run.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    for stage, source, _, records_out in iter_stage_sentinels(pretrain_dir, FUNNEL_STAGES):
        counts[source][stage] = records_out
    for stage, per_source in (extra_counts or {}).items():
        for source, value in per_source.items():
            counts[source][stage] = value

    stats_dir = pretrain_dir / "07_statistics" / "per_dataset"
    if not stats_dir.is_dir():
        raise FileNotFoundError(f"no corpus statistics at {stats_dir}; run the statistics stage first")
    final = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in stats_dir.glob("*.json")}

    rows: List[Dict[str, Any]] = []
    for source, stages in counts.items():
        converted = stages.get("00_convert", 0)
        if source not in final or not converted:
            continue
        row: Dict[str, Any] = {"source": source, "domain": final[source]["domain"]}
        row.update({stage[3:]: stages.get(stage, "") for stage in (*FUNNEL_STAGES, *sorted(extra_counts or ()))})
        row["final"] = final[source]["doc_count"]
        row["final_words"] = final[source]["word_count"]
        row["retained"] = round(final[source]["doc_count"] / converted, 4)
        row["duplicated_input"] = stages.get("01_language", 0) > converted
        rows.append(row)

    rows.sort(key=lambda row: row["retained"])
    total: Dict[str, Any] = {"source": "TOTAL", "domain": ""}
    for stage in (*FUNNEL_STAGES, *sorted(extra_counts or ())):
        total[stage[3:]] = sum(row[stage[3:]] for row in rows if isinstance(row[stage[3:]], int))
    total["final"] = sum(row["final"] for row in rows)
    total["final_words"] = sum(row["final_words"] for row in rows)
    total["retained"] = round(total["final"] / total["convert"], 4) if total["convert"] else ""
    total["duplicated_input"] = sum(1 for row in rows if row["duplicated_input"])
    rows.append(total)
    return rows


def domain_rows(pretrain_dir: Path) -> List[Dict[str, Any]]:
    """Compare the corpus's domain mix before curation with the mix after it.

    Each source declares one domain, so the mix going in is its document counts
    grouped by that declaration. Documents are the only unit available before
    curation — nothing counts words at conversion — so the shares before and
    after are compared as documents, with the finished word share beside them.

    Args:
        pretrain_dir: The curation `output_dir`.

    Returns:
        One row per domain, largest final word share first.
    """
    rows = [row for row in funnel_rows(pretrain_dir) if row["source"] != "TOTAL"]
    before: Dict[str, int] = defaultdict(int)
    after: Dict[str, int] = defaultdict(int)
    words: Dict[str, int] = defaultdict(int)
    for row in rows:
        before[row["domain"]] += row["convert"]
        after[row["domain"]] += row["final"]
        words[row["domain"]] += row["final_words"]
    total_before, total_after, total_words = sum(before.values()), sum(after.values()), sum(words.values())
    return sorted(
        (
            {
                "domain": domain,
                "documents_before": before[domain],
                "share_before": round(before[domain] / total_before, 4),
                "documents_after": after[domain],
                "share_after": round(after[domain] / total_after, 4),
                "words_after": words[domain],
                "word_share_after": round(words[domain] / total_words, 4),
                "retained": round(after[domain] / before[domain], 4) if before[domain] else "",
            }
            for domain in before
        ),
        key=lambda row: -row["word_share_after"],
    )


def gated_rows(pretrain_dir: Path, extract_config: Path) -> List[Dict[str, Any]]:
    """Split the corpus totals by whether a source can be redistributed.

    A public release cannot ship the sources whose licence is login-bound, so
    the corpus has two sizes and the smaller one is what an outside reader can
    reproduce.

    Args:
        pretrain_dir: The curation `output_dir`.
        extract_config: `configs/data/extract.yaml`, read for each source's
            `access` field.

    Returns:
        One row for open sources, one for gated, and one for the total.
    """
    access = _access(extract_config)
    rows = [row for row in funnel_rows(pretrain_dir) if row["source"] != "TOTAL"]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[access.get(row["source"], "open")].append(row)

    out: List[Dict[str, Any]] = []
    for name in ("open", "gated"):
        group = groups.get(name, [])
        out.append(
            {
                "access": name,
                "sources": len(group),
                "documents": sum(row["final"] for row in group),
                "words": sum(row["final_words"] for row in group),
                "source_names": " ".join(sorted(row["source"] for row in group)),
            }
        )
    out.append(
        {
            "access": "total",
            "sources": len(rows),
            "documents": sum(row["final"] for row in rows),
            "words": sum(row["final_words"] for row in rows),
            "source_names": "",
        }
    )
    return out


def _access(extract_config: Path) -> Dict[str, str]:
    """Read whether each source may be redistributed.

    Args:
        extract_config: `configs/data/extract.yaml`.

    Returns:
        `open` or `gated` per source.
    """
    catalog = yaml.safe_load(extract_config.read_text(encoding="utf-8")) or {}
    entries = catalog.get("datasets", catalog)
    return {key: (entry or {}).get("access", "open") for key, entry in entries.items() if isinstance(entry, dict)}


def corpus_dataset_rows(funnel: List[Dict[str, Any]], extract_config: Path) -> List[Dict[str, Any]]:
    """Describe every source of the curated corpus, for the record's Datasets section.

    Args:
        funnel: Rows from `funnel_rows`.
        extract_config: `configs/data/extract.yaml`, read for each source's access.

    Returns:
        One row per source with its domain, kind, access and sizes before and
        after curation.
    """
    access = _access(extract_config)
    return [
        {
            "source": row["source"],
            "domain": row["domain"],
            "kind": "raw web crawl" if row["domain"] == "web" else "curated",
            "access": access.get(row["source"], "open"),
            "converted": row["convert"],
            "finished": row["final"],
            "finished_words": row["final_words"],
        }
        for row in sorted(funnel, key=lambda row: row["source"])
        if row["source"] != "TOTAL"
    ]


def sample_dataset_rows(sample: List[Dict[str, Any]], verdicts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Count the judged sample per stage and decision.

    Args:
        sample: Sample rows, each carrying the `cells` it was drawn into.
        verdicts: Judge verdicts by document id.

    Returns:
        One row per stage and decision, in pipeline order.
    """
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sample:
        for cell in row["cells"]:
            cells[(cell["stage"], cell["decision"])].append(row)
    return [
        {
            "stage": stage,
            "decision": decision,
            "documents": len(rows),
            "judged": sum(1 for row in rows if row["id"] in verdicts),
            "sources": len({row["dataset"] for row in rows}),
        }
        for stage in STAGES
        for decision in ("dropped", "kept")
        if (rows := cells.get((stage, decision)))
    ]


def pairs_dataset_rows(pairs: List[Dict[str, Any]], answers: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Count the pairwise drawing per stage.

    Args:
        pairs: The drawing, one row per comparison.
        answers: The judge's answers, keyed by comparison id.

    Returns:
        One row per stage with its pairs, comparisons answered and sources.
    """
    rows = []
    for stage in STAGES:
        comparisons = [pair for pair in pairs if pair["stage"] == stage]
        if not comparisons:
            continue
        rows.append(
            {
                "stage": stage,
                "pairs": len({(pair["dataset"], pair["pair"]) for pair in comparisons}),
                "comparisons": len(comparisons),
                "answered": sum(1 for pair in comparisons if pair["id"] in answers),
                "sources": len({pair["dataset"] for pair in comparisons}),
            }
        )
    return rows


def labels_dataset_rows(
    calibration: List[Dict[str, Any]], adjudication: List[Dict[str, Any]], sample: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Describe the two sets of hand labels.

    Args:
        calibration: The frozen calibration labels.
        adjudication: The frozen adjudication labels.
        sample: Sample rows, to find each labelled document's source.

    Returns:
        One row per label set.
    """
    source = {row["id"]: row["dataset"] for row in sample}
    return [
        {
            "label_set": name,
            "documents": len(labels),
            "sources": len({source[label["id"]] for label in labels if label["id"] in source}),
            "bad_text_share": round(sum(is_bad(label) for label in labels) / len(labels), 3) if labels else "",
        }
        for name, labels in (("calibration", calibration), ("adjudication", adjudication))
    ]


def profile_rows(profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the corpus profile into one row per source.

    Args:
        profiles: Output of `slm4ie.data.curate.profile.profile_corpus`, read
            from the JSON it was written to.

    Returns:
        One row per source, longest median document first.
    """
    rows = [
        {
            "source": source,
            "documents_sampled": profile["documents_sampled"],
            "words_p5": int(profile["words"]["p5"]),
            "words_p50": int(profile["words"]["p50"]),
            "words_p95": int(profile["words"]["p95"]),
            "chars_p50": int(profile["chars"]["p50"]),
            "type_token_ratio": profile["type_token_ratio"],
            "type_token_budget": profile["type_token_budget"],
            "oov_rate": profile.get("oov_rate", ""),
            "slovene_share": round(profile.get("language", {}).get("in_language_share", 0.0), 4),
            "slovene_confidence_p5": round(profile.get("language", {}).get("confidence", {}).get("p5", 0.0), 4),
            "slovene_confidence_p50": round(profile.get("language", {}).get("confidence", {}).get("p50", 0.0), 4),
            "other_languages": " ".join(
                f"{code}:{count}"
                for code, count in profile.get("language", {}).get("predicted", {}).items()
                if code != "sl"
            ),
            "duplicate_count_mean": round(profile["duplicate_count_mean"], 3),
        }
        for source, profile in profiles.items()
    ]
    return sorted(rows, key=lambda row: -row["words_p50"])


def _block_label(name: str) -> str:
    """Reduce a datatrove block name to the part worth reading.

    Args:
        name: The block's name as datatrove writes it, with its emoji.

    Returns:
        The role and implementation, such as `FILTER: Gopher Quality`.
    """
    return " ".join("".join(char for char in name.split(" - ", 1)[-1] if char.isascii()).split())


def _step_timings(logging_dir: Path) -> Optional[Dict[str, Any]]:
    """Read one executor's task timings out of its log folder.

    A task's end is its stats file's modification time, which is the only
    trustworthy clock here: the task logs are rewritten by whichever executor
    used the folder last, so their opening timestamps can belong to a different
    run than the stats beside them. Its duration is its slowest block rather
    than the sum of its blocks, because a block's timer stays open while the
    blocks downstream of it consume what it yields — the filter's total already
    contains the writer's, so summing them double-counts. Subtracting the
    duration from the end recovers when the executor began.

    datatrove never removes a stats file, so a folder reused by a smaller
    executor still holds the higher ranks of the larger one before it. Those
    ranks are skipped, which is what keeps a scoped stage's numbers to the one
    bucket that ran last rather than silently mixing buckets.

    Args:
        logging_dir: An executor's `logging_dir` under `_logs/`.

    Returns:
        Task and worker counts, documents read, CPU and wall seconds, the day
        it finished and CPU seconds per pipeline block; None if the executor
        never ran or finished no task. The CPU seconds are the pacing block's,
        so they are the step's compute and not the blocks' sum.
    """
    executor_path = logging_dir / "executor.json"
    if not executor_path.is_file():
        return None
    executor = json.loads(executor_path.read_text(encoding="utf-8"))
    ends: List[float] = []
    durations: List[float] = []
    blocks: Dict[str, float] = defaultdict(float)
    documents = 0
    for path in sorted((logging_dir / "stats").glob("*.json")):
        if int(path.stem) >= executor["tasks"]:
            continue
        stats = json.loads(path.read_text(encoding="utf-8"))
        durations.append(max(block["time_stats"]["total"] for block in stats))
        ends.append(path.stat().st_mtime)
        for block in stats:
            blocks[_block_label(block["name"])] += block["time_stats"]["total"]
        documents += int(stats[0]["stats"].get("documents", {}).get("total", 0))
    if not ends:
        return None
    start = min(end - duration for end, duration in zip(ends, durations))
    return {
        "tasks": executor["tasks"],
        "workers": executor["workers"],
        "documents": documents,
        "cpu_seconds": sum(durations),
        "wall_seconds": max(ends) - start,
        "finished": datetime.fromtimestamp(max(ends)).strftime("%Y-%m-%d"),
        "blocks": dict(blocks),
    }


def throughput_rows(pretrain_dir: Path) -> List[Dict[str, Any]]:
    """Report what every pipeline step cost in machine time.

    Two numbers say different things. CPU hours is what the step consumed and
    is comparable between steps; wall hours is what it took on this machine at
    the worker count it was given, so it changes with the machine. The two
    divided give how many workers the step kept busy on average, which is well
    under the pool wherever tasks finish at uneven speeds and the last few run
    alone.

    The scoped stages are reported for one config bucket only, because every
    bucket logs into the same folder and the last one overwrites the rest
    (issue #10). The corpus stages run once over everything, so theirs are
    complete and only those are totalled.

    Args:
        pretrain_dir: The curation `output_dir`, holding `_logs/`.

    Returns:
        One row per executor step in pipeline order, each naming the block
        that set its pace, with a totals row over the corpus-wide steps last.
    """
    rows: List[Dict[str, Any]] = []
    for stage, step in LOG_STEPS:
        timings = _step_timings(pretrain_dir / "_logs" / stage / step)
        if timings is None:
            continue
        wall, cpu, documents = timings["wall_seconds"], timings["cpu_seconds"], timings["documents"]
        slowest = max(timings["blocks"], key=lambda block: timings["blocks"][block])
        rows.append(
            {
                "stage": stage,
                "step": step,
                "scope": "corpus" if stage in CORPUS_STAGES else "one bucket",
                "finished": timings["finished"],
                "tasks": timings["tasks"],
                "workers": timings["workers"],
                "documents": documents or "",
                "cpu_hours": round(cpu / 3600, 2),
                "wall_hours": round(wall / 3600, 2),
                "docs_per_second": round(documents / wall, 1) if documents and wall else "",
                "docs_per_cpu_second": round(documents / cpu, 1) if documents and cpu else "",
                "worker_use": round(cpu / (wall * timings["workers"]), 3) if wall else "",
                "pacing_block": slowest,
            }
        )
    corpus = [row for row in rows if row["scope"] == "corpus"]
    total = {key: "" for key in rows[0]}
    total.update(
        {
            "stage": "TOTAL",
            "scope": "corpus",
            "cpu_hours": round(sum(row["cpu_hours"] for row in corpus), 2),
            "wall_hours": round(sum(row["wall_hours"] for row in corpus), 2),
        }
    )
    rows.append(total)
    return rows


def pairwise_rows(pairs: List[Dict[str, Any]], verdicts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Report how often the judge preferred what each stage kept over what it dropped.

    Every pair was asked twice with the documents swapped, and a pair counts
    for one side only when both orders agree: the kept document won both, the
    dropped document won both, or both were called a tie. Anything else is a
    pair whose answer depended on the order, which is the position bias the
    swap exists to expose.

    Args:
        pairs: The drawing, one row per comparison, naming the slot that held
            the kept document.
        verdicts: The judge's answers, keyed by comparison id.

    Returns:
        One row per stage pooled over sources (source `ALL`), then one per
        stage and source.
    """
    outcomes: Dict[Tuple[str, str, int], List[str]] = defaultdict(list)
    for comparison in pairs:
        verdict = verdicts.get(comparison["id"])
        if verdict is None:
            continue
        better = verdict["better"]
        side = "tie" if better == "tie" else ("kept" if better == comparison["kept"] else "dropped")
        outcomes[(comparison["stage"], comparison["dataset"], comparison["pair"])].append(side)

    grouped: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for (stage, source, _), sides in outcomes.items():
        call = sides[0] if len(sides) == 2 and sides[0] == sides[1] else "orders_differ"
        grouped[(stage, "ALL")][call] += 1
        grouped[(stage, source)][call] += 1

    rows: List[Dict[str, Any]] = []
    for (stage, source), calls in sorted(
        grouped.items(), key=lambda item: (item[0][1] != "ALL", item[0][1], STAGES.index(item[0][0]))
    ):
        total = sum(calls.values())
        row: Dict[str, Any] = {"stage": stage, "source": source, "pairs": total}
        row.update({call: round(calls[call] / total, 4) for call in PAIRWISE_CALLS})
        rows.append(row)
    return rows


def loss_rows(funnel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split each source's converted documents by the stage that dropped them.

    A source the stale-shard defect inflated (issue #9) leaves the language
    stage with more documents than it entered with, so its shares are taken
    over that stage's output instead: the language loss reads zero and the
    duplicates surface as exact_dedup loss, which is where they are removed.

    Args:
        funnel: Rows from `funnel_rows`, holding the exact_dedup count.

    Returns:
        One row per source, in the funnel's order, with one share per stage and
        the share kept, summing to one.
    """
    rows: List[Dict[str, Any]] = []
    for row in funnel:
        if row["source"] == "TOTAL":
            continue
        inflated = row["duplicated_input"] is True
        base = row["language"] if inflated else row["convert"]
        loss: Dict[str, Any] = {"source": row["source"], "base": base, "inflated_input": inflated}
        previous = base
        for stage in STAGES:
            current = row["final"] if stage == "sentence_dedup" else row[stage]
            if stage == "language" and inflated:
                current = base
            loss[stage] = round((previous - current) / base, 4)
            previous = current
        loss["kept"] = round(row["final"] / base, 4)
        rows.append(loss)
    return rows


def draw_figures(
    figures_dir: Path,
    pooled: List[Dict[str, Any]],
    per_source: List[Dict[str, Any]],
    pairwise: List[Dict[str, Any]],
    losses: List[Dict[str, Any]],
    throughput: List[Dict[str, Any]],
) -> List[Path]:
    """Draw the record's figures from the tables `main` has already built.

    Each figure carries one finding and is drawn from a table that is also
    written to `tables/`, so a number read off a figure can be checked there.

    Args:
        figures_dir: Where the SVGs go.
        pooled: Rows from `stage_rows` pooled over sources.
        per_source: Rows from `stage_rows` per source.
        pairwise: Rows from `pairwise_rows`.
        losses: Rows from `loss_rows`.
        throughput: Rows from `throughput_rows`.

    Returns:
        The paths written.
    """
    import matplotlib.pyplot as plt
    from datachart.charts import BarChart, DumbbellChart, Heatmap
    from datachart.constants import BAR_MODE, FIG_SIZE, LEGEND_LOCATION, ORIENTATION, VALUE_FORMAT

    plt.rcParams["svg.hashsalt"] = "curation-quality-slovenian"
    figures_dir.mkdir(parents=True, exist_ok=True)
    horizontal = {"orientation": ORIENTATION.HORIZONTAL, "show_values": True}
    # a legend inside the axes would sit on marks that reach their edge
    beside = {"show_legend": True, "legend": {"location": LEGEND_LOCATION.OUTSIDE_RIGHT}}
    figures: Dict[str, Any] = {}

    # the gap between the dots is how well a stage separates bad text from good
    figures["judge-rubric-drop-precision-by-stage"] = DumbbellChart(
        [{"label": row["stage"], "start": row["residual_bad_rate"], "end": row["drop_precision"]} for row in pooled],
        start_name="kept documents that are bad",
        end_name="dropped documents that are bad",
        xlabel="share judged bad",
        # room left of zero for the labels of dots that sit near it
        xmin=-0.05,
        xmax=0.4,
        show_values=True,
        value_format=VALUE_FORMAT.PERCENT_INT,
        figsize=FIG_SIZE.FULL_SHORT,
        **beside,
    )

    # DumbbellChart takes no xticks, and a share has no negative tick to show
    figures["judge-rubric-drop-precision-by-stage"].axes[0].set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])

    sources = sorted({row["source"] for row in per_source})
    cells = {(row["source"], row["stage"]): row["drop_precision"] for row in per_source if row["drop_precision"] != ""}
    figures["judge-rubric-drop-precision-by-source"] = Heatmap(
        {
            "x": list(STAGES),
            "y": sources,
            "z": [[cells.get((source, stage)) for stage in STAGES] for source in sources],
        },
        xlabel="stage",
        vmin=0.0,
        vmax=1.0,
        value_format=VALUE_FORMAT.PERCENT_INT,
        show_values=True,
        show_colorbar=True,
        xtickrotate=30,
        figsize=FIG_SIZE.FULL_TALL,
    )

    pooled_pairs = [row for row in reversed(pairwise) if row["source"] == "ALL"]
    figures["judge-pairwise-outcome-by-stage"] = BarChart(
        [[{"label": row["stage"], "y": row[call]} for row in pooled_pairs] for call in PAIRWISE_CALLS],
        subtitle=["kept document better", "dropped document better", "tie", "orders disagree"],
        xlabel="share of pairs",
        xmax=1.0,
        bar_mode=BAR_MODE.STACK,
        figsize=FIG_SIZE.FULL_SHORT,
        orientation=ORIENTATION.HORIZONTAL,
        **beside,
    )

    parts = (*STAGES, "kept")
    figures["curate-losses-by-source"] = BarChart(
        [
            [
                {"label": row["source"] + (" *" if row["inflated_input"] else ""), "y": row[part]}
                for row in reversed(losses)
            ]
            for part in parts
        ],
        subtitle=[f"dropped by {stage}" for stage in STAGES] + ["kept"],
        xlabel="share of converted documents (* of language output, issue #9)",
        xmax=1.0,
        bar_mode=BAR_MODE.STACK,
        orientation=ORIENTATION.HORIZONTAL,
        # the palette holds six colours, so the seventh part gets a neutral one
        style=[None] * len(STAGES) + [{"plot_bar_color": "#b8b8b8"}],
        figsize=FIG_SIZE.FULL_MEDIUM,
        **beside,
    )

    steps = [row for row in throughput if row["stage"] != "TOTAL"]
    figures["curate-cpu-hours-by-step"] = BarChart(
        [
            {
                "label": f"{row['stage']}/{row['step']}" if row["step"] else f"{row['stage']} (one bucket)",
                "y": row["cpu_hours"],
            }
            for row in reversed(steps)
        ],
        xlabel="CPU hours",
        value_format=VALUE_FORMAT.DECIMAL,
        figsize=FIG_SIZE.FULL_MEDIUM,
        **horizontal,
    )

    written = []
    for name, figure in figures.items():
        path = figures_dir / f"{name}.svg"
        # no date, so a rerun on unchanged tables leaves the committed SVG untouched
        figure.savefig(path, bbox_inches="tight", metadata={"Date": None})
        written.append(path)
    return written


def _kappa(pairs: List[Tuple[bool, bool]]) -> Tuple[float, float]:
    """Return Cohen's kappa and raw agreement for two raters' yes-or-no calls.

    Args:
        pairs: One (first rater, second rater) pair per document.

    Returns:
        Kappa and the share of documents both raters called alike.
    """
    total = len(pairs)
    observed = sum(first == second for first, second in pairs) / total
    first_yes = sum(first for first, _ in pairs) / total
    second_yes = sum(second for _, second in pairs) / total
    expected = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)
    return ((observed - expected) / (1 - expected) if expected < 1 else 1.0), observed


def agreement_rows(
    labels: List[Dict[str, Any]], runs: Dict[str, Dict[str, Dict[str, Any]]], reference: str
) -> List[Dict[str, Any]]:
    """Report how far each judge run agrees with the person, and the runs with each other.

    Agreement is taken on the collapsed keep-or-drop decision at both bars
    (D8), which is what the calibration gate reads (D5); coherence is reported
    beside it on its 1-5 scale.

    Args:
        labels: The person's calibration labels.
        runs: Judge verdicts by run name, each keyed by document id.
        reference: The run the others are compared against as well.

    Returns:
        One row per run against the person, then one per other run against
        the reference run over every document both judged.
    """
    comparisons: List[Tuple[str, str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]] = []
    for name, verdicts in runs.items():
        comparisons.append(
            (name, "person", [(label, verdicts[label["id"]]) for label in labels if label["id"] in verdicts])
        )
    for name, verdicts in runs.items():
        if name == reference:
            continue
        shared = [(runs[reference][doc_id], verdicts[doc_id]) for doc_id in runs[reference] if doc_id in verdicts]
        comparisons.append((name, reference, shared))

    rows: List[Dict[str, Any]] = []
    for name, against, pairs in comparisons:
        lenient, raw = _kappa([(is_bad(first), is_bad(second)) for first, second in pairs])
        strict, _ = _kappa([(is_bad(first, strict=True), is_bad(second, strict=True)) for first, second in pairs])
        gaps = [abs(first["coherence"] - second["coherence"]) for first, second in pairs]
        rows.append(
            {
                "judge_run": name,
                "compared_with": against,
                "documents": len(pairs),
                "kappa": round(lenient, 3),
                "raw_agreement": round(raw, 3),
                "kappa_strict": round(strict, 3),
                "coherence_exact": round(sum(gap == 0 for gap in gaps) / len(gaps), 3),
                "coherence_within_one": round(sum(gap <= 1 for gap in gaps) / len(gaps), 3),
            }
        )
    return rows


def draw_adjudications(
    sample: List[Dict[str, Any]],
    verdicts: Dict[str, Dict[str, Any]],
    labelled_sources: Iterable[str],
    size: int = 30,
    seed: int = 20260930,
) -> List[Dict[str, Any]]:
    """Choose the documents where the judge and the pipeline flatly conflict.

    A flat conflict is a content filter dropping text the judge calls clean
    even at the strict bar, or keeping text it calls bad at the lenient one;
    the dedup stages are left out because a duplicate is not bad text (D9).
    Only sources the calibration labels never reached are drawn from, since
    agreement there is what D5 leaves untested. The draw is stratified by
    source and conflict direction and interleaved, so stopping early still
    covers every source.

    The conflicts are found with the bar as it stood when the set was drawn,
    without the language filter's own rule, so the labelled set can be drawn
    again exactly.

    Args:
        sample: Sample rows, each carrying the `cells` it was drawn into.
        verdicts: The judge's verdicts, keyed by document id.
        labelled_sources: Sources the calibration labels already cover.
        size: How many documents to draw.
        seed: Seed fixing the draw and its order.

    Returns:
        The chosen rows in labelling order, each with the conflicting cells
        under `conflicts`.
    """
    skip = set(labelled_sources)
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for document in sample:
        verdict = verdicts.get(document["id"])
        if verdict is None or document["dataset"] in skip:
            continue
        conflicts = [
            cell
            for cell in document["cells"]
            if cell["stage"] not in DEDUP_STAGES
            and (
                (cell["decision"] == "dropped" and not is_bad(verdict, strict=True))
                or (cell["decision"] == "kept" and is_bad(verdict))
            )
        ]
        if conflicts:
            direction = "dropped_clean" if conflicts[0]["decision"] == "dropped" else "kept_bad"
            strata[(document["dataset"], direction)].append({**document, "conflicts": conflicts})
    return interleave(strata, size, seed)


def adjudication_rows(adjudications: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Report how often a person sides with the judge where it and the pipeline conflict.

    Every document here was drawn because the two disagree, so agreement
    statistics such as kappa mean nothing on it; the question is simply which
    side the person takes. The person's keep-or-drop call uses the D8 bar.

    Args:
        adjudications: Rows from `draw_adjudications`, with their `conflicts`.
        labels: The person's labels on them; the last label for an id wins.

    Returns:
        One row per conflict direction and a pooled row, each with the share
        of labelled documents where the person sided with the judge and its
        Wilson interval.
    """
    by_id = {label["id"]: label for label in labels}
    sided: Dict[str, List[bool]] = defaultdict(list)
    for document in adjudications:
        label = by_id.get(document["id"])
        if label is None:
            continue
        dropped = document["conflicts"][0]["decision"] == "dropped"
        direction = "pipeline dropped, judge clean" if dropped else "pipeline kept, judge bad"
        stage = document["conflicts"][0]["stage"]
        with_judge = not is_bad(label, stage=stage) if dropped else is_bad(label, stage=stage)
        sided[direction].append(with_judge)
        sided["all"].append(with_judge)

    rows: List[Dict[str, Any]] = []
    for direction, calls in sorted(sided.items(), key=lambda item: item[0] == "all"):
        low, high = wilson(sum(calls), len(calls))
        rows.append(
            {
                "conflict": direction,
                "labelled": len(calls),
                "sides_with_judge": sum(calls),
                "share": round(sum(calls) / len(calls), 4),
                "share_low": round(low, 4),
                "share_high": round(high, 4),
            }
        )
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


def print_funnel(rows: List[Dict[str, Any]]) -> None:
    """Print the survival funnel the way the record quotes it.

    Args:
        rows: Rows from `funnel_rows`.
    """
    print(f"\n{'source':18s}{'converted':>12s}{'final':>12s}{'retained':>10s}{'words':>16s}")
    for row in rows:
        flag = "  (inflated input, #9)" if row["duplicated_input"] is True else ""
        retained = f"{row['retained']:.1%}" if row["retained"] != "" else ""
        print(f"{row['source']:18s}{row['convert']:12,}{row['final']:12,}{retained:>10s}{row['final_words']:16,}{flag}")


def print_throughput(rows: List[Dict[str, Any]]) -> None:
    """Print the machine cost of each step the way the record quotes it.

    Args:
        rows: Rows from `throughput_rows`.
    """
    print(f"\n{'step':26s}{'docs':>12s}{'cpu h':>9s}{'wall h':>8s}{'docs/core s':>13s}{'workers used':>14s}")
    for row in rows:
        name = f"{row['stage']}/{row['step']}" if row["step"] else row["stage"]
        docs = f"{row['documents']:,}" if row["documents"] != "" else ""
        rate = f"{row['docs_per_cpu_second']:,.1f}" if row["docs_per_cpu_second"] != "" else ""
        used = f"{row['worker_use'] * row['workers']:.1f} of {row['workers']}" if row["worker_use"] != "" else ""
        flag = "  (one bucket only)" if row["scope"] == "one bucket" else ""
        print(f"{name:26s}{docs:>12s}{row['cpu_hours']:>9.2f}{row['wall_hours']:>8.2f}{rate:>13s}{used:>14s}{flag}")


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
    parser.add_argument(
        "--calibration-verdicts",
        default="verdicts-sonnet.jsonl",
        help="The judge run the calibration gate was read on, under interim/.",
    )
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Where the CSV tables are written.")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR, help="Where the SVG figures are written.")
    parser.add_argument(
        "--pairwise", default="pairwise-opus.jsonl", help="Pairwise answers under interim/; the drawing sits beside it."
    )
    parser.add_argument(
        "--pretrain-dir", type=Path, default=Path("data/pretrain"), help="The curation output_dir holding the stages."
    )
    parser.add_argument(
        "--extract-config", type=Path, default=Path("configs/data/extract.yaml"), help="Read for each source's access."
    )
    parser.add_argument(
        "--draw-adjudications",
        type=int,
        metavar="N",
        help="Draw N judge-vs-pipeline conflicts to interim/adjudication.jsonl for hand labelling, then stop.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Write the record's tables and figures and print the headline ones."""
    args = parse_args()
    sample = read_jsonl(args.data_root / "interim" / "sample.jsonl")
    verdicts = {row["id"]: row for row in read_jsonl(args.data_root / "interim" / args.verdicts)}
    print(f"{len(verdicts)} verdicts over {len(sample)} sampled documents")

    if args.draw_adjudications:
        destination = args.data_root / "interim" / "adjudication.jsonl"
        if destination.exists():
            raise FileExistsError(f"{destination} exists; labels may already refer to it")
        meta = json.loads((args.data_root / "final" / "human-labels-calibration.meta.json").read_text())
        drawn = draw_adjudications(sample, verdicts, meta["sources"], size=args.draw_adjudications)
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drawn), encoding="utf-8")
        print(f"wrote {len(drawn)} conflicts to {destination}")
        return

    grouped = _cells(sample, verdicts)
    pooled = stage_rows(grouped, by_source=False)
    per_source = stage_rows(grouped, by_source=True)
    print_stage_table(pooled)
    written = [
        write_csv(args.tables / "stage-decisions.csv", pooled),
        write_csv(args.tables / "source-stage-decisions.csv", per_source),
    ]

    # Two runs of one judge: the calibration pass the gate was read on, and the
    # full-sample pass every stage metric uses.
    calibration_labels = read_jsonl(args.data_root / "final" / "human-labels-calibration.jsonl")
    runs = {
        "calibration pass": {
            row["id"]: row for row in read_jsonl(args.data_root / "interim" / args.calibration_verdicts)
        },
        "full-sample pass": verdicts,
    }
    agreement = agreement_rows(calibration_labels, runs, reference="calibration pass")
    for row in agreement:
        print(
            f"{row['judge_run']:18s} vs {row['compared_with']:18s} n={row['documents']:4d} kappa {row['kappa']:.2f}, raw {row['raw_agreement']:.1%}"
        )
    written.append(write_csv(args.tables / "calibration-agreement.csv", agreement))

    answers_path = args.data_root / "interim" / args.pairwise
    answers = {row["id"]: row for row in read_jsonl(answers_path)}
    pairwise = pairwise_rows(read_jsonl(answers_path.with_suffix(".pairs.jsonl")), answers)
    written.append(write_csv(args.tables / "pairwise-outcomes.csv", pairwise))

    # Written by the counting job for the stages that keep no per-source
    # sentinel; the funnel simply leaves their column out when it is absent.
    counts_path = args.data_root / "interim" / "stage-counts.json"
    extra_counts = json.loads(counts_path.read_text(encoding="utf-8")) if counts_path.is_file() else None
    funnel = funnel_rows(args.pretrain_dir, extra_counts)
    print_funnel(funnel)
    throughput = throughput_rows(args.pretrain_dir)
    print_throughput(throughput)
    written += [
        write_csv(args.tables / "source-funnel.csv", funnel),
        write_csv(args.tables / "domain-mix.csv", domain_rows(args.pretrain_dir)),
        write_csv(args.tables / "gated-totals.csv", gated_rows(args.pretrain_dir, args.extract_config)),
        write_csv(args.tables / "throughput.csv", throughput),
    ]

    profile_path = args.data_root / "interim" / "corpus-profile.json"
    if profile_path.is_file():
        written.append(
            write_csv(args.tables / "corpus-profile.csv", profile_rows(json.loads(profile_path.read_text())))
        )

    # Written by `curate_pretraining_corpus.py duplication`, which has to read
    # the corpus; absent until that has run.
    unmatched_path = args.data_root / "interim" / "dedup-unmatched.jsonl"
    if unmatched_path.is_file():
        lost = lost_text_rows(read_jsonl(unmatched_path), verdicts)
        for row in lost:
            print(
                f"{row['stage']:15s}{row['lost']:5d} lost, {row['good_text']} good text "
                f"({row['good_text_share']:.1%} [{row['good_text_low']:.1%}-{row['good_text_high']:.1%}])"
            )
        written.append(write_csv(args.tables / "dedup-lost-text.csv", lost))

    adjudication_labels_path = args.data_root / "final" / "human-labels-adjudication.jsonl"
    written += [
        write_csv(args.tables / "dataset-corpus-statistics.csv", corpus_dataset_rows(funnel, args.extract_config)),
        write_csv(args.tables / "dataset-sample-statistics.csv", sample_dataset_rows(sample, verdicts)),
        write_csv(
            args.tables / "dataset-pairs-statistics.csv",
            pairs_dataset_rows(read_jsonl(answers_path.with_suffix(".pairs.jsonl")), answers),
        ),
        write_csv(
            args.tables / "dataset-labels-statistics.csv",
            labels_dataset_rows(
                calibration_labels,
                read_jsonl(adjudication_labels_path) if adjudication_labels_path.is_file() else [],
                sample,
            ),
        ),
    ]

    losses = loss_rows(funnel)
    written.append(write_csv(args.tables / "source-losses.csv", losses))
    # Labelled by hand in label.py (--set adjudication), then frozen to final/.
    adjudicated_path = args.data_root / "final" / "human-labels-adjudication.jsonl"
    if adjudicated_path.is_file():
        adjudicated = adjudication_rows(
            read_jsonl(args.data_root / "interim" / "adjudication.jsonl"), read_jsonl(adjudicated_path)
        )
        for row in adjudicated:
            print(
                f"{row['conflict']:32s}{row['sides_with_judge']:3d} of {row['labelled']:2d} side with the judge "
                f"({row['share']:.0%} [{row['share_low']:.0%}-{row['share_high']:.0%}])"
            )
        written.append(write_csv(args.tables / "adjudication.csv", adjudicated))

    written += draw_figures(args.figures, pooled, per_source, pairwise, losses, throughput)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
