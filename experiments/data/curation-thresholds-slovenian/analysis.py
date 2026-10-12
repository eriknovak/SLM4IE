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

The third pass reads a candidate build, the corpus rebuilt with every source on
its route (M1), beside the baseline: how much of each source survives, which
stage gained or lost it, and whether the corpus and every domain still clear
the volume targets (D4).

Run it with:

    uv run --group corpus --group analysis python experiments/data/curation-thresholds-slovenian/analysis.py
"""

import argparse
import copy
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import yaml
from datatrove.data import Document
from datatrove.utils.typeshelper import Languages
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the shared helper resolves

from slm4ie.data.curate.config import (  # noqa: E402
    effective_stage_config,
    load_curate_config,
    lock_path_for,
    resolve_profiles,
)
from slm4ie.data.curate.inspect.profile import count_source_documents  # noqa: E402
from slm4ie.data.curate.resources.stopwords import load_stopwords  # noqa: E402
from slm4ie.data.curate.stages.quality import LetterShareGopherQualityFilter, _build_quality_config  # noqa: E402
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

#: The stage folder the quality filter reads, holding every document that reaches it.
QUALITY_INPUT = "02_spam"

#: A document's id in a shard line, compact (datatrove) or spaced (convert), escapes kept.
ID_PATTERN = re.compile(r'"id":\s*"((?:[^"\\]|\\.)*)"')

#: Version of the text-fetch cache layout; bump it when the lookup changes.
FETCH_FORMAT = 2

#: The quality setting behind each drop reason; setting it to 0 switches that rule off.
REASON_SETTINGS = {
    "gopher_short_doc": "min_doc_words",
    "gopher_long_doc": "max_doc_words",
    "gopher_below_avg_threshold": "min_avg_word_length",
    "gopher_above_avg_threshold": "max_avg_word_length",
    "gopher_too_many_hashes": "max_symbol_word_ratio",
    "gopher_too_many_ellipsis": "max_symbol_word_ratio",
    "gopher_too_many_bullets": "max_bullet_lines_ratio",
    "gopher_too_many_end_ellipsis": "max_ellipsis_lines_ratio",
    "gopher_below_alpha_threshold": "max_non_alpha_words_ratio",
    "gopher_enough_stop_words": "min_stop_words",
}

#: The corpus-wide stages after exact dedup, each with one sentinel for the whole corpus.
CORPUS_STAGES: Tuple[str, ...] = ("06_sentence_dedup", "07_statistics")

#: The stages that drop documents, named as in the record.
DROP_STAGES: Tuple[str, ...] = ("language", "spam", "quality", "repetition", "exact_dedup", "sentence_dedup")

#: Subword tokens per word until the project tokenizer exists (GLOSSARY: estimated tokens).
TOKENS_PER_WORD = 2

#: KPI 3, estimated tokens the whole corpus must exceed.
KPI3_TOKENS = 5_000_000_000

#: KPI 4, estimated tokens each domain must exceed.
KPI4_TOKENS = 500_000

#: The candidate build's config; its stem names the tables and figures read from that build.
ROUTES_CONFIG = Path(__file__).resolve().parent / "configs" / "curation-routes.yaml"

#: Each route's name in the record's words, for figure labels.
ROUTE_NAMES = {"raw_crawl": "raw crawl", "curated_prose": "curated prose", "medical_academic": "medical, academic"}

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


def loss_shares(row: Dict[str, Any]) -> Dict[str, float]:
    """Share of one source's converted documents that each drop stage removed, unrounded.

    Args:
        row: One source's row from `funnel_rows`.

    Returns:
        The share per stage in `DROP_STAGES`.
    """
    shares: Dict[str, float] = {}
    previous = row["convert"]
    for stage in DROP_STAGES:
        current = row["final"] if stage == "sentence_dedup" else row[stage]
        shares[stage] = (previous - current) / row["convert"]
        previous = current
    return shares


def loss_rows(funnel: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split each source's converted documents by the stage that dropped them.

    Args:
        funnel: Rows from `funnel_rows`.

    Returns:
        One row per source, in the funnel's order, with one share per stage and
        the share kept, summing to one.
    """
    return [
        {
            "source": row["source"],
            **{stage: round(share, 4) for stage, share in loss_shares(row).items()},
            "kept": row["retained"],
        }
        for row in funnel
        if row["source"] != "TOTAL"
    ]


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


def domain_token_rows(
    funnel: List[Dict[str, Any]], earlier_funnel: List[Dict[str, Any]], earlier: str = "audit"
) -> List[Dict[str, Any]]:
    """Total the finished corpus per domain in estimated tokens, against the KPIs.

    Domains are the labels each source declares (D4). Each domain is held to
    KPI 4 and the whole corpus to KPI 3.

    Args:
        funnel: Rows from `funnel_rows`.
        earlier_funnel: The funnel of the build compared against, for its totals.
        earlier: Name of that build, suffixed to its token column.

    Returns:
        One row per domain, fewest tokens first, then the corpus total.
    """
    tokens: Dict[str, int] = defaultdict(int)
    documents: Dict[str, int] = defaultdict(int)
    sources: Dict[str, int] = defaultdict(int)
    earlier_tokens: Dict[str, int] = defaultdict(int)
    for row in funnel:
        if row["source"] == "TOTAL":
            continue
        tokens[row["domain"]] += row["tokens_estimated"]
        documents[row["domain"]] += row["final"]
        sources[row["domain"]] += 1
    for row in earlier_funnel:
        if row["source"] != "TOTAL":
            earlier_tokens[row["domain"]] += TOKENS_PER_WORD * int(row["final_words"])

    total = sum(tokens.values())
    rows = [
        {
            "domain": domain,
            "sources": sources[domain],
            "documents": documents[domain],
            "tokens_estimated": tokens[domain],
            f"tokens_estimated_{earlier}": earlier_tokens.get(domain, ""),
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
            f"tokens_estimated_{earlier}": sum(earlier_tokens.values()),
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


def read_routes(config: Path) -> Dict[str, str]:
    """Read the route each source takes in a candidate build's config.

    Args:
        config: A curation config whose `overrides:` block names a `profile`
            per source.

    Returns:
        The route name per source; a source without one is absent.
    """
    overrides = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("overrides") or {}
    return {source: entry["profile"] for source, entry in overrides.items() if (entry or {}).get("profile")}


def route_comparison_rows(
    baseline: List[Dict[str, Any]], candidate: List[Dict[str, Any]], routes: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Set each source's retention in a candidate build beside the baseline's.

    Args:
        baseline: Rows from `funnel_rows` over the baseline corpus.
        candidate: Rows from `funnel_rows` over the candidate build.
        routes: The route per source, from `read_routes`.

    Returns:
        One row per source, the largest change in retention first, then the
        corpus total.

    Raises:
        ValueError: If the two builds hold different sources, or a source
            converted to a different number of documents in each.
    """
    before = {row["source"]: row for row in baseline}
    after = {row["source"]: row for row in candidate}
    if set(before) != set(after):
        raise ValueError(f"the builds hold different sources: {sorted(set(before) ^ set(after))}")
    unequal = sorted(source for source in after if after[source]["convert"] != before[source]["convert"])
    if unequal:
        raise ValueError(f"the builds converted different documents for {unequal}")
    rows = [
        {
            "source": source,
            "domain": row["domain"],
            "route": routes.get(source, ""),
            "retained_baseline": before[source]["retained"],
            "retained_candidate": row["retained"],
            "change": round((row["final"] - before[source]["final"]) / row["convert"], 4),
            "words_baseline": before[source]["final_words"],
            "words_candidate": row["final_words"],
        }
        for source, row in after.items()
    ]
    rows.sort(key=lambda row: (row["source"] == "TOTAL", -abs(row["change"])))
    return rows


def loss_change_rows(
    baseline: List[Dict[str, Any]], candidate: List[Dict[str, Any]], routes: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Report how much more or less of each source every stage drops in a candidate build.

    Shares are over the source's converted documents, which both builds share
    (`route_comparison_rows` checks it), so a negative value is text the stage
    no longer drops, and a source's changes sum to minus its change in retention.

    Args:
        baseline: Rows from `funnel_rows` over the baseline corpus.
        candidate: Rows from `funnel_rows` over the candidate build.
        routes: The route per source, from `read_routes`.

    Returns:
        One row per source, in the candidate's funnel order, with the change
        per drop stage.
    """
    before = {row["source"]: loss_shares(row) for row in baseline if row["source"] != "TOTAL"}
    rows: List[Dict[str, Any]] = []
    for row in candidate:
        if row["source"] == "TOTAL":
            continue
        after = loss_shares(row)
        changes = {stage: round(after[stage] - before[row["source"]][stage], 4) for stage in DROP_STAGES}
        rows.append({"source": row["source"], "route": routes.get(row["source"], ""), **changes})
    return rows


def check_lock(pretrain_dir: Path, config: Path) -> None:
    """Check that a build on disk is the one its config's lock file records, at the config's settings.

    A candidate config writes to one fixed folder, so a later build overwrites
    it; tables named after the config must come from the build it locked. The
    config may also have changed since that build, while the audit re-score
    applies the config's settings as they stand.

    Args:
        pretrain_dir: The build's `output_dir`.
        config: The config that made it; its lock file sits beside it.

    Raises:
        ValueError: If a unit on disk differs from its lock entry, is missing,
            or was built at settings other than the config's.
    """
    lock = yaml.safe_load(lock_path_for(config).read_text(encoding="utf-8"))
    curate = load_curate_config(None, None, config, None)
    folders = {folder[3:]: folder for folder in (*SCOPED_STAGES, COUNTED_STAGE, *CORPUS_STAGES)}
    mismatched: List[str] = []
    for stage, folder in folders.items():
        entries = lock[stage] if stage in (name[3:] for name in SCOPED_STAGES) else {"": lock[stage]}
        for source, entry in entries.items():
            sentinel_path = pretrain_dir / folder / source / ".complete"
            sentinel = json.loads(sentinel_path.read_text(encoding="utf-8")) if sentinel_path.is_file() else {}
            differs = any(sentinel.get(key) != entry[key] for key in ("document_digest", "records_in", "records_out"))
            if differs or entry["config_hash"] != curate.expected_hash(stage, source or None):
                mismatched.append(f"{stage}/{source}" if source else stage)
    if mismatched:
        raise ValueError(f"{pretrain_dir} is not the build {lock_path_for(config).name} records: {mismatched}")


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


def rate_columns(
    name: str, verdicts: List[Dict[str, Any]], test: Optional[Callable[[Dict[str, Any]], bool]], call: bool
) -> Dict[str, Any]:
    """Score a group of verdicts on one test, with its Wilson interval.

    Args:
        name: Prefix of the column names, such as `target`.
        verdicts: The judged documents in the group.
        test: Whether one verdict counts as a hit; None leaves the columns empty.
        call: Whether to call the rate against the record's bars (D2).

    Returns:
        `<name>_hits`, `_precision`, `_low`, `_high` and `_call`, all empty
        when there is no test or no verdict.
    """
    if test is None or not verdicts:
        return {f"{name}_{key}": "" for key in ("hits", "precision", "low", "high", "call")}
    hits = sum(1 for verdict in verdicts if test(verdict))
    low, high = wilson(hits, len(verdicts))
    return {
        f"{name}_hits": hits,
        f"{name}_precision": round(hits / len(verdicts), 4),
        f"{name}_low": round(low, 4),
        f"{name}_high": round(high, 4),
        f"{name}_call": call_rate(hits / len(verdicts)) if call else "",
    }


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
            bad_test = is_bad_text if stage in BAD_TEXT_STAGES else None
            rows.append(
                {
                    "stage": stage,
                    "source": source,
                    "drops_judged": len(verdicts),
                    "read": read,
                    **rate_columns("target", verdicts, is_target, read),
                    **rate_columns("bad_text", verdicts, bad_test, read),
                }
            )
    return rows


def quality_drops(
    sample: Iterable[Dict[str, Any]], verdicts: Dict[str, Dict[str, Any]], sources: Set[str]
) -> Dict[str, Dict[str, str]]:
    """Pick the judged documents the quality filter dropped in the earlier audit.

    Args:
        sample: Sample rows, each carrying `id`, `dataset`, `text` and its `cells`.
        verdicts: Judge verdicts by document id.
        sources: The selected sources; a sampled source outside them is skipped.

    Returns:
        Per source, the text the judge read (the document's opening) by document id.
    """
    picked: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in sample:
        if row["id"] not in verdicts or row["dataset"] not in sources:
            continue
        if any(cell["stage"] == "quality" and cell["decision"] == "dropped" for cell in row["cells"]):
            picked[row["dataset"]][row["id"]] = row["text"]
    return dict(picked)


def _documents_in_shard(task: Tuple[Path, Dict[str, str]]) -> List[Tuple[str, Optional[str]]]:
    """Read the wanted documents out of one shard.

    Args:
        task: The shard, and the opening the judge read per wanted id.

    Returns:
        Per wanted document the shard holds, repeats included, its id and its
        full text, or None for the text when it does not start with the opening.
    """
    path, openings = task
    found: List[Tuple[str, Optional[str]]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            # A quote inside text is escaped, so the first match is the id key.
            match = ID_PATTERN.search(line)
            if match is None or json.loads(f'"{match.group(1)}"') not in openings:
                continue
            document = json.loads(line)
            opening = openings.get(document["id"])
            if opening is not None:
                found.append((document["id"], document["text"] if document["text"].startswith(opening) else None))
    return found


def fetch_full_texts(
    stage_dir: Path, openings: Dict[str, Dict[str, str]], cache: Path, workers: int
) -> Tuple[Dict[str, str], Set[str]]:
    """Fetch the full text of documents the judge read only the opening of.

    The sampler cut long documents short, and a filter has to see the whole
    text, so each one is read again from a stage folder. A document missing
    there was dropped before that stage. The result is cached and reused
    while every source's sentinel and the wanted openings are unchanged.

    Args:
        stage_dir: The stage folder to read, one subfolder per source.
        openings: Per source, the text the judge read by document id.
        cache: JSON file holding the result and the inputs it was read from.
        workers: Shards read at once.

    Returns:
        Full text by document id, for the documents the folder holds with the
        text the judge read; and the ids the folder holds only with other text.
    """
    key = {
        "format": FETCH_FORMAT,
        "openings": hashlib.sha256(json.dumps(openings, sort_keys=True).encode("utf-8")).hexdigest(),
        "digests": {
            source: json.loads((stage_dir / source / ".complete").read_text(encoding="utf-8"))["document_digest"]
            for source in sorted(openings)
        },
    }
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached["key"] == key:
            return cached["texts"], set(cached["changed"])
    tasks = [
        (shard, openings[source])
        for source in sorted(openings)
        for shard in sorted((stage_dir / source).glob("*.jsonl.gz"))
    ]
    texts: Dict[str, str] = {}
    seen: Set[str] = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for found in tqdm(pool.map(_documents_in_shard, tasks), total=len(tasks), desc=f"reading {stage_dir.name}"):
            for doc_id, text in found:
                seen.add(doc_id)
                # One id can carry several texts (issue #6); keep the one the judge read.
                if text is not None:
                    texts.setdefault(doc_id, text)
    changed = seen - set(texts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    partial.write_text(
        json.dumps({"key": key, "texts": texts, "changed": sorted(changed)}, ensure_ascii=False), encoding="utf-8"
    )
    partial.replace(cache)
    return texts, changed


def quality_filters(config: Path, sources: Iterable[str]) -> Dict[str, Optional[LetterShareGopherQualityFilter]]:
    """Build the quality filter each source meets in a candidate build.

    Args:
        config: The candidate build's curation config.
        sources: The sources to build a filter for.

    Returns:
        Per source, its quality filter with its route's and its own settings;
        None where the route switches the filter off.
    """
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    overrides = resolve_profiles(cfg.get("profiles"), cfg.get("overrides"))
    stopwords, _ = load_stopwords(cfg["stopwords"])
    filters: Dict[str, Optional[LetterShareGopherQualityFilter]] = {}
    for source in sources:
        knobs = effective_stage_config(cfg, overrides, source, "quality")
        if knobs.get("enabled") is False:
            filters[source] = None
            continue
        # The stage's own builder, so defaults and casts match the build exactly.
        settings = _build_quality_config(knobs)
        filters[source] = LetterShareGopherQualityFilter(
            language=Languages.slovenian, stop_words=sorted(stopwords), **asdict(settings)
        )
    return filters


def refilter_rows(
    openings: Dict[str, Dict[str, str]],
    texts: Dict[str, str],
    changed: Set[str],
    verdicts: Dict[str, Dict[str, Any]],
    filters: Dict[str, Optional[LetterShareGopherQualityFilter]],
    routes: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Re-apply a candidate's quality filter to the documents it dropped in the earlier audit.

    A document the candidate filter keeps is rescued; the rest are its drops,
    scored on the quality target and on bad text as `precision_rows` scores
    the audit's own drops.

    Args:
        openings: Per source, the audit's judged quality drops, from `quality_drops`.
        texts: Their full text, from `fetch_full_texts`.
        changed: The ids the candidate holds only with text other than the
            judged one; they are counted and left out, since the verdict was
            on other text. Any other id missing from `texts` is absent: an
            earlier filter drops it, or the source's ids changed since the audit.
        verdicts: Judge verdicts by document id.
        filters: The candidate's quality filter per source, from `quality_filters`.
        routes: The route per source, from `read_routes`.

    Returns:
        A row pooled over every source, one per route, then one per source,
        each with the rescued documents' bad-text share and the remaining
        drops' rates. A group with fewer than `MIN_SOURCE_DROPS` remaining
        drops gets no call.
    """
    rescued: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dropped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source, ids in sorted(openings.items()):
        quality_filter = filters[source]
        for doc_id in sorted(ids):
            if doc_id not in texts:
                continue
            kept = quality_filter is None or quality_filter.filter(Document(text=texts[doc_id], id=doc_id)) is True
            (rescued if kept else dropped)[source].append(verdicts[doc_id])

    def group(route: str, source: str, members: List[str]) -> Dict[str, Any]:
        """Score one group of sources pooled together."""
        saved = [verdict for member in members for verdict in rescued[member]]
        still_dropped = [verdict for member in members for verdict in dropped[member]]
        read = len(still_dropped) >= MIN_SOURCE_DROPS
        return {
            "route": route,
            "source": source,
            "drops_judged": sum(len(openings[member]) for member in members),
            "text_changed": sum(1 for member in members for doc_id in openings[member] if doc_id in changed),
            "absent": sum(
                1 for member in members for doc_id in openings[member] if doc_id not in texts and doc_id not in changed
            ),
            "reach_quality": len(saved) + len(still_dropped),
            "rescued": len(saved),
            **rate_columns("rescued_bad_text", saved, is_bad_text, call=False),
            "still_dropped": len(still_dropped),
            **rate_columns("target", still_dropped, TARGETS["quality"], read),
            **rate_columns("bad_text", still_dropped, is_bad_text, read),
        }

    by_route: Dict[str, List[str]] = defaultdict(list)
    for source in sorted(openings):
        by_route[routes.get(source, "")].append(source)
    rows = [group("all", "pooled", sorted(openings))]
    rows += [group(route, "pooled", members) for route, members in sorted(by_route.items())]
    rows += [group(route, source, [source]) for route, members in sorted(by_route.items()) for source in members]
    return rows


def failing_rules(quality_filter: Any, doc: Document) -> List[str]:
    """List every quality setting a document fails, in the filter's order.

    The filter stops at the first rule a document fails, so each failing rule
    is switched off in turn and the filter run again.

    Args:
        quality_filter: The quality filter; its rules are switched off on copies.
        doc: The document.

    Returns:
        The settings behind the failed rules, first failure first; empty for
        a kept document.
    """
    failed: List[str] = []
    result = quality_filter.filter(doc)
    while result is not True:
        setting = REASON_SETTINGS[result[1]]
        failed.append(setting)
        quality_filter = copy.copy(quality_filter)
        setattr(quality_filter, setting, 0)
        result = quality_filter.filter(doc)
    return failed


def drop_rule_rows(
    openings: Dict[str, Dict[str, str]],
    texts: Dict[str, str],
    verdicts: Dict[str, Dict[str, Any]],
    filters: Dict[str, Optional[LetterShareGopherQualityFilter]],
    routes: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Group the documents a candidate's quality filter still drops by the rules they fail.

    Args:
        openings: Per source, the audit's judged quality drops, from `quality_drops`.
        texts: Their full text, from `fetch_full_texts`; other ids are skipped.
        verdicts: Judge verdicts by document id.
        filters: The candidate's quality filter per source, from `quality_filters`.
        routes: The route per source, from `read_routes`.

    Returns:
        One row per route, source and set of failed rules, with how many
        documents fail exactly that set and how many of them the judge found
        to be the quality target or bad text. A set of one rule counts the
        documents that loosening that rule alone would keep.
    """
    groups: Dict[Tuple[str, str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for source, ids in sorted(openings.items()):
        if filters[source] is None:
            continue
        for doc_id in sorted(ids):
            if doc_id not in texts:
                continue
            failed = failing_rules(filters[source], Document(text=texts[doc_id], id=doc_id))
            if failed:
                groups[(routes.get(source, ""), source, tuple(failed))].append(verdicts[doc_id])
    return [
        {
            "route": route,
            "source": source,
            "first_rule": failed[0],
            "failed_rules": "+".join(failed),
            "still_dropped": len(group),
            "target_hits": sum(1 for verdict in group if TARGETS["quality"](verdict)),
            "bad_text_hits": sum(1 for verdict in group if is_bad_text(verdict)),
        }
        for (route, source, failed), group in sorted(groups.items())
    ]


def draw_figures(
    figures_dir: Path,
    losses: List[Dict[str, Any]],
    comparison: List[Dict[str, Any]],
    route_comparison: List[Dict[str, Any]],
    stem: str,
) -> List[Path]:
    """Draw the record's figures from the tables `main` has already built.

    Args:
        figures_dir: Where the SVGs go.
        losses: Rows from `loss_rows`.
        comparison: Rows from `build_comparison_rows`.
        route_comparison: Rows from `route_comparison_rows`.
        stem: The candidate config's stem, naming its figures.

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

    sources = sorted((row for row in route_comparison if row["source"] != "TOTAL"), key=lambda row: row["change"])
    figures[f"{stem}-retention-by-build-and-source"] = DumbbellChart(
        [
            {
                "label": f"{row['source']} ({ROUTE_NAMES.get(row['route'], row['route'])})",
                "start": row["retained_baseline"],
                "end": row["retained_candidate"],
            }
            for row in sources
        ],
        title="Retention per source under routes",
        start_name="baseline build",
        end_name="build with routes",
        xlabel="share of converted documents kept",
        ylabel="source (route)",
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
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="The candidate build; defaults to --routes-config's output_dir.",
    )
    parser.add_argument("--routes-config", type=Path, default=ROUTES_CONFIG, help="The candidate build's config.")
    parser.add_argument("--extract-config", type=Path, default=Path("configs/data/extract.yaml"))
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Where the CSV tables are written.")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR, help="Where the SVG figures are written.")
    parser.add_argument(
        "--workers", type=int, default=30, help="Shards read at once (exact-dedup counts, audit texts)."
    )
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

    access = read_access(args.extract_config)
    exact_counts = counted_stage(args.pretrain_dir, args.data_root / "interim" / "stage-counts.json", args.workers)
    funnel = funnel_rows(args.pretrain_dir, exact_counts, access)
    audit_funnel = read_csv(AUDIT_TABLES / "source-funnel.csv")
    losses = loss_rows(funnel)
    comparison = build_comparison_rows(funnel, audit_funnel)
    domains = domain_token_rows(funnel, audit_funnel)

    for row in domains:
        print(f"{row['domain']:14s}{row['tokens_estimated']:>16,d} tokens  clears KPI: {row['clears_kpi']}")

    verdicts = {verdict["id"]: verdict for verdict in read_jsonl(args.verdicts)}
    sources = {row["source"] for row in funnel if row["source"] != "TOTAL"}
    sample = read_jsonl(args.sample)
    precision = precision_rows(judged_drops(sample, verdicts, sources))
    for row in precision:
        if row["source"] == "pooled":
            print(
                f"{row['stage']:12s} target {row['target_precision']} ({row['target_call']})"
                f"  bad text {row['bad_text_precision']} ({row['bad_text_call']})  n={row['drops_judged']}"
            )
    stem = args.routes_config.stem
    routes = read_routes(args.routes_config)
    candidate_dir = args.candidate_dir or Path(
        yaml.safe_load(args.routes_config.read_text(encoding="utf-8"))["output_dir"]
    )
    check_lock(candidate_dir, args.routes_config)
    _, candidate_commits = scoped_counts(candidate_dir)
    print(
        f"{stem} build matches {lock_path_for(args.routes_config).name}; built at {', '.join(sorted(candidate_commits))}"
    )
    candidate_cache = args.data_root / "interim" / f"{stem}-stage-counts.json"
    candidate = funnel_rows(candidate_dir, counted_stage(candidate_dir, candidate_cache, args.workers), access)
    route_comparison = route_comparison_rows(funnel, candidate, routes)
    loss_changes = loss_change_rows(funnel, candidate, routes)
    candidate_domains = domain_token_rows(candidate, funnel, earlier="baseline")
    for row in candidate_domains:
        print(f"{stem}: {row['domain']:14s}{row['tokens_estimated']:>16,d} tokens  clears KPI: {row['clears_kpi']}")

    openings = quality_drops(sample, verdicts, sources)
    texts_cache = args.data_root / "interim" / f"{stem}-audit-quality-drops.json"
    texts, changed = fetch_full_texts(candidate_dir / QUALITY_INPUT, openings, texts_cache, args.workers)
    filters = quality_filters(args.routes_config, openings)
    refilter = refilter_rows(openings, texts, changed, verdicts, filters, routes)
    drop_rules = drop_rule_rows(openings, texts, verdicts, filters, routes)
    for row in refilter:
        if row["source"] == "pooled":
            print(
                f"{stem}: quality re-applied, {row['route']:16s} rescued {row['rescued']}/{row['reach_quality']}"
                f" (bad text {row['rescued_bad_text_precision']}), still dropped {row['still_dropped']}"
                f" target {row['target_precision']} ({row['target_call']})"
            )

    written = [
        write_csv(args.tables / f"{stem}-documents-by-source.csv", candidate),
        write_csv(args.tables / f"{stem}-retention-by-build-and-source.csv", route_comparison),
        write_csv(args.tables / f"{stem}-loss-change-by-stage-and-source.csv", loss_changes),
        write_csv(args.tables / f"{stem}-tokens-by-domain.csv", candidate_domains),
        write_csv(args.tables / f"{stem}-audit-quality-refilter-by-route-and-source.csv", refilter),
        write_csv(args.tables / f"{stem}-audit-quality-drops-by-rule-and-source.csv", drop_rules),
        write_csv(args.tables / "curate-documents-by-source.csv", funnel),
        write_csv(args.tables / "curate-losses-by-source.csv", losses),
        write_csv(args.tables / "curate-retention-by-build-and-source.csv", comparison),
        write_csv(args.tables / "curate-tokens-by-domain.csv", domains),
        write_csv(args.tables / "dataset-corpus-statistics.csv", corpus_dataset_rows(funnel)),
        write_csv(args.tables / "audit-precision-by-filter-and-source.csv", precision),
    ]
    written += draw_figures(args.figures, losses, comparison, route_comparison, stem)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
