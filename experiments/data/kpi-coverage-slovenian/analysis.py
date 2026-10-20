"""Turn the two corpus builds, the judged gold and the catalogue into the record's tables.

Every input is a frozen file of an earlier experiment, read in place and never
copied: the loosened build of the curation-thresholds experiment, the baseline
build on the shared pretraining path, the judged sample of the curation-quality
experiment and the catalogue of the data-landscape survey. This pass writes the
statistics behind the record's `## Datasets` rows.

Run it with:

    uv run --group analysis python experiments/data/kpi-coverage-slovenian/analysis.py
"""

import csv
import hashlib
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from datachart.charts import BarChart
from datachart.constants import BAR_MODE, FIG_SIZE, LEGEND_LOCATION, ORIENTATION, SORT

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402  — path set above so the shared helper resolves

from slm4ie.data.curate.inspect.profile import cached_source_counts, iter_stage_sentinels
from slm4ie.data.curate.lineage import read_sentinel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
EXTRACT_CONFIG = ROOT / "configs" / "data" / "extract.yaml"

# The validated build (D1) and the shared build tabled beside it, with the commit each was built at.
BUILDS: Dict[str, Path] = {
    "loosened": ROOT
    / "data"
    / "experiments"
    / "data"
    / "curation-thresholds-slovenian"
    / "interim"
    / "pretrain-loosened",
    "baseline": ROOT / "data" / "pretrain",
}
BUILD_COMMITS = {"loosened": "8e21f67", "baseline": "2ece205"}

# The stages that stamp per-source counts in their sentinels, in pipeline order (M1).
SCOPED_STAGES = ("00_convert", "01_language", "02_spam", "03_quality", "04_repetition")
# Exact dedup runs over the whole corpus, so its per-source counts are read off its shards.
COUNTED_STAGE = "05_exact_dedup"
COUNT_WORKERS = 20


def stage_column(stage: str) -> str:
    """Name a funnel column after a stage folder, without the folder's order prefix.

    Args:
        stage: A stage folder name such as `03_quality`.

    Returns:
        The column name, such as `quality`.
    """
    return stage[3:]


FUNNEL_COLUMNS = (
    *(stage_column(stage) for stage in SCOPED_STAGES),
    "exact_dedup",
    "sentence_dedup",
    "words_final",
    "tokens_estimated",
)

GOLD_DATA = ROOT / "data" / "experiments" / "data" / "curation-quality-slovenian"
GOLD_SAMPLE = GOLD_DATA / "interim" / "sample.jsonl"
GOLD_VERDICTS = GOLD_DATA / "final" / "judge-verdicts-sample.jsonl"
GOLD_META = GOLD_DATA / "final" / "judge-verdicts-sample.meta.json"
# The sample sits under the parent's `interim/` with no frozen sidecar, so its hash is pinned here.
GOLD_SAMPLE_SHA256 = "9a3e7b9c80426ca0b47aea88a63a32795b7697ab7e4aca0278188f6afaad5104"

CATALOGUE = HERE.parent / "data-landscape-slovenian" / "tables" / "catalogue.csv"
# The survey used the issue's domain names; the judge's label set (D2) renames three.
CATALOGUE_DOMAINS = {"encyclopedic": "wiki", "forum/social": "forum", "general-web": "web"}

EXPERIMENT_DATA = ROOT / "data" / "experiments" / "data" / "kpi-coverage-slovenian"
# The gold's two halves (D3): the classifier is tuned on one and its gate is read on the other.
SPLIT_DIR = EXPERIMENT_DATA / "final" / "gold-tune-test-split"
SPLIT_SEED = 20261020
# Copies of one text are grouped on this many opening characters, so no copy lands on both sides.
GROUP_CHARS = 500
WHITESPACE = re.compile(r"\s+")

# A token is estimated as two words, the conversion every KPI reading in the project uses.
TOKENS_PER_WORD = 2


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read one JSON object per line.

    Args:
        path: File to read.

    Returns:
        The objects in file order.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_sources(extract_config: Path) -> Dict[str, Dict[str, str]]:
    """Read each source's domain label and whether it may be redistributed.

    Args:
        extract_config: `configs/data/extract.yaml`.

    Returns:
        `domain` and `access` (`open` or `gated`) per source.
    """
    catalog = yaml.safe_load(extract_config.read_text(encoding="utf-8")) or {}
    entries = catalog.get("datasets", catalog)
    return {
        key: {"domain": entry["domain"], "access": entry.get("access", "open")}
        for key, entry in entries.items()
        if isinstance(entry, dict)
    }


def check_build_commit(build: str, pretrain_dir: Path) -> None:
    """Fail unless the build's statistics stage was written at the recorded commit.

    Args:
        build: Build name, a key of `BUILD_COMMITS`.
        pretrain_dir: The build's curation output folder.

    Raises:
        ValueError: If the statistics sentinel names another commit.
    """
    sentinel = read_sentinel(pretrain_dir / "07_statistics")
    commit = sentinel.info.get("git_commit", "unknown") if sentinel else "unknown"
    if not commit.startswith(BUILD_COMMITS[build]):
        raise ValueError(f"{build} build statistics written at {commit}, record names {BUILD_COMMITS[build]}")


def build_rows(builds: Dict[str, Path], sources: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Read the final-stage size of every source in each build.

    Args:
        builds: Curation output folder per build name.
        sources: `domain` and `access` per source, from `read_sources`.

    Returns:
        One row per build and source with its domain label, access, documents,
        words, mean words per document and estimated tokens.
    """
    rows: List[Dict[str, Any]] = []
    for build, pretrain_dir in builds.items():
        check_build_commit(build, pretrain_dir)
        aggregate = json.loads((pretrain_dir / "07_statistics" / "aggregate.json").read_text(encoding="utf-8"))
        for source, final in read_final_statistics(pretrain_dir).items():
            rows.append(
                {
                    "build": build,
                    "source": source,
                    "source_domain": sources[source]["domain"],
                    "access": sources[source]["access"],
                    "documents": final["doc_count"],
                    "words": final["word_count"],
                    "mean_words_per_document": round(final["avg_doc_words"], 1),
                    "tokens_estimated": final["word_count"] * TOKENS_PER_WORD,
                }
            )
        totals = [row for row in rows if row["build"] == build]
        if sum(row["documents"] for row in totals) != aggregate["total_docs"]:
            raise ValueError(f"{build} per-source documents do not sum to the aggregate {aggregate['total_docs']}")
    return rows


def read_final_statistics(pretrain_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read the statistics stage's file per source.

    Args:
        pretrain_dir: The build's curation output folder.

    Returns:
        The statistics per source, in source order. A source that reached the
        final corpus with no document has no file and is absent.
    """
    files = sorted((pretrain_dir / "07_statistics" / "per_dataset").glob("*.json"))
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in files}


def funnel_source_rows(
    build: str, pretrain_dir: Path, exact_counts: Dict[str, int], sources: Dict[str, Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Follow each source's documents through every curation stage of one build.

    The per-source stages stamp the documents they read and wrote in their
    sentinels; exact dedup is counted off its shards; the final stage's
    documents and words come from the statistics stage, which writes no file
    for a source that lost its last document earlier, so such a source ends at
    zero rather than vanishing.

    Args:
        build: Build name.
        pretrain_dir: The build's curation output folder.
        exact_counts: Documents per source after exact dedup, from `cached_source_counts`.
        sources: `domain` and `access` per source, from `read_sources`.

    Returns:
        One row per converted source with its documents after every stage, its
        final words and estimated tokens, and the share of its converted
        documents that reached the final corpus.

    Raises:
        ValueError: If a stage read other than what the stage before it wrote,
            or the statistics stage names a source no stage converted.
    """
    scoped: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    for stage, source, read, written in iter_stage_sentinels(pretrain_dir, SCOPED_STAGES):
        scoped[source][stage] = (read, written)
    finals = read_final_statistics(pretrain_dir)
    if unconverted := sorted(set(finals) - set(scoped)):
        raise ValueError(f"{build}: statistics name sources no stage converted: {unconverted}")
    rows: List[Dict[str, Any]] = []
    for source in sorted(scoped):
        row: Dict[str, Any] = {
            "build": build,
            "source": source,
            "source_domain": sources[source]["domain"],
            "access": sources[source]["access"],
        }
        previous = None
        for stage in SCOPED_STAGES:
            read, written = scoped[source][stage]
            if previous is not None and read != previous:
                raise ValueError(f"{build} {source}: {stage} read {read} documents, the stage before wrote {previous}")
            row[stage_column(stage)] = previous = written
        final = finals.get(source, {"doc_count": 0, "word_count": 0})
        row["exact_dedup"] = exact_counts.get(source, 0)
        row["sentence_dedup"] = final["doc_count"]
        row["words_final"] = final["word_count"]
        row["tokens_estimated"] = final["word_count"] * TOKENS_PER_WORD
        row["retained"] = round(row["sentence_dedup"] / row["convert"], 4)
        rows.append(row)
    check_corpus_stages(build, pretrain_dir, rows)
    return rows


def check_corpus_stages(build: str, pretrain_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Fail unless the per-source counts add up to what the two dedup stages recorded.

    Args:
        build: Build name.
        pretrain_dir: The build's curation output folder.
        rows: Rows from `funnel_source_rows`.

    Raises:
        ValueError: If a column's sum differs from the corpus-wide sentinel.
    """
    exact = read_sentinel(pretrain_dir / COUNTED_STAGE)
    sentence = read_sentinel(pretrain_dir / "06_sentence_dedup")
    if exact is None or sentence is None:
        raise ValueError(f"{build}: a dedup stage under {pretrain_dir} has no completed sentinel")
    expected = [
        ("repetition", exact.records_in),
        ("exact_dedup", exact.records_out),
        ("exact_dedup", sentence.records_in),
        ("sentence_dedup", sentence.records_out),
    ]
    for column, total in expected:
        counted = sum(row[column] for row in rows)
        if counted != total:
            raise ValueError(
                f"{build}: per-source {column} documents sum to {counted}, the dedup sentinels record {total}"
            )


def funnel_domain_rows(source_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sum the per-source funnel over the sources sharing a domain label.

    Args:
        source_rows: Rows from `funnel_source_rows`, any number of builds.

    Returns:
        One row per build and source domain, largest final corpus first, with
        its source count, its documents after every stage, its final words and
        estimated tokens, and the share of its converted documents kept.
    """
    totals: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in source_rows:
        total = totals.setdefault(
            (row["build"], row["source_domain"]),
            {"build": row["build"], "source_domain": row["source_domain"], "sources": 0}
            | dict.fromkeys(FUNNEL_COLUMNS, 0),
        )
        total["sources"] += 1
        for column in FUNNEL_COLUMNS:
            total[column] += row[column]
    rows = sorted(totals.values(), key=lambda total: (total["build"], -total["tokens_estimated"]))
    for row in rows:
        row["retained"] = round(row["sentence_dedup"] / row["convert"], 4)
    return rows


def check_frozen(path: Path, expected: str) -> None:
    """Fail unless a frozen file still has the hash recorded when it was frozen.

    Args:
        path: The frozen file.
        expected: Its recorded sha256 hex digest.

    Raises:
        ValueError: If the file's hash differs from the recorded one.
    """
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{path.name} hashes to {actual}, frozen as {expected}")


def gold_documents(sample_path: Path, verdicts_path: Path) -> List[Dict[str, Any]]:
    """Join the judged sample to its verdicts, keeping what the classifier is scored on.

    The kept flag is the decision of the stage the document was drawn at, in
    the September build the sample came from, so it says whether that one
    stage kept it there, not whether it reached either final corpus.

    Args:
        sample_path: The curation-quality experiment's judged sample.
        verdicts_path: Its frozen judge verdicts.

    Returns:
        One row per judged document with its id, source, source domain, judge
        domain, whether its drawn stage kept it, length in characters and
        whether the judge saw a cut.

    Raises:
        ValueError: If a sample document has no verdict or the reverse.
    """
    verdicts = {verdict["id"]: verdict for verdict in read_jsonl(verdicts_path)}
    sample = read_jsonl(sample_path)
    unmatched = set(verdicts) ^ {document["id"] for document in sample}
    if unmatched:
        raise ValueError(f"{len(unmatched)} ids are in only one of the sample and the verdicts, e.g. {min(unmatched)}")
    return [
        {
            "id": document["id"],
            "source": document["dataset"],
            "source_domain": document["domain"],
            "judge_domain": verdicts[document["id"]]["domain"],
            "kept_at_drawn_stage": all(cell["decision"] == "kept" for cell in document["cells"]),
            "chars": document["chars"],
            "truncated": document["truncated"],
            "group": text_group(document["text"]),
        }
        for document in sample
    ]


def text_group(text: str) -> str:
    """Key under which copies of one text fall together.

    Args:
        text: The document text the judge read.

    Returns:
        The hash of its opening characters with whitespace and case folded.
    """
    opening = WHITESPACE.sub(" ", text.lower()).strip()[:GROUP_CHARS]
    return hashlib.sha256(opening.encode("utf-8")).hexdigest()[:16]


def split_gold(gold: List[Dict[str, Any]], seed: int) -> Dict[str, str]:
    """Halve the gold into a tune and a test side, copies together, domains balanced.

    Each group of copies is filed under the judge domain most of its members
    carry, shuffled per domain under the seed and dealt to whichever side
    holds fewer of that domain's documents, so no copy sits on both sides
    and each domain is split as evenly as its group sizes allow.

    Args:
        gold: Rows from `gold_documents`.
        seed: The shuffle seed.

    Returns:
        `tune` or `test` per document id.
    """
    members_of: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for document in gold:
        members_of[document["group"]].append(document)
    groups: Dict[str, List[List[str]]] = defaultdict(list)
    for members in members_of.values():
        domain = Counter(document["judge_domain"] for document in members).most_common(1)[0][0]
        groups[domain].append(sorted(document["id"] for document in members))
    side_of: Dict[str, str] = {}
    for domain in sorted(groups):
        members = sorted(groups[domain])
        random.Random(f"{seed}:{domain}").shuffle(members)
        held = {"tune": 0, "test": 0}
        for ids in members:
            side = "tune" if held["tune"] <= held["test"] else "test"
            held[side] += len(ids)
            side_of.update(dict.fromkeys(ids, side))
    return side_of


def write_split(gold: List[Dict[str, Any]], side_of: Dict[str, str], split_dir: Path, seed: int) -> None:
    """Write the two halves to `final/`, or check that what is there is the same split.

    Args:
        gold: Rows from `gold_documents`.
        side_of: `tune` or `test` per document id.
        split_dir: Folder for `tune.jsonl`, `test.jsonl` and `split.meta.json`.
        seed: The shuffle seed, recorded in the meta file.

    Raises:
        ValueError: If a half already on disk differs from the one computed now.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    for side in ("tune", "test"):
        lines = [
            json.dumps({key: document[key] for key in ("id", "source", "source_domain", "judge_domain", "group")})
            for document in gold
            if side_of[document["id"]] == side
        ]
        content = "\n".join(lines) + "\n"
        path = split_dir / f"{side}.jsonl"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"{path} differs from the split computed under seed {seed}; final/ is fixed")
        path.write_text(content, encoding="utf-8")
    meta = {
        "seed": seed,
        "group_chars": GROUP_CHARS,
        "sample_sha256": GOLD_SAMPLE_SHA256,
        "tune": sum(side == "tune" for side in side_of.values()),
        "test": sum(side == "test" for side in side_of.values()),
        "note": "Gold halved per judge domain, copies together (D3): tune picks the classifier, test reads its gate.",
    }
    (split_dir / "split.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def gold_rows(gold: List[Dict[str, Any]], side_of: Dict[str, str]) -> List[Dict[str, Any]]:
    """Summarise the judged gold per judge domain.

    Args:
        gold: Rows from `gold_documents`.
        side_of: `tune` or `test` per document id, from `split_gold`.

    Returns:
        One row per judge domain, largest first, with its documents, how many
        fall on the tune and the test side, how many the stage they were drawn
        at kept, how many carry the same source label, how many the judge read
        only the opening of, and the median full length in characters.
    """
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for document in gold:
        by_domain[document["judge_domain"]].append(document)
    rows = [
        {
            "judge_domain": domain,
            "documents": len(documents),
            "tune": sum(side_of[document["id"]] == "tune" for document in documents),
            "test": sum(side_of[document["id"]] == "test" for document in documents),
            "kept_at_drawn_stage": sum(document["kept_at_drawn_stage"] for document in documents),
            "same_source_label": sum(document["source_domain"] == domain for document in documents),
            "truncated": sum(document["truncated"] for document in documents),
            "median_chars": int(statistics.median(document["chars"] for document in documents)),
        }
        for domain, documents in by_domain.items()
    ]
    return sorted(rows, key=lambda row: -row["documents"])


def catalogue_rows(catalogue_path: Path) -> List[Dict[str, Any]]:
    """Summarise the landscape catalogue per domain and access.

    A dataset listing several domains counts once under each.

    Args:
        catalogue_path: The data-landscape survey's `catalogue.csv`.

    Returns:
        One row per domain and access with the datasets, how many fit the
        examples target, how many report a size, and their summed documents
        and estimated tokens.
    """
    with catalogue_path.open(encoding="utf-8", newline="") as handle:
        datasets = list(csv.DictReader(handle))
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for dataset in datasets:
        for domain in dataset["domains"].split(","):
            groups[(CATALOGUE_DOMAINS.get(domain, domain), dataset["access"])].append(dataset)
    rows = []
    for (domain, access), members in sorted(groups.items()):
        rows.append(
            {
                "domain": domain,
                "access": access,
                "datasets": len(members),
                "fits_examples_target": sum(member["kpi2_fit"] == "yes" for member in members),
                "examples_fit_unknown": sum(member["kpi2_fit"] == "unknown" for member in members),
                "sized": sum(bool(member["tokens_estimated"]) for member in members),
                "documents": sum(int(member["documents"]) for member in members if member["documents"]),
                "tokens_estimated": sum(
                    int(member["tokens_estimated"]) for member in members if member["tokens_estimated"]
                ),
            }
        )
    return rows


def draw_gold_figure(rows: List[Dict[str, Any]]) -> None:
    """Draw the judged gold's documents per judge domain, split by the drawn stage's decision.

    Args:
        rows: Rows from `gold_rows`.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    save_figure(
        BarChart(
            [
                [{"label": row["judge_domain"], "y": row["kept_at_drawn_stage"]} for row in rows],
                [{"label": row["judge_domain"], "y": row["documents"] - row["kept_at_drawn_stage"]} for row in rows],
            ],
            title="Judged gold documents per judge domain",
            xlabel="documents",
            ylabel="domain the judge assigned",
            subtitle=["kept at the stage it was drawn", "dropped at the stage it was drawn"],
            bar_mode=BAR_MODE.STACK,
            orientation=ORIENTATION.HORIZONTAL,
            sort=SORT.ASCENDING,
            show_legend=True,
            legend={"location": LEGEND_LOCATION.OUTSIDE_BOTTOM},
            xmin=0,
            figsize=FIG_SIZE.FULL_MEDIUM,
        ),
        FIGURES_DIR / "dataset-gold-documents-by-domain.svg",
    )


def main() -> None:
    """Write the dataset statistics, the per-stage funnel tables and the figure, and print the totals."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sources = read_sources(EXTRACT_CONFIG)
    builds = build_rows(BUILDS, sources)
    write_csv(TABLES_DIR / "dataset-builds-statistics.csv", builds)
    for build in BUILDS:
        documents = sum(row["documents"] for row in builds if row["build"] == build)
        print(f"{build}: {documents:,} documents in the final corpus")

    funnel: List[Dict[str, Any]] = []
    for build, pretrain_dir in BUILDS.items():
        cache = EXPERIMENT_DATA / "interim" / f"{build}-build-exact-dedup-counts.json"
        exact_counts = cached_source_counts(pretrain_dir / COUNTED_STAGE, cache, COUNT_WORKERS)
        funnel += funnel_source_rows(build, pretrain_dir, exact_counts, sources)
    write_csv(TABLES_DIR / "funnel-documents-by-source.csv", funnel)
    domains = funnel_domain_rows(funnel)
    write_csv(TABLES_DIR / "funnel-documents-by-domain.csv", domains)
    print(f"funnel: {len(funnel)} source rows and {len(domains)} domain rows over {len(BUILDS)} builds")

    check_frozen(GOLD_VERDICTS, json.loads(GOLD_META.read_text(encoding="utf-8"))["sha256"])
    check_frozen(GOLD_SAMPLE, GOLD_SAMPLE_SHA256)
    documents = gold_documents(GOLD_SAMPLE, GOLD_VERDICTS)
    side_of = split_gold(documents, SPLIT_SEED)
    write_split(documents, side_of, SPLIT_DIR, SPLIT_SEED)
    gold = gold_rows(documents, side_of)
    write_csv(TABLES_DIR / "dataset-gold-statistics.csv", gold)
    draw_gold_figure(gold)
    print(f"gold: {sum(row['documents'] for row in gold):,} judged documents over {len(gold)} judge domains")

    catalogue = catalogue_rows(CATALOGUE)
    write_csv(TABLES_DIR / "dataset-catalogue-statistics.csv", catalogue)
    print(f"catalogue: {len(catalogue)} domain and access groups")


if __name__ == "__main__":
    main()
