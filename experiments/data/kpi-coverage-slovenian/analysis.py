"""Turn the two corpus builds, the judged gold and the catalogue into the record's tables.

Every input is a frozen file of an earlier experiment, read in place and never
copied: the loosened build of the curation-thresholds experiment, the baseline
build on the shared pretraining path, the judged sample of the curation-quality
experiment and the catalogue of the data-landscape survey. Beside them it reads
the two corpus passes of the `classify` subcommand. It writes the statistics
behind the record's `## Datasets` rows, the classifier gates, the per-target
verdicts, the gap attribution and the figures.

Run it with:

    uv run --group analysis python experiments/data/kpi-coverage-slovenian/analysis.py
"""

import csv
import hashlib
import json
import logging
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import sklearn
import yaml
from datachart.charts import BarChart, DumbbellChart, LineChart
from datachart.constants import AXIS_SCALE, BAR_MODE, FIG_SIZE, LEGEND_LOCATION, ORIENTATION, SORT

# make the shared figure helper importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_figures import save_figure  # noqa: E402

from slm4ie.data.curate.inspect.domains import DecisionRule, DomainLexicons, assign_domain, load_lexicons, score_text
from slm4ie.data.curate.inspect.judge import DOMAINS
from slm4ie.data.curate.inspect.profile import cached_source_counts, iter_stage_sentinels
from slm4ie.data.curate.inspect.supervised import DomainClassifier, fit_classifier, load_candidate
from slm4ie.data.curate.lineage import read_sentinel
from slm4ie.utils.config import load_yaml

# ===============================
# Defaults
# ===============================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
EXTRACT_CONFIG = ROOT / "configs" / "data" / "extract.yaml"

# The validated build (D1) and the shared build tabled beside it, with the commit each was built at.
BUILDS: dict[str, Path] = {
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
SENTENCE_STAGE = "06_sentence_dedup"
STATISTICS_STAGE = "07_statistics"
COUNT_WORKERS = 20
# a stage folder starts with its two-digit order and an underscore
STAGE_PREFIX_CHARS = 3

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
GROUP_KEY_CHARS = 16
WHITESPACE = re.compile(r"\s+")

# A token is estimated as two words, the conversion every KPI reading in the project uses.
TOKENS_PER_WORD = 2

# The lexicon seed (M2): words of the tune half ranked by log-odds per judge domain.
SEED_WORD = re.compile(r"[^\W_]+", re.UNICODE)
SEED_TOP = 150
SEED_MIN_COUNT = 5
SEED_MIN_CHARS = 2
SEED_PRIOR = 0.01
SEED_SCORE_DECIMALS = 2

#: The lexicon classifier's config; its rule must carry the tune-side pick (M3).
LEXICON_CONFIG = HERE / "configs" / "classifier-lexicon.yaml"
#: The hash of the lexicons the test side was first read with; a changed lexicon reads it again.
LEXICON_FREEZE = EXPERIMENT_DATA / "final" / "classifier-lexicon.frozen.json"
#: The supervised candidates tried when the lexicons fail the gate (M4), and where each fit is kept.
SUPERVISED_CONFIG = HERE / "configs" / "classifier-supervised.yaml"
SUPERVISED_CANDIDATES: tuple[str, ...] = ("tfidf", "embeddings")
SUPERVISED_MODELS = EXPERIMENT_DATA / "final" / "classifier-supervised"
#: The thresholds and margins searched on the tune side, in weighted hits per thousand words.
GRID_THRESHOLDS: tuple[float, ...] = tuple(float(value) for value in range(0, 101, 5))
GRID_MARGINS: tuple[float, ...] = tuple(float(value) for value in range(0, 41, 5))
#: The gate's floors on the test side (D3), and the domain whose precision it reads off that domain's own source.
GATE_MACRO_F1 = 0.7
GATE_PRECISION = 0.8
GATE_DOMAIN = "medical"
#: Judge labels folded into one for scoring and fitting (D2): both are science writing, and the judge's
#: line between them is the classifier's largest confusion.
LABEL_MERGE: dict[str, str] = {"scientific": "science", "academic": "science"}
#: The labels scored, the judge's order with the merged ones collapsed into one at the first's place.
LABELS: tuple[str, ...] = tuple(dict.fromkeys(LABEL_MERGE.get(label, label) for label in DOMAINS))
#: Scores and metrics are tabled, and rules compared, at this many decimals.
DECIMALS = 4

#: Each build's corpus pass (M5): documents and words per source and assigned label, and the gold ids it holds.
CORPUS_LABELS: dict[str, Path] = {build: EXPERIMENT_DATA / "final" / f"corpus-labels-{build}.json" for build in BUILDS}
#: The candidate whose saved fit labelled both corpora (M5), and the hash both tracked passes logged for it;
#: a refit stops the analysis, since the corpus counts would no longer match the gate.
CORPUS_CLASSIFIER = "tfidf"
CORPUS_CLASSIFIER_SHA256 = "1ca0684142cd5113e1cdeb83971965138e7684613bd307fa75a126c3a7d26256"
#: The data targets the clauses read (M6): estimated tokens in the corpus, and tokens and documents per domain.
CORPUS_TOKEN_TARGET = 5_000_000_000
DOMAIN_TOKEN_TARGET = 500_000
DOMAIN_DOCUMENT_TARGET = 10_000
#: The domains the per-domain targets are read for, science folded (D2).
TARGET_DOMAINS: tuple[str, ...] = ("science", "medical")
#: What a count counts: words read as two tokens each, words read as one token each, and documents.
MEASURES: tuple[str, ...] = ("estimated tokens", "words", "documents")
#: A document counts under its source's label, or under the label the classifier assigned it.
LABELLING_RULES: tuple[str, ...] = ("source", "document")
#: Every source, or only those a public release may carry, mapped to whether the gated ones are left out.
SOURCE_SETS: dict[str, bool] = {"all": False, "open": True}
#: The catalogue's annotation value for datasets annotated with extraction spans, KPI 2's second reading.
IE_ANNOTATION = "IE spans"
#: The notes written beside the frozen split and the frozen lexicon hash.
SPLIT_NOTE = "Gold halved per judge domain, copies together (D3): tune picks the classifier, test reads its gate."
LEXICON_FREEZE_NOTE = "The lexicons the test side was first read with (M3); a changed lexicon reads it again."

# ===============================
# Helper functions
# ===============================


def stage_column(stage: str) -> str:
    """Name a funnel column after a stage folder, without the folder's order prefix.

    Args:
        stage: A stage folder name such as `03_quality`.

    Returns:
        The column name, such as `quality`.
    """
    return stage[STAGE_PREFIX_CHARS:]


#: The documents after each stage, in pipeline order (M1).
STAGE_COLUMNS = (*(stage_column(stage) for stage in SCOPED_STAGES), "exact_dedup", "sentence_dedup")
FUNNEL_COLUMNS = (*STAGE_COLUMNS, "words_final", "tokens_estimated")


class Labelled(NamedTuple):
    """One labelled gold document.

    Attributes:
        judge: The judge's domain.
        assigned: The label the rule assigned.
        fallback: The label it keeps when no domain clears the rule, its source label mapped.
    """

    judge: str
    assigned: str
    fallback: str


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line.

    Args:
        path: File to read.

    Returns:
        The objects in file order.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_sources(extract_config: Path) -> dict[str, dict[str, str]]:
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
    sentinel = read_sentinel(pretrain_dir / STATISTICS_STAGE)
    commit = sentinel.info.get("git_commit", "unknown") if sentinel else "unknown"
    if not commit.startswith(BUILD_COMMITS[build]):
        raise ValueError(f"{build} build statistics written at {commit}, record names {BUILD_COMMITS[build]}")


def build_rows(builds: dict[str, Path], sources: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Read the final-stage size of every source in each build.

    Args:
        builds: Curation output folder per build name.
        sources: `domain` and `access` per source, from `read_sources`.

    Returns:
        One row per build and source with its domain label, access, documents,
        words, mean words per document and estimated tokens.
    """
    rows: list[dict[str, Any]] = []
    for build, pretrain_dir in builds.items():
        check_build_commit(build, pretrain_dir)
        aggregate = json.loads((pretrain_dir / STATISTICS_STAGE / "aggregate.json").read_text(encoding="utf-8"))
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


def read_final_statistics(pretrain_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the statistics stage's file per source.

    Args:
        pretrain_dir: The build's curation output folder.

    Returns:
        The statistics per source, in source order. A source that reached the
        final corpus with no document has no file and is absent.
    """
    files = sorted((pretrain_dir / STATISTICS_STAGE / "per_dataset").glob("*.json"))
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in files}


def funnel_source_rows(
    build: str, pretrain_dir: Path, exact_counts: dict[str, int], sources: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
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
    # read the per-source stage sentinels and the final statistics
    scoped: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for stage, source, read, written in iter_stage_sentinels(pretrain_dir, SCOPED_STAGES):
        scoped[source][stage] = (read, written)
    finals = read_final_statistics(pretrain_dir)
    if unconverted := sorted(set(finals) - set(scoped)):
        raise ValueError(f"{build}: statistics name sources no stage converted: {unconverted}")

    # follow each source through the stages
    rows: list[dict[str, Any]] = []
    for source in sorted(scoped):
        row: dict[str, Any] = {
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
        row["retained"] = round(row["sentence_dedup"] / row["convert"], DECIMALS)
        rows.append(row)

    # check the sums against the dedup sentinels
    check_corpus_stages(build, pretrain_dir, rows)
    return rows


def check_corpus_stages(build: str, pretrain_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Fail unless the per-source counts add up to what the two dedup stages recorded.

    Args:
        build: Build name.
        pretrain_dir: The build's curation output folder.
        rows: Rows from `funnel_source_rows`.

    Raises:
        ValueError: If a column's sum differs from the corpus-wide sentinel.
    """
    exact = read_sentinel(pretrain_dir / COUNTED_STAGE)
    sentence = read_sentinel(pretrain_dir / SENTENCE_STAGE)
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


def funnel_domain_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sum the per-source funnel over the sources sharing a domain label.

    Args:
        source_rows: Rows from `funnel_source_rows`, any number of builds.

    Returns:
        One row per build and source domain, largest final corpus first, with
        its source count, its documents after every stage, its final words and
        estimated tokens, and the share of its converted documents kept.
    """
    totals: dict[tuple[str, str], dict[str, Any]] = {}
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
        row["retained"] = round(row["sentence_dedup"] / row["convert"], DECIMALS)
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


def gold_documents(sample_path: Path, verdicts_path: Path) -> list[dict[str, Any]]:
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
    return hashlib.sha256(opening.encode("utf-8")).hexdigest()[:GROUP_KEY_CHARS]


def split_gold(gold: list[dict[str, Any]], seed: int) -> dict[str, str]:
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
    # group the copies under their majority judge domain
    members_of: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in gold:
        members_of[document["group"]].append(document)
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for members in members_of.values():
        domain = Counter(document["judge_domain"] for document in members).most_common(1)[0][0]
        groups[domain].append(sorted(document["id"] for document in members))

    # deal the groups of each domain to the smaller side
    side_of: dict[str, str] = {}
    for domain in sorted(groups):
        members = sorted(groups[domain])
        random.Random(f"{seed}:{domain}").shuffle(members)
        held = {"tune": 0, "test": 0}
        for ids in members:
            side = "tune" if held["tune"] <= held["test"] else "test"
            held[side] += len(ids)
            side_of.update(dict.fromkeys(ids, side))
    return side_of


def write_split(gold: list[dict[str, Any]], side_of: dict[str, str], split_dir: Path, seed: int) -> None:
    """Write the two halves to `final/`, or check that what is there is the same split.

    Args:
        gold: Rows from `gold_documents`.
        side_of: `tune` or `test` per document id.
        split_dir: Folder for `tune.jsonl`, `test.jsonl` and `split.meta.json`.
        seed: The shuffle seed, recorded in the meta file.

    Raises:
        ValueError: If a half already on disk differs from the one computed now.
    """
    # write each half, or check it against the one on disk
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

    # write the meta file
    meta = {
        "seed": seed,
        "group_chars": GROUP_CHARS,
        "sample_sha256": GOLD_SAMPLE_SHA256,
        "tune": sum(side == "tune" for side in side_of.values()),
        "test": sum(side == "test" for side in side_of.values()),
        "note": SPLIT_NOTE,
    }
    (split_dir / "split.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def lexicon_seed_rows(sample_path: Path, gold: list[dict[str, Any]], side_of: dict[str, str]) -> list[dict[str, Any]]:
    """Rank, per judge domain, the words most distinctive of its tune-side documents (the M2 seed).

    A word's score is its log-odds of appearing in the domain's documents
    against the other domains', smoothed with a prior proportional to the
    word's overall count and divided by its standard error, after Monroe,
    Colaresi and Quinn (2008). Only the tune half is read, so the test half
    stays unseen by the lexicons.

    Args:
        sample_path: The judged sample, for the texts.
        gold: Rows from `gold_documents`, for each document's judge domain.
        side_of: `tune` or `test` per document id.

    Returns:
        The top words per judge domain with their score and counts, best first.
    """
    # count the words of the tune-side documents per domain
    domain_of = {document["id"]: document["judge_domain"] for document in gold if side_of[document["id"]] == "tune"}
    per_domain: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for document in read_jsonl(sample_path):
        if document["id"] not in domain_of:
            continue
        words = [word.lower() for word in SEED_WORD.findall(document["text"])]
        counts = Counter(word for word in words if len(word) > SEED_MIN_CHARS and not word.isdigit())
        per_domain[domain_of[document["id"]]].update(counts)
        overall.update(counts)

    # rank each domain's words by smoothed log-odds
    total = sum(overall.values())
    rows: list[dict[str, Any]] = []
    for domain in sorted(per_domain):
        counts = per_domain[domain]
        in_domain = sum(counts.values())
        scored = []
        for word, count in counts.items():
            if count < SEED_MIN_COUNT:
                continue
            rest = overall[word] - count
            prior = SEED_PRIOR * overall[word]
            odds_in = math.log((count + prior) / (in_domain + SEED_PRIOR * total - count - prior))
            odds_out = math.log((rest + prior) / (total - in_domain + SEED_PRIOR * total - rest - prior))
            error = math.sqrt(1 / (count + prior) + 1 / (rest + prior))
            scored.append(((odds_in - odds_out) / error, word, count, overall[word]))
        scored.sort(reverse=True)
        rows += [
            {
                "judge_domain": domain,
                "rank": rank,
                "word": word,
                "z": round(z, SEED_SCORE_DECIMALS),
                "tune_domain": count,
                "tune_all": all_,
            }
            for rank, (z, word, count, all_) in enumerate(scored[:SEED_TOP], start=1)
        ]
    return rows


def gold_texts(sample_path: Path, gold: list[dict[str, Any]]) -> dict[str, str]:
    """Read the text of every judged document.

    Args:
        sample_path: The judged sample.
        gold: Rows from `gold_documents`, naming the documents wanted.

    Returns:
        Each document's text by its id, in the sample's order.
    """
    wanted = {document["id"] for document in gold}
    return {document["id"]: document["text"] for document in read_jsonl(sample_path) if document["id"] in wanted}


def merge_label(label: str) -> str:
    """Return the label a judge or classifier label is scored as, folding the merged ones ([D2]).

    Args:
        label: A judge or classifier label.

    Returns:
        The label it is scored as.
    """
    return LABEL_MERGE.get(label, label)


def score_gold(texts: dict[str, str], gold: list[dict[str, Any]], lexicons: DomainLexicons) -> list[dict[str, Any]]:
    """Score every judged document against the lexicons once, so every rule is read on the same scores.

    Args:
        texts: Each judged document's text by id, from `gold_texts`.
        gold: Rows from `gold_documents`, for each document's judge domain and source label.
        lexicons: The compiled lexicons.

    Returns:
        One row per judged document with its id, judge domain, the label it
        keeps when no domain clears the rule, its score per domain and its
        word count.
    """
    rows = []
    for document in gold:
        scores, words = score_text(texts[document["id"]], lexicons)
        rows.append(
            {
                "id": document["id"],
                "judge_domain": merge_label(document["judge_domain"]),
                "fallback": merge_label(lexicons.fallback_label(document["source_domain"])),
                "scores": scores,
                "words": words,
            }
        )
    return rows


def label_scored(scored: list[dict[str, Any]], rule: DecisionRule) -> list[Labelled]:
    """Read one decision rule over scored documents.

    Args:
        scored: Rows from `score_gold`.
        rule: The threshold, margin and length floor to read them with.

    Returns:
        Per document, its judge domain, the label the rule assigns and its fallback label.
    """
    return [
        Labelled(
            row["judge_domain"],
            merge_label(assign_domain(row["scores"], row["words"], rule, row["fallback"])),
            row["fallback"],
        )
        for row in scored
    ]


def label_metrics(labelled: list[Labelled], labels: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Compute precision, recall and F1 per label.

    Args:
        labelled: Judge domain, assigned label and fallback label per document.
        labels: The labels to report, each present whether or not it occurs.

    Returns:
        Per label, its gold documents, how many documents were assigned it,
        and its precision, recall and F1, zero where undefined.
    """
    gold_count = Counter(row.judge for row in labelled)
    assigned_count = Counter(row.assigned for row in labelled)
    correct = Counter(row.judge for row in labelled if row.judge == row.assigned)
    metrics = {}
    for label in labels:
        precision = correct[label] / assigned_count[label] if assigned_count[label] else 0.0
        recall = correct[label] / gold_count[label] if gold_count[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "documents": gold_count[label],
            "assigned": assigned_count[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    """Average F1 over every label of `label_metrics`, each label counting once whatever its size.

    Args:
        metrics: The result of `label_metrics`.

    Returns:
        The macro-averaged F1.
    """
    return statistics.mean(label["f1"] for label in metrics.values())


def off_source(labelled: list[Labelled], label: str) -> list[Labelled]:
    """Keep the documents whose fallback label is not `label`, where only the lexicons can assign it.

    Args:
        labelled: Judge domain, assigned label and fallback label per document.
        label: The label whose own-source documents are left out.

    Returns:
        The documents kept.
    """
    return [row for row in labelled if row.fallback != label]


def rounded(row: dict[str, Any]) -> dict[str, Any]:
    """Round every float of a table row to `DECIMALS`, leaving counts and labels as they are.

    Args:
        row: A table row.

    Returns:
        The row with its floats rounded.
    """
    return {key: round(value, DECIMALS) if isinstance(value, float) else value for key, value in row.items()}


def rule_grid_rows(
    scored: list[dict[str, Any]], min_words: int, thresholds: tuple[float, ...], margins: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Read every threshold and margin pair on the tune side and mark the pick.

    The pick is the pair with the highest macro-F1; ties go to the higher
    medical precision, then to the lower threshold, then the lower margin, so
    equal fits prefer the rule that assigns more documents.

    Args:
        scored: Tune-side rows from `score_gold`.
        min_words: The length floor, fixed by the config.
        thresholds: The thresholds to try.
        margins: The margins to try.

    Returns:
        One row per pair with its macro-F1 and medical precision, and whether it is the pick.
    """
    rows = []
    for threshold in thresholds:
        for margin in margins:
            metrics = label_metrics(label_scored(scored, DecisionRule(threshold, margin, min_words)), LABELS)
            rows.append(
                rounded(
                    {
                        "threshold": threshold,
                        "margin": margin,
                        "macro_f1": macro_f1(metrics),
                        f"{GATE_DOMAIN}_precision": metrics[GATE_DOMAIN]["precision"],
                    }
                )
            )
    picked = max(
        rows, key=lambda row: (row["macro_f1"], row[f"{GATE_DOMAIN}_precision"], -row["threshold"], -row["margin"])
    )
    return [{**row, "picked": row is picked} for row in rows]


def check_rule_tuned(rule: DecisionRule, grid: list[dict[str, Any]], config_path: Path) -> None:
    """Stop unless the config's rule is the tune-side pick, so the test side is never read on an untuned rule.

    Args:
        rule: The rule the config carries.
        grid: Rows from `rule_grid_rows`, one of them marked as the pick.
        config_path: The config, named in the error.

    Raises:
        ValueError: If the config's threshold or margin differs from the pick.
    """
    picked = next(row for row in grid if row["picked"])
    if (rule.threshold, rule.margin) != (picked["threshold"], picked["margin"]):
        raise ValueError(
            f"{config_path.name} rule threshold={rule.threshold:g} margin={rule.margin:g} is not the tune-side pick "
            f"threshold={picked['threshold']:g} margin={picked['margin']:g}; write the pick into the config"
        )


def settings_digest(settings: Any) -> str:
    """Hash a JSON-serialisable value, the same way wherever a read is frozen.

    Args:
        settings: The value to hash.

    Returns:
        Its sha256 hex digest.
    """
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def check_lexicons_frozen(config_path: Path, freeze_path: Path) -> None:
    """Freeze the lexicons at the first test-side read, and stop when a later run reads it with other lexicons.

    Args:
        config_path: The classifier config; its `weights` and `lexicons` are hashed.
        freeze_path: Where the hash is kept, written when missing.

    Raises:
        ValueError: If the lexicons differ from the frozen ones.
    """
    config = load_yaml(config_path)
    digest = settings_digest({key: config[key] for key in ("weights", "lexicons")})
    if not freeze_path.exists():
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            json.dumps({"sha256": digest, "note": LEXICON_FREEZE_NOTE}, indent=2) + "\n", encoding="utf-8"
        )
        return
    expected = json.loads(freeze_path.read_text(encoding="utf-8"))["sha256"]
    if digest != expected:
        raise ValueError(
            f"{config_path.name} lexicons hash to {digest}, frozen as {expected} in {freeze_path} when the test side "
            "was first read; a changed lexicon reads the test side again, which D3 allows only by decision"
        )


def gate_rows(labelled_by_side: dict[str, list[Labelled]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the gate's two numbers per side of the gold, with medical precision also off its own source.

    Args:
        labelled_by_side: `tune` and `test` to their labelled documents.
        settings: The classifier settings picked on the tune side, one column each.

    Returns:
        One row per side with the settings, macro-F1, medical precision over
        all documents and over those whose source is not the medical one,
        and whether both floors are met.
    """
    rows = []
    for side, labelled in labelled_by_side.items():
        metrics = label_metrics(labelled, LABELS)
        off_source_metrics = label_metrics(off_source(labelled, GATE_DOMAIN), LABELS)[GATE_DOMAIN]
        row = rounded(
            {
                "side": side,
                "documents": len(labelled),
                **settings,
                "macro_f1": macro_f1(metrics),
                f"{GATE_DOMAIN}_precision": metrics[GATE_DOMAIN]["precision"],
                f"{GATE_DOMAIN}_precision_off_source": off_source_metrics["precision"],
                f"{GATE_DOMAIN}_recall_off_source": off_source_metrics["recall"],
            }
        )
        row["passes_gate"] = (
            row["macro_f1"] >= GATE_MACRO_F1 and row[f"{GATE_DOMAIN}_precision_off_source"] >= GATE_PRECISION
        )
        rows.append(row)
    return rows


def fit_or_load_candidate(
    name: str, config_path: Path, tune: list[dict[str, Any]], sample_sha256: str, models_dir: Path
) -> tuple[DomainClassifier, list[dict[str, Any]]]:
    """Fit a supervised candidate on the tune side once, and reuse the saved fit after.

    The saved fit is what the test side is read with, so a change to anything
    it was fitted from stops the analysis rather than refit in silence;
    deleting the saved fit is the deliberate way to read the test side again.

    Args:
        name: The candidate's key in the config.
        config_path: The supervised classifier config.
        tune: The tune-side documents, each with its id, text, judge domain and copy group.
        sample_sha256: The hash of the judged sample the texts come from.
        models_dir: Where the fit and its regularisation grid are kept.

    Returns:
        The fitted classifier and its grid, one row per regularisation strength.

    Raises:
        ValueError: If the saved fit came from other settings, documents or library than now.
    """
    # hash everything the fit depends on
    config = load_yaml(config_path)
    digest = settings_digest(
        {
            "settings": {key: value for key, value in config.items() if key != "candidates"},
            "candidate": config["candidates"][name],
            "sample_sha256": sample_sha256,
            "tune": [(document["id"], document["judge_domain"]) for document in tune],
            "scikit-learn": sklearn.__version__,
        }
    )
    # reuse the saved fit when its inputs are unchanged
    model_path = models_dir / f"{name}.pkl"
    grid_path = models_dir / f"{name}.grid.json"
    if model_path.exists():
        saved = json.loads(grid_path.read_text(encoding="utf-8"))
        if saved["sha256"] != digest:
            raise ValueError(
                f"candidate '{name}' was fitted from inputs hashing to {saved['sha256']}, which now hash to {digest}; "
                f"delete {model_path} to refit, which reads the test side again"
            )
        return DomainClassifier.load(model_path), saved["grid"]

    # fit on the tune side and save
    classifier, grid = fit_classifier(
        [document["text"] for document in tune],
        [document["judge_domain"] for document in tune],
        [document["group"] for document in tune],
        LABELS,
        load_candidate(config_path, name),
    )
    classifier.save(model_path)
    grid_path.write_text(json.dumps({"sha256": digest, "grid": grid}, indent=2) + "\n", encoding="utf-8")
    return classifier, grid


def label_predicted(scored: list[dict[str, Any]], predicted: list[str]) -> list[Labelled]:
    """Pair each gold document with the label a classifier gave it.

    Args:
        scored: Rows from `score_gold`, for the judge domain and fallback label.
        predicted: One label per row, in the same order.

    Returns:
        The labelled documents.
    """
    return [Labelled(row["judge_domain"], label, row["fallback"]) for row, label in zip(scored, predicted)]


def write_gate_tables(name: str, labelled: dict[str, list[Labelled]], settings: dict[str, Any]) -> bool:
    """Write one classifier's gate, per-domain and confusion tables and print its test-side verdict.

    Args:
        name: The classifier's name, the stem of its tables.
        labelled: `tune` and `test` to their labelled documents.
        settings: The settings picked on the tune side, tabled beside the gate.

    Returns:
        Whether the classifier passes the gate on the test side.
    """
    gate = gate_rows(labelled, settings)
    write_csv(TABLES_DIR / f"classifier-{name}-gate.csv", gate)
    write_csv(TABLES_DIR / f"classifier-{name}-gate-by-domain.csv", gate_domain_rows(labelled))
    write_csv(TABLES_DIR / f"classifier-{name}-confusion-by-domain.csv", confusion_rows(labelled))
    test_gate = next(row for row in gate if row["side"] == "test")
    verdict = "passes" if test_gate["passes_gate"] else "fails"
    print(
        f"{name} gate: test macro-F1 {test_gate['macro_f1']:.3f}, {GATE_DOMAIN} precision off its source "
        f"{test_gate[f'{GATE_DOMAIN}_precision_off_source']:.3f}, {verdict}"
    )
    return bool(test_gate["passes_gate"])


def gate_domain_rows(labelled_by_side: dict[str, list[Labelled]]) -> list[dict[str, Any]]:
    """Table precision, recall and F1 per judge domain and side, over all documents and off the domain's own source.

    Args:
        labelled_by_side: `tune` and `test` to their rows from `label_scored`.

    Returns:
        One row per side and domain in the judge's label order.
    """
    rows = []
    for side, labelled in labelled_by_side.items():
        metrics = label_metrics(labelled, LABELS)
        for domain in LABELS:
            off_source_metrics = label_metrics(off_source(labelled, domain), LABELS)[domain]
            rows.append(
                rounded(
                    {
                        "side": side,
                        "judge_domain": domain,
                        "documents": metrics[domain]["documents"],
                        "assigned": metrics[domain]["assigned"],
                        "precision": metrics[domain]["precision"],
                        "recall": metrics[domain]["recall"],
                        "f1": metrics[domain]["f1"],
                        "off_source_documents": off_source_metrics["documents"],
                        "off_source_assigned": off_source_metrics["assigned"],
                        "off_source_precision": off_source_metrics["precision"],
                        "off_source_recall": off_source_metrics["recall"],
                    }
                )
            )
    return rows


def confusion_rows(labelled_by_side: dict[str, list[Labelled]]) -> list[dict[str, Any]]:
    """Count, per side, the documents of each judge domain under each assigned label.

    Args:
        labelled_by_side: `tune` and `test` to their rows from `label_scored`.

    Returns:
        One row per side, judge domain and assigned label, zeros included.
    """
    rows = []
    for side, labelled in labelled_by_side.items():
        counts = Counter((row.judge, row.assigned) for row in labelled)
        rows += [
            {"side": side, "judge_domain": judge, "assigned_domain": assigned, "documents": counts[judge, assigned]}
            for judge in LABELS
            for assigned in LABELS
        ]
    return rows


def gold_rows(gold: list[dict[str, Any]], side_of: dict[str, str]) -> list[dict[str, Any]]:
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
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
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


def catalogue_rows(datasets: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarise the landscape catalogue per domain and access.

    A dataset listing several domains counts once under each.

    Args:
        datasets: Rows from `read_catalogue`.

    Returns:
        One row per domain and access with the datasets, how many fit the
        examples target, how many report a size, and their summed documents
        and estimated tokens.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
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


def read_corpus_labels(path: Path, build: str, funnel: list[dict[str, Any]]) -> dict[str, Any]:
    """Read one build's corpus pass and check it against the statistics stage, source by source.

    Args:
        path: The pass's JSON, written by the `classify` subcommand ([M5]).
        build: Build name.
        funnel: Rows from `funnel_source_rows`, any number of builds.

    Returns:
        The pass: `documents` and `words` per source and assigned label, and
        the `gold_ids` the build still holds.

    Raises:
        ValueError: If the pass names a source the build's funnel lacks, or
            counts a source's documents or words other than the statistics stage did.
    """
    labels = json.loads(path.read_text(encoding="utf-8"))
    finals = {row["source"]: row for row in funnel if row["build"] == build}
    if unknown := sorted(set(labels["documents"]) - set(finals)):
        raise ValueError(f"{build}: the corpus pass names sources the funnel lacks: {unknown}")
    for source, final in finals.items():
        documents = sum(labels["documents"].get(source, {}).values())
        words = sum(labels["words"].get(source, {}).values())
        if (documents, words) != (final["sentence_dedup"], final["words_final"]):
            raise ValueError(
                f"{build} {source}: the corpus pass counts {documents} documents and {words} words, "
                f"the statistics stage {final['sentence_dedup']} and {final['words_final']}"
            )
    return labels


def domain_totals(
    labels: dict[str, Any], sources: dict[str, dict[str, str]], rule: str, open_only: bool
) -> dict[str, tuple[int, int]]:
    """Sum one corpus pass's documents and words per domain under one labelling rule.

    Under `source` every document counts under its source's registry label,
    unfolded, so scientific and academic stay apart; under `document` it
    counts under the label the classifier assigned it.

    Args:
        labels: A pass from `read_corpus_labels`.
        sources: `domain` and `access` per source, from `read_sources`.
        rule: `source` or `document`, a member of `LABELLING_RULES`.
        open_only: Leave out the gated sources, which a public release cannot carry.

    Returns:
        Documents and words per domain.
    """
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for source, by_label in labels["documents"].items():
        if open_only and sources[source]["access"] == "gated":
            continue
        for label, documents in by_label.items():
            total = totals[sources[source]["domain"] if rule == "source" else label]
            total[0] += documents
            total[1] += labels["words"][source][label]
    return {domain: (documents, words) for domain, (documents, words) in totals.items()}


def measure_count(documents: int, words: int, measure: str) -> int:
    """Return the count one measure of `MEASURES` reads from a domain's documents and words.

    Args:
        documents: The domain's documents.
        words: The domain's words.
        measure: A member of `MEASURES`.

    Returns:
        The count the measure reads.
    """
    return {"estimated tokens": words * TOKENS_PER_WORD, "words": words, "documents": documents}[measure]


def kpi_row(
    build: str, domain: str, measure: str, labels: str, sources: str, count: int, is_readable: bool = True
) -> dict[str, Any]:
    """Read one count against its target.

    Args:
        build: Build name.
        domain: `corpus`, or a member of `TARGET_DOMAINS`.
        measure: A member of `MEASURES`.
        labels: The labelling rule, or `either` for the corpus, whose total no rule changes.
        sources: A key of `SOURCE_SETS`.
        count: The count read.
        is_readable: Whether the count may be read against its target; a count
            under document labels may not when the classifier failed its gate ([D3]).

    Returns:
        The row, with the target, whether the count is above it (none when it
        may not be read), and the share of the count that must truly belong to
        the domain for it to stay above.
    """
    if domain == "corpus":
        target = CORPUS_TOKEN_TARGET
    else:
        target = DOMAIN_DOCUMENT_TARGET if measure == "documents" else DOMAIN_TOKEN_TARGET
    return {
        "build": build,
        "domain": domain,
        "measure": measure,
        "labels": labels,
        "sources": sources,
        "count": count,
        "target": target,
        "passes": count > target if is_readable else None,
        "share_needed": round(target / count, DECIMALS) if count else None,
    }


def kpi_rows(
    build: str, labels: dict[str, Any], sources: dict[str, dict[str, str]], classifier_passes: bool
) -> list[dict[str, Any]]:
    """Read every target under every counting rule on one build ([M6]).

    The corpus is read in estimated tokens and in words; each target domain
    in estimated tokens, words and documents, under source and document
    labels; everything with every source and with the gated ones removed.

    Args:
        build: Build name.
        labels: The build's pass from `read_corpus_labels`.
        sources: `domain` and `access` per source, from `read_sources`.
        classifier_passes: Whether the classifier that labelled the corpus passed
            its gate; when it did not, no count under document labels is read ([D3]).

    Returns:
        One row per count read, from `kpi_row`: the corpus first, then each
        domain measure by measure.
    """
    totals = {
        (rule, source_set): domain_totals(labels, sources, rule, open_only)
        for rule in LABELLING_RULES
        for source_set, open_only in SOURCE_SETS.items()
    }
    rows = []
    for measure in MEASURES[:2]:
        for source_set in SOURCE_SETS:
            words = sum(words for _, words in totals["source", source_set].values())
            rows.append(kpi_row(build, "corpus", measure, "either", source_set, measure_count(0, words, measure)))
    for domain in TARGET_DOMAINS:
        for measure in MEASURES:
            for rule in LABELLING_RULES:
                for source_set in SOURCE_SETS:
                    members = [
                        counts for label, counts in totals[rule, source_set].items() if merge_label(label) == domain
                    ]
                    documents, words = (sum(column) for column in zip(*members)) if members else (0, 0)
                    count = measure_count(documents, words, measure)
                    is_readable = rule == "source" or classifier_passes
                    rows.append(kpi_row(build, domain, measure, rule, source_set, count, is_readable))
    return rows


def corpus_domain_rows(build: str, labels: dict[str, Any], sources: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Total one build's documents and estimated tokens per domain under each labelling rule.

    Args:
        build: Build name.
        labels: The build's pass from `read_corpus_labels`.
        sources: `domain` and `access` per source, from `read_sources`.

    Returns:
        One row per labelling rule and domain, the target domains first, then
        by documents, with the same totals over the open sources beside.
    """
    rows = []
    for rule in LABELLING_RULES:
        every = domain_totals(labels, sources, rule, open_only=False)
        public = domain_totals(labels, sources, rule, open_only=True)
        for domain, (documents, words) in every.items():
            open_documents, open_words = public.get(domain, (0, 0))
            rows.append(
                {
                    "build": build,
                    "labels": rule,
                    "domain": domain,
                    "documents": documents,
                    "tokens_estimated": words * TOKENS_PER_WORD,
                    "open_documents": open_documents,
                    "open_tokens_estimated": open_words * TOKENS_PER_WORD,
                }
            )
    return sorted(
        rows, key=lambda row: (row["labels"], merge_label(row["domain"]) not in TARGET_DOMAINS, -row["documents"])
    )


def corpus_source_rows(build: str, labels: dict[str, Any], sources: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Table one build's documents and estimated tokens per source and assigned label.

    Args:
        build: Build name.
        labels: The build's pass from `read_corpus_labels`.
        sources: `domain` and `access` per source, from `read_sources`.

    Returns:
        One row per source and assigned label, the target domains first, then
        by documents.
    """
    rows = [
        {
            "build": build,
            "assigned_domain": label,
            "source": source,
            "source_domain": sources[source]["domain"],
            "access": sources[source]["access"],
            "documents": documents,
            "tokens_estimated": labels["words"][source][label] * TOKENS_PER_WORD,
        }
        for source, by_label in labels["documents"].items()
        for label, documents in by_label.items()
    ]
    order = {domain: index for index, domain in enumerate(TARGET_DOMAINS)}
    return sorted(rows, key=lambda row: (order.get(row["assigned_domain"], len(order)), -row["documents"]))


def domain_stage_counts(funnel: list[dict[str, Any]], build: str, domain: str, open_only: bool) -> dict[str, int]:
    """Sum the documents after each stage over the sources whose label folds into one domain.

    Args:
        funnel: Rows from `funnel_source_rows`, any number of builds.
        build: Build name.
        domain: The folded domain, such as `science`.
        open_only: Leave out the gated sources.

    Returns:
        Documents per stage column of `STAGE_COLUMNS`.
    """
    members = [
        row
        for row in funnel
        if row["build"] == build
        and merge_label(row["source_domain"]) == domain
        and not (open_only and row["access"] == "gated")
    ]
    return {column: sum(row[column] for row in members) for column in STAGE_COLUMNS}


def gap_rows(kpi: list[dict[str, Any]], funnel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attribute every missed target to a missing source or to the curation stage that lost it ([M7]).

    A domain whose sources held no more than the target once converted has a
    source gap. One that held more and fell to the target or below is a
    curation loss, named by the first stage after which it holds no more.
    Only documents under source labels have a count per stage: words are
    counted at the final stage alone and document labels exist only there,
    so any other missed target is tabled as not attributable.

    Args:
        kpi: Rows from `kpi_rows`.
        funnel: Rows from `funnel_source_rows`, any number of builds.

    Returns:
        One row per missed target, in the order of `kpi`, with its converted
        and final documents, the cause, and the stage a curation loss names.
    """
    rows = []
    for shortfall in kpi:
        # a count that may not be read against its target ([D3]) is no shortfall to attribute
        if shortfall["passes"] is not False:
            continue
        row = {key: shortfall[key] for key in ("build", "domain", "measure", "labels", "sources")}
        if shortfall["labels"] != "source" or shortfall["measure"] != "documents":
            rows.append(
                row | {"converted": None, "final": shortfall["count"], "cause": "not attributable", "stage": ""}
            )
            continue
        counts = domain_stage_counts(funnel, shortfall["build"], shortfall["domain"], shortfall["sources"] == "open")
        target = shortfall["target"]
        if counts["convert"] > target:
            cause = "curation loss"
            stage = next(column for column in STAGE_COLUMNS if counts[column] <= target)
        else:
            cause, stage = "source gap", ""
        rows.append(
            row | {"converted": counts["convert"], "final": counts["sentence_dedup"], "cause": cause, "stage": stage}
        )
    return rows


def read_catalogue(catalogue_path: Path) -> list[dict[str, str]]:
    """Read the landscape catalogue, one row per dataset.

    Args:
        catalogue_path: The data-landscape survey's `catalogue.csv`.

    Returns:
        The catalogue's rows in file order.
    """
    with catalogue_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def catalogue_domains(dataset: dict[str, str]) -> list[str]:
    """Return a catalogue dataset's domains under the judge's labels, science folded ([D2]).

    Args:
        dataset: A catalogue row.

    Returns:
        Its distinct domains, in catalogue order.
    """
    return list(
        dict.fromkeys(merge_label(CATALOGUE_DOMAINS.get(domain, domain)) for domain in dataset["domains"].split(","))
    )


def catalogue_dataset_row(domain: str, dataset: dict[str, str]) -> dict[str, Any]:
    """Table one catalogue dataset under one of its domains.

    Args:
        domain: The domain it is listed under.
        dataset: Its catalogue row.

    Returns:
        Its name, access, provenance, annotation, documents (none when the
        survey found no count), whether the survey judged it to fit the
        examples target, and its address.
    """
    return {
        "domain": domain,
        "dataset": dataset["name"],
        "access": dataset["access"],
        "provenance": dataset["provenance_class"],
        "annotation": dataset["annotation"],
        "documents": int(dataset["documents"]) if dataset["documents"] else None,
        "fits_examples_target": dataset["kpi2_fit"],
        "url": dataset["url"],
    }


def by_domain_then_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort catalogue rows with the target domains first, then by documents, uncounted last.

    Args:
        rows: Rows from `catalogue_dataset_row`.

    Returns:
        The rows sorted.
    """
    order = {domain: index for index, domain in enumerate(TARGET_DOMAINS)}
    return sorted(
        rows, key=lambda row: (order.get(row["domain"], len(order)), row["domain"], -(row["documents"] or -1))
    )


def gap_catalogue_rows(gaps: list[dict[str, Any]], catalogue: list[dict[str, str]]) -> list[dict[str, Any]]:
    """List the open catalogue datasets of each domain that misses a target and the corpus does not yet hold.

    Args:
        gaps: Rows from `gap_rows`.
        catalogue: Rows from `read_catalogue`.

    Returns:
        One row per domain and dataset, from `catalogue_dataset_row`, largest first.
    """
    domains = dict.fromkeys(gap["domain"] for gap in gaps)
    rows = [
        catalogue_dataset_row(domain, dataset)
        for domain in domains
        for dataset in catalogue
        if domain in catalogue_domains(dataset) and dataset["access"] == "open" and dataset["in_registry"] != "yes"
    ]
    return by_domain_then_documents(rows)


def ie_example_rows(catalogue: list[dict[str, str]]) -> list[dict[str, Any]]:
    """List the catalogue datasets annotated for extraction, the second reading of the examples target.

    Args:
        catalogue: Rows from `read_catalogue`.

    Returns:
        One row per domain and dataset annotated with extraction spans, from
        `catalogue_dataset_row`, the target domains first.
    """
    rows = [
        catalogue_dataset_row(domain, dataset)
        for dataset in catalogue
        if dataset["annotation"] == IE_ANNOTATION
        for domain in catalogue_domains(dataset)
    ]
    return by_domain_then_documents(rows)


def final_corpus_gate_rows(
    test: list[dict[str, Any]], labelled: list[Labelled], gold_ids: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Read the gate's numbers again on the test-side documents each final corpus still holds.

    Args:
        test: The test side's rows from `score_gold`, for their ids.
        labelled: The same rows labelled by the classifier that labelled the corpora.
        gold_ids: The gold ids each build's final corpus holds, per build.

    Returns:
        One row per build, as `gate_rows` reads it, without its pass column:
        the gate is read on the whole test side alone ([D3]).
    """
    held = {build: set(ids) for build, ids in gold_ids.items()}
    rows = gate_rows(
        {
            f"test in the {build} final corpus": [
                document for row, document in zip(test, labelled) if row["id"] in held[build]
            ]
            for build in held
        },
        {},
    )
    return [{key: value for key, value in row.items() if key != "passes_gate"} for row in rows]


def kpi_figure_rows(kpi: list[dict[str, Any]], build: str) -> list[dict[str, Any]]:
    """Pair each target's count under source and under document labels, as a multiple of the target.

    Args:
        kpi: Rows from `kpi_rows`.
        build: Build name.

    Returns:
        One dumbbell record per target read with every source, plus the corpus
        with the gated sources removed: the source-label multiple at the
        start, the document-label one at the end.
    """
    multiple = {
        (row["domain"], row["measure"], row["labels"], row["sources"]): row["count"] / row["target"]
        for row in kpi
        if row["build"] == build
    }
    records = [
        {
            "label": f"corpus tokens, {source_set} sources",
            "start": multiple["corpus", "estimated tokens", "either", source_set],
            "end": multiple["corpus", "estimated tokens", "either", source_set],
        }
        for source_set in SOURCE_SETS
    ]
    for measure, name in (("estimated tokens", "tokens"), ("documents", "documents")):
        records += [
            {
                "label": f"{domain} {name}",
                "start": multiple[domain, measure, "source", "all"],
                "end": multiple[domain, measure, "document", "all"],
            }
            for domain in TARGET_DOMAINS
        ]
    return records


def draw_kpi_figure(kpi: list[dict[str, Any]]) -> None:
    """Draw every target's count as a multiple of it, under source and document labels, per build.

    Args:
        kpi: Rows from `kpi_rows`.
    """
    save_figure(
        lambda: DumbbellChart(
            [kpi_figure_rows(kpi, build) for build in BUILDS],
            title="Data targets per labelling rule",
            xlabel="count as a multiple of its target",
            ylabel="target",
            subtitle=[f"{build} build" for build in BUILDS],
            start_name="source labels",
            end_name="document labels",
            scaley=AXIS_SCALE.LOG,
            vlines=[{"x": 1, "label": "target"}],
            legend={"location": LEGEND_LOCATION.OUTSIDE_BOTTOM},
            figsize=FIG_SIZE.FULL_TALL,
        ),
        FIGURES_DIR / "kpi-target-multiple-by-rule.svg",
    )


def draw_medical_funnel_figure(funnel: list[dict[str, Any]]) -> None:
    """Draw the medical source's documents after every curation stage, per build.

    Args:
        funnel: Rows from `funnel_source_rows`, both builds.
    """
    series = [
        [
            {"x": index, "y": count}
            for index, count in enumerate(domain_stage_counts(funnel, build, "medical", open_only=False).values())
        ]
        for build in BUILDS
    ]
    save_figure(
        lambda: LineChart(
            series,
            title="Medical source documents per stage",
            xlabel="curation stage",
            ylabel="documents",
            subtitle=[f"{build} build" for build in BUILDS],
            scaley=AXIS_SCALE.LOG,
            xticks=list(range(len(STAGE_COLUMNS))),
            xticklabels=[column.replace("_", " ") for column in STAGE_COLUMNS],
            xtickrotate=30,
            hlines=[{"y": DOMAIN_DOCUMENT_TARGET, "label": "examples target"}],
            show_legend=True,
            legend={"location": LEGEND_LOCATION.OUTSIDE_BOTTOM},
            figsize=FIG_SIZE.FULL_MEDIUM,
        ),
        FIGURES_DIR / "funnel-medical-documents-by-stage.svg",
    )


def draw_medical_sources_figure(by_source: list[dict[str, Any]]) -> None:
    """Draw the documents the classifier labels medical in each source, per build.

    Args:
        by_source: Rows from `corpus_source_rows`, both builds.
    """
    medical = {
        (row["build"], row["source"]): row["documents"] for row in by_source if row["assigned_domain"] == "medical"
    }
    # horizontal bars draw the first label at the bottom, so the largest source goes last
    names = sorted({source for _, source in medical}, key=lambda source: medical.get(("loosened", source), 0))
    save_figure(
        lambda: BarChart(
            [[{"label": source, "y": medical.get((build, source), 0)} for source in names] for build in BUILDS],
            title="Medical documents per source",
            xlabel="documents labelled medical",
            ylabel="source",
            subtitle=[f"{build} build" for build in BUILDS],
            orientation=ORIENTATION.HORIZONTAL,
            show_legend=True,
            legend={"location": LEGEND_LOCATION.OUTSIDE_BOTTOM},
            xmin=0,
            figsize=FIG_SIZE.FULL_TALL,
        ),
        FIGURES_DIR / "corpus-medical-documents-by-source.svg",
    )


def draw_gold_figure(rows: list[dict[str, Any]]) -> None:
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


# ===============================
# Main function
# ===============================


def main() -> None:
    """Write the dataset statistics, the funnel, the gates, the per-target verdicts and the figures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # tabulate the final size of every source in each build
    sources = read_sources(EXTRACT_CONFIG)
    builds = build_rows(BUILDS, sources)
    write_csv(TABLES_DIR / "dataset-builds-statistics.csv", builds)
    for build in BUILDS:
        documents = sum(row["documents"] for row in builds if row["build"] == build)
        print(f"{build}: {documents:,} documents in the final corpus")

    # follow the documents through the curation stages
    funnel: list[dict[str, Any]] = []
    for build, pretrain_dir in BUILDS.items():
        cache = EXPERIMENT_DATA / "interim" / f"{build}-build-exact-dedup-counts.json"
        exact_counts = cached_source_counts(pretrain_dir / COUNTED_STAGE, cache, COUNT_WORKERS)
        funnel += funnel_source_rows(build, pretrain_dir, exact_counts, sources)
    write_csv(TABLES_DIR / "funnel-documents-by-source.csv", funnel)
    domains = funnel_domain_rows(funnel)
    write_csv(TABLES_DIR / "funnel-documents-by-domain.csv", domains)
    print(f"funnel: {len(funnel)} source rows and {len(domains)} domain rows over {len(BUILDS)} builds")

    # check the frozen gold and split it into tune and test
    check_frozen(GOLD_VERDICTS, json.loads(GOLD_META.read_text(encoding="utf-8"))["sha256"])
    check_frozen(GOLD_SAMPLE, GOLD_SAMPLE_SHA256)
    documents = gold_documents(GOLD_SAMPLE, GOLD_VERDICTS)
    side_of = split_gold(documents, SPLIT_SEED)
    write_split(documents, side_of, SPLIT_DIR, SPLIT_SEED)

    # summarise the gold and seed the lexicons
    gold = gold_rows(documents, side_of)
    write_csv(TABLES_DIR / "dataset-gold-statistics.csv", gold)
    draw_gold_figure(gold)
    print(f"gold: {sum(row['documents'] for row in gold):,} judged documents over {len(gold)} judge domains")
    seed = lexicon_seed_rows(GOLD_SAMPLE, documents, side_of)
    write_csv(TABLES_DIR / "classifier-lexicon-seed-words-by-domain.csv", seed)
    print(f"lexicon seed: {len(seed)} ranked words over {len({row['judge_domain'] for row in seed})} judge domains")

    # tune the lexicon rule on the tune side
    lexicons = load_lexicons(LEXICON_CONFIG)
    texts = gold_texts(GOLD_SAMPLE, documents)
    scored = score_gold(texts, documents, lexicons)
    by_side = {side: [row for row in scored if side_of[row["id"]] == side] for side in ("tune", "test")}
    rule_grid = rule_grid_rows(by_side["tune"], lexicons.rule.min_words, GRID_THRESHOLDS, GRID_MARGINS)
    write_csv(TABLES_DIR / "classifier-lexicon-gate-by-rule.csv", rule_grid)
    check_rule_tuned(lexicons.rule, rule_grid, LEXICON_CONFIG)
    check_lexicons_frozen(LEXICON_CONFIG, LEXICON_FREEZE)

    # read the lexicon gate on both sides
    lexicon_labelled = {side: label_scored(rows, lexicons.rule) for side, rows in by_side.items()}
    rule = lexicons.rule
    passes_by_name = {}
    lexicon_passes = passes_by_name["lexicon"] = write_gate_tables(
        "lexicon", lexicon_labelled, {"threshold": rule.threshold, "margin": rule.margin, "min_words": rule.min_words}
    )

    # fit the supervised candidates when the lexicons fail the gate
    group_of = {document["id"]: document["group"] for document in documents}
    tune = [
        {"id": row["id"], "text": texts[row["id"]], "judge_domain": row["judge_domain"], "group": group_of[row["id"]]}
        for row in by_side["tune"]
    ]
    candidate_labelled_by_name: dict[str, dict[str, list[Labelled]]] = {}
    for name in SUPERVISED_CANDIDATES if not lexicon_passes else ():
        classifier, regularisation_grid = fit_or_load_candidate(
            name, SUPERVISED_CONFIG, tune, GOLD_SAMPLE_SHA256, SUPERVISED_MODELS
        )
        write_csv(TABLES_DIR / f"classifier-{name}-gate-by-regularisation.csv", regularisation_grid)
        candidate_labelled = {
            side: label_predicted(rows, classifier.predict([texts[row["id"]] for row in rows]))
            for side, rows in by_side.items()
        }
        passes_by_name[name] = write_gate_tables(
            name, candidate_labelled, {"regularisation": classifier.regularisation}
        )
        candidate_labelled_by_name[name] = candidate_labelled

    # tabulate the catalogue
    catalogue = read_catalogue(CATALOGUE)
    catalogue_statistics = catalogue_rows(catalogue)
    write_csv(TABLES_DIR / "dataset-catalogue-statistics.csv", catalogue_statistics)
    print(f"catalogue: {len(catalogue_statistics)} domain and access groups")

    # read the corpus passes and the gate on the final corpora
    check_frozen(SUPERVISED_MODELS / f"{CORPUS_CLASSIFIER}.pkl", CORPUS_CLASSIFIER_SHA256)
    labels = {build: read_corpus_labels(path, build, funnel) for build, path in CORPUS_LABELS.items()}
    classifier_passes = passes_by_name.get(CORPUS_CLASSIFIER, False)
    if CORPUS_CLASSIFIER in candidate_labelled_by_name:
        gold_held = {build: labels[build]["gold_ids"] for build in BUILDS}
        final_gate = final_corpus_gate_rows(
            by_side["test"], candidate_labelled_by_name[CORPUS_CLASSIFIER]["test"], gold_held
        )
        write_csv(TABLES_DIR / f"classifier-{CORPUS_CLASSIFIER}-gate-by-final-corpus.csv", final_gate)

    # read every target and tabulate the corpus
    kpi = [row for build in BUILDS for row in kpi_rows(build, labels[build], sources, classifier_passes)]
    write_csv(TABLES_DIR / "kpi-pass-by-rule.csv", kpi)
    print(
        f"targets: {sum(row['passes'] is True for row in kpi)} of {len(kpi)} counts above their target, "
        f"{sum(row['passes'] is None for row in kpi)} not read"
    )
    by_source = [row for build in BUILDS for row in corpus_source_rows(build, labels[build], sources)]
    write_csv(TABLES_DIR / "corpus-documents-by-source.csv", by_source)
    by_domain = [row for build in BUILDS for row in corpus_domain_rows(build, labels[build], sources)]
    write_csv(TABLES_DIR / "corpus-documents-by-domain.csv", by_domain)

    # attribute the gaps and list the catalogue datasets
    gaps = gap_rows(kpi, funnel)
    write_csv(TABLES_DIR / "gap-attribution-by-stage.csv", gaps)
    write_csv(TABLES_DIR / "gap-catalogue-datasets.csv", gap_catalogue_rows(gaps, catalogue))
    print(f"gaps: {len(gaps)} missed targets, {sum(gap['cause'] == 'curation loss' for gap in gaps)} curation losses")
    write_csv(TABLES_DIR / "catalogue-ie-examples-by-domain.csv", ie_example_rows(catalogue))

    # draw the figures
    draw_kpi_figure(kpi)
    draw_medical_funnel_figure(funnel)
    draw_medical_sources_figure(by_source)


if __name__ == "__main__":
    main()
