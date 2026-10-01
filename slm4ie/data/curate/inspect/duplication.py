"""Decide whether the dedup stages dropped anything the corpus then lost.

The two dedup stages cannot be scored the way the content filters are. A
duplicate is not bad text — it reads exactly as well as the copy that
survived — so a judge asked which of the two is better has nothing to go on,
and the audit measured 3.4% against 3.9% at `exact_dedup`. What a dedup stage
must be right about is different: every document it drops should still be
present in the corpus as another copy.

That is what this module measures. For each dropped document it looks for a
twin among the documents the corpus kept:

* `exact_dedup` drops whole-document duplicates, so a twin is a kept document
  whose text hashes the same.
* `sentence_dedup` removes sentence windows it has seen before and then drops
  documents left under the length floors, so a twin is a kept document sharing
  the dropped document's windows. Coverage — the share of a dropped document's
  windows found elsewhere — says how much of it survived rather than whether
  any of it did.

A drop with no twin is text the corpus simply lost, and those ids are written
out so the judge can read them. Windows are built from a regex sentence split
rather than datatrove's spaCy tokenizer: the faithful split costs about ten
core-hours over a corpus this size, and both sides of every comparison here
use the same split, so the measurement stays internally consistent.
"""

import gzip
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import xxhash

logger = logging.getLogger(__name__)

#: Sentence boundary: terminal punctuation followed by space, or a line break.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…:;])\s+|\n+")

#: Windows are three sentences long, matching `sentence_dedup.n_sentences`.
WINDOW_SENTENCES: int = 3

#: The stages this module assesses, in pipeline order.
DEDUP_STAGES: Tuple[str, ...] = ("exact_dedup", "sentence_dedup")

#: The stage folder each dedup stage reads, where its dropped documents still are.
STAGE_INPUT: Dict[str, str] = {"exact_dedup": "04_repetition", "sentence_dedup": "05_exact_dedup"}

#: Where twins are looked for: the last folder holding untrimmed document text.
TWIN_SEARCH_DIR: str = "05_exact_dedup"

#: The finished corpus, used to ask whether a twin itself survived to the end.
FINAL_DIR: str = "06_sentence_dedup"


def sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: The document's text.

    Returns:
        Non-empty sentences, stripped.
    """
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]


def text_hash(text: str) -> int:
    """Hash a whole document's text.

    Args:
        text: The document's text.

    Returns:
        A 64-bit hash, so two documents with identical text agree.
    """
    return xxhash.xxh64_intdigest(text)


def window_hashes(text: str, window: int = WINDOW_SENTENCES) -> Set[int]:
    """Hash every sliding window of sentences in a document.

    Args:
        text: The document's text.
        window: Sentences per window.

    Returns:
        One hash per window; a document shorter than one window yields a single
        hash of everything it has, so short documents are still matchable.
    """
    parts = sentences(text)
    if len(parts) <= window:
        return {xxhash.xxh64_intdigest(" ".join(parts))} if parts else set()
    return {xxhash.xxh64_intdigest(" ".join(parts[i : i + window])) for i in range(len(parts) - window + 1)}


def iter_shard(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield every document in a gzipped JSONL shard.

    Args:
        path: Path to a `*.jsonl.gz` shard.

    Yields:
        Each document as a dict.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dropped_documents(sample_path: Path, stage: str, pretrain_dir: Path) -> List[Dict[str, Any]]:
    """Read the full text of the sampled documents a dedup stage dropped.

    The sample keeps only the first 2,000 characters, which is not enough to
    hash, so each document is fetched again from the shard the sampler found it
    in — a shard of the stage's input, since a dropped document never reaches
    the stage's output.

    Args:
        sample_path: The stratified sample's JSONL file.
        stage: `exact_dedup` or `sentence_dedup`.
        pretrain_dir: The curation `output_dir` holding the stage folders.

    Returns:
        One dict per dropped document, carrying `id`, `dataset` and `text`.

    Raises:
        KeyError: If *stage* is not a dedup stage.
    """
    wanted: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for line in sample_path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        for cell in row["cells"]:
            if cell["stage"] == stage and cell["decision"] == "dropped":
                wanted[(row["dataset"], cell["shard"])].add(row["id"])

    stage_dir = pretrain_dir / STAGE_INPUT[stage]
    # By id, not a list: coleslaw repeats ids with different text (issue #6), and
    # one id must contribute one document or it would be assessed twice.
    found: Dict[str, Dict[str, Any]] = {}
    for (dataset, shard), ids in sorted(wanted.items()):
        path = stage_dir / dataset / shard
        if not path.is_file():
            logger.warning("%s: no such shard, %d documents skipped", path, len(ids))
            continue
        for document in iter_shard(path):
            if document["id"] in ids:
                found.setdefault(document["id"], {"id": document["id"], "dataset": dataset, "text": document["text"]})
    logger.info("%s: recovered %d of %d dropped documents", stage, len(found), sum(len(v) for v in wanted.values()))
    return list(found.values())


def _targets(documents: Sequence[Dict[str, Any]]) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[str, int]]:
    """Index the dropped documents by the hashes a twin would share.

    Args:
        documents: Dropped documents carrying `id` and `text`.

    Returns:
        Document ids by text hash, document ids by window hash, and the number
        of windows each document has.
    """
    by_text: Dict[int, List[str]] = defaultdict(list)
    by_window: Dict[int, List[str]] = defaultdict(list)
    window_count: Dict[str, int] = {}
    for document in documents:
        by_text[text_hash(document["text"])].append(document["id"])
        hashes = window_hashes(document["text"])
        window_count[document["id"]] = len(hashes)
        for value in hashes:
            by_window[value].append(document["id"])
    return dict(by_text), dict(by_window), window_count


#: Set once per worker process, so the indexes are not pickled per shard.
_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(by_text: Dict[int, List[str]], by_window: Dict[int, List[str]], own_ids: Set[str]) -> None:
    """Hand a worker process the indexes it matches shards against.

    Args:
        by_text: Dropped ids by text hash.
        by_window: Dropped ids by window hash.
        own_ids: The dropped ids themselves, so a document never matches itself.
    """
    _WORKER_STATE.update(by_text=by_text, by_window=by_window, own_ids=own_ids)


def _scan_shard(path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Tuple[int, str]], int]:
    """Look for twins of the dropped documents in one shard.

    Args:
        path: Shard to read.

    Returns:
        Exact twins by dropped id, the kept documents sharing windows with each
        dropped id, the best window match per dropped id as (windows, kept id),
        and how many documents the shard held.
    """
    by_text = _WORKER_STATE["by_text"]
    by_window = _WORKER_STATE["by_window"]
    own_ids = _WORKER_STATE["own_ids"]
    exact: Dict[str, str] = {}
    sharers: Dict[str, Set[str]] = defaultdict(set)
    best: Dict[str, Tuple[int, str]] = {}
    seen = 0
    for document in iter_shard(path):
        seen += 1
        if document["id"] in own_ids:
            continue
        candidates = by_text.get(text_hash(document["text"]))
        if candidates:
            for dropped_id in candidates:
                exact.setdefault(dropped_id, document["id"])
        if by_window:
            hits: Dict[str, int] = defaultdict(int)
            for value in window_hashes(document["text"]):
                for dropped_id in by_window.get(value, ()):
                    hits[dropped_id] += 1
            for dropped_id, count in hits.items():
                sharers[dropped_id].add(document["id"])
                if count > best.get(dropped_id, (0, ""))[0]:
                    best[dropped_id] = (count, document["id"])
    return exact, dict(sharers), best, seen


def _shard_paths(root: Path, datasets: Optional[Sequence[str]] = None) -> List[Path]:
    """List every shard under a stage folder.

    Args:
        root: Stage folder holding one directory per dataset.
        datasets: Restrict to these datasets, or None for all of them.

    Returns:
        Shard paths, largest first, so long shards start before short ones.
    """
    paths = [
        shard
        for folder in sorted(root.iterdir())
        if folder.is_dir() and (datasets is None or folder.name in datasets)
        for shard in sorted(folder.glob("*.jsonl.gz"))
    ]
    return sorted(paths, key=lambda path: path.stat().st_size, reverse=True)


def find_twins(
    documents: Sequence[Dict[str, Any]],
    search_dir: Path,
    workers: int = 10,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Search the corpus for a twin of every dropped document.

    Args:
        documents: Dropped documents carrying `id` and `text`.
        search_dir: Stage folder to search, holding untrimmed document text.
        workers: Shards read at once.
        datasets: Restrict the search to these datasets, or None for the corpus.

    Returns:
        Per dropped id: `exact_twin` (a kept id or None), `sharers` (how many
        kept documents share a window), `best_twin` (the kept document holding
        most of its windows), `window_coverage` (that document's share) and
        `windows`.
    """
    by_text, by_window, window_count = _targets(documents)
    own_ids = {document["id"] for document in documents}
    paths = _shard_paths(search_dir, datasets)
    logger.info("searching %d shards for twins of %d documents", len(paths), len(documents))

    exact: Dict[str, str] = {}
    sharers: Dict[str, Set[str]] = defaultdict(set)
    best: Dict[str, Tuple[int, str]] = {}
    scanned = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(by_text, by_window, own_ids)
    ) as pool:
        for index, (shard_exact, shard_sharers, shard_windows, seen) in enumerate(
            pool.map(_scan_shard, paths), start=1
        ):
            scanned += seen
            for dropped_id, twin in shard_exact.items():
                exact.setdefault(dropped_id, twin)
            for dropped_id, ids in shard_sharers.items():
                sharers[dropped_id].update(ids)
            for dropped_id, found in shard_windows.items():
                if found[0] > best.get(dropped_id, (0, ""))[0]:
                    best[dropped_id] = found
            if index % 50 == 0 or index == len(paths):
                logger.info("%d of %d shards, %d documents read", index, len(paths), scanned)

    findings = {}
    for document in documents:
        doc_id = document["id"]
        matched, twin = best.get(doc_id, (0, None))
        total = window_count[doc_id]
        findings[doc_id] = {
            "exact_twin": exact.get(doc_id),
            "sharers": len(sharers.get(doc_id, ())),
            "best_twin": exact.get(doc_id) or twin,
            "windows": total,
            "window_coverage": matched / total if total else 0.0,
        }
    return findings


def _is_matched(stage: str, row: Dict[str, Any], coverage_floor: float) -> bool:
    """Decide whether one dropped document's content is still in the corpus.

    Args:
        stage: The dedup stage that dropped it.
        row: That document's twin findings.
        coverage_floor: Window share a twin must hold at `sentence_dedup`.

    Returns:
        True when a surviving twin carries the document's content.
    """
    if not row.get("twin_survives", True):
        return False
    if stage == "exact_dedup":
        return bool(row["exact_twin"])
    return row["window_coverage"] >= coverage_floor


def summarise(stage: str, twins: Dict[str, Dict[str, Any]], coverage_floor: float = 0.5) -> Dict[str, Any]:
    """Reduce per-document twin findings to the record's numbers.

    Args:
        stage: The dedup stage assessed.
        twins: Output of `find_twins`.
        coverage_floor: Share of windows a twin must hold for a
            `sentence_dedup` drop to count as matched.

    Returns:
        Counts and rates keyed for a CSV row.
    """
    total = len(twins)
    exact = sum(1 for row in twins.values() if row["exact_twin"])
    any_window = sum(1 for row in twins.values() if row["sharers"])
    covered = sum(1 for row in twins.values() if row["window_coverage"] >= coverage_floor)
    matched = sum(1 for row in twins.values() if _is_matched(stage, row, coverage_floor))
    survives = sum(1 for row in twins.values() if row.get("twin_survives"))
    coverages = sorted(row["window_coverage"] for row in twins.values())
    median = coverages[len(coverages) // 2] if coverages else 0.0
    return {
        "stage": stage,
        "dropped_assessed": total,
        "exact_twin": exact,
        "shares_any_window": any_window,
        "covered_at_floor": covered,
        "coverage_floor": coverage_floor,
        "twin_survives_to_corpus": survives,
        "twin_match_rate": round(matched / total, 4) if total else "",
        "median_window_coverage": round(median, 4),
        "unmatched": total - matched,
    }


def _collect_ids(path: Path) -> Set[str]:
    """Return the wanted ids present in one shard.

    Args:
        path: Shard to read.

    Returns:
        The ids from the worker's set that this shard holds.
    """
    wanted = _WORKER_STATE["wanted_ids"]
    return {document["id"] for document in iter_shard(path) if document["id"] in wanted}


def _init_id_worker(wanted_ids: Set[str]) -> None:
    """Hand a worker process the ids it looks for.

    Args:
        wanted_ids: Document ids to search for.
    """
    _WORKER_STATE.update(wanted_ids=wanted_ids)


def surviving_ids(ids: Set[str], final_dir: Path, workers: int = 10) -> Set[str]:
    """Report which of *ids* reached the finished corpus.

    A twin found one stage earlier may itself have been dropped by sentence
    dedup, in which case the content it stood for is gone after all — so the
    twin has to be looked for again in the final folder.

    Args:
        ids: Document ids to look for.
        final_dir: The finished corpus folder.
        workers: Shards read at once.

    Returns:
        The subset of *ids* the corpus still holds.
    """
    if not ids:
        return set()
    paths = _shard_paths(final_dir)
    logger.info("checking %d twins against %d shards of the finished corpus", len(ids), len(paths))
    present: Set[str] = set()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_id_worker, initargs=(ids,)) as pool:
        for found in pool.map(_collect_ids, paths):
            present.update(found)
    return present


def assess_dedup(
    sample_path: Path,
    pretrain_dir: Path,
    stages: Sequence[str] = DEDUP_STAGES,
    workers: int = 10,
    coverage_floor: float = 0.5,
    unmatched_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Measure both dedup stages' drops against the corpus that kept them.

    Both stages are assessed in one pass over the corpus: a document dropped by
    exact dedup never reaches sentence dedup, so the two sets of ids are
    disjoint and can share an index. A second, cheaper pass then asks whether
    each twin itself survived to the end.

    Args:
        sample_path: The stratified sample's JSONL file.
        pretrain_dir: The curation `output_dir`.
        stages: Dedup stages to assess.
        workers: Shards read at once.
        coverage_floor: Window share a twin must hold for a `sentence_dedup`
            drop to count as matched.
        unmatched_path: Where to write the drops with no twin, for judging.

    Returns:
        One summary row per stage, and the per-document findings keyed by
        document id with the stage recorded on each.
    """
    documents: Dict[str, List[Dict[str, Any]]] = {
        stage: dropped_documents(sample_path, stage, pretrain_dir) for stage in stages
    }
    every = [document for stage in stages for document in documents[stage]]
    twins = find_twins(every, pretrain_dir / TWIN_SEARCH_DIR, workers=workers)

    candidates = {row["best_twin"] for row in twins.values() if row["best_twin"]}
    survivors = surviving_ids(candidates, pretrain_dir / FINAL_DIR, workers=workers)
    for row in twins.values():
        row["twin_survives"] = bool(row["best_twin"] and row["best_twin"] in survivors)

    rows: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    for stage in stages:
        stage_twins = {document["id"]: twins[document["id"]] for document in documents[stage]}
        for row in stage_twins.values():
            row["stage"] = stage
        rows.append(summarise(stage, stage_twins, coverage_floor))
        texts = {document["id"]: document for document in documents[stage]}
        unmatched.extend(
            {**texts[doc_id], **row}
            for doc_id, row in stage_twins.items()
            if not _is_matched(stage, row, coverage_floor)
        )

    if unmatched_path is not None:
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        with unmatched_path.open("w", encoding="utf-8") as handle:
            for row in unmatched:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("wrote %d unmatched drops to %s", len(unmatched), unmatched_path)
    return rows, twins
