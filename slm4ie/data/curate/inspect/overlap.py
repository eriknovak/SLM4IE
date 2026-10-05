"""Measure how far the extracted corpora overlap, as evidence for the containment audit.

The audit in `docs/datasets.md` settles most `contains` / `overlaps` verdicts
from the publishers' documentation; the pairs it cannot settle (the
CommonCrawl family, two .si web crawls, corpora rebuilt from one source) are
measured here, read-only, over `extracted/<key>.jsonl`. Two measures:

* URL overlap, for corpora that record each document's URL (`metadata.url`,
  or `metadata.u` for HPLT): the share of one side's distinct normalised URLs
  found on the other side. Exact, over every document.
* Text overlap, for every corpus: the share of one side's sentence units
  found anywhere in the other. A unit is a sentence reduced to its lowercased
  word tokens, so the extractors' differing tokenisation and line layout do
  not matter; units under `MIN_UNIT_CHARS` are skipped as boilerplate. The
  measured side is a seeded sample of documents; on both sides only units
  whose hash falls in a seeded 1-in-`keep_one_in` slice are kept, the same
  slice everywhere, so the estimate stays unbiased at a fraction of the memory.

Nothing is tracked in MLflow: the tool runs once for the audit, its table is
written to `out_dir`, and the figures are committed as documentation.
"""

import csv
import logging
import re
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import orjson
import xxhash

logger = logging.getLogger(__name__)

#: Metadata keys a document's URL is stored under (`u` is HPLT's).
URL_KEYS = ("url", "u")

#: Query parameters that only track a visit and never change the page.
_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid)$")

#: Sentence boundary: terminal punctuation followed by space, or a line break.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n+")

#: A word token; units keep only these, lowercased.
_WORD = re.compile(r"\w+")

#: Units shorter than this (after normalisation) are skipped as boilerplate.
MIN_UNIT_CHARS = 40

#: Bytes of a corpus file one task scans, so large corpora spread over workers.
CHUNK_BYTES = 1 << 30

#: The hash arrays kept per corpus: distinct URLs, all units, sampled units.
ARRAYS: Tuple[str, ...] = ("urls", "units", "sample")

#: Columns of the table `measure_overlap` returns and writes.
COLUMNS: Tuple[str, ...] = (
    "a",
    "b",
    "docs_a",
    "sample_docs",
    "sample_units",
    "text_share",
    "urls_a",
    "shared_urls",
    "url_share",
)


def normalize_url(url: str) -> str:
    """Reduce a URL to the form two crawls of one page share.

    Drops the scheme, a leading `www.`, the fragment, tracking parameters and
    a trailing slash, lowercases the host and sorts the query.

    Args:
        url: The URL as the corpus recorded it.

    Returns:
        `host/path?query`.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING.match(k))
    path = parts.path.rstrip("/")
    return f"{host}{path}" + (f"?{urlencode(query)}" if query else "")


def sentence_units(text: str) -> List[str]:
    """Split *text* into normalised sentence units.

    Args:
        text: A document's text.

    Returns:
        Each sentence as its lowercased word tokens joined by single spaces,
        keeping those of at least `MIN_UNIT_CHARS` characters.
    """
    units = (" ".join(_WORD.findall(s.lower())) for s in _SENTENCE_BREAK.split(text))
    return [unit for unit in units if len(unit) >= MIN_UNIT_CHARS]


def _count_lines(path: Path, start: int, end: int) -> int:
    """Count the newlines in bytes `[start, end)` of *path* without parsing them."""
    count = 0
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining > 0 and (chunk := handle.read(min(1 << 24, remaining))):
            count += chunk.count(b"\n")
            remaining -= len(chunk)
    return count


def _chunks(path: Path) -> List[Tuple[int, int]]:
    """Split *path* into `CHUNK_BYTES` byte ranges covering the whole file."""
    size = path.stat().st_size
    return [(start, min(start + CHUNK_BYTES, size)) for start in range(0, max(size, 1), CHUNK_BYTES)]


def _scan(path: Path, start: int, end: int, threshold: float, seed: int, keep_one_in: int) -> Dict[str, Any]:
    """Hash the URLs and sentence units of the lines starting in bytes `[start, end)` of *path*.

    Args:
        path: The corpus's `extracted/<key>.jsonl`.
        start: First byte of the range. A line belongs to the range it starts in.
        end: Byte the range stops at; a line starting before it is read to its end.
        threshold: A document is sampled when its uid hash is below this.
        seed: Seed of the document sample and the unit slice.
        keep_one_in: Keep a unit when its hash is divisible by this.

    Returns:
        Sorted unique `urls`, `units` and `sample` hash arrays and the
        `sampled` document count.
    """
    urls, units, sample = array("Q"), array("Q"), array("Q")
    sampled = 0
    with path.open("rb") as handle:
        if start:
            # Finish the line the previous range owns: the one holding byte start - 1.
            handle.seek(start - 1)
            handle.readline()
        while handle.tell() < end:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = orjson.loads(line)
            metadata = record.get("metadata") or {}
            url = next((metadata[k] for k in URL_KEYS if isinstance(metadata.get(k), str)), None)
            if url:
                urls.append(xxhash.xxh3_64_intdigest(normalize_url(url), seed))
            in_sample = xxhash.xxh3_64_intdigest(str(record.get("uid", "")), seed) < threshold
            sampled += in_sample
            for unit in sentence_units(record.get("text") or ""):
                digest = xxhash.xxh3_64_intdigest(unit, seed)
                if digest % keep_one_in == 0:
                    units.append(digest)
                    if in_sample:
                        sample.append(digest)
    arrays = {
        name: np.unique(np.frombuffer(v, dtype=np.uint64))
        for name, v in (("urls", urls), ("units", units), ("sample", sample))
    }
    return {**arrays, "sampled": sampled}


def _share(part: int, whole: int) -> Optional[float]:
    """Return `part / whole` rounded to four places, or None for an empty whole."""
    return round(part / whole, 4) if whole else None


def measure_overlap(
    extracted_dir: Path,
    *,
    out_dir: Path,
    keys: Optional[Sequence[str]] = None,
    seed: int = 0,
    sample_docs: int = 1_000_000,
    keep_one_in: int = 16,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """Measure URL and text overlap between every ordered pair of extracted corpora.

    Args:
        extracted_dir: The extracted tier holding `<key>.jsonl` files.
        out_dir: Folder for the hash arrays (`<key>.{urls,units,sample}.npy`)
            and `overlap.csv`; created if needed.
        keys: Corpora to compare, or None for every `<key>.jsonl` present.
        seed: Seed of the document sample and of the unit slice.
        sample_docs: Documents sampled from the measured side of each pair.
        keep_one_in: Keep 1 in this many units on both sides (1 keeps all).
        workers: Byte ranges (`CHUNK_BYTES` each) scanned at once.

    Returns:
        One row per ordered pair `(a, b)` with the `COLUMNS` fields:
        `text_share` is the share of `a`'s sampled units found in `b`, and
        `url_share` the share of `a`'s distinct URLs found in `b` (None when
        either side records no URL).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(keys) if keys else sorted(p.stem for p in extracted_dir.glob("*.jsonl"))
    paths = {key: extracted_dir / f"{key}.jsonl" for key in keys}
    ranges = {key: _chunks(path) for key, path in paths.items()}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        line_counts = {
            key: [pool.submit(_count_lines, paths[key], a, b) for a, b in spans] for key, spans in ranges.items()
        }
        docs = {key: sum(f.result() for f in futures) for key, futures in line_counts.items()}
        scans = {
            key: [
                pool.submit(
                    _scan,
                    paths[key],
                    a,
                    b,
                    min(1.0, sample_docs / max(docs[key], 1)) * 2**64,
                    seed,
                    keep_one_in,
                )
                for a, b in spans
            ]
            for key, spans in ranges.items()
        }
        arrays: Dict[str, Dict[str, Any]] = {}
        for key, futures in scans.items():
            parts = [f.result() for f in futures]
            arrays[key] = {name: np.unique(np.concatenate([part[name] for part in parts])) for name in ARRAYS}
            arrays[key]["sampled"] = sum(part["sampled"] for part in parts)
            for name in ARRAYS:
                np.save(out_dir / f"{key}.{name}.npy", arrays[key][name])
            logger.info(
                "[overlap] %s: %d docs, %d sampled, %d urls, %d units",
                key,
                docs[key],
                arrays[key]["sampled"],
                len(arrays[key]["urls"]),
                len(arrays[key]["units"]),
            )
    rows: List[Dict[str, Any]] = []
    for a in keys:
        for b in keys:
            if a == b:
                continue
            mine, theirs = arrays[a], arrays[b]
            shared_units = len(np.intersect1d(mine["sample"], theirs["units"], assume_unique=True))
            has_urls = len(mine["urls"]) > 0 and len(theirs["urls"]) > 0
            shared_urls = len(np.intersect1d(mine["urls"], theirs["urls"], assume_unique=True)) if has_urls else None
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "docs_a": docs[a],
                    "sample_docs": arrays[a]["sampled"],
                    "sample_units": len(mine["sample"]),
                    "text_share": _share(shared_units, len(mine["sample"])),
                    "urls_a": len(mine["urls"]),
                    "shared_urls": shared_urls,
                    "url_share": _share(shared_urls, len(mine["urls"])) if shared_urls is not None else None,
                }
            )
    with (out_dir / "overlap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
