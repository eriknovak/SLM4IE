"""Curation pipeline that produces the final SLM4IE pretraining corpus.

Reads `<input_dir>/<key>.jsonl` extraction outputs and writes durable,
sentinel-tracked artifacts under `<output_dir>/`:

* `convert` (stage 0): per-dataset folders of datatrove `Document`-shaped
  gzipped JSONL shards, lifted straight from the extraction output.
* `language`: every document is tagged with a lingua-py language label
  and a target-language confidence score.
* `spam`: adult/SEO-spam removal via per-language lexicons, a URL/domain
  blocklist, and an optional pluggable model scorer.
* `quality`, `repetition`: per-document Gopher heuristics.
* `exact_dedup`, `sentence_dedup`: corpus-wide whole-document and
  N-sentence dedup.
* `statistics`: corpus-wide totals plus per-domain and per-dataset
  breakdowns and a global top-K word-frequency table.

Layout of the package:

* `stages/` — the registry (`stages/__init__.py`) and one module per stage
  holding everything that runs it; a stage's version is its module's hash.
* `run.py` — the run loop (`curate`); `status.py` — reporting and adoption.
* `config.py` — the curation config, overrides and config hashes;
  `lineage.py` — sentinels, currency, atomic swap, lock file;
  `paths.py` — the output tree and shard helpers; `tracking.py` — MLflow.
* `inspect/` — read-only tools over a finished corpus (leakage diagnosis,
  stratified sampling, the LM judge, dedup assessment, profiling).

This module re-exports only the stage registry, so importing it stays cheap.
It eagerly imports `importlib.metadata` and `importlib.util` so that
datatrove's lazy dependency probing (which uses
`importlib.metadata.distributions` without an explicit submodule import)
works under Python 3.13.
"""

import importlib.metadata  # noqa: F401  (eager import; see module docstring)
import importlib.util  # noqa: F401  (eager import; see module docstring)

from slm4ie.data.curate.stages import (
    ALL_STAGE_NAMES,
    CORPUS_STAGES,
    SCOPED_STAGES,
    STAGE_DIRS,
    STAGE_NAMES,
    STAGE_VERSIONS,
    cascade_from,
    config_slice_keys,
    final_corpus_dir,
    statistics_dir,
    upstream_stage,
)

__all__ = [
    "ALL_STAGE_NAMES",
    "CORPUS_STAGES",
    "SCOPED_STAGES",
    "STAGE_DIRS",
    "STAGE_NAMES",
    "STAGE_VERSIONS",
    "cascade_from",
    "config_slice_keys",
    "final_corpus_dir",
    "statistics_dir",
    "upstream_stage",
]
