"""Curation pipeline that produces the final SLM4IE pretraining corpus.

Reads `<output_dir>/datatrove/<key>.jsonl.gz` and writes one consolidated
output: `<output_dir>/final/` containing the deduplicated training shards
and a `statistics/` subfolder. Three concerns are wired into a single
five-executor datatrove pipeline:

* `language`: every document is tagged with a lingua-py language label
  and a target-language confidence score.
* `dedup`: cross-dataset whole-document and 3-sentence exact dedup
  (datatrove's six-block ladder, fused with the surrounding lang and
  stats steps so only five executors are spawned).
* `stats`: corpus-wide totals plus per-domain and per-dataset
  breakdowns, top-K word and n-gram tables, and classla-lemmatized
  TF-IDF keywords.

The package eagerly imports `importlib.metadata` and `importlib.util`
at module load so that `datatrove`'s lazy dependency probing
(which uses `importlib.metadata.distributions` without an explicit
submodule import) works under Python 3.13.
"""

import importlib.metadata  # noqa: F401  (eager import; see module docstring)
import importlib.util  # noqa: F401  (eager import; see module docstring)

from slm4ie.data.curate.stages import (
    ALL_STAGE_NAMES,
    STAGE_DIRS,
    STAGE_NAMES,
    cascade_from,
    config_slice_keys,
    final_corpus_dir,
    stats_dir,
)

__all__ = [
    "ALL_STAGE_NAMES",
    "STAGE_DIRS",
    "STAGE_NAMES",
    "cascade_from",
    "config_slice_keys",
    "final_corpus_dir",
    "stats_dir",
]
