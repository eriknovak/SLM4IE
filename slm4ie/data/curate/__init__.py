"""Curation pipeline that produces the final SLM4IE pretraining corpus.

Reads `<input_dir>/<key>.jsonl` extraction outputs and writes durable,
sentinel-tracked artifacts under `<output_dir>/`:

* `convert` (stage 0): per-dataset folders of datatrove `Document`-shaped
  gzipped JSONL shards. Wires the extraction step into the curate
  pipeline without going through a separate `to_datatrove` script.
* `language`: every document is tagged with a lingua-py language label
  and a target-language confidence score.
* `spam`: adult/SEO-spam removal via per-language lexicons, a URL/domain
  blocklist, and an optional pluggable model scorer.
* `quality`, `repetition`: per-document Gopher heuristics.
* `exact_dedup`, `sentence_dedup`: corpus-wide whole-document and
  N-sentence dedup (datatrove's six-block ladder).
* `stats`: corpus-wide totals plus per-domain and per-dataset
  breakdowns and a global top-K word-frequency table.

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
    upstream_stage,
)
from slm4ie.data.curate.sentinel import (
    Sentinel,
    SENTINEL_NAME,
    cascade_invalidate,
    config_hash,
    read_sentinel,
    sentinel_is_current,
    write_sentinel,
)
from slm4ie.data.curate.overrides import (
    STAGE_KNOBS,
    OverrideConfigError,
    effective_stage_config,
    validate_overrides,
)
from slm4ie.data.curate.manifest import (
    DEFAULT_SHARD_GLOBS,
    ROWS_NOT_COUNTED,
    corpus_digest,
    shard_manifest,
)

__all__ = [
    "ALL_STAGE_NAMES",
    "STAGE_DIRS",
    "STAGE_NAMES",
    "cascade_from",
    "config_slice_keys",
    "final_corpus_dir",
    "stats_dir",
    "upstream_stage",
]

__all__ += [
    "Sentinel",
    "SENTINEL_NAME",
    "cascade_invalidate",
    "config_hash",
    "read_sentinel",
    "sentinel_is_current",
    "write_sentinel",
]

__all__ += [
    "STAGE_KNOBS",
    "OverrideConfigError",
    "effective_stage_config",
    "validate_overrides",
]

__all__ += [
    "DEFAULT_SHARD_GLOBS",
    "ROWS_NOT_COUNTED",
    "corpus_digest",
    "shard_manifest",
]
