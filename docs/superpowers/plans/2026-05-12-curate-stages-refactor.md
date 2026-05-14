# Curate stages refactor implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fused six-executor `build_curate_executors(...)` with six user-facing, sentinel-skippable pipeline stages, each writing a durable on-disk artifact under `<output_dir>/`. Stage execution is resumable and cascade-invalidates downstream stages on config change.

**Architecture:** Library code in `slm4ie/data/curate/` exposes (a) a constants table mapping stage names to folder names and config slices, (b) a sentinel module for read/write/hash with cascade invalidation, and (c) one builder function per stage that returns the executor chain needed for that stage. The CLI in `scripts/data/curate.py` drives the stages: it loads `curate.yaml`, slices config per stage, checks each sentinel, runs the stages whose hash differs (cascading downstream), and writes a JSON sentinel on success.

**Tech Stack:** Python 3.13, datatrove (`LocalPipelineExecutor`, `ExactDedup*`, `SentDedup*`, `GopherQualityFilter`, `GopherRepetitionFilter`, `JsonlReader`, `JsonlWriter`), uv, pytest, ruff (Google docstrings via pydocstyle).

---

## Stage taxonomy

| CLI name | Folder | Description |
|---|---|---|
| `language` | `01_language/` | lingua-py language detection (per-doc tag or filter). |
| `quality` | `02_quality/` | Gopher within-document quality heuristics. |
| `repetition` | `03_repetition/` | Gopher within-document repetition heuristics. |
| `exact_dedup` | `04_1_dedup/` | Whole-document exact dedup across the corpus. |
| `sentence_dedup` | `04_2_dedup/` | 3-sentence sliding-window dedup across the corpus. Final corpus. |
| `stats` | `05_statistics/` | Corpus statistics (`aggregate.json`, `per_dataset/<key>.json`). |
| `all` | n/a | Run all six in order, skipping any whose sentinel hash matches current config. |

Each stage writes:
- `<output_dir>/<folder>/<dataset>/<rank>.jsonl.gz` (data shards) — except `05_statistics/` which writes JSON files.
- `<output_dir>/<folder>/.complete` (sentinel JSON, see Task 2).

Dedup scratch (`exact_sigs/`, `exact_dups/`, `sent_sigs/`, `sent_dups/`) lives under `<output_dir>/_dedup_state/` and is purged when the respective dedup stage's sentinel lands.

Logs live under `<output_dir>/_logs/<stage>/`.

---

## YAML schema (locked in)

```yaml
input_dir: /vault/data/SLM4IE/processed/datatrove
output_dir: /vault/data/SLM4IE/final

stopwords: configs/data/stopwords_sl.txt

language:
  targets: [sl]
  candidates: [sl, hr, sr, bs, mk, en, de, it, hu]
  minimum_relative_distance: 0.1
  mode: filter
  low_accuracy: true
  max_chars: 2048

quality:
  min_doc_words: 50
  max_doc_words: 100000
  min_avg_word_length: 3
  max_avg_word_length: 10
  max_symbol_word_ratio: 0.1
  max_bullet_lines_ratio: 0.9
  max_ellipsis_lines_ratio: 0.3
  max_non_alpha_words_ratio: 0.8
  min_stop_words: 2

repetition: {}

exact_dedup:
  precision: 64
  hash_fc: xxhash
  only_dedup_in_index: true

sentence_dedup:
  n_sentences: 3
  min_doc_words: 50
  min_num_sentences: 2
  split_sentences: true

stats:
  top_k_words: 5000
  top_k_ngrams: 5000
  keyword_top_k: 200
  ngram_orders: [2, 3]
  compute_keywords: true
```

Sentinel hash slice per stage:
- `01_language` ← `language:` dict.
- `02_quality` ← `quality:` dict + **contents** of the file at `stopwords:`.
- `03_repetition` ← `repetition:` dict (empty today).
- `04_1_dedup` ← `exact_dedup:` dict.
- `04_2_dedup` ← `sentence_dedup:` dict.
- `05_statistics` ← `stats:` dict + **contents** of the file at `stopwords:`.

Runtime-only flags (`--max-workers`, `--force`, `--stage`, `--input-dir`, `--output-dir`, paths to YAMLs) are **excluded** from every hash.

---

## CLI surface (locked in)

```
uv run python scripts/data/curate.py --all                     # run all stages, skip completed
uv run python scripts/data/curate.py --all --stage quality     # run only the quality stage
uv run python scripts/data/curate.py --all --stage all         # explicit; same as no --stage flag
uv run python scripts/data/curate.py kzb solar                 # subset (still all stages)
uv run python scripts/data/curate.py --all --force             # nuke entire <output_dir>
uv run python scripts/data/curate.py --all --force --stage quality  # nuke quality + downstream
uv run python scripts/data/curate.py --all --max-workers 8     # 8 parallel workers
uv run python scripts/data/curate.py --all --max-workers 0     # cpu_default
```

Removed: `--debug`, `--debug-dir`, `--no-keywords` (use `stats.compute_keywords: false` in YAML).

`--max-workers` semantics:
- Default = `1` (serial).
- `0` = `cpu_default(len(stages))` (i.e. `cpu_count // 2`).
- N ≥ 1 = N.

---

## File structure (what changes)

Files to **create**:
- `slm4ie/data/curate/stages.py` — stage name constants, folder mapping, config slice helpers.
- `slm4ie/data/curate/sentinel.py` — sentinel I/O + per-stage config hashing + cascade invalidation.
- `tests/data/test_curate_stages.py` — tests for the stages module.
- `tests/data/test_curate_sentinel.py` — tests for sentinel logic.

Files to **modify**:
- `slm4ie/data/curate/__init__.py` — re-export the public API.
- `slm4ie/data/curate/dedup.py` — accept `HashConfig`/`only_dedup_in_index` knobs.
- `slm4ie/data/curate/pipeline.py` — replace `build_curate_executors` with six per-stage builders.
- `scripts/data/curate.py` — new `--stage` CLI, stage-driven runner.
- `configs/data/curate.yaml` — new flat schema (see above).
- `tests/data/test_curate_dedup.py` — extend for the new `make_exact_config(...)` knobs.
- `tests/data/test_curate_pipeline.py` — replace `build_curate_executors` tests with per-stage tests; replace symlink-CLI tests with `--stage` tests; update the smoke test to drive the new builders.
- `README.md` — update the curate section if it references the old layout (`final/<dataset>/...`).

Files to **leave alone**:
- `slm4ie/data/curate/language.py`, `stats.py` — the `LinguaLanguageFilter` and `CorpusStats` classes are unchanged; only how they're wired changes.
- `tests/data/test_curate_language.py`, `test_curate_stats.py` — class behavior unchanged.

---

## Task 1: Stage constants module

**Files:**
- Create: `slm4ie/data/curate/stages.py`
- Create: `tests/data/test_curate_stages.py`
- Modify: `slm4ie/data/curate/__init__.py` (re-export `STAGE_NAMES`, `STAGE_DIRS`, `final_corpus_dir`, `stats_dir`, etc.)

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_curate_stages.py
"""Tests for the stage name/folder mapping in slm4ie.data.curate.stages."""

from slm4ie.data.curate.stages import (
    ALL_STAGE_NAMES,
    STAGE_DIRS,
    STAGE_NAMES,
    cascade_from,
    config_slice_keys,
    final_corpus_dir,
    stats_dir,
)


def test_stage_names_are_in_pipeline_order() -> None:
    """STAGE_NAMES lists the six stages in execution order."""
    assert STAGE_NAMES == (
        "language",
        "quality",
        "repetition",
        "exact_dedup",
        "sentence_dedup",
        "stats",
    )


def test_all_stage_names_includes_sentinel() -> None:
    """ALL_STAGE_NAMES is STAGE_NAMES plus the 'all' sentinel."""
    assert ALL_STAGE_NAMES == STAGE_NAMES + ("all",)


def test_stage_dirs_use_numeric_prefix() -> None:
    """Each stage maps to its numbered folder name."""
    assert STAGE_DIRS == {
        "language": "01_language",
        "quality": "02_quality",
        "repetition": "03_repetition",
        "exact_dedup": "04_1_dedup",
        "sentence_dedup": "04_2_dedup",
        "stats": "05_statistics",
    }


def test_final_corpus_dir_is_sentence_dedup() -> None:
    """The final pretraining corpus lives under 04_2_dedup/."""
    assert final_corpus_dir() == "04_2_dedup"


def test_stats_dir_matches_mapping() -> None:
    """Stats lives under 05_statistics/."""
    assert stats_dir() == "05_statistics"


def test_config_slice_keys_per_stage() -> None:
    """Each stage advertises the top-level YAML key(s) that govern it."""
    assert config_slice_keys("language") == ("language",)
    assert config_slice_keys("quality") == ("quality",)
    assert config_slice_keys("repetition") == ("repetition",)
    assert config_slice_keys("exact_dedup") == ("exact_dedup",)
    assert config_slice_keys("sentence_dedup") == ("sentence_dedup",)
    assert config_slice_keys("stats") == ("stats",)


def test_cascade_from_returns_stage_and_successors() -> None:
    """cascade_from yields the stage and every downstream stage in order."""
    assert cascade_from("language") == STAGE_NAMES
    assert cascade_from("quality") == STAGE_NAMES[1:]
    assert cascade_from("exact_dedup") == STAGE_NAMES[3:]
    assert cascade_from("stats") == ("stats",)


def test_cascade_from_rejects_unknown_stage() -> None:
    """An unknown stage name raises KeyError."""
    import pytest

    with pytest.raises(KeyError):
        cascade_from("not_a_stage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_curate_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: slm4ie.data.curate.stages`.

- [ ] **Step 3: Write the stages module**

```python
# slm4ie/data/curate/stages.py
"""Stage names, folder mapping, and config-slice helpers for the curate pipeline.

The curation pipeline runs six sequential stages, each producing a
durable on-disk artifact under `<output_dir>/<folder>/`. This module is
the single source of truth that ties together the user-facing CLI name
of each stage (`--stage <name>`), the folder it writes to, and the
top-level `curate.yaml` key(s) whose contents determine its sentinel
hash. Consumers should import from here rather than hard-coding stage
names or folder paths.
"""

from typing import Tuple


#: Stage names in pipeline execution order.
STAGE_NAMES: Tuple[str, ...] = (
    "language",
    "quality",
    "repetition",
    "exact_dedup",
    "sentence_dedup",
    "stats",
)


#: Stage names plus the `"all"` sentinel that the CLI uses for the default
#: "run everything that needs running" mode.
ALL_STAGE_NAMES: Tuple[str, ...] = STAGE_NAMES + ("all",)


#: Mapping from stage name to the folder under `<output_dir>/` it writes.
STAGE_DIRS = {
    "language": "01_language",
    "quality": "02_quality",
    "repetition": "03_repetition",
    "exact_dedup": "04_1_dedup",
    "sentence_dedup": "04_2_dedup",
    "stats": "05_statistics",
}


#: Per-stage top-level YAML keys that go into the sentinel config hash.
_CONFIG_SLICE_KEYS = {
    "language": ("language",),
    "quality": ("quality",),
    "repetition": ("repetition",),
    "exact_dedup": ("exact_dedup",),
    "sentence_dedup": ("sentence_dedup",),
    "stats": ("stats",),
}


def final_corpus_dir() -> str:
    """Return the folder name (under `<output_dir>/`) holding the final corpus.

    Returns:
        The folder name of the final, fully-deduplicated training
        corpus (the output of the `sentence_dedup` stage).
    """
    return STAGE_DIRS["sentence_dedup"]


def stats_dir() -> str:
    """Return the folder name (under `<output_dir>/`) holding statistics output.

    Returns:
        The folder name of the corpus statistics output.
    """
    return STAGE_DIRS["stats"]


def config_slice_keys(stage: str) -> Tuple[str, ...]:
    """Return the top-level YAML keys whose contents drive *stage*'s sentinel hash.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        Tuple of `curate.yaml` top-level keys whose values are included
        in the stage's config hash slice. Stopword *file contents* are
        also included for `quality` and `stats`, but that's handled by
        the sentinel module — those keys live outside `curate.yaml`.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    return _CONFIG_SLICE_KEYS[stage]


def cascade_from(stage: str) -> Tuple[str, ...]:
    """Return *stage* followed by every downstream stage, in execution order.

    Used by the sentinel runner to cascade-invalidate downstream stages
    when *stage*'s config has changed.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        Tuple starting with *stage* and ending with the last pipeline
        stage in execution order.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    if stage not in STAGE_NAMES:
        raise KeyError(stage)
    idx = STAGE_NAMES.index(stage)
    return STAGE_NAMES[idx:]
```

- [ ] **Step 4: Re-export from `__init__.py`**

```python
# slm4ie/data/curate/__init__.py — extend the existing file by appending:
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
```

(Keep the existing `importlib.metadata` / `importlib.util` imports and module docstring intact.)

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/data/test_curate_stages.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/stages.py slm4ie/data/curate/__init__.py tests/data/test_curate_stages.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add slm4ie/data/curate/stages.py slm4ie/data/curate/__init__.py tests/data/test_curate_stages.py
git commit -m "$(cat <<'EOF'
feat(curate): add stages module for pipeline taxonomy

Establishes the single source of truth for the six curate stages:
their CLI names, their `<output_dir>/<folder>/` mapping, and the
top-level `curate.yaml` keys each stage's sentinel hashes.
EOF
)"
```

---

## Task 2: Sentinel module (read/write/hash/cascade)

**Files:**
- Create: `slm4ie/data/curate/sentinel.py`
- Create: `tests/data/test_curate_sentinel.py`
- Modify: `slm4ie/data/curate/__init__.py` (extend `__all__`).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_curate_sentinel.py
"""Tests for the per-stage sentinel I/O + config-hash module."""

import json
from pathlib import Path

import pytest

from slm4ie.data.curate.sentinel import (
    Sentinel,
    cascade_invalidate,
    config_hash,
    read_sentinel,
    sentinel_is_current,
    write_sentinel,
)


def test_config_hash_is_deterministic() -> None:
    """The hash is stable across calls with identical input."""
    cfg = {"min_doc_words": 50, "max_doc_words": 100000}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_differs_on_value_change() -> None:
    """Changing any value changes the hash."""
    a = {"min_doc_words": 50}
    b = {"min_doc_words": 100}
    assert config_hash(a) != config_hash(b)


def test_config_hash_ignores_key_order() -> None:
    """Two dicts with the same keys/values in different insertion order hash equal."""
    a = {"a": 1, "b": 2}
    b = {"b": 2, "a": 1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_includes_extra_payload(tmp_path: Path) -> None:
    """Optional extra payload (e.g. stopword file contents) affects the hash."""
    a = config_hash({"min_doc_words": 50})
    b = config_hash({"min_doc_words": 50}, extra=b"different bytes")
    assert a != b


def test_write_then_read_sentinel_roundtrip(tmp_path: Path) -> None:
    """Writing and reading a sentinel returns the same structured data."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={"min_doc_words": 50},
        config_hash_value="sha256:abc",
        records_in=100,
        records_out=80,
    )
    sentinel = read_sentinel(folder)
    assert sentinel is not None
    assert sentinel.config_hash == "sha256:abc"
    assert sentinel.config_slice == {"min_doc_words": 50}
    assert sentinel.records_in == 100
    assert sentinel.records_out == 80
    assert sentinel.completed_at  # ISO timestamp present


def test_read_sentinel_returns_none_when_missing(tmp_path: Path) -> None:
    """Missing sentinel returns None instead of raising."""
    assert read_sentinel(tmp_path / "01_language") is None


def test_sentinel_is_current_matches_hash(tmp_path: Path) -> None:
    """sentinel_is_current returns True iff the recorded hash equals the new hash."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={"min_doc_words": 50},
        config_hash_value="sha256:abc",
        records_in=1,
        records_out=1,
    )
    assert sentinel_is_current(folder, "sha256:abc") is True
    assert sentinel_is_current(folder, "sha256:different") is False


def test_sentinel_is_current_false_when_missing(tmp_path: Path) -> None:
    """A missing sentinel is never current."""
    assert sentinel_is_current(tmp_path / "missing", "sha256:abc") is False


def test_cascade_invalidate_removes_sentinels(tmp_path: Path) -> None:
    """cascade_invalidate removes sentinels for stage + every downstream stage."""
    for name in ("02_quality", "03_repetition", "04_1_dedup", "04_2_dedup", "05_statistics"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / ".complete").write_text("{}")
    (tmp_path / "01_language").mkdir()
    (tmp_path / "01_language" / ".complete").write_text("{}")

    removed = cascade_invalidate(tmp_path, "quality")
    assert "quality" in removed
    assert "stats" in removed
    # language was before quality — must NOT be invalidated.
    assert (tmp_path / "01_language" / ".complete").exists()
    # quality + downstream sentinels gone.
    for name in ("02_quality", "03_repetition", "04_1_dedup", "04_2_dedup", "05_statistics"):
        assert not (tmp_path / name / ".complete").exists()


def test_cascade_invalidate_handles_missing_sentinels_silently(tmp_path: Path) -> None:
    """It's fine to invalidate when sentinels don't exist; result lists requested stages."""
    removed = cascade_invalidate(tmp_path, "exact_dedup")
    assert removed == ("exact_dedup", "sentence_dedup", "stats")


def test_sentinel_filename_is_complete(tmp_path: Path) -> None:
    """The sentinel file is named .complete, matching to_datatrove.py convention."""
    folder = tmp_path / "02_quality"
    folder.mkdir()
    write_sentinel(
        folder,
        config_slice={},
        config_hash_value="sha256:x",
        records_in=0,
        records_out=0,
    )
    assert (folder / ".complete").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_curate_sentinel.py -v`
Expected: FAIL with `ModuleNotFoundError: slm4ie.data.curate.sentinel`.

- [ ] **Step 3: Write the sentinel module**

```python
# slm4ie/data/curate/sentinel.py
"""Per-stage sentinel files for the curate pipeline.

Each stage writes a JSON `.complete` sentinel into its output folder
once the underlying datatrove executors finish cleanly. The sentinel
records the config slice that drove the run, a SHA-256 hash of that
slice (so future runs can detect config drift and cascade-invalidate
downstream stages), the completion timestamp, and the input/output
record counts.

Conventions:

* Per-stage sentinel: each `<output_dir>/<stage_dir>/.complete` is
  independent. A stage is "current" if and only if its sentinel exists
  AND its recorded hash matches the freshly computed slice hash.
* Cascade invalidation: when a stage is determined stale, every
  downstream stage's sentinel is removed as well (their inputs are now
  potentially different).
* Runtime-only inputs (CLI flags, worker counts, output paths) MUST NOT
  enter the hash; only output-affecting config does.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from slm4ie.data.curate.stages import STAGE_DIRS, cascade_from


#: Sentinel filename, matching the `to_datatrove.py` convention.
SENTINEL_NAME = ".complete"


@dataclass(frozen=True)
class Sentinel:
    """Parsed contents of a stage's `.complete` sentinel file.

    Attributes:
        completed_at: ISO-8601 UTC timestamp the stage finished.
        config_hash: SHA-256 hex digest of the stage's config slice.
        config_slice: The raw config slice the hash was computed from
            (useful for human inspection).
        records_in: Number of records read into the stage.
        records_out: Number of records written out (i.e. surviving).
    """

    completed_at: str
    config_hash: str
    config_slice: Dict[str, Any]
    records_in: int
    records_out: int


def config_hash(slice_: Dict[str, Any], extra: Optional[bytes] = None) -> str:
    """Compute a stable hash for a config slice.

    The slice is serialized as canonical JSON (sorted keys, no
    whitespace) before hashing. Optional `extra` bytes are appended to
    the hash input — used for hashing the *contents* of files
    referenced by the config (e.g. the stopword file), which are
    output-affecting but not part of the slice itself.

    Args:
        slice_: A JSON-serializable dict of output-affecting config.
        extra: Optional extra bytes to fold into the hash (e.g. the
            stopword file contents).

    Returns:
        Lower-case hex digest prefixed with `"sha256:"`.
    """
    h = hashlib.sha256()
    h.update(json.dumps(slice_, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if extra is not None:
        h.update(b"\x00")
        h.update(extra)
    return "sha256:" + h.hexdigest()


def write_sentinel(
    stage_folder: Path,
    *,
    config_slice: Dict[str, Any],
    config_hash_value: str,
    records_in: int,
    records_out: int,
) -> Path:
    """Write the sentinel JSON file for *stage_folder*.

    Args:
        stage_folder: The stage's output folder (e.g. `<output_dir>/02_quality`).
        config_slice: The config slice the run consumed.
        config_hash_value: Pre-computed hash of *config_slice* (plus
            any extra payload like stopword file contents).
        records_in: Records read.
        records_out: Records written.

    Returns:
        Path to the written sentinel file.
    """
    stage_folder.mkdir(parents=True, exist_ok=True)
    sentinel_path = stage_folder / SENTINEL_NAME
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash_value,
        "config_slice": config_slice,
        "records_in": records_in,
        "records_out": records_out,
    }
    sentinel_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sentinel_path


def read_sentinel(stage_folder: Path) -> Optional[Sentinel]:
    """Read the sentinel JSON from *stage_folder*, or return None.

    Args:
        stage_folder: The stage's output folder.

    Returns:
        Parsed `Sentinel` instance, or `None` if the sentinel file is
        missing or malformed.
    """
    sentinel_path = stage_folder / SENTINEL_NAME
    if not sentinel_path.exists():
        return None
    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Sentinel(
        completed_at=str(data.get("completed_at", "")),
        config_hash=str(data.get("config_hash", "")),
        config_slice=dict(data.get("config_slice") or {}),
        records_in=int(data.get("records_in", 0)),
        records_out=int(data.get("records_out", 0)),
    )


def sentinel_is_current(stage_folder: Path, expected_hash: str) -> bool:
    """Return True iff *stage_folder*'s sentinel exists and matches *expected_hash*.

    Args:
        stage_folder: The stage's output folder.
        expected_hash: The hash recomputed from current config.

    Returns:
        True if the recorded hash matches; False otherwise (including
        when the sentinel is missing).
    """
    sentinel = read_sentinel(stage_folder)
    return sentinel is not None and sentinel.config_hash == expected_hash


def cascade_invalidate(output_dir: Path, stage: str) -> Tuple[str, ...]:
    """Remove the sentinel of *stage* and every downstream stage.

    Args:
        output_dir: The curation output root (which contains the
            `01_language/`, `02_quality/`, ... folders).
        stage: First stage to invalidate. `cascade_from(stage)` is used
            to compute the downstream set.

    Returns:
        Tuple of stage names whose sentinels were targeted (regardless
        of whether the sentinel file actually existed beforehand).
    """
    affected = cascade_from(stage)
    for name in affected:
        sentinel_path = output_dir / STAGE_DIRS[name] / SENTINEL_NAME
        if sentinel_path.exists():
            sentinel_path.unlink()
    return affected
```

- [ ] **Step 4: Extend `__init__.py` exports**

```python
# Append to slm4ie/data/curate/__init__.py
from slm4ie.data.curate.sentinel import (
    Sentinel,
    SENTINEL_NAME,
    cascade_invalidate,
    config_hash,
    read_sentinel,
    sentinel_is_current,
    write_sentinel,
)

# Extend the existing __all__ list with these names (don't replace it):
__all__ += [
    "Sentinel",
    "SENTINEL_NAME",
    "cascade_invalidate",
    "config_hash",
    "read_sentinel",
    "sentinel_is_current",
    "write_sentinel",
]
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/data/test_curate_sentinel.py -v`
Expected: PASS (11 tests).

- [ ] **Step 6: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/sentinel.py slm4ie/data/curate/__init__.py tests/data/test_curate_sentinel.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add slm4ie/data/curate/sentinel.py slm4ie/data/curate/__init__.py tests/data/test_curate_sentinel.py
git commit -m "$(cat <<'EOF'
feat(curate): add per-stage sentinel module with config-hash invalidation

Sentinels record a SHA-256 hash of the stage's config slice, plus
optional extra bytes (e.g. stopword file contents). On rerun, a
stage whose recorded hash diverges from the current slice is
invalidated together with every downstream stage.
EOF
)"
```

---

## Task 3: Dedup config knobs

**Files:**
- Modify: `slm4ie/data/curate/dedup.py`
- Modify: `tests/data/test_curate_dedup.py`

- [ ] **Step 1: Extend the dedup tests**

Replace the existing class `TestExactDedupHelpers` body in `tests/data/test_curate_dedup.py` with:

```python
from datatrove.utils.hashing import HashConfig
from slm4ie.data.curate.dedup import default_exact_config, doc_text, make_exact_config


class TestExactDedupHelpers:
    """Helpers used by the exact-dedup signature stage."""

    def test_doc_text_returns_text_payload(self) -> None:
        """`doc_text` extracts the text body for hashing."""
        d = Document(text="hello", id="1", metadata={})
        assert doc_text(d) == "hello"

    def test_default_exact_config_uses_doc_text(self) -> None:
        """The default ExactDedupConfig hashes `doc.text`, not metadata."""
        cfg = default_exact_config()
        assert cfg.content_getter is doc_text

    def test_make_exact_config_defaults_match_default_helper(self) -> None:
        """make_exact_config with no args returns the same shape as default_exact_config."""
        cfg = make_exact_config()
        default = default_exact_config()
        assert cfg.content_getter is doc_text
        assert cfg.hash_config.precision == default.hash_config.precision
        assert cfg.hash_config.hash_fc == default.hash_config.hash_fc
        assert cfg.only_dedup_in_index == default.only_dedup_in_index

    def test_make_exact_config_overrides_precision(self) -> None:
        """make_exact_config threads the precision arg into HashConfig."""
        cfg = make_exact_config(precision=32)
        assert cfg.hash_config.precision == 32

    def test_make_exact_config_overrides_hash_fc(self) -> None:
        """make_exact_config threads hash_fc into HashConfig."""
        cfg = make_exact_config(hash_fc="sha1")
        assert cfg.hash_config.hash_fc == "sha1"

    def test_make_exact_config_overrides_only_dedup_in_index(self) -> None:
        """make_exact_config threads the only_dedup_in_index flag."""
        cfg = make_exact_config(only_dedup_in_index=False)
        assert cfg.only_dedup_in_index is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_curate_dedup.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_exact_config'`.

- [ ] **Step 3: Add `make_exact_config` to `slm4ie/data/curate/dedup.py`**

Append this function to the existing module (keep `doc_text`, `default_exact_config` intact):

```python
from typing import Literal

from datatrove.utils.hashing import HashConfig


def make_exact_config(
    *,
    precision: Literal[32, 64] = 64,
    hash_fc: Literal["sha1", "xxhash"] = "xxhash",
    only_dedup_in_index: bool = True,
) -> ExactDedupConfig:
    """Build an `ExactDedupConfig` parameterized for the curate pipeline.

    Wraps datatrove's `ExactDedupConfig` so the CLI can pass through the
    output-affecting knobs declared under `exact_dedup:` in `curate.yaml`.
    The content getter is always `doc_text` — exact dedup operates on
    document text, never on metadata.

    Args:
        precision: Hash width in bits. Choose `32` only for very small
            corpora (collision risk grows past ~10M docs); `64` is the
            collision-safe default up to ~10B docs.
        hash_fc: Hash function. `"xxhash"` is faster; `"sha1"` is
            cryptographically strong but unnecessary for dedup.
        only_dedup_in_index: When True, only deduplicate within the
            current run's index (datatrove default). Set to False when
            extending an existing dedup index across runs.

    Returns:
        A configured `ExactDedupConfig` with `doc_text` as the content
        getter.
    """
    return ExactDedupConfig(
        content_getter=doc_text,
        hash_config=HashConfig(precision=precision, hash_fc=hash_fc),
        only_dedup_in_index=only_dedup_in_index,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/data/test_curate_dedup.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/dedup.py tests/data/test_curate_dedup.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/dedup.py tests/data/test_curate_dedup.py
git commit -m "$(cat <<'EOF'
feat(curate): expose ExactDedupConfig knobs via make_exact_config

precision, hash_fc and only_dedup_in_index are output-affecting and
therefore need to live in curate.yaml so the per-stage sentinel hash
covers them. default_exact_config() stays for backwards-compatible
test fixtures.
EOF
)"
```

---

## Task 4: Pipeline refactor — per-stage builders

**Files:**
- Modify: `slm4ie/data/curate/pipeline.py` (replace `build_curate_executors` with six stage builders)
- Modify: `tests/data/test_curate_pipeline.py` (replace structural tests; smoke test rewritten in Task 9)

This is the biggest single change. It rewrites `pipeline.py` from scratch. The replacement keeps `CuratePaths` and `QualityConfig` (extending `CuratePaths`).

- [ ] **Step 1: Write the failing tests**

Replace the top half of `tests/data/test_curate_pipeline.py` (everything from line 1 through the line ending in `def test_lang_minimum_relative_distance_is_threaded`, i.e. up to and including the `TestBuildCurateExecutors` class — keep the smoke test marker `@pytest.mark.slow` section for now, that's rewritten in Task 9) with:

```python
"""Tests for the per-stage curate pipeline builders.

Structural assertions only; the heavy end-to-end smoke test lives at
the bottom of this file (marked `@pytest.mark.slow`).
"""

import importlib.metadata  # noqa: F401  (datatrove workaround)
import importlib.util  # noqa: F401  (datatrove workaround)
from pathlib import Path

import pytest

pytest.importorskip("datatrove")

from datatrove.pipeline.dedup import (  # noqa: E402
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    SentDedupConfig,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.filters import (  # noqa: E402
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.pipeline.writers.jsonl import JsonlWriter  # noqa: E402
from datatrove.utils.typeshelper import Languages  # noqa: E402

from slm4ie.data.curate.language import LinguaLanguageFilter  # noqa: E402
from slm4ie.data.curate.pipeline import (  # noqa: E402
    CuratePaths,
    QualityConfig,
    build_exact_dedup_executors,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_sentence_dedup_executors,
    build_stats_executors,
)
from slm4ie.data.curate.stats import CorpusStats  # noqa: E402


def _paths(tmp_path: Path) -> CuratePaths:
    """Build a CuratePaths anchored under *tmp_path* for structural tests."""
    return CuratePaths(
        input_folder=tmp_path / "datatrove",
        output_dir=tmp_path / "curated",
    )


class TestLanguageStage:
    """The language stage is a single parallel executor."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        """The language stage runs as a single executor."""
        execs = build_language_executors(_paths(tmp_path))
        assert len(execs) == 1
        assert execs[0].depends is None

    def test_pipeline_contains_lingua_and_writer(self, tmp_path: Path) -> None:
        """The pipeline reads input, applies lingua, writes to 01_language/."""
        execs = build_language_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert JsonlReader in types_
        assert LinguaLanguageFilter in types_
        assert JsonlWriter in types_

    def test_writes_to_language_folder(self, tmp_path: Path) -> None:
        """The writer's output_folder is `<output_dir>/01_language`."""
        paths = _paths(tmp_path)
        execs = build_language_executors(paths)
        writer = next(s for s in execs[0].pipeline if isinstance(s, JsonlWriter))
        assert str(paths.stage_dir("language")) in writer.output_folder

    def test_lang_minimum_relative_distance_is_threaded(self, tmp_path: Path) -> None:
        """`minimum_relative_distance` reaches the LinguaLanguageFilter."""
        execs = build_language_executors(
            _paths(tmp_path), lang_minimum_relative_distance=0.15
        )
        lang = next(s for s in execs[0].pipeline if isinstance(s, LinguaLanguageFilter))
        assert lang.minimum_relative_distance == 0.15


class TestQualityStage:
    """The quality stage reads 01_language/ and writes 02_quality/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        execs = build_quality_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_gopher_quality(self, tmp_path: Path) -> None:
        execs = build_quality_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert GopherQualityFilter in types_
        assert GopherRepetitionFilter not in types_  # repetition is its own stage

    def test_quality_config_threaded(self, tmp_path: Path) -> None:
        cfg = QualityConfig(min_doc_words=10, max_doc_words=200, min_stop_words=0)
        execs = build_quality_executors(_paths(tmp_path), quality_config=cfg)
        quality = next(s for s in execs[0].pipeline if isinstance(s, GopherQualityFilter))
        assert quality.min_doc_words == 10
        assert quality.max_doc_words == 200
        assert quality.min_stop_words == 0

    def test_stopwords_become_gopher_stop_words(self, tmp_path: Path) -> None:
        execs = build_quality_executors(_paths(tmp_path), stopwords={"in", "je", "na"})
        quality = next(s for s in execs[0].pipeline if isinstance(s, GopherQualityFilter))
        assert {"in", "je", "na"}.issubset(quality.stop_words)


class TestRepetitionStage:
    """The repetition stage reads 02_quality/ and writes 03_repetition/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        execs = build_repetition_executors(_paths(tmp_path))
        assert len(execs) == 1

    def test_pipeline_contains_repetition_filter(self, tmp_path: Path) -> None:
        execs = build_repetition_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert GopherRepetitionFilter in types_
        # No quality filter here — quality is its own preceding stage.
        assert GopherQualityFilter not in types_


class TestExactDedupStage:
    """Exact dedup is three internal executors: sig → find → filter+write."""

    def test_returns_three_executors_chained(self, tmp_path: Path) -> None:
        execs = build_exact_dedup_executors(_paths(tmp_path))
        assert len(execs) == 3
        assert execs[0].depends is None
        assert execs[1].depends is execs[0]
        assert execs[2].depends is execs[1]

    def test_executor_blocks(self, tmp_path: Path) -> None:
        execs = build_exact_dedup_executors(_paths(tmp_path))
        types_ = [[type(s) for s in ex.pipeline] for ex in execs]
        assert ExactDedupSignature in types_[0]
        assert ExactFindDedups in types_[1]
        assert ExactDedupFilter in types_[2]
        assert JsonlWriter in types_[2]
        # No sentence-dedup blocks bleed in here.
        assert SentenceDedupSignature not in types_[0] + types_[1] + types_[2]
        assert SentenceDedupFilter not in types_[0] + types_[1] + types_[2]

    def test_finder_workers_propagates(self, tmp_path: Path) -> None:
        execs = build_exact_dedup_executors(_paths(tmp_path), finder_workers=4)
        sig = next(s for s in execs[0].pipeline if isinstance(s, ExactDedupSignature))
        assert sig.finder_workers == 4
        assert execs[1].tasks == 4


class TestSentenceDedupStage:
    """Sentence dedup is three internal executors: sig → find → filter+write."""

    def test_returns_three_executors_chained(self, tmp_path: Path) -> None:
        execs = build_sentence_dedup_executors(_paths(tmp_path))
        assert len(execs) == 3
        assert execs[0].depends is None
        assert execs[1].depends is execs[0]
        assert execs[2].depends is execs[1]

    def test_executor_blocks(self, tmp_path: Path) -> None:
        execs = build_sentence_dedup_executors(_paths(tmp_path))
        types_ = [[type(s) for s in ex.pipeline] for ex in execs]
        assert SentenceDedupSignature in types_[0]
        assert SentenceFindDedups in types_[1]
        assert SentenceDedupFilter in types_[2]
        assert JsonlWriter in types_[2]
        assert ExactDedupSignature not in types_[0] + types_[1] + types_[2]
        assert ExactDedupFilter not in types_[0] + types_[1] + types_[2]

    def test_sentence_blocks_run_in_slovenian(self, tmp_path: Path) -> None:
        execs = build_sentence_dedup_executors(_paths(tmp_path))
        sent_sig = next(s for s in execs[0].pipeline if isinstance(s, SentenceDedupSignature))
        sent_filter = next(s for s in execs[2].pipeline if isinstance(s, SentenceDedupFilter))
        assert sent_sig.language == Languages.slovenian
        assert sent_filter.language == Languages.slovenian

    def test_sentence_config_threaded(self, tmp_path: Path) -> None:
        cfg = SentDedupConfig(
            n_sentences=4, min_doc_words=10, min_num_sentences=1, split_sentences=True
        )
        execs = build_sentence_dedup_executors(_paths(tmp_path), sentence_config=cfg)
        sig = next(s for s in execs[0].pipeline if isinstance(s, SentenceDedupSignature))
        assert sig.config.n_sentences == 4


class TestStatsStage:
    """The stats stage runs single-process and reads 04_2_dedup/."""

    def test_returns_one_executor(self, tmp_path: Path) -> None:
        execs = build_stats_executors(_paths(tmp_path))
        assert len(execs) == 1
        assert execs[0].tasks == 1
        assert execs[0].workers == 1

    def test_pipeline_contains_corpus_stats(self, tmp_path: Path) -> None:
        execs = build_stats_executors(_paths(tmp_path))
        types_ = [type(s) for s in execs[0].pipeline]
        assert JsonlReader in types_
        assert CorpusStats in types_

    def test_stats_reads_from_final_corpus_folder(self, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        execs = build_stats_executors(paths)
        reader = next(s for s in execs[0].pipeline if isinstance(s, JsonlReader))
        assert str(paths.stage_dir("sentence_dedup")) in reader.data_folder.path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_curate_pipeline.py -v -k "not slow"`
Expected: FAIL (functions don't exist yet).

- [ ] **Step 3: Rewrite `slm4ie/data/curate/pipeline.py`**

Replace the file's contents entirely with:

```python
"""Per-stage executor builders for the curate pipeline.

Each builder function returns the `LocalPipelineExecutor`(s) needed to
run one user-facing stage. Builders are independent — a caller can run
just the language stage, just the quality stage, etc. Cross-stage
ordering (and downstream invalidation when one stage's output is stale)
is owned by the CLI runner, not by the builders.

I/O layout — every reader walks `<input_folder>/<dataset>/<part>.jsonl.gz`
recursively, and every writer emits `<output_folder>/<dataset>/<rank>.jsonl.gz`,
matching the upstream `to_datatrove.py` per-dataset shard layout. This
preserves dataset provenance through every stage.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    ExactDedupConfig,
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    SentDedupConfig,
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.filters import GopherQualityFilter, GopherRepetitionFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.typeshelper import Languages

from slm4ie.data.curate.dedup import default_exact_config
from slm4ie.data.curate.language import LinguaLanguageFilter
from slm4ie.data.curate.stages import STAGE_DIRS
from slm4ie.data.curate.stats import CorpusStats

logger = logging.getLogger(__name__)


@dataclass
class CuratePaths:
    """Filesystem locations for one curation run.

    Attributes:
        input_folder: Upstream `<key>/<part>.jsonl.gz` shards
            (typically the output of `to_datatrove.py`).
        output_dir: Curation output root. Stage folders
            (`01_language/`, `02_quality/`, ...) live directly under
            this path, alongside `_dedup_state/` and `_logs/`.
    """

    input_folder: Path
    output_dir: Path

    def stage_dir(self, stage: str) -> Path:
        """Return the folder under `output_dir` that holds *stage*'s output.

        Args:
            stage: Stage name (see `slm4ie.data.curate.stages.STAGE_NAMES`).

        Returns:
            Absolute path to the stage's output folder.

        Raises:
            KeyError: If *stage* is not a known stage name.
        """
        return self.output_dir / STAGE_DIRS[stage]

    @property
    def dedup_state_dir(self) -> Path:
        """Folder holding the dedup sig/dup scratch between dedup sub-stages."""
        return self.output_dir / "_dedup_state"

    def logs_dir(self, stage: str) -> Path:
        """Per-stage logging directory under `<output_dir>/_logs/<stage>/`."""
        return self.output_dir / "_logs" / stage


@dataclass
class QualityConfig:
    """Knobs for the Gopher quality heuristic filter.

    Mirrors `GopherQualityFilter.__init__` defaults so the CLI can
    override individual values from `curate.yaml` without listing
    every parameter.

    Attributes:
        min_doc_words: Minimum word count; shorter docs are dropped.
        max_doc_words: Maximum word count; longer docs are dropped.
        min_avg_word_length: Minimum average word length in chars.
        max_avg_word_length: Maximum average word length in chars.
        max_symbol_word_ratio: Max fraction of word-like tokens that
            are pure symbols.
        max_bullet_lines_ratio: Max fraction of lines starting with a
            bullet glyph.
        max_ellipsis_lines_ratio: Max fraction of lines ending in an
            ellipsis.
        max_non_alpha_words_ratio: *Minimum* fraction of words that
            must contain at least one alphabetic character (datatrove
            keeps the legacy Gopher name).
        min_stop_words: Minimum number of stopword tokens that must
            appear in the document.
    """

    min_doc_words: int = 50
    max_doc_words: int = 100_000
    min_avg_word_length: int = 3
    max_avg_word_length: int = 10
    max_symbol_word_ratio: float = 0.1
    max_bullet_lines_ratio: float = 0.9
    max_ellipsis_lines_ratio: float = 0.3
    max_non_alpha_words_ratio: float = 0.8
    min_stop_words: int = 2


def _writer(stage_folder: Path) -> JsonlWriter:
    """Return a JsonlWriter that emits `<stage_folder>/<dataset>/<rank>.jsonl.gz`."""
    stage_folder.mkdir(parents=True, exist_ok=True)
    return JsonlWriter(
        output_folder=str(stage_folder),
        output_filename="${dataset}/${rank}.jsonl.gz",
    )


def _reader(folder: Path) -> JsonlReader:
    """Return a JsonlReader that walks `<folder>/**/*.jsonl.gz` recursively."""
    return JsonlReader(
        str(folder),
        glob_pattern="**/*.jsonl.gz",
        shuffle_files=False,
        recursive=True,
    )


def build_language_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    target_languages: Sequence[str] = ("sl",),
    candidate_languages: Optional[List[str]] = None,
    lang_mode: str = "filter",
    lang_minimum_relative_distance: float = 0.0,
    lang_low_accuracy: bool = False,
    lang_max_chars: Optional[int] = None,
) -> List[LocalPipelineExecutor]:
    """Build the language stage: read input → lingua filter → write 01_language/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count for this stage.
        target_languages: ISO 639-1 codes considered "in-language".
        candidate_languages: ISO 639-1 candidate set for lingua.
        lang_mode: `"tag"` keeps every doc; `"filter"` drops
            out-of-target docs.
        lang_minimum_relative_distance: Required confidence gap before
            lingua commits. `0.0` disables.
        lang_low_accuracy: Use lingua's trigram-only model.
        lang_max_chars: Truncate doc text to this many chars before
            detection. `None` disables truncation.

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    out = paths.stage_dir("language")
    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(paths.input_folder),
            LinguaLanguageFilter(
                targets=list(target_languages),
                candidates=candidate_languages,
                mode=lang_mode,
                minimum_relative_distance=lang_minimum_relative_distance,
                low_accuracy=lang_low_accuracy,
                max_chars=lang_max_chars,
            ),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("language")),
        skip_completed=False,
    )
    return [executor]


def build_quality_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    quality_config: Optional[QualityConfig] = None,
    language: str = Languages.slovenian,
    stopwords: Optional[Set[str]] = None,
) -> List[LocalPipelineExecutor]:
    """Build the quality stage: read 01_language/ → Gopher quality → write 02_quality/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        quality_config: `GopherQualityFilter` knob bundle; defaults to
            Gopher paper values.
        language: ISO-3 code for the word/sentence tokenizer.
        stopwords: Stopword set used by `GopherQualityFilter`.

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    cfg = quality_config or QualityConfig()
    in_ = paths.stage_dir("language")
    out = paths.stage_dir("quality")
    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            GopherQualityFilter(
                language=language,
                stop_words=sorted(stopwords) if stopwords else None,
                min_doc_words=cfg.min_doc_words,
                max_doc_words=cfg.max_doc_words,
                min_avg_word_length=cfg.min_avg_word_length,
                max_avg_word_length=cfg.max_avg_word_length,
                max_symbol_word_ratio=cfg.max_symbol_word_ratio,
                max_bullet_lines_ratio=cfg.max_bullet_lines_ratio,
                max_ellipsis_lines_ratio=cfg.max_ellipsis_lines_ratio,
                max_non_alpha_words_ratio=cfg.max_non_alpha_words_ratio,
                min_stop_words=cfg.min_stop_words,
            ),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("quality")),
        skip_completed=False,
    )
    return [executor]


def build_repetition_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    language: str = Languages.slovenian,
) -> List[LocalPipelineExecutor]:
    """Build the repetition stage: read 02_quality/ → Gopher repetition → write 03_repetition/.

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count.
        language: ISO-3 code for the word/sentence tokenizer the
            repetition filter uses.

    Returns:
        A list with one `LocalPipelineExecutor`.
    """
    in_ = paths.stage_dir("quality")
    out = paths.stage_dir("repetition")
    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            GopherRepetitionFilter(language=language),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("repetition")),
        skip_completed=False,
    )
    return [executor]


def build_exact_dedup_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    finder_workers: int = 1,
    exact_config: Optional[ExactDedupConfig] = None,
) -> List[LocalPipelineExecutor]:
    """Build the exact-dedup stage: sig → find → filter+write 04_1_dedup/.

    Three executors chained via `depends`:
        1. (parallel) read 03_repetition/ → ExactDedupSignature → exact_sigs/
        2. (single)   ExactFindDedups(exact_sigs/) → exact_dups/
        3. (parallel) read 03_repetition/ → ExactDedupFilter → write 04_1_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count for executors 1 and 3.
        finder_workers: Worker count for the single-worker find
            executor 2 (and the `finder_workers` argument of the sig
            executor 1).
        exact_config: Optional `ExactDedupConfig`; defaults to one whose
            `content_getter` hashes `doc.text`.

    Returns:
        Three chained `LocalPipelineExecutor`s.
    """
    cfg = exact_config or default_exact_config()
    in_ = paths.stage_dir("repetition")
    out = paths.stage_dir("exact_dedup")
    sigs = paths.dedup_state_dir / "exact_sigs"
    dups = paths.dedup_state_dir / "exact_dups"

    sig = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            ExactDedupSignature(
                output_folder=str(sigs), config=cfg, finder_workers=finder_workers
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("exact_dedup") / "1_sig"),
        skip_completed=False,
    )
    find = LocalPipelineExecutor(
        pipeline=[ExactFindDedups(data_folder=str(sigs), output_folder=str(dups), config=cfg)],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(paths.logs_dir("exact_dedup") / "2_find"),
        depends=sig,
        skip_completed=False,
    )
    filt = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            ExactDedupFilter(data_folder=str(dups), config=cfg),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("exact_dedup") / "3_filter"),
        depends=find,
        skip_completed=False,
    )
    return [sig, find, filt]


def build_sentence_dedup_executors(
    paths: CuratePaths,
    *,
    tasks: int = 1,
    finder_workers: int = 1,
    sentence_config: Optional[SentDedupConfig] = None,
    language: str = Languages.slovenian,
) -> List[LocalPipelineExecutor]:
    """Build the sentence-dedup stage: sig → find → filter+write 04_2_dedup/.

    Three executors chained via `depends`, mirroring the exact stage:
        1. (parallel) read 04_1_dedup/ → SentenceDedupSignature → sent_sigs/
        2. (single)   SentenceFindDedups(sent_sigs/) → sent_dups/
        3. (parallel) read 04_1_dedup/ → SentenceDedupFilter → write 04_2_dedup/

    Args:
        paths: Resolved input/output locations.
        tasks: Parallel worker count for executors 1 and 3.
        finder_workers: Worker count for the find executor.
        sentence_config: Optional `SentDedupConfig`.
        language: ISO-3 code for the sentence tokenizer.

    Returns:
        Three chained `LocalPipelineExecutor`s.
    """
    cfg = sentence_config or SentDedupConfig()
    in_ = paths.stage_dir("exact_dedup")
    out = paths.stage_dir("sentence_dedup")
    sigs = paths.dedup_state_dir / "sent_sigs"
    dups = paths.dedup_state_dir / "sent_dups"

    sig = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            SentenceDedupSignature(
                output_folder=str(sigs),
                config=cfg,
                finder_workers=finder_workers,
                language=language,
            ),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "1_sig"),
        skip_completed=False,
    )
    find = LocalPipelineExecutor(
        pipeline=[SentenceFindDedups(data_folder=str(sigs), output_folder=str(dups), config=cfg)],
        tasks=finder_workers,
        workers=finder_workers,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "2_find"),
        depends=sig,
        skip_completed=False,
    )
    filt = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            SentenceDedupFilter(data_folder=str(dups), config=cfg, language=language),
            _writer(out),
        ],
        tasks=tasks,
        workers=tasks,
        logging_dir=str(paths.logs_dir("sentence_dedup") / "3_filter"),
        depends=find,
        skip_completed=False,
    )
    return [sig, find, filt]


def build_stats_executors(
    paths: CuratePaths,
    *,
    language: str = Languages.slovenian,
    stopwords: Optional[Set[str]] = None,
    top_k_words: int = 5_000,
    top_k_ngrams: int = 5_000,
    keyword_top_k: int = 200,
    compute_keywords: bool = True,
    ngram_orders: Sequence[int] = (2, 3),
) -> List[LocalPipelineExecutor]:
    """Build the stats stage: read 04_2_dedup/ → CorpusStats → 05_statistics/.

    Single-process by design: `CorpusStats` keeps every counter on the
    instance, so worker fan-out is not supported.

    Args:
        paths: Resolved input/output locations.
        language: ISO-3 code for the tokenizer.
        stopwords: Stopword set used by `CorpusStats`.
        top_k_words: Word-frequency table size.
        top_k_ngrams: Per-order n-gram table size.
        keyword_top_k: TF-IDF keywords per bucket.
        compute_keywords: Disable to skip the classla pass.
        ngram_orders: N-gram orders to compute.

    Returns:
        A list with one single-process `LocalPipelineExecutor`.
    """
    in_ = paths.stage_dir("sentence_dedup")
    out = paths.stage_dir("stats")
    out.mkdir(parents=True, exist_ok=True)

    executor = LocalPipelineExecutor(
        pipeline=[
            _reader(in_),
            CorpusStats(
                output_path=out / "aggregate.json",
                per_dataset_dir=out / "per_dataset",
                language=language,
                stopwords=stopwords or set(),
                top_k_words=top_k_words,
                top_k_ngrams=top_k_ngrams,
                keyword_top_k=keyword_top_k,
                compute_keywords=compute_keywords,
                ngram_orders=ngram_orders,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(paths.logs_dir("stats")),
        skip_completed=False,
    )
    return [executor]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/data/test_curate_pipeline.py -v -k "not slow"`
Expected: PASS for all `TestLanguageStage`, `TestQualityStage`, `TestRepetitionStage`, `TestExactDedupStage`, `TestSentenceDedupStage`, `TestStatsStage` (~20 tests). The `test_final_corpus_drops_cross_dataset_duplicates` smoke test will fail — that's expected; it's rewritten in Task 9.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/pipeline.py tests/data/test_curate_pipeline.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/pipeline.py tests/data/test_curate_pipeline.py
git commit -m "$(cat <<'EOF'
feat(curate): split build_curate_executors into per-stage builders

Each builder owns one user-facing stage and returns the executor
chain it needs. CuratePaths now carries output_dir + stage_dir()
helpers; the dedup sig/find scratch is rooted at _dedup_state/.
Quality and repetition are separate stages.
EOF
)"
```

---

## Task 5: Update `configs/data/curate.yaml`

**Files:**
- Modify: `configs/data/curate.yaml`

- [ ] **Step 1: Replace the YAML content with the new schema**

Replace the entire file with:

```yaml
input_dir: /vault/data/SLM4IE/processed/datatrove
output_dir: /vault/data/SLM4IE/final

# Stopword file used by both the quality stage (Gopher's min_stop_words)
# and the stats stage (word/n-gram/keyword filtering).
stopwords: configs/data/stopwords_sl.txt

# 01_language: lingua-py language detection. In `mode: filter`, only
# docs whose detected top-1 language is in `targets` survive.
language:
  targets: [sl]
  candidates: [sl, hr, sr, bs, mk, en, de, it, hu]
  minimum_relative_distance: 0.1
  mode: filter
  low_accuracy: true
  max_chars: 2048

# 02_quality: Gopher WITHIN-DOCUMENT quality heuristics. Drops docs
# based on length, word-length, symbol/bullet/ellipsis ratios,
# alphabetic content, and stopword floor. Never compares across docs.
quality:
  min_doc_words: 50
  max_doc_words: 100000
  min_avg_word_length: 3
  max_avg_word_length: 10
  max_symbol_word_ratio: 0.1
  max_bullet_lines_ratio: 0.9
  max_ellipsis_lines_ratio: 0.3
  max_non_alpha_words_ratio: 0.8
  min_stop_words: 2

# 03_repetition: Gopher WITHIN-DOCUMENT repetition heuristics. Drops
# docs whose own text repeats too much internally (duplicate
# paragraphs/lines, top-n-gram saturation, dup-n-gram fractions).
# datatrove's defaults from the Gopher paper are used.
repetition: {}

# 04_1_dedup: whole-document exact dedup ACROSS THE CORPUS. Two docs
# with byte-identical text collapse to one. `precision` and `hash_fc`
# rarely change — 64-bit xxhash is collision-safe up to ~10B docs.
exact_dedup:
  precision: 64
  hash_fc: xxhash
  only_dedup_in_index: true

# 04_2_dedup: N-sentence sliding-window dedup ACROSS THE CORPUS.
# Removes windows that recur across docs; drops docs whose surviving
# text falls below the floors.
sentence_dedup:
  n_sentences: 3
  min_doc_words: 50
  min_num_sentences: 2
  split_sentences: true

# 05_statistics: word/n-gram tables and (optional) classla TF-IDF
# keywords. Single-process by design.
stats:
  top_k_words: 5000
  top_k_ngrams: 5000
  keyword_top_k: 200
  ngram_orders: [2, 3]
  compute_keywords: true
```

- [ ] **Step 2: Commit**

```bash
git add configs/data/curate.yaml
git commit -m "$(cat <<'EOF'
feat(curate): flatten curate.yaml to per-stage top-level sections

Replaces nested dedup.sentence with separate exact_dedup and
sentence_dedup sections. Promotes stopwords to a top-level key
shared by quality and stats. exact_dedup now exposes precision,
hash_fc and only_dedup_in_index so the stage's sentinel hash binds
to output-affecting knobs.
EOF
)"
```

---

## Task 6: Rewrite `scripts/data/curate.py` CLI

**Files:**
- Modify: `scripts/data/curate.py` (full rewrite)

This task replaces the existing CLI end-to-end. The new CLI loads `curate.yaml`, iterates the requested stages, checks each sentinel, runs the stage if its hash is stale, writes a fresh sentinel on success, and cascades-invalidates downstream stages whenever a stage actually executes.

- [ ] **Step 1: Replace `scripts/data/curate.py` with the new implementation**

```python
"""Build the final SLM4IE pretraining corpus stage-by-stage.

`scripts/data/curate.py` runs the six-stage curation pipeline:

    1. language       -> <output_dir>/01_language/
    2. quality        -> <output_dir>/02_quality/
    3. repetition     -> <output_dir>/03_repetition/
    4. exact_dedup    -> <output_dir>/04_1_dedup/
    5. sentence_dedup -> <output_dir>/04_2_dedup/   (final corpus)
    6. stats          -> <output_dir>/05_statistics/

Each stage writes a `.complete` sentinel into its output folder. On
rerun, a stage's sentinel is compared against a fresh hash of its
config slice; on mismatch the stage and every downstream stage are
invalidated and re-executed.

Examples:
    # Run all stages, skipping any whose config slice hash is unchanged.
    uv run python scripts/data/curate.py --all

    # Run only the quality stage. Re-run downstream stats etc. on next --all.
    uv run python scripts/data/curate.py --all --stage quality

    # Force-rebuild quality + downstream (drops their sentinels).
    uv run python scripts/data/curate.py --all --force --stage quality

    # Subset of datasets — dedup operates within the given subset only.
    uv run python scripts/data/curate.py kzb solar

    # Saturate the box.
    uv run python scripts/data/curate.py --all --max-workers 0
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from datatrove.pipeline.dedup import SentDedupConfig

from slm4ie.data.curate import (
    ALL_STAGE_NAMES,
    STAGE_DIRS,
    STAGE_NAMES,
    cascade_invalidate,
    config_hash,
    sentinel_is_current,
    write_sentinel,
)
from slm4ie.data.curate.dedup import make_exact_config
from slm4ie.data.curate.pipeline import (
    CuratePaths,
    QualityConfig,
    build_exact_dedup_executors,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_sentence_dedup_executors,
    build_stats_executors,
)
from slm4ie.data.io_utils import find_project_root as _find_project_root
from slm4ie.data.parallel import cpu_default, resolve_workers

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to `sys.argv`).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the SLM4IE pretraining corpus stage-by-stage. Each "
            "stage writes a durable artifact under <output_dir>/ and a "
            ".complete sentinel; on rerun, stale stages auto-invalidate."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("datasets", nargs="*", default=[], help="Dataset keys.")
    target.add_argument("--all", action="store_true", help="Process every dataset.")
    parser.add_argument(
        "--stage",
        choices=ALL_STAGE_NAMES,
        default="all",
        help="Stage to run. Default: all (skips finished stages).",
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Override curate.yaml::input_dir.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override curate.yaml::output_dir.")
    parser.add_argument("--curate-config", type=Path, default=None,
                        help="Path to curate.yaml (default: configs/data/curate.yaml).")
    parser.add_argument("--extract-config", type=Path, default=None,
                        help="Path to extract.yaml (default: configs/data/extract.yaml).")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force re-run. With --stage X: drop X's sentinel and all "
            "downstream sentinels. Without --stage: nuke <output_dir>."
        ),
    )
    parser.add_argument(
        "--max-workers", "--tasks",
        dest="workers", type=int, default=1,
        help="Parallel workers. 1=serial (default), 0=cpu_count//2, N=N.",
    )
    args = parser.parse_args(argv)
    if args.all and args.datasets:
        parser.error("argument --all: not allowed with positional datasets")
    if not args.all and not args.datasets:
        parser.error("one of the arguments datasets --all is required")
    return args


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file, returning {} when missing."""
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _list_datasets(extract_config: Path) -> List[str]:
    """Return dataset keys declared in extract.yaml."""
    cfg = _load_yaml(extract_config)
    return list((cfg.get("datasets") or {}).keys())


def _resolve_dirs(
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Resolve input/output dirs from CLI flags or curate.yaml."""
    raw_input = args.input_dir if args.input_dir is not None else cfg.get("input_dir")
    raw_output = args.output_dir if args.output_dir is not None else cfg.get("output_dir")
    if raw_input is None or raw_output is None:
        raise FileNotFoundError(
            "Curation paths not set. Provide --input-dir/--output-dir or set "
            "curate.yaml::input_dir / output_dir."
        )
    return Path(raw_input), Path(raw_output)


def _load_stopwords(project_root: Path, cfg: Dict[str, Any]) -> Tuple[Set[str], bytes]:
    """Load the stopword set and return (set, raw_bytes_for_hashing).

    Args:
        project_root: Project root for resolving relative stopword paths.
        cfg: Parsed curate.yaml.

    Returns:
        Tuple of `(stopword set, raw file bytes)`. When `stopwords:` is
        not configured or the file is missing, returns `(set(), b"")`.
    """
    rel = cfg.get("stopwords")
    if not rel:
        return set(), b""
    path = project_root / rel
    if not path.exists():
        logger.warning("stopwords file %s not found; using empty set.", path)
        return set(), b""
    raw = path.read_bytes()
    out: Set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.lower())
    return out, raw


def _filter_input_keys(input_dir: Path, keys: List[str]) -> Path:
    """Materialize a tempdir of symlinks restricted to *keys*.

    Args:
        input_dir: Folder of `<key>/<NNNNN>.jsonl.gz` shards.
        keys: Dataset keys to keep.

    Returns:
        Path to a tempdir mirroring the requested keys via symlinks.

    Raises:
        FileNotFoundError: If any requested shard folder is missing or empty.
    """
    missing: List[str] = []
    for key in keys:
        src = input_dir / key
        if not src.is_dir() or not any(src.glob("*.jsonl.gz")):
            missing.append(key)
    if missing:
        raise FileNotFoundError(
            f"No datatrove shard folder(s) under {input_dir} for dataset(s): "
            + ", ".join(repr(k) for k in missing)
        )
    holder = Path(tempfile.mkdtemp(prefix="slm4ie-curate-subset-"))
    for key in keys:
        src = input_dir / key
        holder_key = holder / key
        holder_key.mkdir()
        for shard in src.glob("*.jsonl.gz"):
            (holder_key / shard.name).symlink_to(shard.resolve())
    return holder


def _count_records(folder: Path) -> int:
    """Cheap record-count helper: sum of decompressed JSONL line counts.

    Args:
        folder: A stage output folder containing `<dataset>/<rank>.jsonl.gz`.

    Returns:
        Total surviving record count across every shard. Returns 0 if
        the folder doesn't exist.
    """
    if not folder.is_dir():
        return 0
    import gzip

    total = 0
    for shard in folder.glob("**/*.jsonl.gz"):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            total += sum(1 for _ in fh)
    return total


def _stage_input_dir(paths: CuratePaths, stage: str) -> Path:
    """Return the folder a stage reads from."""
    upstream = {
        "language": paths.input_folder,
        "quality": paths.stage_dir("language"),
        "repetition": paths.stage_dir("quality"),
        "exact_dedup": paths.stage_dir("repetition"),
        "sentence_dedup": paths.stage_dir("exact_dedup"),
        "stats": paths.stage_dir("sentence_dedup"),
    }
    return upstream[stage]


def _purge_dedup_state(paths: CuratePaths, which: str) -> None:
    """Purge the dedup scratch for *which* stage (`exact_dedup` or `sentence_dedup`)."""
    prefix = {"exact_dedup": "exact", "sentence_dedup": "sent"}[which]
    for sub in (paths.dedup_state_dir / f"{prefix}_sigs", paths.dedup_state_dir / f"{prefix}_dups"):
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)


def _stage_runner(
    stage: str,
    paths: CuratePaths,
    cfg: Dict[str, Any],
    workers: int,
    stopwords: Set[str],
) -> Callable[[], None]:
    """Return a zero-arg callable that runs *stage*'s executors."""
    if stage == "language":
        lang_cfg = cfg.get("language") or {}
        def run() -> None:
            execs = build_language_executors(
                paths,
                tasks=workers,
                target_languages=lang_cfg.get("targets") or ["sl"],
                candidate_languages=lang_cfg.get("candidates"),
                lang_mode=str(lang_cfg.get("mode", "filter")),
                lang_minimum_relative_distance=float(
                    lang_cfg.get("minimum_relative_distance", 0.0)
                ),
                lang_low_accuracy=bool(lang_cfg.get("low_accuracy", False)),
                lang_max_chars=lang_cfg.get("max_chars"),
            )
            execs[-1].run()
        return run

    if stage == "quality":
        qcfg = cfg.get("quality") or {}
        quality_config = QualityConfig(
            min_doc_words=int(qcfg.get("min_doc_words", 50)),
            max_doc_words=int(qcfg.get("max_doc_words", 100_000)),
            min_avg_word_length=int(qcfg.get("min_avg_word_length", 3)),
            max_avg_word_length=int(qcfg.get("max_avg_word_length", 10)),
            max_symbol_word_ratio=float(qcfg.get("max_symbol_word_ratio", 0.1)),
            max_bullet_lines_ratio=float(qcfg.get("max_bullet_lines_ratio", 0.9)),
            max_ellipsis_lines_ratio=float(qcfg.get("max_ellipsis_lines_ratio", 0.3)),
            max_non_alpha_words_ratio=float(qcfg.get("max_non_alpha_words_ratio", 0.8)),
            min_stop_words=int(qcfg.get("min_stop_words", 2)),
        )

        def run() -> None:
            execs = build_quality_executors(
                paths,
                tasks=workers,
                quality_config=quality_config,
                stopwords=stopwords,
            )
            execs[-1].run()

        return run

    if stage == "repetition":
        def run() -> None:
            execs = build_repetition_executors(paths, tasks=workers)
            execs[-1].run()

        return run

    if stage == "exact_dedup":
        edcfg = cfg.get("exact_dedup") or {}
        exact_cfg = make_exact_config(
            precision=int(edcfg.get("precision", 64)),
            hash_fc=str(edcfg.get("hash_fc", "xxhash")),
            only_dedup_in_index=bool(edcfg.get("only_dedup_in_index", True)),
        )

        def run() -> None:
            try:
                execs = build_exact_dedup_executors(
                    paths,
                    tasks=workers,
                    exact_config=exact_cfg,
                )
                execs[-1].run()
            finally:
                _purge_dedup_state(paths, "exact_dedup")

        return run

    if stage == "sentence_dedup":
        scfg = cfg.get("sentence_dedup") or {}
        sent_cfg = SentDedupConfig(
            n_sentences=int(scfg.get("n_sentences", 3)),
            min_doc_words=int(scfg.get("min_doc_words", 50)),
            min_num_sentences=int(scfg.get("min_num_sentences", 2)),
            split_sentences=bool(scfg.get("split_sentences", True)),
        )

        def run() -> None:
            try:
                execs = build_sentence_dedup_executors(
                    paths,
                    tasks=workers,
                    sentence_config=sent_cfg,
                )
                execs[-1].run()
            finally:
                _purge_dedup_state(paths, "sentence_dedup")

        return run

    if stage == "stats":
        stcfg = cfg.get("stats") or {}

        def run() -> None:
            execs = build_stats_executors(
                paths,
                stopwords=stopwords,
                top_k_words=int(stcfg.get("top_k_words", 5_000)),
                top_k_ngrams=int(stcfg.get("top_k_ngrams", 5_000)),
                keyword_top_k=int(stcfg.get("keyword_top_k", 200)),
                compute_keywords=bool(stcfg.get("compute_keywords", True)),
                ngram_orders=stcfg.get("ngram_orders") or (2, 3),
            )
            execs[-1].run()

        return run

    raise ValueError(f"Unknown stage: {stage}")


def _stage_slice(stage: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the config slice that drives *stage*'s sentinel hash."""
    return dict(cfg.get(stage) or {})


def _stage_extra(stage: str, stopwords_bytes: bytes) -> bytes:
    """Return extra bytes folded into the hash for stages that consume side files."""
    if stage in ("quality", "stats"):
        return stopwords_bytes
    return b""


def main() -> None:
    """Entry point for the curate CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    workers = resolve_workers(args.workers, len(STAGE_NAMES), cpu_default(len(STAGE_NAMES)))

    project_root = _find_project_root()
    curate_path = args.curate_config or (project_root / "configs" / "data" / "curate.yaml")
    extract_path = args.extract_config or (project_root / "configs" / "data" / "extract.yaml")
    cfg = _load_yaml(curate_path)
    input_dir, output_dir = _resolve_dirs(args, cfg)
    stopwords, stopwords_raw = _load_stopwords(project_root, cfg)

    # Subset resolution: validate that requested keys exist on disk.
    subset_holder: Optional[Path] = None
    if args.all:
        all_keys = _list_datasets(extract_path)
        logger.info("Running on all %d datasets (workers=%d)", len(all_keys), workers)
        input_folder = input_dir
    else:
        logger.info("Running on %d dataset(s): %s (workers=%d)",
                    len(args.datasets), ", ".join(args.datasets), workers)
        subset_holder = _filter_input_keys(input_dir, args.datasets)
        input_folder = subset_holder

    paths = CuratePaths(input_folder=input_folder, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --force without --stage = nuke output_dir entirely.
    if args.force and args.stage == "all":
        if output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            logger.warning("--force: cleared %s", output_dir)

    # --force with --stage X = drop X's sentinel + downstream.
    if args.force and args.stage != "all":
        removed = cascade_invalidate(output_dir, args.stage)
        logger.warning("--force --stage %s: invalidated %s", args.stage, removed)

    requested_stages = STAGE_NAMES if args.stage == "all" else (args.stage,)

    try:
        cascaded = False  # once True, every subsequent stage MUST run regardless of hash.
        for stage in requested_stages:
            slice_ = _stage_slice(stage, cfg)
            extra = _stage_extra(stage, stopwords_raw)
            current_hash = config_hash(slice_, extra=extra)
            stage_folder = paths.stage_dir(stage)

            if not cascaded and sentinel_is_current(stage_folder, current_hash):
                logger.info("[%s] sentinel current; skipping.", stage)
                continue

            if not cascaded:
                removed = cascade_invalidate(output_dir, stage)
                if any((output_dir / STAGE_DIRS[r] / ".complete").exists() for r in removed):
                    logger.warning("[%s] cascade-invalidating %s", stage, removed)
                cascaded = True

            records_in_before = _count_records(_stage_input_dir(paths, stage))
            logger.info("[%s] starting (input records ~%d)", stage, records_in_before)
            runner = _stage_runner(stage, paths, cfg, workers, stopwords)
            runner()
            records_out = _count_records(stage_folder) if stage != "stats" else 0
            write_sentinel(
                stage_folder,
                config_slice=slice_,
                config_hash_value=current_hash,
                records_in=records_in_before,
                records_out=records_out,
            )
            logger.info("[%s] done (records_in=%d, records_out=%d)",
                        stage, records_in_before, records_out)
    finally:
        if subset_holder is not None:
            shutil.rmtree(subset_holder, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
```

- [ ] **Step 2: Sanity-check parse_args**

Run: `uv run python -c "from scripts.data.curate import parse_args; print(parse_args(['--all']))"`
Expected: `Namespace(datasets=[], all=True, stage='all', input_dir=None, output_dir=None, curate_config=None, extract_config=None, force=False, workers=1)`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check --select D scripts/data/curate.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/data/curate.py
git commit -m "$(cat <<'EOF'
feat(curate): rewrite CLI to drive six sentinel-skippable stages

--stage selects one of language/quality/repetition/exact_dedup/
sentence_dedup/stats (default all). Each stage's .complete sentinel
records a SHA-256 hash of its YAML slice plus stopword bytes; on
rerun, a stale hash invalidates that stage and cascades downstream.
Removes --debug, --debug-dir, --no-keywords (move to YAML),
_stats_only, _scratch_root and _suppress_writer_rank_warning.
EOF
)"
```

---

## Task 7: CLI tests for the new --stage flag

**Files:**
- Modify: `tests/data/test_curate_pipeline.py` (replace the `TestCurateCLISelection` class)

- [ ] **Step 1: Replace the `TestCurateCLISelection` class with --stage tests**

Find the existing `TestCurateCLISelection` class at the bottom of `tests/data/test_curate_pipeline.py` and replace it with:

```python
from scripts.data import curate as curate_cli  # noqa: E402


class TestCurateCLISelection:
    """parse_args contract for the new --stage CLI."""

    def test_parse_args_accepts_single_key(self) -> None:
        args = curate_cli.parse_args(["kzb"])
        assert args.datasets == ["kzb"]
        assert args.all is False

    def test_parse_args_accepts_multiple_keys(self) -> None:
        args = curate_cli.parse_args(["kzb", "solar"])
        assert args.datasets == ["kzb", "solar"]
        assert args.all is False

    def test_parse_args_accepts_all_flag(self) -> None:
        args = curate_cli.parse_args(["--all"])
        assert args.all is True
        assert args.datasets == []

    def test_parse_args_errors_when_nothing_selected(self) -> None:
        with pytest.raises(SystemExit):
            curate_cli.parse_args([])

    def test_parse_args_default_stage_is_all(self) -> None:
        args = curate_cli.parse_args(["--all"])
        assert args.stage == "all"

    def test_parse_args_accepts_each_stage_name(self) -> None:
        for name in (
            "language", "quality", "repetition",
            "exact_dedup", "sentence_dedup", "stats", "all",
        ):
            args = curate_cli.parse_args(["--all", "--stage", name])
            assert args.stage == name

    def test_parse_args_rejects_unknown_stage(self) -> None:
        with pytest.raises(SystemExit):
            curate_cli.parse_args(["--all", "--stage", "lang"])

    def test_parse_args_default_workers_is_serial(self) -> None:
        args = curate_cli.parse_args(["--all"])
        assert args.workers == 1

    def test_parse_args_max_workers_zero(self) -> None:
        args = curate_cli.parse_args(["--all", "--max-workers", "0"])
        assert args.workers == 0

    def test_parse_args_tasks_alias(self) -> None:
        args = curate_cli.parse_args(["--all", "--tasks", "4"])
        assert args.workers == 4

    def test_filter_input_keys_mirrors_multiple_datasets(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "datatrove"
        for key in ("kzb", "solar"):
            shard_dir = input_dir / key
            shard_dir.mkdir(parents=True)
            (shard_dir / "00000.jsonl.gz").write_bytes(b"")

        holder = curate_cli._filter_input_keys(input_dir, ["kzb", "solar"])
        try:
            assert (holder / "kzb" / "00000.jsonl.gz").is_symlink()
            assert (holder / "solar" / "00000.jsonl.gz").is_symlink()
        finally:
            import shutil
            shutil.rmtree(holder, ignore_errors=True)

    def test_filter_input_keys_lists_all_missing_keys(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "datatrove"
        (input_dir / "kzb").mkdir(parents=True)
        (input_dir / "kzb" / "00000.jsonl.gz").write_bytes(b"")

        with pytest.raises(FileNotFoundError) as excinfo:
            curate_cli._filter_input_keys(input_dir, ["kzb", "missing1", "missing2"])
        msg = str(excinfo.value)
        assert "missing1" in msg
        assert "missing2" in msg
        assert "'kzb'" not in msg
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/data/test_curate_pipeline.py::TestCurateCLISelection -v`
Expected: PASS (12 tests).

- [ ] **Step 3: Lint**

Run: `uv run ruff check --select D tests/data/test_curate_pipeline.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_curate_pipeline.py
git commit -m "test(curate): cover the new --stage CLI flag"
```

---

## Task 8: Sentinel-driven runner integration test

**Files:**
- Create: `tests/data/test_curate_runner.py`

This test asserts the sentinel-skip and cascade-invalidation behavior end-to-end against a tiny synthetic input, without paying for real datatrove executors. It does so by monkey-patching `_stage_runner` to record what got run.

- [ ] **Step 1: Write the test**

```python
# tests/data/test_curate_runner.py
"""Integration tests for the stage runner in scripts/data/curate.py.

Asserts the sentinel-skip + cascade-invalidate contract by stubbing
out the actual executor builds. The real builders are tested in
test_curate_pipeline.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

import scripts.data.curate as curate_cli
from slm4ie.data.curate import (
    STAGE_DIRS,
    STAGE_NAMES,
    config_hash,
    read_sentinel,
    write_sentinel,
)


def _setup_output(tmp_path: Path) -> Path:
    """Build a minimal <output_dir>/ with empty stage folders."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return output_dir


def _stub_runner(
    monkeypatch: pytest.MonkeyPatch, ran: List[str]
) -> None:
    """Replace `_stage_runner` so each stage just records its name."""

    def fake_runner(
        stage: str,
        paths: Any,
        cfg: Dict[str, Any],
        workers: int,
        stopwords: Set[str],
    ):
        def run() -> None:
            ran.append(stage)
            # Materialize the stage folder so `_count_records` finds it.
            paths.stage_dir(stage).mkdir(parents=True, exist_ok=True)

        return run

    monkeypatch.setattr(curate_cli, "_stage_runner", fake_runner)


def _common_cfg(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "language": {"targets": ["sl"]},
        "quality": {"min_doc_words": 50},
        "repetition": {},
        "exact_dedup": {"precision": 64, "hash_fc": "xxhash"},
        "sentence_dedup": {"n_sentences": 3},
        "stats": {"top_k_words": 5000},
    }


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    cfg: Dict[str, Any],
    args: List[str],
    project_root: Path,
) -> None:
    """Drive `curate_cli.main()` with a stubbed YAML loader."""
    monkeypatch.setattr(curate_cli, "_load_yaml", lambda _p: cfg)
    monkeypatch.setattr(
        curate_cli, "_load_stopwords", lambda _root, _cfg: (set(), b"")
    )
    monkeypatch.setattr(curate_cli, "_find_project_root", lambda: project_root)
    monkeypatch.setattr(curate_cli, "_list_datasets", lambda _p: [])
    monkeypatch.setattr(curate_cli.sys, "argv", ["curate.py", *args])
    curate_cli.main()


def test_first_run_executes_every_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A clean output_dir causes every stage to run."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == list(STAGE_NAMES)


def test_unchanged_rerun_skips_every_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Running twice with identical config runs everything once then nothing."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == []


def test_quality_config_change_cascades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editing quality config invalidates quality + downstream, not language."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    cfg["quality"]["min_doc_words"] = 100
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["quality", "repetition", "exact_dedup", "sentence_dedup", "stats"]


def test_stats_config_change_only_reruns_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editing stats config invalidates only stats, leaves dedup alone."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    cfg["stats"]["top_k_words"] = 9999
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    assert ran == ["stats"]


def test_force_stage_invalidates_downstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--force --stage exact_dedup reruns exact_dedup + sentence_dedup + stats."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all"], tmp_path)
    ran.clear()
    _run_cli(monkeypatch, cfg, ["--all", "--force", "--stage", "exact_dedup"], tmp_path)
    assert ran == ["exact_dedup", "sentence_dedup", "stats"]


def test_run_only_one_stage_skips_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--stage quality runs only quality (assuming it was stale)."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    assert ran == ["quality"]


def test_sentinel_records_config_slice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sentinel JSON stores the actual config slice the stage saw."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = _setup_output(tmp_path)
    cfg = _common_cfg(input_dir, output_dir)
    ran: List[str] = []
    _stub_runner(monkeypatch, ran)
    _run_cli(monkeypatch, cfg, ["--all", "--stage", "quality"], tmp_path)
    sentinel = read_sentinel(output_dir / STAGE_DIRS["quality"])
    assert sentinel is not None
    assert sentinel.config_slice == {"min_doc_words": 50}
    assert sentinel.config_hash == config_hash({"min_doc_words": 50})
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/data/test_curate_runner.py -v`
Expected: PASS (7 tests). If the runner's stage-iteration logic has bugs, fix them in `scripts/data/curate.py` and re-run.

- [ ] **Step 3: Lint**

Run: `uv run ruff check --select D tests/data/test_curate_runner.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_curate_runner.py
git commit -m "$(cat <<'EOF'
test(curate): cover sentinel-driven stage runner

Asserts first-run executes everything, unchanged rerun skips
everything, per-stage config change cascades to downstream only,
and --force --stage X invalidates X + downstream.
EOF
)"
```

---

## Task 9: Update the end-to-end smoke test

**Files:**
- Modify: `tests/data/test_curate_pipeline.py` (rewrite the `test_final_corpus_drops_cross_dataset_duplicates` smoke test).

- [ ] **Step 1: Replace the smoke test**

Find the existing `test_final_corpus_drops_cross_dataset_duplicates` function near the bottom of the file (between the synthetic-doc constants and `TestCurateCLISelection`) and replace it with:

```python
@pytest.mark.slow
def test_final_corpus_drops_cross_dataset_duplicates(tmp_path: Path) -> None:
    """Two shards with one full-doc dup and one shared span produce 3 survivors."""
    input_folder = tmp_path / "datatrove"
    _write_shard(
        input_folder / "alpha" / "00000.jsonl.gz",
        dataset="alpha",
        domain="scientific",
        docs=[
            {"id": "alpha:1", "text": SHARED_DOC},
            {"id": "alpha:2", "text": A2_TEXT},
            {"id": "alpha:3", "text": "Solnce sveti nad gorami in dolinami slovenskih krajev. " * 8},
        ],
    )
    _write_shard(
        input_folder / "beta" / "00000.jsonl.gz",
        dataset="beta",
        domain="legal",
        docs=[
            {"id": "beta:1", "text": SHARED_DOC},
            {"id": "beta:2", "text": B2_TEXT},
            {"id": "beta:3", "text": "Pravna doktrina se razvija s časom in družbenimi spremembami. " * 8},
        ],
    )

    output_dir = tmp_path / "curated"
    paths = CuratePaths(input_folder=input_folder, output_dir=output_dir)

    loose_quality = QualityConfig(
        min_doc_words=5,
        min_stop_words=0,
        max_non_alpha_words_ratio=0.6,
        max_avg_word_length=15,
    )
    loose_sentence = SentDedupConfig(
        n_sentences=3,
        min_doc_words=5,
        min_num_sentences=1,
        split_sentences=True,
    )

    build_language_executors(paths, tasks=1)[-1].run()
    build_quality_executors(paths, tasks=1, quality_config=loose_quality, stopwords=set())[-1].run()
    build_repetition_executors(paths, tasks=1)[-1].run()
    build_exact_dedup_executors(paths, tasks=1)[-1].run()
    build_sentence_dedup_executors(paths, tasks=1, sentence_config=loose_sentence)[-1].run()
    build_stats_executors(paths, stopwords=set(), compute_keywords=False)[-1].run()

    final_folder = paths.stage_dir("sentence_dedup")
    survivors: List[str] = []
    survivor_dirs: set = set()
    for shard in sorted(final_folder.glob("**/*.jsonl.gz")):
        survivor_dirs.add(shard.parent.name)
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                survivors.append(rec["id"])

    assert len(survivors) == 3
    assert "alpha:1" in survivors
    assert "beta:1" not in survivors
    assert "alpha:3" not in survivors
    assert "beta:3" not in survivors
    assert survivor_dirs == {"alpha", "beta"}

    stats_folder = paths.stage_dir("stats")
    bundle = json.loads((stats_folder / "aggregate.json").read_text(encoding="utf-8"))
    assert bundle["total_docs"] == 3
    assert "alpha" in bundle["by_dataset"]
    assert "beta" in bundle["by_dataset"]
    assert bundle["by_dataset"]["alpha"]["doc_count"] == 2
    assert bundle["by_dataset"]["beta"]["doc_count"] == 1

    per_dataset_dir = stats_folder / "per_dataset"
    assert (per_dataset_dir / "alpha.json").exists()
    assert (per_dataset_dir / "beta.json").exists()
```

Also at the top of the file, in the import block, add the new per-stage builder imports if they're not already present:

```python
from slm4ie.data.curate.pipeline import (
    CuratePaths,
    QualityConfig,
    build_exact_dedup_executors,
    build_language_executors,
    build_quality_executors,
    build_repetition_executors,
    build_sentence_dedup_executors,
    build_stats_executors,
)
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/data/test_curate_pipeline.py::test_final_corpus_drops_cross_dataset_duplicates -v -m slow`
Expected: PASS (~30-60 seconds depending on box).

- [ ] **Step 3: Commit**

```bash
git add tests/data/test_curate_pipeline.py
git commit -m "$(cat <<'EOF'
test(curate): rewrite smoke test against per-stage builders

Drives each stage's builder in sequence, asserts dedup invariants
on the final 04_2_dedup/ folder, and reads stats from 05_statistics/.
EOF
)"
```

---

## Task 10: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full curate test set**

Run: `uv run pytest tests/data/test_curate_*.py -v`
Expected: PASS (all tests).

- [ ] **Step 2: Run the broader test suite to catch unrelated breakage**

Run: `uv run pytest -v --ignore=tests/data/test_download.py --ignore=tests/data/test_extract.py`
Expected: PASS. The two ignored test files hit `/vault/data/SLM4IE/` and the network; running them is environment-dependent.

- [ ] **Step 3: Final lint sweep on every touched file**

Run:
```bash
uv run ruff check --select D \
  slm4ie/data/curate/__init__.py \
  slm4ie/data/curate/stages.py \
  slm4ie/data/curate/sentinel.py \
  slm4ie/data/curate/dedup.py \
  slm4ie/data/curate/pipeline.py \
  scripts/data/curate.py \
  tests/data/test_curate_stages.py \
  tests/data/test_curate_sentinel.py \
  tests/data/test_curate_dedup.py \
  tests/data/test_curate_pipeline.py \
  tests/data/test_curate_runner.py
```
Expected: clean.

- [ ] **Step 4: Sanity-check the CLI is at least loadable**

Run: `uv run python scripts/data/curate.py --help`
Expected: shows usage with `--stage {language,quality,repetition,exact_dedup,sentence_dedup,stats,all}` in the help text.

- [ ] **Step 5: Commit any lint or test fixups produced during verification**

```bash
git status
# If anything is uncommitted, stage and commit it with a focused message:
# git add <files> && git commit -m "fix(curate): <what>"
```

---

## Out of scope (deliberately not handled here)

- **`configs/data/curate.yaml` migration of in-tree corpora.** Existing on-disk `final/<dataset>/...` outputs are not migrated. Operators rerun `--all --force` once on the canonical box to produce the new layout. Document this in the commit message of Task 5 if downstream training configs reference the old `final/` path.
- **README updates.** Update only if `README.md` actively describes the old curate layout. A quick `grep -n "final/" README.md` confirms whether a doc patch is needed.
- **`scripts/data/analyze_dedup_drops.py` reinstatement.** It was deleted in an earlier commit; with no `*_dropped/` folders it has nothing to inspect. Do not reintroduce.
- **Tokenizer-eval / training scripts that read `final/`.** If `scripts/training/...` or `scripts/tokenizer/...` references the old `final/<dataset>/...` layout, they will need to read `<output_dir>/04_2_dedup/<dataset>/...` instead. That migration is its own plan.

## Self-review (already done by author)

1. **Spec coverage:** Six stages × {builder, sentinel, CLI hook} = 18 surfaces; all covered. CLI flag changes covered. YAML restructure covered. Tests updated for every new contract.
2. **Placeholder scan:** No "TBD", no `# TODO`, every step contains either runnable code, a runnable command, or both.
3. **Type consistency:** `STAGE_NAMES`/`STAGE_DIRS`/`cascade_from`/`config_hash`/`sentinel_is_current`/`write_sentinel`/`read_sentinel` are referenced consistently across Tasks 1, 2, 6, 8. `CuratePaths(input_folder=..., output_dir=...)` shape is consistent across Tasks 4, 6, 8, 9. `build_*_executors(...)` signatures match across Tasks 4 and 9.
