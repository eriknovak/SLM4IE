# Incremental Dataset-Scoped Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `to_pretrain.py` curate one dataset through the per-document stages (convert/language/quality/repetition) without reprocessing the rest, then run dedup/stats corpus-wide — so a new dataset like gigafida can be added incrementally.

**Architecture:** Split the seven stages into *scoped* (convert, language, quality, repetition — per-dataset sentinels, subset-filtered reads) and *corpus* (exact_dedup, sentence_dedup, stats — corpus-level roster-sensitive sentinel, run only under `--all`). Scoped reads reuse the existing convert→language symlink-view trick, generalized to every scoped stage. The CLI runs scoped stages for positional keys; `--all` adds the corpus stages.

**Tech Stack:** Python 3.13, uv, pytest, ruff (Google docstrings), datatrove. Design spec: `docs/superpowers/specs/2026-05-30-incremental-dataset-scoped-curation-design.md`.

---

## Background the implementer needs

- Stage registry lives in `slm4ie/data/curate/stages.py`: `STAGE_NAMES = (convert, language, quality, repetition, exact_dedup, sentence_dedup, stats)`, `STAGE_DIRS` maps each to `00_convert … 05_statistics`, `cascade_from(stage)` returns the stage + downstream in order.
- Sentinels live in `slm4ie/data/curate/sentinel.py`: `SENTINEL_NAME = ".complete"`; `config_hash(slice_, extra=None) -> str` (SHA-256); `write_sentinel(stage_folder, *, config_slice, config_hash_value, records_in, records_out)`; `read_sentinel`; `sentinel_is_current(stage_folder, expected_hash)`; `cascade_invalidate(output_dir, stage)`.
- Stage executors are built in `slm4ie/data/curate/pipeline.py`. `build_language_executors(... input_override=None)` already reads from `input_override or paths.stage_dir("convert")`. `build_quality_executors` reads `paths.stage_dir("language")`; `build_repetition_executors` reads `paths.stage_dir("quality")` — neither has an override yet. Every writer emits `<stage_dir>/<dataset>/<rank>.jsonl.gz` (routes by `${dataset}`); readers walk the input folder recursively.
- The CLI orchestration is in `scripts/data/to_pretrain.py`: `_filter_convert_subset(convert_dir, keys)` builds a tempdir of symlinks to `<convert_dir>/<key>/*.jsonl.gz`; `_stage_extra(stage, stopwords_bytes, dataset_keys_bytes)` folds the roster into every stage's hash (and stopwords into quality/stats); `_dataset_keys_payload(keys)` → sorted JSON bytes; `_stage_runner(stage, paths, cfg, workers, stopwords, dataset_keys, convert_view, log_dir)` returns the per-stage run callable; `main()` loops `requested_stages`, checks `sentinel_is_current`, runs, writes sentinels.
- Conventions: `uv run pytest`, `uv run ruff check` / `uv run ruff check --select D`. Google docstrings required (no reST). `typing` generics. The `curate` extra is installed (`uv sync --all-extras`); datatrove import works.
- Pre-existing: `tests/data/test_curate_runner.py` errors on a missing optional import in some envs — ignore it; it is unrelated. Run suites with `--ignore=tests/data/test_curate_runner.py` when it interferes, but the curate extra is installed here so it should collect.

## File structure

- `slm4ie/data/curate/stages.py` — add `SCOPED_STAGES`, `CORPUS_STAGES`, `is_scoped(stage)`.
- `slm4ie/data/curate/sentinel.py` — add `dataset_sentinel_path`, `write_dataset_sentinel`, `dataset_sentinel_is_current`, `invalidate_dataset_sentinels`; extend `cascade_invalidate` with class-awareness via a new `cascade_invalidate_scoped(output_dir, stage, keys)`.
- `slm4ie/data/curate/pipeline.py` — add `input_override` to `build_quality_executors` and `build_repetition_executors`.
- `scripts/data/to_pretrain.py` — generalize `_filter_convert_subset` → `_filter_stage_subset`; update `_stage_extra`; thread `input_override` through `_stage_runner` for quality/repetition; rewrite the `main()` stage loop for scoped/corpus dispatch + per-dataset skip; CLI validation; `--force` matrix.
- Tests under `tests/data/`.

---

## Phase A — Stage taxonomy + per-dataset sentinels (pure, TDD)

### Task A1: Stage taxonomy constants

**Files:**
- Modify: `slm4ie/data/curate/stages.py`
- Test: `tests/data/test_curate_stages.py` (create if absent; otherwise append)

- [ ] **Step 1: Write the failing test**

Create/append `tests/data/test_curate_stages.py`:

```python
"""Tests for the curate stage taxonomy."""

from slm4ie.data.curate.stages import (
    CORPUS_STAGES,
    SCOPED_STAGES,
    STAGE_NAMES,
    is_scoped,
)


def test_scoped_and_corpus_partition_stage_names() -> None:
    """SCOPED_STAGES + CORPUS_STAGES is exactly STAGE_NAMES, in order."""
    assert SCOPED_STAGES + CORPUS_STAGES == STAGE_NAMES
    assert set(SCOPED_STAGES).isdisjoint(CORPUS_STAGES)


def test_scoped_membership() -> None:
    """convert/language/quality/repetition are scoped; dedup/stats are not."""
    assert SCOPED_STAGES == ("convert", "language", "quality", "repetition")
    assert CORPUS_STAGES == ("exact_dedup", "sentence_dedup", "stats")


def test_is_scoped() -> None:
    """is_scoped returns True only for scoped stages."""
    assert is_scoped("quality") is True
    assert is_scoped("exact_dedup") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_curate_stages.py -v`
Expected: FAIL (ImportError: cannot import `SCOPED_STAGES`).

- [ ] **Step 3: Add the constants + predicate to `stages.py`**

After the `STAGE_NAMES` definition, add:

```python
#: Stages that process one dataset at a time. They honor a dataset
#: subset, write to canonical `<stage_dir>/<dataset>/`, and are tracked
#: by per-dataset sentinels.
SCOPED_STAGES: Tuple[str, ...] = ("convert", "language", "quality", "repetition")

#: Stages that operate over the whole corpus at once. They read every
#: dataset under their input folder, are tracked by a corpus-level
#: sentinel whose hash includes the dataset roster, and run only under
#: `--all`.
CORPUS_STAGES: Tuple[str, ...] = ("exact_dedup", "sentence_dedup", "stats")
```

At the end of the module add:

```python
def is_scoped(stage: str) -> bool:
    """Return True if *stage* is a per-dataset scoped stage.

    Args:
        stage: One of the values in `STAGE_NAMES`.

    Returns:
        True for convert/language/quality/repetition; False for the
        corpus-wide dedup/stats stages.

    Raises:
        KeyError: If *stage* is not a known stage name.
    """
    if stage not in STAGE_NAMES:
        raise KeyError(stage)
    return stage in SCOPED_STAGES
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_curate_stages.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/stages.py tests/data/test_curate_stages.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/stages.py tests/data/test_curate_stages.py
git commit -m "feat(curate): add scoped/corpus stage taxonomy"
```

---

### Task A2: Per-dataset sentinel helpers

**Files:**
- Modify: `slm4ie/data/curate/sentinel.py`
- Test: `tests/data/test_curate_sentinel.py` (append; create if absent)

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_curate_sentinel.py`:

```python
def test_dataset_sentinel_roundtrip(tmp_path: "Path") -> None:
    """A per-dataset sentinel is current only when its hash matches."""
    from slm4ie.data.curate.sentinel import (
        dataset_sentinel_is_current,
        dataset_sentinel_path,
        write_dataset_sentinel,
    )

    stage_dir = tmp_path / "02_quality"
    write_dataset_sentinel(
        stage_dir,
        "gigafida",
        config_slice={"min_doc_words": 20},
        config_hash_value="abc123",
        records_in=10,
        records_out=8,
    )
    assert dataset_sentinel_path(stage_dir, "gigafida").exists()
    assert dataset_sentinel_is_current(stage_dir, "gigafida", "abc123") is True
    assert dataset_sentinel_is_current(stage_dir, "gigafida", "different") is False
    # A different dataset under the same stage is independent.
    assert dataset_sentinel_is_current(stage_dir, "kas", "abc123") is False


def test_invalidate_dataset_sentinels(tmp_path: "Path") -> None:
    """Invalidating removes only the named datasets' sentinels."""
    from slm4ie.data.curate.sentinel import (
        dataset_sentinel_is_current,
        invalidate_dataset_sentinels,
        write_dataset_sentinel,
    )

    stage_dir = tmp_path / "01_language"
    for key in ("a", "b"):
        write_dataset_sentinel(
            stage_dir, key, config_slice={}, config_hash_value="h",
            records_in=1, records_out=1,
        )
    invalidate_dataset_sentinels(stage_dir, ["a"])
    assert dataset_sentinel_is_current(stage_dir, "a", "h") is False
    assert dataset_sentinel_is_current(stage_dir, "b", "h") is True
```

If the file lacks a `Path` import at module top, add `from pathlib import Path` and change the annotations from `"Path"` to `Path`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_curate_sentinel.py -k dataset_sentinel -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Add helpers to `sentinel.py`**

Add these functions after `write_sentinel`/`sentinel_is_current` (they reuse `SENTINEL_NAME`, `write_sentinel`, `read_sentinel`):

```python
def dataset_sentinel_path(stage_folder: Path, dataset: str) -> Path:
    """Return the per-dataset sentinel path under *stage_folder*.

    Args:
        stage_folder: A scoped stage's output folder (e.g.
            `<output_dir>/02_quality`).
        dataset: Dataset key whose sentinel is addressed.

    Returns:
        Path to `<stage_folder>/<dataset>/.complete`.
    """
    return stage_folder / dataset / SENTINEL_NAME


def write_dataset_sentinel(
    stage_folder: Path,
    dataset: str,
    *,
    config_slice: Dict[str, Any],
    config_hash_value: str,
    records_in: int,
    records_out: int,
) -> Path:
    """Write a per-dataset `.complete` sentinel for a scoped stage.

    Args:
        stage_folder: The scoped stage's output folder.
        dataset: Dataset key the sentinel covers.
        config_slice: The config slice the run consumed.
        config_hash_value: Pre-computed hash of *config_slice* (the
            dataset roster is intentionally excluded for scoped stages).
        records_in: Records read for this dataset.
        records_out: Records written for this dataset.

    Returns:
        Path to the written sentinel file.
    """
    return write_sentinel(
        stage_folder / dataset,
        config_slice=config_slice,
        config_hash_value=config_hash_value,
        records_in=records_in,
        records_out=records_out,
    )


def dataset_sentinel_is_current(
    stage_folder: Path, dataset: str, expected_hash: str
) -> bool:
    """Return True iff *dataset*'s per-dataset sentinel matches *expected_hash*.

    Args:
        stage_folder: The scoped stage's output folder.
        dataset: Dataset key to check.
        expected_hash: Hash recomputed from current config.

    Returns:
        True if the recorded per-dataset hash matches; False otherwise
        (including when the sentinel is missing).
    """
    return sentinel_is_current(stage_folder / dataset, expected_hash)


def invalidate_dataset_sentinels(stage_folder: Path, datasets: List[str]) -> None:
    """Remove the per-dataset sentinels for *datasets* under *stage_folder*.

    Args:
        stage_folder: The scoped stage's output folder.
        datasets: Dataset keys whose sentinels should be removed. Keys
            without a sentinel are ignored.
    """
    for dataset in datasets:
        dataset_sentinel_path(stage_folder, dataset).unlink(missing_ok=True)
```

Add `List` to the `typing` import line (`from typing import Any, Dict, List, Optional, Tuple`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_curate_sentinel.py -k dataset_sentinel -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/sentinel.py tests/data/test_curate_sentinel.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/sentinel.py tests/data/test_curate_sentinel.py
git commit -m "feat(curate): add per-dataset sentinel helpers"
```

---

### Task A3: Class-aware cascade invalidation for scoped stages

**Files:**
- Modify: `slm4ie/data/curate/sentinel.py`
- Test: `tests/data/test_curate_sentinel.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_cascade_invalidate_scoped_mixes_per_dataset_and_corpus(tmp_path: Path) -> None:
    """Scoped stages drop per-dataset sentinels; corpus stages drop stage sentinels."""
    from slm4ie.data.curate.sentinel import (
        cascade_invalidate_scoped,
        dataset_sentinel_is_current,
        sentinel_is_current,
        write_dataset_sentinel,
        write_sentinel,
    )

    out = tmp_path
    # Scoped stage quality has per-dataset sentinels for a, b.
    q = out / "02_quality"
    for key in ("a", "b"):
        write_dataset_sentinel(q, key, config_slice={}, config_hash_value="h",
                               records_in=1, records_out=1)
    # Corpus stage exact_dedup has a stage-level sentinel.
    d = out / "04_1_dedup"
    write_sentinel(d, config_slice={}, config_hash_value="h",
                   records_in=1, records_out=1)

    # Invalidate from quality, only for dataset 'a'.
    cascade_invalidate_scoped(out, "quality", ["a"])

    # quality/a dropped, quality/b kept (only requested keys invalidated).
    assert dataset_sentinel_is_current(q, "a", "h") is False
    assert dataset_sentinel_is_current(q, "b", "h") is True
    # Downstream corpus stage dropped wholesale (roster/inputs changed).
    assert sentinel_is_current(d, "h") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_curate_sentinel.py -k cascade_invalidate_scoped -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Add `cascade_invalidate_scoped` to `sentinel.py`**

Import the taxonomy at the top (extend the existing stages import):

```python
from slm4ie.data.curate.stages import STAGE_DIRS, cascade_from, is_scoped
```

Add:

```python
def cascade_invalidate_scoped(
    output_dir: Path, stage: str, keys: List[str]
) -> Tuple[str, ...]:
    """Invalidate *stage* and downstream, honoring per-dataset granularity.

    For scoped stages, only the named *keys*' per-dataset sentinels are
    removed; for corpus stages the whole stage-level sentinel is removed.
    A scoped stage's downstream set spans into the corpus stages, so a
    scoped edit also drops the corpus sentinels.

    Args:
        output_dir: The curation output root.
        stage: First stage to invalidate.
        keys: Dataset keys whose scoped-stage sentinels should drop.

    Returns:
        The stage names considered (i.e. `cascade_from(stage)`).
    """
    affected = cascade_from(stage)
    for name in affected:
        stage_folder = output_dir / STAGE_DIRS[name]
        if is_scoped(name):
            invalidate_dataset_sentinels(stage_folder, keys)
        else:
            (stage_folder / SENTINEL_NAME).unlink(missing_ok=True)
    return affected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_curate_sentinel.py -k cascade_invalidate_scoped -v`
Expected: PASS.

- [ ] **Step 5: Lint + full sentinel/stages suite**

Run: `uv run ruff check --select D slm4ie/data/curate/sentinel.py && uv run pytest tests/data/test_curate_sentinel.py tests/data/test_curate_stages.py -q`
Expected: clean + all pass.

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/sentinel.py tests/data/test_curate_sentinel.py
git commit -m "feat(curate): class-aware cascade invalidation for scoped stages"
```

---

## Phase B — Pipeline scoped reads

### Task B1: `input_override` on quality and repetition executors

**Files:**
- Modify: `slm4ie/data/curate/pipeline.py`
- Test: `tests/data/test_curate_pipeline.py` (append; create if absent)

- [ ] **Step 1: Write the failing test**

Append a test that asserts the override is honored. The executors expose their reader's data folder via `executor.pipeline[0].data_folder.path`. Read the existing `test_curate_pipeline.py` first to match its fixture/import style; then add:

```python
def test_quality_executor_honors_input_override(tmp_path: Path) -> None:
    """build_quality_executors reads from input_override when provided."""
    from slm4ie.data.curate.pipeline import CuratePaths, build_quality_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_quality_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path


def test_repetition_executor_honors_input_override(tmp_path: Path) -> None:
    """build_repetition_executors reads from input_override when provided."""
    from slm4ie.data.curate.pipeline import CuratePaths, build_repetition_executors

    paths = CuratePaths(input_folder=tmp_path / "in", output_dir=tmp_path / "out")
    override = tmp_path / "view"
    override.mkdir()
    execs = build_repetition_executors(paths, tasks=1, input_override=override)
    reader = execs[0].pipeline[0]
    assert str(override) in reader.data_folder.path
```

If `reader.data_folder.path` is not the right attribute for the installed datatrove version, the implementer should inspect `execs[0].pipeline[0]` (the `_reader(in_)` object) and assert on whatever attribute exposes the input path; the behavioral requirement is "reads from override". Keep the assertion meaningful (compares to `override`), not trivially true.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_curate_pipeline.py -k input_override -v`
Expected: FAIL (TypeError: unexpected keyword argument `input_override`).

- [ ] **Step 3: Add the parameter to both builders**

In `build_quality_executors`, add `input_override: Optional[Path] = None` to the keyword-only args and change the input line:

```python
    in_ = input_override if input_override is not None else paths.stage_dir("language")
```

Add to its docstring `Args:`:

```
        input_override: Optional folder to read from instead of the
            language stage's output, used to restrict the stage to a
            symlinked subset of datasets.
```

In `build_repetition_executors`, add `input_override: Optional[Path] = None` and:

```python
    in_ = input_override if input_override is not None else paths.stage_dir("quality")
```

with the matching docstring line (referring to the quality stage's output). Confirm `Optional` and `Path` are already imported in `pipeline.py` (they are — `build_language_executors` uses `Optional[Path]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_curate_pipeline.py -k input_override -v`
Expected: PASS.

- [ ] **Step 5: Lint + existing pipeline tests**

Run: `uv run ruff check --select D slm4ie/data/curate/pipeline.py && uv run pytest tests/data/test_curate_pipeline.py -q`
Expected: clean + pass.

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/pipeline.py tests/data/test_curate_pipeline.py
git commit -m "feat(curate): allow input_override on quality and repetition stages"
```

---

## Phase C — Orchestration in `to_pretrain.py`

### Task C1: Generalize the subset-view helper

**Files:**
- Modify: `scripts/data/to_pretrain.py`
- Test: `tests/data/test_to_pretrain.py` (append; create if absent)

- [ ] **Step 1: Write the failing test**

```python
def test_filter_stage_subset_links_requested_keys(tmp_path: Path) -> None:
    """_filter_stage_subset mirrors only the requested keys via symlinks."""
    from scripts.data.to_pretrain import _filter_stage_subset

    stage = tmp_path / "01_language"
    for key in ("a", "b"):
        (stage / key).mkdir(parents=True)
        (stage / key / "000.jsonl.gz").write_bytes(b"x")
    view = _filter_stage_subset(stage, ["a"])
    try:
        assert (view / "a" / "000.jsonl.gz").is_symlink()
        assert not (view / "b").exists()
    finally:
        import shutil
        shutil.rmtree(view, ignore_errors=True)


def test_filter_stage_subset_missing_key_raises(tmp_path: Path) -> None:
    """_filter_stage_subset raises when a key has no shards."""
    import pytest

    from scripts.data.to_pretrain import _filter_stage_subset

    stage = tmp_path / "01_language"
    (stage / "a").mkdir(parents=True)
    (stage / "a" / "000.jsonl.gz").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        _filter_stage_subset(stage, ["a", "missing"])
```

Note: importing from `scripts.data.to_pretrain` requires `scripts/` and `scripts/data/` to be importable. If the import fails, the implementer should check how other tests import script-level helpers (e.g. an existing `tests/data/test_to_pretrain.py`, or a conftest that adds the path) and follow that pattern; do not move the helper into the script's `__main__`-only scope.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_to_pretrain.py -k filter_stage_subset -v`
Expected: FAIL (ImportError: cannot import `_filter_stage_subset`).

- [ ] **Step 3: Rename + generalize the helper**

Rename `_filter_convert_subset(convert_dir, keys)` to `_filter_stage_subset(stage_dir, keys)`. The body is identical (it already operates on `<dir>/<key>/*.jsonl.gz`); only the parameter name and docstring change to refer to a generic stage dir:

```python
def _filter_stage_subset(stage_dir: Path, keys: List[str]) -> Path:
    """Materialize a tempdir of symlinks restricted to *keys* under *stage_dir*.

    Args:
        stage_dir: A scoped stage's output folder (e.g.
            `<output_dir>/01_language/`).
        keys: Dataset keys to expose.

    Returns:
        Path to a tempdir mirroring the requested keys via symlinks, so a
        downstream stage's reader walks only the subset's shards.

    Raises:
        FileNotFoundError: If any requested shard folder is missing or
            empty under *stage_dir*.
    """
```

Keep the body unchanged (the `missing` check, `tempfile.mkdtemp`, per-key symlink loop). Update both call sites in `main`/`_stage_runner` (currently `_filter_convert_subset(stage_folder, dataset_keys)`) to the new name. Grep to be sure: `grep -n _filter_convert_subset scripts/data/to_pretrain.py` must return nothing afterward.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_to_pretrain.py -k filter_stage_subset -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "refactor(curate): generalize subset-view helper to any stage"
```

---

### Task C2: Roster only in corpus-stage hashes

**Files:**
- Modify: `scripts/data/to_pretrain.py`
- Test: `tests/data/test_to_pretrain.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_stage_extra_folds_roster_only_for_corpus_stages() -> None:
    """Scoped stages exclude the roster; corpus stages include it."""
    from scripts.data.to_pretrain import _stage_extra

    roster = b'["a","b"]'
    sw = b"stopwords"
    # Scoped: roster must NOT appear.
    assert _stage_extra("language", sw, roster) == b""
    assert _stage_extra("quality", sw, roster) == sw  # stopwords only, no roster
    # Corpus: roster present.
    assert roster in _stage_extra("exact_dedup", sw, roster)
    assert roster in _stage_extra("stats", sw, roster)
    assert sw in _stage_extra("stats", sw, roster)  # stats also folds stopwords
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_to_pretrain.py -k stage_extra -v`
Expected: FAIL (current `_stage_extra` folds roster for all stages).

- [ ] **Step 3: Rewrite `_stage_extra`**

Replace the function body so the roster is folded only for corpus stages; stopword bytes still fold for quality/stats:

```python
def _stage_extra(stage: str, stopwords_bytes: bytes, dataset_keys_bytes: bytes) -> bytes:
    """Return extra bytes folded into the hash for a stage.

    Corpus stages (exact_dedup, sentence_dedup, stats) fold in the
    dataset roster so adding or removing a dataset invalidates them.
    Scoped stages (convert, language, quality, repetition) exclude the
    roster so per-dataset work survives roster changes. Stopword file
    contents are folded for the stages that consume them (quality,
    stats).

    Args:
        stage: Stage name.
        stopwords_bytes: Raw bytes of the stopword file.
        dataset_keys_bytes: Canonical JSON bytes of the sorted roster.

    Returns:
        Bytes to fold into the stage's sentinel hash.
    """
    from slm4ie.data.curate.stages import is_scoped

    roster = b"" if is_scoped(stage) else dataset_keys_bytes
    if stage in ("quality", "stats"):
        return stopwords_bytes + b"\x00" + roster if roster else stopwords_bytes
    return roster
```

(Place the `is_scoped` import at module top with the other `slm4ie.data.curate` imports instead of inline if the file already imports from `slm4ie.data.curate.stages`; inline is acceptable if not.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_to_pretrain.py -k stage_extra -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "feat(curate): fold dataset roster into corpus-stage hashes only"
```

---

### Task C3: CLI validation — reject corpus `--stage` with positional keys

**Files:**
- Modify: `scripts/data/to_pretrain.py` (`parse_args`)
- Test: `tests/data/test_to_pretrain.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_corpus_stage_with_positional_keys_errors() -> None:
    """--stage exact_dedup with positional keys is rejected."""
    import pytest

    from scripts.data.to_pretrain import parse_args

    with pytest.raises(SystemExit):
        parse_args(["gigafida", "--stage", "exact_dedup"])


def test_corpus_stage_with_all_is_ok() -> None:
    """--all --stage stats parses fine."""
    from scripts.data.to_pretrain import parse_args

    args = parse_args(["--all", "--stage", "stats"])
    assert args.all is True
    assert args.stage == "stats"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_to_pretrain.py -k corpus_stage -v`
Expected: FAIL (the first test does not raise today).

- [ ] **Step 3: Add the validation in `parse_args`**

After the existing mutually-exclusive `--all`/positional validation in `parse_args`, add (using `CORPUS_STAGES`):

```python
    from slm4ie.data.curate.stages import CORPUS_STAGES

    if args.datasets and args.stage in CORPUS_STAGES:
        parser.error(
            f"--stage {args.stage} is corpus-wide; run it with --all, "
            "not with positional dataset keys."
        )
```

Place this where `parser` is still in scope (inside `parse_args`, before `return args`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_to_pretrain.py -k corpus_stage -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "feat(curate): reject corpus --stage with positional keys"
```

---

### Task C4: Scoped/corpus dispatch + per-dataset skip in the stage loop

This is the integration task. Read the current `main()` (the `for stage in requested_stages:` loop), `_stage_runner`, and how `requested_stages`/`subset_holder` are computed. Then make the changes below. Keep all behavior identical for the `--all` full-run case except for the per-dataset sentinel semantics.

**Files:**
- Modify: `scripts/data/to_pretrain.py`
- Test: `tests/data/test_to_pretrain.py` (append — a unit test for `requested_stages` selection; the end-to-end behavior is covered in Phase D)

- [ ] **Step 1: Add a helper that selects requested stages, with a failing test**

Test:

```python
def test_resolve_requested_stages() -> None:
    """Subset 'all' = scoped stages; --all 'all' = every stage."""
    from scripts.data.to_pretrain import _resolve_requested_stages
    from slm4ie.data.curate.stages import SCOPED_STAGES, STAGE_NAMES

    assert _resolve_requested_stages(stage="all", run_all=False) == SCOPED_STAGES
    assert _resolve_requested_stages(stage="all", run_all=True) == STAGE_NAMES
    assert _resolve_requested_stages(stage="quality", run_all=False) == ("quality",)
    assert _resolve_requested_stages(stage="exact_dedup", run_all=True) == ("exact_dedup",)
```

- [ ] **Step 2: Run it (fails — no `_resolve_requested_stages`)**

Run: `uv run pytest tests/data/test_to_pretrain.py -k resolve_requested_stages -v`
Expected: FAIL.

- [ ] **Step 3: Add the helper**

```python
def _resolve_requested_stages(stage: str, run_all: bool) -> Tuple[str, ...]:
    """Resolve which stages a run executes.

    A subset run (`run_all` False) with `--stage all` runs only the
    scoped stages and stops before the corpus stages. With `--all`,
    `all` means every stage. An explicit single stage is returned as-is.

    Args:
        stage: The `--stage` value (a stage name or `"all"`).
        run_all: True when `--all` was passed.

    Returns:
        The stage names to execute, in pipeline order.
    """
    if stage != "all":
        return (stage,)
    return STAGE_NAMES if run_all else SCOPED_STAGES
```

Ensure `STAGE_NAMES`, `SCOPED_STAGES`, `Tuple` are imported in `to_pretrain.py`.

- [ ] **Step 4: Run it (passes)**

Run: `uv run pytest tests/data/test_to_pretrain.py -k resolve_requested_stages -v`
Expected: PASS.

- [ ] **Step 5: Rewrite the `main()` stage loop to use scoped/corpus dispatch**

Replace the computation of `requested_stages` with `_resolve_requested_stages(args.stage, args.all)`. Then rewrite the per-stage loop body so that:

For a **scoped stage** (`is_scoped(stage)` True):
1. Compute `current_hash = config_hash(_stage_slice(stage, cfg), extra=_stage_extra(stage, stopwords_raw, dataset_keys_bytes))` (roster now excluded by C2).
2. Determine the keys to process: `todo = [k for k in dataset_keys if not (cascaded or dataset_sentinel_is_current(stage_folder, k, current_hash))]` — but `cascaded` must itself become per-dataset aware. Simplest correct approach: track a set `force_keys` (keys whose upstream scoped stage actually ran this invocation) so that when an upstream scoped stage re-runs a key, the downstream scoped stage also re-runs that key regardless of its sentinel. Initialize `force_keys: set[str] = set()`. `todo = [k for k in dataset_keys if k in force_keys or not dataset_sentinel_is_current(stage_folder, k, current_hash)]`.
3. If `todo` is empty: log "[<stage>] all requested datasets current; skipping." and continue.
4. Build the input view for `todo` from the **upstream** stage dir (except convert, which needs no view): for `language`, upstream is convert; `quality`→language; `repetition`→quality. Use `_filter_stage_subset(paths.stage_dir(upstream_stage(stage)), todo)`; for `convert`, `view = None`. Wrap in try/finally to `shutil.rmtree` the view.
5. Run the stage for `todo` (pass `dataset_keys=todo`, and the view as `convert_view`/`input_override` per the runner — see Step 6 for the runner change).
6. Write a per-dataset sentinel for each key in `todo`: `write_dataset_sentinel(stage_folder, k, config_slice=slice_, config_hash_value=current_hash, records_in=..., records_out=...)`. The runner returns aggregate `(records_in, records_out)`; per-dataset counts are not separated, so record the aggregate on each key's sentinel (counts are advisory only — the hash is what gates skipping). Add `force_keys.update(todo)` so downstream scoped stages re-run these keys.

For a **corpus stage** (`is_scoped(stage)` False):
1. Only run under `--all`. Because C3 forbids corpus `--stage` with positional keys, and `_resolve_requested_stages` excludes corpus stages from a subset `--stage all`, the only way a corpus stage appears in `requested_stages` is an `--all` run (either `--all` or `--all --stage <corpus>`). Assert/guard `args.all` is True; if somehow False, skip with a warning.
2. Compute `current_hash` with the roster folded in (C2 handles this).
3. Use the existing stage-level sentinel path: if `sentinel_is_current(stage_folder, current_hash)` and not cascaded → skip. Else run via the existing runner (reads the full upstream dir — no view) and `write_sentinel(stage_folder, ...)`.

Keep the existing cascade-invalidation behavior, but route it through `cascade_invalidate_scoped(output_dir, stage, dataset_keys)` when the first not-current stage is hit, so scoped-downstream per-dataset sentinels (for the processed keys) and corpus sentinels are invalidated together. Set `cascaded = True` and `force_keys.update(dataset_keys)` once cascaded (so all downstream scoped stages re-run the processed keys).

The old single `subset_holder` lifecycle in `main` is removed (each scoped stage now builds and tears down its own view inside the loop iteration).

- [ ] **Step 6: Update `_stage_runner` to accept and thread the per-stage input view**

Generalize `_stage_runner`'s `convert_view` parameter to `input_view: Optional[Path]` (a view of the stage's upstream output). Thread it into the builder calls:
- `language`: pass `input_override=input_view` (as today via `convert_view`).
- `quality`: pass `input_override=input_view` to `build_quality_executors`.
- `repetition`: pass `input_override=input_view` to `build_repetition_executors`.
- `convert`: ignores the view (scoped by `dataset_keys` already).
- corpus stages: ignore the view (always full corpus).
Keep the existing return contract `(records_in, records_out)`.

- [ ] **Step 7: Run the unit suite for to_pretrain**

Run: `uv run pytest tests/data/test_to_pretrain.py -q`
Expected: PASS (filter_stage_subset, stage_extra, corpus_stage, resolve_requested_stages tests).

- [ ] **Step 8: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "feat(curate): scoped/corpus stage dispatch with per-dataset skipping"
```

---

### Task C5: `--force` matrix

**Files:**
- Modify: `scripts/data/to_pretrain.py` (`main` force handling)
- Test: `tests/data/test_to_pretrain.py` (append)

Read the current `--force` handling in `main` (the two blocks: `args.force and args.stage == "all"` nukes `output_dir`; `args.force and args.stage != "all"` removes folders + cascade). Generalize per the spec's force matrix.

- [ ] **Step 1: Write the failing test (subset force drops only requested keys' scoped sentinels)**

```python
def test_force_subset_stage_drops_only_requested_keys(tmp_path: Path, monkeypatch) -> None:
    """--force gigafida --stage quality drops gigafida's quality sentinel, keeps others."""
    from slm4ie.data.curate.sentinel import (
        dataset_sentinel_is_current,
        write_dataset_sentinel,
    )
    from scripts.data.to_pretrain import _apply_force

    out = tmp_path / "pretrain"
    q = out / "02_quality"
    for key in ("gigafida", "kas"):
        write_dataset_sentinel(q, key, config_slice={}, config_hash_value="h",
                               records_in=1, records_out=1)
    _apply_force(out, stage="quality", run_all=False, dataset_keys=["gigafida"])
    assert dataset_sentinel_is_current(q, "gigafida", "h") is False
    assert dataset_sentinel_is_current(q, "kas", "h") is True
```

- [ ] **Step 2: Run it (fails — no `_apply_force`)**

Run: `uv run pytest tests/data/test_to_pretrain.py -k force_subset -v`
Expected: FAIL.

- [ ] **Step 3: Extract force handling into `_apply_force` and implement the matrix**

```python
def _apply_force(
    output_dir: Path, *, stage: str, run_all: bool, dataset_keys: List[str]
) -> None:
    """Apply `--force` per the scoped/corpus force matrix.

    Args:
        output_dir: Curation output root.
        stage: The `--stage` value (`"all"` or a stage name).
        run_all: True when `--all` was passed.
        dataset_keys: Keys in play (the full roster under `--all`, else
            the positional subset).

    Behavior:
        - `--all` with `--stage all`: remove the whole `output_dir`.
        - subset with `--stage all`: drop the requested keys' per-dataset
          sentinels and shard subfolders across every scoped stage, plus
          all corpus sentinels.
        - `--stage <scoped>`: drop the requested keys' sentinels/subfolders
          for that scoped stage and its scoped-downstream, plus corpus
          sentinels (and `_dedup_state` if dedup is downstream).
        - `--stage <corpus>` (implies `--all`): drop that corpus stage's
          sentinel + downstream corpus sentinels and their data folders
          (and `_dedup_state` when dedup is affected).
    """
    ...
```

Implement the four branches using `cascade_from`, `is_scoped`, `STAGE_DIRS`, `invalidate_dataset_sentinels`, and `shutil.rmtree`. For scoped stages, remove both the per-dataset sentinel (`invalidate_dataset_sentinels`) and the dataset's shard subfolder (`shutil.rmtree(stage_dir / key, ignore_errors=True)`) for each requested key. For corpus stages, remove the stage folder's data + sentinel. When the affected set includes `exact_dedup` or `sentence_dedup`, also `shutil.rmtree(paths.dedup_state_dir, ignore_errors=True)` (look up the existing `paths.dedup_state_dir` usage in `main`). Replace the two inline `--force` blocks in `main` with a single `if args.force: _apply_force(output_dir, stage=args.stage, run_all=args.all, dataset_keys=dataset_keys)` call placed where `output_dir`, `args`, and `dataset_keys` are known.

- [ ] **Step 4: Run it (passes)**

Run: `uv run pytest tests/data/test_to_pretrain.py -k force_subset -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "feat(curate): scoped/corpus-aware --force matrix"
```

---

## Phase D — End-to-end integration + verification

### Task D1: End-to-end incremental run on a tiny fixture corpus

**Files:**
- Create: `tests/data/test_to_pretrain_e2e.py`

This test exercises the real pipeline (datatrove) on a tiny corpus to prove the incremental contract. It is slower than the unit tests; keep the fixtures minimal (a handful of short docs per dataset).

- [ ] **Step 1: Write the test**

```python
"""End-to-end incremental curation tests on a tiny fixture corpus."""

import gzip
import json
from pathlib import Path

from scripts.data.to_pretrain import _curate  # see note below


def _write_extracted(input_dir: Path, key: str, texts: list[str]) -> None:
    """Write a minimal extracted <key>.jsonl for the convert stage."""
    input_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"text": t, "doc_id": f"{key}-{i}", "domain": "web",
                    "source": key}, ensure_ascii=False)
        for i, t in enumerate(texts)
    ]
    (input_dir / f"{key}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_dirs(stage_dir: Path) -> set[str]:
    """Return dataset subfolder names that contain shards under stage_dir."""
    return {
        p.name for p in stage_dir.iterdir()
        if p.is_dir() and any(p.glob("*.jsonl.gz"))
    } if stage_dir.exists() else set()


def test_subset_run_then_all_is_incremental(tmp_path: Path) -> None:
    """Subset run touches only its dataset; --all skips it and dedups all."""
    in_dir = tmp_path / "extracted"
    out_dir = tmp_path / "pretrain"
    # Two tiny Slovenian datasets so the language filter keeps them.
    _write_extracted(in_dir, "alfa", ["To je prvi slovenski dokument o vremenu."] * 3)
    _write_extracted(in_dir, "beta", ["Drugi slovenski dokument govori o hrani."] * 3)

    # 1) Subset run for 'alfa' only: scoped stages, no corpus stages.
    _curate(datasets=["alfa"], run_all=False, stage="all",
            input_dir=in_dir, output_dir=out_dir, force=False, workers=1)

    assert _dataset_dirs(out_dir / "01_language") == {"alfa"}
    assert _dataset_dirs(out_dir / "03_repetition") == {"alfa"}
    # Corpus stages did not run.
    assert not (out_dir / "04_1_dedup" / ".complete").exists()
    # Per-dataset sentinel exists for alfa/quality.
    assert (out_dir / "02_quality" / "alfa" / ".complete").exists()

    # 2) --all: 'alfa' scoped work is skipped (sentinel current), 'beta'
    #    is processed, then corpus dedup/stats run across both.
    _curate(datasets=[], run_all=True, stage="all",
            input_dir=in_dir, output_dir=out_dir, force=False, workers=1)

    assert _dataset_dirs(out_dir / "03_repetition") == {"alfa", "beta"}
    assert (out_dir / "04_1_dedup" / ".complete").exists()
    assert (out_dir / "05_statistics" / ".complete").exists()
```

Note on `_curate`: the test calls a thin, importable entry point rather than the argv-parsing `main()`. As part of this task, refactor `main()` so its body (after `parse_args`) is a function `_curate(*, datasets, run_all, stage, input_dir, output_dir, force, workers, pretrain_config=None, extract_config=None) -> None` that `main()` calls with values from `args`. This keeps `main()` a thin argv adapter and makes the pipeline testable without monkeypatching argv. If the existing `main()` already factors cleanly, expose the minimal seam needed; keep the public CLI behavior identical.

- [ ] **Step 2: Run it (fails until `_curate` exists / dispatch is wired)**

Run: `uv run pytest tests/data/test_to_pretrain_e2e.py -v`
Expected: FAIL initially (ImportError or assertion), then PASS after the refactor + correct dispatch.

- [ ] **Step 3: Refactor `main()` to delegate to `_curate(...)`**

Extract the post-`parse_args` body of `main()` into `_curate(...)` with the signature above (keyword-only). `main()` becomes: `args = parse_args(); _curate(datasets=args.datasets, run_all=args.all, stage=args.stage, input_dir=args.input_dir, output_dir=args.output_dir, force=args.force, workers=args.workers, pretrain_config=args.pretrain_config, extract_config=args.extract_config)`. Resolve config/dirs inside `_curate` as `main` did. Keep logging setup in `main`.

- [ ] **Step 4: Run it (passes)**

Run: `uv run pytest tests/data/test_to_pretrain_e2e.py -v`
Expected: PASS. If the language filter drops the fixture docs (empty `01_language`), make the fixture texts longer/clearly Slovenian, or set the test's pretrain config `language.minimum_relative_distance: 0.0` and `quality.min_doc_words` low via a temporary config file — but prefer fixing the fixture text first.

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py tests/data/test_to_pretrain_e2e.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain_e2e.py
git commit -m "test(curate): end-to-end incremental subset-then-all run"
```

---

### Task D2: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Full data test suite**

Run: `uv run pytest tests/data/ -q`
Expected: PASS (if `test_curate_runner.py` errors on a missing optional import in this env, re-run with `--ignore=tests/data/test_curate_runner.py` and confirm everything else passes; that error is pre-existing and unrelated).

- [ ] **Step 2: Project-wide docstring + lint sweep**

Run: `uv run ruff check --select D slm4ie/ scripts/ && uv run ruff check slm4ie/ scripts/ tests/`
Expected: clean.

- [ ] **Step 3: CLI smoke checks (no heavy compute)**

```bash
uv run python scripts/data/to_pretrain.py gigafida --stage exact_dedup   # expect: error, corpus-wide
uv run python scripts/data/to_pretrain.py --help                          # expect: prints
```
Expected: the first exits non-zero with the corpus-wide error; `--help` prints.

- [ ] **Step 4: Commit (if any cleanup was needed)**

```bash
git add -A && git commit -m "chore(curate): finalize incremental dataset-scoped curation"
```

---

## Self-review notes

- **Spec coverage:** taxonomy (A1) ✓; per-dataset sentinels + corpus sentinel (A2, C2) ✓; cascade class-awareness (A3) ✓; scoped reads via `input_override` + generalized view (B1, C1, C4) ✓; CLI matrix incl. corpus-`--stage` rejection and subset-stops-before-dedup (C3, C4) ✓; `--force` matrix (C5) ✓; backward-compat one-time reprocess is emergent (no code — scoped stages simply find no per-dataset sentinel) and is exercised implicitly by D1's first run; testing (A/B/C unit + D1 e2e) ✓.
- **Known seam:** D1 introduces `_curate(...)` as the testable entry point; C4/C5 build the dispatch it drives. If executed strictly in order, C4/C5 land the dispatch and D1 adds the seam + e2e proof — acceptable since each commits independently and the unit tests gate C-phase behavior before D1.
- **Per-dataset record counts:** the datatrove runner returns aggregate `(records_in, records_out)`; per-dataset sentinels store the aggregate (counts are advisory; the hash gates skipping). Documented in C4.
- **Datatrove reader attribute** for the B1 assertion may vary by version; the task says inspect and assert meaningfully rather than hard-coding a possibly-wrong attribute.
```