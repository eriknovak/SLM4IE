# Pretrain Per-Dataset Config Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let individual datasets deep-merge per-stage config overrides onto the global `pretrain.yaml` defaults, so e.g. a news corpus can relax one quality knob without forking the whole config or re-running the other datasets.

**Architecture:** Add an optional top-level `overrides:` block in `pretrain.yaml`, keyed by dataset → scoped-stage → knobs. A new `slm4ie/data/curate/overrides.py` validates the block (fail-fast) and resolves each dataset's effective stage config via the existing `_deep_merge`. The `to_pretrain.py` scoped-stage loop computes a **per-dataset effective config hash**, buckets the run's datasets by that hash, and runs one datatrove executor per bucket — so the all-defaults datasets stay one batch and each distinct override is isolated. Per-dataset sentinels already exist, so changing one dataset's override invalidates only that dataset and cascades through its downstream + the corpus stages.

**Tech Stack:** Python 3.13, uv, pytest, ruff (Google docstrings, `--select D`), datatrove, PyYAML.

---

## Background (read before starting)

- `to_pretrain.py` runs 8 stages. **Scoped** stages (`convert`, `language`, `spam`, `quality`, `repetition`) run per-dataset with per-dataset `.complete` sentinels; **corpus** stages (`exact_dedup`, `sentence_dedup`, `stats`) run over the whole corpus with one stage-level sentinel.
- Today, each scoped stage reads its single global slice `cfg.get(<stage>)`, computes one `current_hash`, and runs **all** `todo` datasets through one executor over a combined symlink view.
- A sentinel is "current" iff its recorded config hash matches a freshly computed hash. `convert` additionally compares a size+mtime input fingerprint (already implemented: `_convert_input_fingerprint`, `_convert_dataset_current`).
- `_deep_merge(base, override)` lives in `slm4ie/data/catalog.py` (lists/scalars replace, dicts recurse). It is already imported as a "private" symbol by `slm4ie/utils/config.py`, so cross-module use is an established pattern.
- `SCOPED_STAGES` is exported from `slm4ie/data/curate/stages.py`.

### Design decisions (locked)

1. Scope: pretrain only; keep helpers general for later reuse.
2. Location: top-level `overrides:` block in `pretrain.yaml`, keyed by dataset.
3. Execution: bucket `todo` by effective config hash; one executor per bucket.
4. Merge: partial deep-merge (inherit unspecified global knobs).
5. Overridable: scoped stages only; corpus stages / globals → hard error.
6. Validation: dataset keys ∈ roster, sections ∈ scoped stages, knobs ∈ each stage's known set → hard error.
7. Keying: dataset-key only (precedence global → dataset).
8. Cascade as today + log active overrides + store effective slice in the sentinel.

### Config shape (target)

```yaml
# configs/data/pretrain.yaml
quality:
  min_doc_words: 20
  max_ellipsis_lines_ratio: 0.3      # global default

overrides:                            # optional; absent ⇒ behaviour identical to today
  slovenian_news:
    quality:
      max_ellipsis_lines_ratio: 0.9   # deep-merged over global quality
```

---

## File Structure

- **Create** `slm4ie/data/curate/overrides.py` — knob registry, `OverrideConfigError`, `validate_overrides`, `effective_stage_config`. Self-contained; imports only `_deep_merge` and `SCOPED_STAGES` (no datatrove).
- **Create** `tests/data/test_curate_overrides.py` — unit tests for the registry, validation, and resolution.
- **Modify** `slm4ie/data/curate/__init__.py` — re-export the new public symbols (match how `sentinel` symbols are surfaced).
- **Modify** `scripts/data/to_pretrain.py` — load + validate `overrides`; refactor the scoped-stage loop to per-key hashing + config-bucketed execution.
- **Modify** `tests/data/test_to_pretrain.py` — bucketing + per-key currency tests.
- **Modify** `README.md` — document the `overrides:` block.

---

## Task 1: `overrides.py` — knob registry + validation + resolution

**Files:**
- Create: `slm4ie/data/curate/overrides.py`
- Test: `tests/data/test_curate_overrides.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_curate_overrides.py
"""Tests for the per-dataset pretrain override resolver/validator."""

from dataclasses import fields

import pytest

from slm4ie.data.curate.overrides import (
    STAGE_KNOBS,
    OverrideConfigError,
    effective_stage_config,
    validate_overrides,
)
from slm4ie.data.curate.pipeline import QualityConfig
from slm4ie.data.curate.spam import SpamConfig


def test_quality_knobs_match_dataclass() -> None:
    """The quality knob whitelist stays in lockstep with QualityConfig."""
    assert STAGE_KNOBS["quality"] == {f.name for f in fields(QualityConfig)}


def test_spam_knobs_match_dataclass() -> None:
    """The spam knob whitelist stays in lockstep with SpamConfig."""
    assert STAGE_KNOBS["spam"] == {f.name for f in fields(SpamConfig)}


def test_effective_config_deep_merges_over_global() -> None:
    """A dataset override patches only the named knobs; others inherit."""
    cfg = {"quality": {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}}
    overrides = {"slovenian_news": {"quality": {"max_ellipsis_lines_ratio": 0.9}}}
    eff = effective_stage_config(cfg, overrides, "slovenian_news", "quality")
    assert eff == {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.9}


def test_effective_config_no_override_returns_global_copy() -> None:
    """A dataset with no override yields a value equal to the global slice."""
    cfg = {"quality": {"min_doc_words": 20}}
    eff = effective_stage_config(cfg, {}, "kas", "quality")
    assert eff == {"min_doc_words": 20}
    # Must be a copy, not the same object (no mutation of cfg downstream).
    eff["min_doc_words"] = 999
    assert cfg["quality"]["min_doc_words"] == 20


def test_validate_accepts_empty_overrides() -> None:
    """Absent/empty overrides validate trivially."""
    validate_overrides({}, ["a", "b"])
    validate_overrides(None, ["a", "b"])


def test_validate_rejects_unknown_dataset() -> None:
    """A dataset key not in the roster is a hard error."""
    with pytest.raises(OverrideConfigError, match="unknown dataset"):
        validate_overrides({"typo_news": {"quality": {"min_doc_words": 5}}}, ["slovenian_news"])


def test_validate_rejects_corpus_stage() -> None:
    """Overriding a corpus stage is a hard error."""
    with pytest.raises(OverrideConfigError, match="scoped stages"):
        validate_overrides({"a": {"exact_dedup": {"precision": 32}}}, ["a"])


def test_validate_rejects_global_key() -> None:
    """Overriding a global key (e.g. stopwords) is a hard error."""
    with pytest.raises(OverrideConfigError, match="scoped stages"):
        validate_overrides({"a": {"stopwords": "sl"}}, ["a"])


def test_validate_rejects_unknown_knob() -> None:
    """A typo'd knob inside a valid stage is a hard error."""
    with pytest.raises(OverrideConfigError, match="unknown knob"):
        validate_overrides(
            {"a": {"quality": {"max_elipsis_lines_ratio": 0.9}}}, ["a"]
        )


def test_validate_accepts_valid_block() -> None:
    """A well-formed override block passes."""
    validate_overrides(
        {"slovenian_news": {"quality": {"max_ellipsis_lines_ratio": 0.9}, "language": {"mode": "tag"}}},
        ["slovenian_news", "kas"],
    )


def test_validate_rejects_non_mapping_section() -> None:
    """A dataset whose value is not a stage->knobs mapping is rejected."""
    with pytest.raises(OverrideConfigError, match="mapping"):
        validate_overrides({"a": ["quality"]}, ["a"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/data/test_curate_overrides.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm4ie.data.curate.overrides'`.

- [ ] **Step 3: Write the implementation**

```python
# slm4ie/data/curate/overrides.py
"""Per-dataset config overrides for the pretrain curation pipeline.

The pretrain pipeline is driven by a single `pretrain.yaml`. Each scoped
stage (convert, language, spam, quality, repetition) consumes the global
section named after it. An optional top-level `overrides:` block lets an
individual dataset deep-merge changes onto those defaults, so a corpus
can tweak one knob without forking the whole config or disturbing the
other datasets.

Only scoped stages are overridable. Corpus stages (exact_dedup,
sentence_dedup, stats) and global keys (input_dir, output_dir,
stopwords) operate over the whole corpus and reject per-dataset
overrides. The block is validated at load time against each stage's
known knob set so a typo fails fast instead of silently no-opping.
"""

from typing import Any, Dict, FrozenSet, List, Optional

from slm4ie.data.catalog import _deep_merge
from slm4ie.data.curate.stages import SCOPED_STAGES

#: Knobs each scoped stage accepts as an override. The quality and spam
#: sets mirror `QualityConfig` / `SpamConfig`; `test_curate_overrides.py`
#: asserts they stay in lockstep. `repetition` exposes no knobs today, so
#: it is effectively non-overridable until some are surfaced.
STAGE_KNOBS: Dict[str, FrozenSet[str]] = {
    "convert": frozenset(
        {
            "text_field",
            "id_field",
            "metadata_fields",
            "include_annotations",
            "max_shard_bytes",
        }
    ),
    "language": frozenset(
        {
            "targets",
            "candidates",
            "mode",
            "minimum_relative_distance",
            "low_accuracy",
            "max_chars",
        }
    ),
    "spam": frozenset(
        {
            "min_adult_hits",
            "min_spam_hits",
            "keep_fraction",
            "default_language",
            "url_blocklist",
            "use_ldnoobw",
            "model",
            "model_threshold",
        }
    ),
    "quality": frozenset(
        {
            "min_doc_words",
            "max_doc_words",
            "min_avg_word_length",
            "max_avg_word_length",
            "max_symbol_word_ratio",
            "max_bullet_lines_ratio",
            "max_ellipsis_lines_ratio",
            "max_non_alpha_words_ratio",
            "min_stop_words",
        }
    ),
    "repetition": frozenset(),
}


class OverrideConfigError(ValueError):
    """Raised when the `overrides:` block is malformed or out of bounds."""


def validate_overrides(
    overrides: Optional[Dict[str, Any]], roster: List[str]
) -> None:
    """Validate the `overrides:` block against the dataset roster.

    Args:
        overrides: The parsed `overrides:` mapping (dataset -> stage ->
            knobs), or `None`/empty when absent.
        roster: Every dataset key declared in `extract.yaml`.

    Raises:
        OverrideConfigError: If a dataset key is not in `roster`, a
            section is not a scoped stage, a knob is unknown for its
            stage, or a section/knob value has the wrong shape.
    """
    if not overrides:
        return
    roster_set = set(roster)
    for dataset, sections in overrides.items():
        if dataset not in roster_set:
            raise OverrideConfigError(
                f"overrides: unknown dataset '{dataset}' "
                f"(not declared in extract.yaml)"
            )
        if not isinstance(sections, dict):
            raise OverrideConfigError(
                f"overrides.{dataset}: expected a mapping of stage -> knobs"
            )
        for stage, knobs in sections.items():
            if stage not in SCOPED_STAGES:
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: only scoped stages "
                    f"{sorted(SCOPED_STAGES)} may be overridden"
                )
            if not isinstance(knobs, dict):
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: expected a mapping of knob -> value"
                )
            unknown = set(knobs) - STAGE_KNOBS[stage]
            if unknown:
                raise OverrideConfigError(
                    f"overrides.{dataset}.{stage}: unknown knob(s) "
                    f"{sorted(unknown)}; allowed: {sorted(STAGE_KNOBS[stage])}"
                )


def effective_stage_config(
    cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    dataset: str,
    stage: str,
) -> Dict[str, Any]:
    """Return *dataset*'s effective config for *stage*.

    Deep-merges the dataset's stage override (if any) over the global
    stage section. A dataset with no override yields a fresh copy of the
    global slice, byte-identical in content to the pre-overrides
    behaviour.

    Args:
        cfg: Parsed `pretrain.yaml`.
        overrides: The `overrides:` mapping, or `None`/empty.
        dataset: Dataset key.
        stage: Scoped stage name.

    Returns:
        A new mapping of the effective knobs for `(dataset, stage)`.
    """
    base = dict(cfg.get(stage) or {})
    override = ((overrides or {}).get(dataset) or {}).get(stage) or {}
    return _deep_merge(base, override)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/data/test_curate_overrides.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check --select D slm4ie/data/curate/overrides.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/curate/overrides.py tests/data/test_curate_overrides.py
git commit -m "feat(curate): add per-dataset pretrain override resolver + validator"
```

---

## Task 2: Re-export overrides symbols from the curate package

**Files:**
- Modify: `slm4ie/data/curate/__init__.py`

- [ ] **Step 1: Inspect the current exports**

Run: `sed -n '1,80p' slm4ie/data/curate/__init__.py`
Expected: a module that imports from `.sentinel`, `.stages`, etc. and lists names in `__all__`.

- [ ] **Step 2: Add the overrides re-exports**

Add an import block alongside the existing ones (match local style/ordering), e.g.:

```python
from slm4ie.data.curate.overrides import (
    OverrideConfigError,
    effective_stage_config,
    validate_overrides,
)
```

And add `"OverrideConfigError"`, `"effective_stage_config"`, `"validate_overrides"` to `__all__` if the module defines one. If `__init__.py` does **not** aggregate exports (verify in Step 1), skip this task entirely and import directly from `slm4ie.data.curate.overrides` in Task 3.

- [ ] **Step 3: Verify import resolves**

Run: `uv run python -c "from slm4ie.data.curate import validate_overrides, effective_stage_config; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add slm4ie/data/curate/__init__.py
git commit -m "chore(curate): export override resolver/validator from package"
```

---

## Task 3: Load + validate the overrides block in `to_pretrain.py`

**Files:**
- Modify: `scripts/data/to_pretrain.py` (imports near line 78–90; `run_pipeline` near line 979)
- Test: `tests/data/test_to_pretrain.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/data/test_to_pretrain.py
def test_run_pipeline_rejects_bad_override(tmp_path, monkeypatch) -> None:
    """run_pipeline fails fast when overrides name an unknown knob."""
    import yaml

    from scripts.data import to_pretrain
    from slm4ie.data.curate.overrides import OverrideConfigError

    cfgs = tmp_path / "configs" / "data"
    cfgs.mkdir(parents=True)
    (cfgs / "extract.yaml").write_text(
        yaml.safe_dump({"datasets": {"news": {"extractor": "jsonl", "domain": "news"}}})
    )
    (cfgs / "pretrain.yaml").write_text(
        yaml.safe_dump(
            {
                "input_dir": str(tmp_path / "in"),
                "output_dir": str(tmp_path / "out"),
                "quality": {"min_doc_words": 20},
                "overrides": {"news": {"quality": {"max_elipsis_lines_ratio": 0.9}}},
            }
        )
    )
    with pytest.raises(OverrideConfigError, match="unknown knob"):
        to_pretrain.run_pipeline(
            datasets=["news"],
            run_all=False,
            pretrain_config=cfgs / "pretrain.yaml",
            extract_config=cfgs / "extract.yaml",
        )
```

NOTE: if `run_pipeline`'s signature differs (verify with `grep -n "def run_pipeline" scripts/data/to_pretrain.py`), adapt the call's keyword args to match. The assertion (raises `OverrideConfigError` before any stage runs) is the contract.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/data/test_to_pretrain.py::test_run_pipeline_rejects_bad_override -q`
Expected: FAIL (no validation yet — either no error raised, or a different error).

- [ ] **Step 3: Add the import**

In `scripts/data/to_pretrain.py`, add to the curate imports (near the existing `from slm4ie.data.curate.sentinel import (...)` block):

```python
from slm4ie.data.curate.overrides import effective_stage_config, validate_overrides
```

- [ ] **Step 4: Load and validate overrides in `run_pipeline`**

Immediately after `cfg = _load_yaml(pretrain_path)` (currently line 979), insert:

```python
    overrides = cfg.get("overrides") or {}
    validate_overrides(overrides, _list_datasets(extract_path))
```

(`_list_datasets(extract_path)` returns the full roster regardless of any positional subset, so an override for an out-of-run dataset is still validated.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/data/test_to_pretrain.py::test_run_pipeline_rejects_bad_override -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "feat(to_pretrain): load + fail-fast validate the overrides block"
```

---

## Task 4: Per-key hashing + config-bucketed execution in the scoped loop

**Files:**
- Modify: `scripts/data/to_pretrain.py` — the `if is_scoped(stage_name):` block inside the `for stage_name in requested_stages:` loop (currently lines ~1024–1125). Do **not** touch the corpus `else:` branch; it keeps using `slice_` / `current_hash` (lines 1019–1022), which remain correct for whole-corpus stages.
- Test: `tests/data/test_to_pretrain.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/data/test_to_pretrain.py
def test_bucket_keys_by_effective_hash_groups_shared_configs(tmp_path) -> None:
    """Datasets sharing an effective config land in one bucket; overrides split out."""
    from scripts.data.to_pretrain import _bucket_keys_by_effective_hash
    from slm4ie.data.curate import config_hash

    cfg = {"quality": {"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}}
    overrides = {"news": {"quality": {"max_ellipsis_lines_ratio": 0.9}}}
    extra = b""
    buckets = _bucket_keys_by_effective_hash(
        ["a", "b", "news"], "quality", cfg, overrides, extra
    )
    # Two groups: {a, b} default, {news} override.
    groups = sorted(sorted(v) for v in buckets.values())
    assert groups == [["a", "b"], ["news"]]
    # The default bucket's hash equals the plain global-slice hash (rollout-safe).
    default_hash = config_hash({"min_doc_words": 20, "max_ellipsis_lines_ratio": 0.3}, extra=extra)
    assert default_hash in buckets
    assert sorted(buckets[default_hash]) == ["a", "b"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/data/test_to_pretrain.py::test_bucket_keys_by_effective_hash_groups_shared_configs -q`
Expected: FAIL with `ImportError: cannot import name '_bucket_keys_by_effective_hash'`.

- [ ] **Step 3: Add the bucketing helper**

Add near `_convert_dataset_current` (before `_dataset_keys_payload`) in `scripts/data/to_pretrain.py`:

```python
def _bucket_keys_by_effective_hash(
    keys: List[str],
    stage: str,
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    extra: bytes,
) -> Dict[str, List[str]]:
    """Group *keys* by their effective-config hash for *stage*.

    Datasets that resolve to the same effective stage config share a
    hash and run together in one executor; each distinct override forms
    its own bucket. A dataset with no override hashes identically to the
    plain global slice, so the all-defaults case stays a single bucket.

    Args:
        keys: Dataset keys to bucket (in run order).
        stage: Scoped stage name.
        cfg: Parsed `pretrain.yaml`.
        overrides: The `overrides:` mapping.
        extra: Stage-level extra bytes folded into the hash (stopwords /
            spam lexicon / roster), identical across keys of a stage.

    Returns:
        Mapping of effective-config hash to the keys sharing it.
        Iteration order preserves first appearance of each bucket.
    """
    buckets: Dict[str, List[str]] = {}
    for key in keys:
        slice_ = effective_stage_config(cfg, overrides, key, stage)
        h = config_hash(slice_, extra=extra)
        buckets.setdefault(h, []).append(key)
    return buckets
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `uv run pytest tests/data/test_to_pretrain.py::test_bucket_keys_by_effective_hash_groups_shared_configs -q`
Expected: PASS.

- [ ] **Step 5: Replace the scoped-stage block body**

Replace the entire body of `if is_scoped(stage_name):` (from `todo = [` through the end of the `for key in todo:` sentinel-write loop and its trailing `logger.info("[%s] done ...")`, i.e. lines ~1024–1125) with:

```python
        if is_scoped(stage_name):
            # Convert reads the extracted tier directly, so its currency
            # also depends on the source file's size+mtime fingerprint;
            # later scoped stages read regenerated upstream output. Each
            # key is hashed against its EFFECTIVE (override-merged) config.
            def _effective_hash(k: str) -> str:
                return config_hash(
                    effective_stage_config(cfg, overrides, k, stage_name),
                    extra=extra,
                )

            def _is_current(k: str) -> bool:
                expected = _effective_hash(k)
                if stage_name == "convert":
                    inc = bool(
                        effective_stage_config(
                            cfg, overrides, k, "convert"
                        ).get("include_annotations", False)
                    )
                    return _convert_dataset_current(
                        stage_folder, k, expected, paths.input_folder, inc
                    )
                return dataset_sentinel_is_current(stage_folder, k, expected)

            todo = [
                k for k in dataset_keys
                if k in force_keys or not _is_current(k)
            ]
            if not todo:
                logger.info("[%s] all requested datasets current; skipping.", stage_name)
                continue

            # Drop datasets with no upstream output (declared but never
            # downloaded, or fully filtered upstream). convert reads the
            # extraction tier directly and tolerates missing input.
            upstream = upstream_stage(stage_name)
            up_dir = paths.stage_dir(upstream) if upstream is not None else None
            if up_dir is not None:
                missing = [k for k in todo if not _has_stage_output(up_dir, k)]
                if missing:
                    logger.info(
                        "[%s] skipping %d dataset(s) with no upstream output: %s",
                        stage_name, len(missing), ", ".join(missing),
                    )
                    todo = [k for k in todo if k not in missing]
                if not todo:
                    logger.info("[%s] no datasets with upstream output; skipping.", stage_name)
                    continue

            # Invalidate downstream + force later scoped stages to re-run
            # these keys (idempotent across stages).
            cascade_invalidate_scoped(output_dir, stage_name, todo)
            force_keys.update(todo)

            # Bucket by effective config so datasets sharing a config run
            # in one executor and each override is isolated.
            buckets = _bucket_keys_by_effective_hash(
                todo, stage_name, cfg, overrides, extra
            )
            logger.info(
                "[%s] %d dataset(s) in %d config group(s)",
                stage_name, len(todo), len(buckets),
            )

            for bucket_hash, bucket_keys in buckets.items():
                effective = effective_stage_config(
                    cfg, overrides, bucket_keys[0], stage_name
                )
                overridden = [
                    k for k in bucket_keys
                    if (overrides.get(k) or {}).get(stage_name)
                ]
                if overridden:
                    logger.info(
                        "[%s] override group %s <- %s",
                        stage_name, overridden, effective,
                    )

                if stage_name == "convert":
                    n_datasets, input_bytes = _extracted_input_summary(
                        paths.input_folder, bucket_keys
                    )
                    logger.info(
                        "[convert] starting (%d dataset(s), %s)",
                        n_datasets, _human_bytes(input_bytes),
                    )
                else:
                    logger.info(
                        "[%s] starting%s",
                        stage_name, _starting_input_hint(paths, stage_name),
                    )

                view = (
                    _filter_stage_subset(up_dir, bucket_keys)
                    if up_dir is not None
                    else None
                )
                # Hand the bucket's effective slice to the runner by
                # swapping just this stage's section in a shallow cfg copy;
                # _stage_runner reads cfg.get(stage_name).
                bucket_cfg = {**cfg, stage_name: effective}
                try:
                    runner = _stage_runner(
                        stage_name,
                        paths,
                        bucket_cfg,
                        workers,
                        stopwords,
                        spam_assets,
                        dataset_keys=bucket_keys,
                        input_view=view,
                        log_dir=convert_log_dir if stage_name == "convert" else None,
                    )
                    records_in, records_out = runner()
                finally:
                    if view is not None:
                        shutil.rmtree(view, ignore_errors=True)

                for key in bucket_keys:
                    fingerprint = (
                        _convert_input_fingerprint(
                            paths.input_folder,
                            key,
                            bool(effective.get("include_annotations", False)),
                        )
                        if stage_name == "convert"
                        else None
                    )
                    write_dataset_sentinel(
                        stage_folder,
                        key,
                        config_slice=effective,
                        config_hash_value=bucket_hash,
                        records_in=records_in,
                        records_out=records_out,
                        input_fingerprint=fingerprint,
                    )
                logger.info(
                    "[%s] done for %d dataset(s) (records_in=%d, records_out=%d)",
                    stage_name, len(bucket_keys), records_in, records_out,
                )
```

NOTE: this replaces the prior single-executor convert special-case (the `convert_include_annotations` local and the single `write_dataset_sentinel` loop) introduced by the input-fingerprint change. Effective per-key `include_annotations` now comes from `effective`.

- [ ] **Step 6: Run the full to_pretrain + curate suites**

Run: `uv run pytest tests/data/test_to_pretrain.py tests/data/test_to_pretrain_e2e.py tests/data/test_curate_runner.py -q`
Expected: PASS (the e2e test exercises the real scoped→corpus flow with no overrides; it must still pass, proving the no-override path is unchanged).

- [ ] **Step 7: Lint**

Run: `uv run ruff check --select D scripts/data/to_pretrain.py`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add scripts/data/to_pretrain.py tests/data/test_to_pretrain.py
git commit -m "feat(to_pretrain): per-dataset config overrides via effective-hash buckets"
```

---

## Task 5: End-to-end test — an override re-runs only its dataset

**Files:**
- Test: `tests/data/test_to_pretrain_e2e.py` (append)

- [ ] **Step 1: Inspect the existing e2e fixtures**

Run: `sed -n '1,80p' tests/data/test_to_pretrain_e2e.py`
Expected: see how the test builds a tiny `extracted/` tree + `pretrain.yaml` and calls `run_pipeline`. Reuse that fixture/helper style verbatim in the next step (do not invent a new harness).

- [ ] **Step 2: Write the failing test**

Using the existing e2e helpers (adapt names to what Step 1 shows), add a test that:
1. Builds a 2-dataset corpus (`a`, `b`) and runs scoped stages once (no overrides) → record both datasets' `03_quality/.../.complete` `config_hash`.
2. Adds `overrides: {b: {quality: {min_doc_words: 999}}}` to the `pretrain.yaml` and re-runs the `quality` stage.
3. Asserts dataset `a`'s quality sentinel hash is **unchanged** (not re-run) and dataset `b`'s quality sentinel hash **changed** (re-run with the override).

```python
def test_override_reruns_only_target_dataset(tmp_path) -> None:
    """Adding an override re-runs only that dataset's scoped stage."""
    from slm4ie.data.curate.sentinel import read_sentinel
    # ... build corpus + pretrain.yaml via the existing e2e helper ...
    # run quality once (no overrides)
    # hash_a0 = read_sentinel(out/"03_quality"/"a").config_hash
    # hash_b0 = read_sentinel(out/"03_quality"/"b").config_hash
    # rewrite pretrain.yaml adding overrides.b.quality.min_doc_words = 999
    # run quality again
    # hash_a1 = read_sentinel(out/"03_quality"/"a").config_hash
    # hash_b1 = read_sentinel(out/"03_quality"/"b").config_hash
    # assert hash_a1 == hash_a0
    # assert hash_b1 != hash_b0
    ...
```

Fill the `...` with the concrete fixture calls discovered in Step 1 — no placeholders in the committed test.

- [ ] **Step 3: Run to verify it fails, then (it should already pass given Task 4) confirm green**

Run: `uv run pytest tests/data/test_to_pretrain_e2e.py::test_override_reruns_only_target_dataset -q`
Expected: PASS (Task 4 implements the behavior; this test is the regression guard). If it FAILS, debug Task 4 before proceeding.

- [ ] **Step 4: Full suite**

Run: `uv run pytest -q`
Expected: PASS (was 571 before this plan; now 571 + new tests).

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_to_pretrain_e2e.py
git commit -m "test(to_pretrain): override re-runs only the targeted dataset"
```

---

## Task 6: Document the overrides block in the README

**Files:**
- Modify: `README.md` (the pretrain sentinel section, just after the "Each stage's sentinel hash covers…" paragraph near line 271)

- [ ] **Step 1: Add the documentation paragraph**

Insert after the sentinel-hash note (and after the input-fingerprint note added earlier):

```markdown
> **Note — per-dataset overrides.** An optional top-level `overrides:` block in `pretrain.yaml` lets a single dataset patch any **scoped** stage's config without forking the file. It is keyed by dataset, then by stage, and deep-merges over the global section (unspecified knobs inherit the default):
>
> ```yaml
> overrides:
>   slovenian_news:
>     quality:
>       max_ellipsis_lines_ratio: 0.9
> ```
>
> Only scoped stages (`convert`, `language`, `spam`, `quality`, `repetition`) are overridable — naming a corpus stage (`exact_dedup`/`sentence_dedup`/`stats`) or a global key (`input_dir`/`output_dir`/`stopwords`), an unknown dataset, or an unknown knob is a hard error at load. Datasets are bucketed by their effective config: those sharing one run together in a single executor, so only overridden datasets pay isolation cost. Each dataset's sentinel hashes its effective (merged) config, so adding/editing an override re-runs only that dataset's stage plus its downstream and the corpus dedup/stats; a dataset with no override is byte-identical to before and never re-runs spuriously.
```

- [ ] **Step 2: Verify the markdown renders sanely**

Run: `sed -n '270,300p' README.md`
Expected: the new block is well-formed and adjacent to the sentinel discussion.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document per-dataset pretrain config overrides"
```

---

## Task 7 (GATED — only after the in-flight `--all` run finishes): apply the news ellipsis override

> Do NOT start this task while the current `to_pretrain.py --all` run is active (it would contend on the same output tree). Confirm the run is done first.

**Files:**
- Modify: `configs/data/pretrain.yaml`

- [ ] **Step 1: Confirm no run is active**

Run: `pgrep -af "to_pretrain" | grep -v pgrep || echo "no run active"`
Expected: `no run active`.

- [ ] **Step 2: Add the override**

Append to `configs/data/pretrain.yaml`:

```yaml
overrides:
  slovenian_news:
    quality:
      # News prose (interviews/Q&A, tabloid style) uses "…" mid-article;
      # the Gopher web-crawl default of 0.3 dropped ~928K real articles.
      max_ellipsis_lines_ratio: 0.9
```

- [ ] **Step 3: Dry check the resolution (no pipeline run)**

Run:
```bash
uv run python -c "
import yaml
from pathlib import Path
from slm4ie.data.curate.overrides import effective_stage_config, validate_overrides
cfg = yaml.safe_load(Path('configs/data/pretrain.yaml').read_text())
ov = cfg.get('overrides') or {}
validate_overrides(ov, ['slovenian_news'])
print(effective_stage_config(cfg, ov, 'slovenian_news', 'quality'))
"
```
Expected: prints the merged quality dict with `max_ellipsis_lines_ratio: 0.9` and all other global quality knobs intact. No exception.

- [ ] **Step 4: Re-run the affected pipeline** (in tmux `slm4ie`, per project workflow)

Run (new pane): `uv run python scripts/data/to_pretrain.py --all --max-workers 12`
Expected log: `[quality] override group ['slovenian_news'] <- {...max_ellipsis_lines_ratio: 0.9...}`; only `slovenian_news` re-runs quality + repetition; corpus `exact_dedup`/`sentence_dedup`/`stats` re-run; the other 24 datasets' quality/repetition are skipped as current.

- [ ] **Step 5: Verify recovery**

After completion, compare `06_statistics/per_dataset/slovenian_news.json` doc count to the prior run — expect a large increase (most of the ~928K previously dropped now retained).

- [ ] **Step 6: Commit**

```bash
git add configs/data/pretrain.yaml
git commit -m "feat(pretrain): relax ellipsis filter for slovenian_news via override"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Decisions 1–8 map to Tasks 1 (validate/resolve/merge/scope), 3 (load+validate), 4 (bucketed execution + per-key hash + cascade + audit logging + effective slice in sentinel), 6 (docs). The motivating ellipsis case is Task 7.
- **Rollout safety:** Task 4's no-override path must hash identically to today — the e2e regression test (Task 5) and the unchanged `test_to_pretrain_e2e` guard this. If `test_to_pretrain_e2e` fails after Task 4, the refactor changed default behavior — fix before continuing.
- **Type consistency:** `effective_stage_config(cfg, overrides, dataset, stage)`, `validate_overrides(overrides, roster)`, `_bucket_keys_by_effective_hash(keys, stage, cfg, overrides, extra)`, `STAGE_KNOBS` — names used identically across tasks.
- **Interaction with the input-fingerprint feature:** convert currency = effective-config hash match AND fingerprint match; `include_annotations` is resolved per-key from the effective convert slice. Verify the convert branch in Task 4 reads `effective.get("include_annotations")`, not a global.
```
