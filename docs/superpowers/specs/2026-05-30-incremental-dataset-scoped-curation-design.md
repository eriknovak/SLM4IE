# Incremental Dataset-Scoped Curation — Design

**Status:** Approved (brainstorm)
**Date:** 2026-05-30
**Component:** `scripts/data/to_pretrain.py` + `slm4ie/data/curate/` (pipeline, sentinel, stages)

## Problem

The pretraining pipeline (`to_pretrain.py`, datatrove-backed) cannot add a single
dataset to an existing corpus without reprocessing everything. Two limitations
combine to force a full rebuild:

1. **Only `convert` and `language` can be scoped to a dataset subset.** `language`
   accepts an `input_override` (a symlink view of `00_convert/` restricted to the
   requested keys, built by `_filter_convert_subset`). But `quality` reads the full
   `01_language/` directory and `repetition` reads the full `02_quality/` directory —
   neither is dataset-scoped. From `quality` onward, every stage reprocesses the
   entire upstream directory.

2. **The sentinel hash folds the whole dataset roster into every stage**
   (`_stage_extra` returns `dataset_keys_bytes` for all stages). Adding a dataset
   changes the sorted key list, so *every* stage's sentinel goes stale and the
   whole pipeline re-runs.

Concretely: 19 datasets are already built through `04_1_dedup`. Adding `gigafida`
today means re-running `language`→`stats` over the full corpus (including the huge
web sets culturax/c4/fineweb2/hplt) — many hours to day-scale.

## Goal

Enable true incremental curation: run the per-document stages for just the new
dataset(s), leaving existing datasets' per-document outputs intact, then run the
corpus-wide dedup/stats across the whole corpus.

Target workflow:

```bash
# 1. Bring gigafida through the per-document stages only (incremental).
uv run python scripts/data/to_pretrain.py gigafida

# 2. Dedup + stats across the full corpus (now including gigafida).
uv run python scripts/data/to_pretrain.py --all
```

Step 2's per-document stages skip every dataset whose work is already current
(the 19 + gigafida), so it jumps straight to the corpus stages.

## Non-goals

- No change to datatrove stage internals (filters, dedup algorithm).
- No automatic migration of the existing 19-dataset sentinels (see Backward
  Compatibility — a one-time reprocess is accepted).
- No change to the `download`/`extract`/task-converter pipelines.

## Design

### 1. Stage taxonomy

Two stage classes, declared once in `slm4ie/data/curate/stages.py` (derived from
`STAGE_NAMES`, not hard-coded elsewhere):

- **Scoped stages** — `convert`, `language`, `quality`, `repetition`. Process only
  the requested dataset keys; write to canonical `<stage_dir>/<dataset>/`; tracked
  by **per-dataset sentinels**.
- **Corpus stages** — `exact_dedup`, `sentence_dedup`, `stats`. Always read the
  entire upstream directory (all datasets); tracked by a **corpus-level sentinel**;
  run **only under `--all`**.

```python
SCOPED_STAGES: Tuple[str, ...] = ("convert", "language", "quality", "repetition")
CORPUS_STAGES: Tuple[str, ...] = ("exact_dedup", "sentence_dedup", "stats")
# Invariant: SCOPED_STAGES + CORPUS_STAGES == STAGE_NAMES
```

### 2. CLI semantics

| Invocation | Scoped stages | Corpus stages |
| --- | --- | --- |
| `to_pretrain.py KEY…` | run for `KEY…` (skip per-dataset-current) | **not run** |
| `to_pretrain.py --all` | run for all keys (skip per-dataset-current) | run if corpus sentinel stale |
| `to_pretrain.py KEY… --stage <scoped>` | run that stage for `KEY…` | — |
| `to_pretrain.py --all --stage <scoped>` | run that stage for all keys | — |
| `to_pretrain.py --all --stage <corpus>` | — | run that corpus stage (+ cascade) |
| `to_pretrain.py KEY… --stage <corpus>` | **error**: "dedup/stats are corpus-wide; use --all" | — |

- In a subset run, `--stage all` resolves to the scoped stages only (convert →
  repetition) and stops. Under `--all`, `--stage all` is all seven stages.
- `requested_stages` is computed from `(args.all, args.stage)`: for a subset run
  with `--stage all`, it is `SCOPED_STAGES`; for `--all --stage all`, it is
  `STAGE_NAMES`.

### 3. Scoped reads (Approach A — generalized symlink view)

- Generalize `_filter_convert_subset(convert_dir, keys)` →
  `_filter_stage_subset(stage_dir, keys)`: builds a tempdir of symlinks to the
  requested `<stage_dir>/<key>/*.jsonl.gz` shards. Same missing-key
  `FileNotFoundError` behavior as today.
- Add `input_override: Optional[Path] = None` to `build_quality_executors` and
  `build_repetition_executors` (mirroring `build_language_executors`). When set,
  the stage reads from the view instead of `paths.stage_dir(<upstream>)`.
- `_stage_runner` builds each scoped stage's input view from its **upstream** stage
  dir for the requested keys and passes it as `input_override`:
  - `convert` — already scoped by `dataset_keys` inside `run_convert_stage`; no view
    needed.
  - `language` — view of `00_convert/` (unchanged behavior, now via the renamed
    helper).
  - `quality` — view of `01_language/`.
  - `repetition` — view of `02_quality/`.
- Writes always land in canonical `<stage_dir>/<dataset>/` because datatrove's
  writer routes by `${dataset}` (per the pipeline I/O contract). A scoped run
  therefore only touches the processed datasets' subfolders; existing datasets'
  subfolders are untouched.
- All views are created inside `_stage_runner`'s run and removed in a `finally`
  (replacing the single `subset_holder` lifecycle in `main`).

### 4. Sentinel & cascade model

**Scoped stages — per-dataset sentinels.**
- Location: `<stage_dir>/<dataset>/.complete` (JSON, same schema as today).
- Hash: `config_hash(slice_)` over the stage's config slice **only** — the dataset
  roster is dropped. Stopword-file contents remain in the `quality` hash.
- New sentinel helpers in `slm4ie/data/curate/sentinel.py`:
  - `write_dataset_sentinel(stage_dir, dataset, *, config_slice, config_hash_value, records_in, records_out)`
  - `dataset_sentinel_is_current(stage_dir, dataset, expected_hash) -> bool`
- A scoped stage iterates the requested keys, skips any whose per-dataset sentinel
  is current, runs the rest (via one filtered-view executor over the not-current
  keys), then writes a per-dataset sentinel for each key it processed.

**Corpus stages — corpus-level sentinel.**
- Location: `<stage_dir>/.complete` (unchanged).
- Hash: `config_hash(slice_, extra=roster_bytes [+ stopword bytes for stats])`. The
  roster is the full sorted key list from `extract.yaml`, so adding/removing any
  dataset invalidates the corpus stages.

**`_stage_extra` change.** Today it folds the roster into every stage. New behavior:
fold the roster only for `CORPUS_STAGES`; scoped stages get config-slice-only hashes
(plus stopword bytes for `quality`).

**Cascade.**
- `cascade_from(stage)` is unchanged (downstream set in execution order).
- Editing a scoped stage's config invalidates that stage's per-dataset sentinels
  (for the keys in play) and cascades downstream — which includes the corpus stages,
  so dedup/stats re-run.
- Editing a corpus stage's config invalidates that corpus stage and downstream
  corpus stages.
- `cascade_invalidate` becomes class-aware: for scoped stages it removes the
  per-dataset sentinels (for the relevant keys); for corpus stages it removes the
  stage-level sentinel. A `keys` argument scopes scoped-stage invalidation.

### 5. `--force` matrix

| Invocation | Effect |
| --- | --- |
| `--force KEY… --stage <scoped>` | drop `KEY…`'s per-dataset sentinels + subfolders for that scoped stage and its scoped-downstream; corpus sentinels also dropped |
| `--force --all --stage <scoped>` | as above for all keys |
| `--force --all --stage <corpus>` | drop that corpus stage's sentinel + downstream corpus sentinels and data folders (incl. `_dedup_state` when dedup is affected) |
| `--force KEY…` (no `--stage`) | drop `KEY…`'s scoped-stage per-dataset sentinels + subfolders across all scoped stages; drop corpus sentinels |
| `--force --all` (no `--stage`) | nuke `output_dir` (today's behavior) |

### 6. Backward compatibility

No migration (chosen). The existing 19-dataset tree has *stage-level* `.complete`
files for the scoped stages (keyed to the old 19-roster hash). The new scoped
stages look for *per-dataset* sentinels, which don't exist, so the first `--all`
re-runs the four scoped stages once for all 20 datasets, writes per-dataset
sentinels, then runs dedup/stats. This one-time reprocess also applies the loosened
`quality` thresholds (currently uncommitted in `pretrain.yaml`) to the existing
corpus — desirable. The old stage-level sentinel files for scoped stages are
ignored and removed by the run. After the first `--all`, curation is fully
incremental.

## Affected files

- `slm4ie/data/curate/stages.py` — add `SCOPED_STAGES` / `CORPUS_STAGES`; a
  predicate (`is_scoped(stage)`).
- `slm4ie/data/curate/sentinel.py` — add `write_dataset_sentinel`,
  `dataset_sentinel_is_current`; make `cascade_invalidate` class-aware with an
  optional `keys` argument.
- `slm4ie/data/curate/pipeline.py` — add `input_override` to
  `build_quality_executors` and `build_repetition_executors`.
- `scripts/data/to_pretrain.py` — rename/generalize `_filter_convert_subset` →
  `_filter_stage_subset`; rewrite the stage loop in `main` to: compute
  `requested_stages` from `(all, stage)`; for scoped stages, build the upstream
  view, skip per-dataset-current keys, run, write per-dataset sentinels; for corpus
  stages, gate on `--all`, use corpus sentinels; update `_stage_extra` to fold the
  roster only for corpus stages; update `--force` handling per the matrix; reject
  `--stage <corpus>` with positional keys.

## Testing

**Unit (`tests/data/`):**
- `write_dataset_sentinel` / `dataset_sentinel_is_current`: round-trip; stale on
  hash change; missing → not current.
- `_filter_stage_subset`: multi-key view; missing/empty key raises
  `FileNotFoundError`.
- `_stage_extra`: roster folded for corpus stages, excluded for scoped stages;
  stopword bytes still folded for `quality`/`stats`.
- `cascade_invalidate`: scoped stage removes per-dataset sentinels for given keys;
  corpus stage removes stage-level sentinel; downstream cascade spans the boundary.

**Integration (tiny fixture corpus, real datatrove, 2–3 tiny datasets under
`tests/data/` fixtures):**
- Subset run for dataset `a`: only `a`'s subfolders + per-dataset sentinels appear
  under each scoped stage; dataset `b`'s outputs are untouched; no corpus-stage
  output produced.
- `--all` after the subset run: scoped stages skip `a` (sentinel current), process
  `b`, then dedup/stats run and their input spans both `a` and `b`.
- `--all --stage exact_dedup` with positional keys → exits with the corpus-stage
  error.

## Open questions

None. Design approved in brainstorm.
