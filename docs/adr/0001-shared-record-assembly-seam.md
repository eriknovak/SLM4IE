# 1. Shared record→Document assembly seam in extractors

Status: Accepted

## Context

Four extractors (`json`, `jsonl`, `coleslaw`, `huggingface`) each turn a raw
record into a `Document`. Two steps of that assembly are format-independent:
probing a `doc_id` from candidate fields, and projecting the remaining record
fields into `Document.metadata` (reserved-field exclusion plus `None`-dropping).

All four re-implemented these steps inline, and the copies had drifted:

- PR #33's `None`-drop fix landed in `json` but not `jsonl`, so the two produced
  different metadata for the same record.
- `json`/`jsonl` returned the raw `id_field` value as `doc_id`; `coleslaw`/
  `huggingface` `str()`-coerced it. Ids were therefore typed inconsistently.

With the policy copied four ways, any future fix has to be applied four times,
and divergence is the default outcome.

## Decision

Introduce one shared module, `slm4ie/data/extractors/assembly.py`, exposing two
granular helpers that each extractor composes into its own `Document(...)`:

- `probe_doc_id(record, keys)` — walk an ordered list of candidate keys and
  return the first present, non-empty value, `str()`-coerced; else `None`.
- `project_metadata(record, *, exclude, whitelist, value_transform)` — keep a
  whitelist of present fields or every field not in `exclude`, always dropping
  `None` values, applying an optional `value_transform`, preserving order.

Only the doc_id probe and metadata projection are shared. Text selection,
annotation parsing, and the `huggingface` positional-id fallback are genuinely
format-specific and stay in each extractor. The helpers are two functions, not a
single `assemble_document`, so each extractor keeps ownership of its field
mapping.

Behavior converges the `json` way: `None` values are dropped everywhere
(including `jsonl`), and doc_ids are `str()`-coerced everywhere. Extractors pass
their own candidate keys and `exclude`/`whitelist`; `huggingface` opts into
`value_transform=_to_jsonable`, and `coleslaw` merges its injected `subcorpus`
around the shared projection.

## Consequences

- Metadata and doc_id policy now lives in one module; a fix applies once.
- `json` and `jsonl` produce identical metadata for the same record by
  construction, guarded by a dedicated test that would have caught the PR #33
  divergence.
- `jsonl` now drops `None` from metadata and `str()`-coerces its `doc_id` — the
  intended convergence, not a regression.
- The unified probe skips empty-string ids (`if str(value):`) for
  `json`/`jsonl`/`coleslaw`, which previously returned an empty-string id.
  `huggingface` already did this. It is a small deliberate behavior change that
  only affects degenerate empty-string ids, and it keeps `json` ≡ `jsonl`.
- `schema.py` stays a pure data-shape vocabulary; the record-projection policy
  lives next to its only callers, the extractors.
- Prerequisite for the related extractor issues: #38 (task-converter driver),
  #39 (orchestrator identity), #40 (`Annotations.from_sentences`), and #41
  (extractor-interface collapse).
