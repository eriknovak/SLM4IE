---
title: SuperGLUE format
---

# SuperGLUE format (Slovene SuperGLUE evaluation)

`scripts/data/to_superglue.py` converts the extracted Slovene SuperGLUE
distribution into per-task per-split JSONL files for fine-tuning and
SloBENCH-style evaluation.

Each task is materialized in its native SuperGLUE schema (BoolQ, CB, COPA,
RTE, ReCoRD, WiC, WSC pass through unchanged). MultiRC is flattened to
one row per answer by default for classification convenience.

```bash
# All 8 tasks, all splits, HumanT variant
uv run python scripts/data/to_superglue.py

# Only CB and RTE, val split, GoogleMT variant
uv run python scripts/data/to_superglue.py --tasks CB RTE --splits val --variant googlemt

# Keep MultiRC in its native nested shape
uv run python scripts/data/to_superglue.py --tasks MultiRC --no-flatten-multirc
```

## Output

```text
<raw-dir>/eval/superglue_sl/<variant>/<task>/<split>.jsonl.gz
```

Override with `--output-dir`. Existing outputs are skipped unless
`--force` is passed.

## Expected raw layout

The converter expects the raw download to contain a `SuperGLUE-HumanT/`
or `SuperGLUE-GoogleMT/` directory with one subdirectory per task and
`train.jsonl` / `val.jsonl` / `test.jsonl` inside.
