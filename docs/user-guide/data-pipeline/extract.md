---
title: Extract
---

# Extract to unified JSONL

!!! success "Verified end-to-end"
    This stage has been run end-to-end across the project corpus.

`scripts/data/extract.py` normalizes raw downloads into a unified JSONL shape
according to
[`configs/data/extract.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/extract.yaml).
The dataset key list also drives stage 0 of the
[pretraining corpus](pretrain.md). Selection is explicit: positional keys or
`--all`.

```bash
# Extract every dataset declared in extract.yaml
uv run python scripts/data/extract.py --all

# Extract specific datasets (positional, mutually exclusive with --all)
uv run python scripts/data/extract.py macocu_sl

# Re-extract a dataset whose output already exists
uv run python scripts/data/extract.py macocu_sl --force

# Several datasets in parallel (process pool)
uv run python scripts/data/extract.py macocu_sl classla_web_sl kzb --max-workers 3

# Override the configured input/output directories
uv run python scripts/data/extract.py --all \
    --input-dir /vault/data/SLM4IE/raw \
    --output-dir /vault/data/SLM4IE/extracted
```

## Output: text + annotations split

For annotated corpora (CoNLL-U, TEI with `<w>`, CLASSLA-web JSONL, COLESLAW),
extraction writes **two files per dataset** under `extracted/`:

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`. Consumed
  both by `to_pretrain.py`'s stage 0 and by the task converters.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations as parallel
  arrays (`forms`, `lemmas`, `upos`, `feats`, `sentences`), plus `spans` when
  present. Kept separate to avoid loading them during text-only training.

The task converters ([Spans](spans.md), [Sentiment](sentiment.md),
[SuperGLUE](superglue.md)) join these two files on the fly via
`slm4ie.data.io_utils.iter_joined_records`, so **no intermediate merged file is
materialized**.

!!! tip "Why split text and annotations?"
    Pretraining only needs text; loading parallel annotation arrays for every
    document would waste memory and disk during the largest stage of the
    pipeline. Keeping them split lets each downstream consumer pull only what
    it needs.

## Next step

After extraction, fork into one of the three conversion routes:

- **Pretraining**: [Pretraining corpus](pretrain.md) — the verified route.
- **Tasks**: [Spans (NER)](spans.md), [Sentiment](sentiment.md),
  [SuperGLUE](superglue.md).
- **Tokenizer gold**: [Tokenizer-quality data](tokenizer-eval.md).
