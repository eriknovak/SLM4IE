---
title: Extract
---

# Extract to unified JSONL

`scripts/data/extract.py` converts raw downloads to a unified JSONL shape
according to
[`configs/data/extract.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/extract.yaml).

```bash
uv run python scripts/data/extract.py
uv run python scripts/data/extract.py --datasets macocu_sl

# re-extract a dataset whose output already exists
uv run python scripts/data/extract.py --datasets macocu_sl --force
```

## Output: text + annotations split

For annotated corpora (CoNLL-U, TEI with `<w>`, CLASSLA-web JSONL,
COLESLAW), extraction writes **two files per dataset**:

- `<key>.jsonl` — text + `source` / `domain` / `doc_id` / `metadata`.
  Used directly for pretraining.
- `<key>.annotations.jsonl.gz` — gzipped per-document annotations as
  parallel arrays (`forms`, `lemmas`, `upos`, `feats`, `sentences`),
  plus `spans` when present. Kept separate to avoid loading them during
  text-only training.

The downstream converters ([Datatrove](datatrove.md), [Spans](spans.md))
join these two files on the fly via
`slm4ie.data.io_utils.iter_joined_records`, so **no intermediate merged
file is materialized**.

!!! tip "Why split text and annotations?"
    Pretraining only needs text; loading parallel annotation arrays for
    every document would waste memory and disk during the largest stage
    of the pipeline. Keeping them split lets each downstream consumer
    pull only what it needs.

## Next step

After extraction, fork into one of:

- **Pretraining**: [Datatrove format](datatrove.md), then
  [Curation](curate.md) for the final pretraining corpus.
- **IE / spans**: [Spans format](spans.md).
- **Benchmarks**: [Sentiment](sentiment.md), [SuperGLUE](superglue.md),
  [Tokenizer eval](tokenizer-eval.md).
