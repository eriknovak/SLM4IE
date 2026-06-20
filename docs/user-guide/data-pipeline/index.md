---
title: Data Pipeline
---

# Data Pipeline

The data pipeline turns raw Slovenian corpora into training data. Two stages
are shared by everything downstream, then the pipeline forks into three
independent **conversion routes** that own disjoint output trees.

```text
raw/<key>/...                                       # download.py
extracted/<key>.jsonl (+ .annotations.jsonl.gz)     # extract.py — canonical form
        │
        ├─► pretrain/00_convert … 06_statistics/    # to_pretrain.py  (pretraining corpus)
        ├─► tasks/<task>/<dataset>/<split>.jsonl.gz # to_spans / to_sentiment / to_superglue
        └─► tokenization/<dataset>.jsonl.gz         # to_tokenization.py (tokenizer gold)
```

## Shared stages

1. **[Download](download.md)** — fetch raw archives from CLARIN.SI, the
   HuggingFace Hub, and direct HTTP endpoints into the data store
   (`/vault/data/SLM4IE/` by default).
2. **[Extract](extract.md)** — normalize each raw corpus into a unified JSONL
   shape with shared metadata. Annotated corpora keep text and annotations in
   two parallel files, joined on the fly downstream.

!!! success "Verified end-to-end"
    Download, Extract, and the [Pretraining corpus](pretrain.md) route have
    been run end-to-end. The resulting corpus is profiled on the
    [Corpus Statistics](../../datasets/corpus-statistics.md) page.

## The three conversion routes

Downstream consumers diverge after extraction. **Keep them separate** — each
route owns its own output tree, and the split is intentional.

| Route | Script | Output | Status |
|-------|--------|--------|--------|
| Pretraining corpus | [`to_pretrain.py`](pretrain.md) | `pretrain/00_convert … 06_statistics/` | **Verified** |
| Tasks — NER | [`to_spans.py`](spans.md) | `tasks/ner/<dataset>/<split>.jsonl.gz` | 🚧 In progress |
| Tasks — Sentiment | [`to_sentiment.py`](sentiment.md) | `tasks/sentiment/<dataset>/<split>.jsonl.gz` | 🚧 In progress |
| Tasks — SuperGLUE | [`to_superglue.py`](superglue.md) | `tasks/{nli,qa,coref,wsd,commonsense}/...` | 🚧 In progress |
| Tokenizer gold | [`to_tokenization.py`](tokenizer-eval.md) | `tokenization/<dataset>.jsonl.gz` | 🚧 In progress |

The three task converters all read a single flat registry,
[`configs/data/tasks.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/tasks.yaml),
keyed `<task>/<dataset>`. Train/test isolation is enforced by each entry's
`role` field (`finetune_and_eval` vs `held_out`), not by directory placement.

## Pretraining corpus

The pretraining route ([`to_pretrain.py`](pretrain.md)) runs eight
sentinel-skippable stages on top of
[datatrove](https://github.com/huggingface/datatrove): convert, language
filtering, adult/SEO-spam removal, Gopher quality and repetition heuristics,
exact and sentence deduplication, and corpus statistics. There is no separate
"datatrove conversion" step — stage 0 lifts `extracted/*.jsonl` into the
datatrove `Document` shape.

## Synthetic data

[`generate_synthetic.py`](synthetic.md) generates synthetic IE training data
via LLM APIs, used to bootstrap low-resource tasks.

## Conventions

- **Selection is explicit.** Every converter takes either one or more dataset
  keys as positional arguments, or `--all`. A bare invocation errors out.
- **Idempotent.** Existing outputs are skipped unless `--force` is passed.
- **Parallelism.** `download.py`, `extract.py`, and the task / tokenization
  converters accept `--max-workers` for per-dataset concurrency; `to_pretrain.py`
  accepts it as a whole-pipeline worker count.
- **Per-dataset logs** are written under `logs/<script>/<UTC-timestamp>/<key>.log`.
