---
title: Data Pipeline
---

# Data Pipeline

The data pipeline produces two distinct flavours of training data from a
single set of raw corpora. Most of the work happens in three stages:

1. **Download** — fetch raw archives from CLARIN.SI, HuggingFace Hub, and
   direct HTTP endpoints into the local data store
   (`/vault/data/SLM4IE/` by default).
2. **Extract** — convert each raw corpus to a unified JSONL shape with
   shared metadata. Annotated corpora keep text and annotations in
   separate files (joined on the fly downstream).
3. **Convert** — fork into a downstream-specific format.

## The two conversion routes

Downstream consumers diverge after extraction. **Keep them separate.**

| Route                           | Script                                                          | Output                                              | Use                                       |
|---------------------------------|-----------------------------------------------------------------|-----------------------------------------------------|-------------------------------------------|
| Pretraining (datatrove)         | [`scripts/data/to_datatrove.py`](datatrove.md)                  | `<out>/datatrove/<key>.jsonl.gz`                    | Filtering, deduping, sharding, sampling   |
| IE / spans                      | [`scripts/data/to_spans.py`](spans.md)                          | `<out>/spans/<schema>/<key>.jsonl.gz`               | GLiNER / CoNLL / generic span training    |
| Sentiment benchmark             | [`scripts/data/to_sentiment.py`](sentiment.md)                  | `<raw>/eval/sentiment/<key>.jsonl.gz`               | SA evaluation                             |
| SuperGLUE benchmark             | [`scripts/data/to_superglue.py`](superglue.md)                  | `<raw>/eval/superglue_sl/<variant>/<task>/...`      | SuperGLUE-SL evaluation                   |
| Tokenizer / morphology eval     | [`scripts/data/to_tokenizer-eval.py`](tokenizer-eval.md)        | `<raw>/eval/tokenizer/<key>.jsonl.gz`               | Tokenizer evaluation against Sloleks      |

## Curation

For pretraining, the datatrove shards still need to be **deduplicated,
language-verified, and statistically profiled**. That is what
[`scripts/data/curate.py`](curate.md) does in a single invocation, fusing
five datatrove executors that share a temporary working directory.

## Synthetic data

[`scripts/data/generate_synthetic.py`](synthetic.md) generates synthetic
IE training data via LLM APIs, used to bootstrap low-resource tasks.

## Diagram

```text
            download.yaml         extract.yaml
                  │                     │
                  ▼                     ▼
           [download.py] ────► [extract.py]
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       [to_datatrove.py]      [to_spans.py]         [to_sentiment.py]
              │                                          + others
              ▼
        [curate.py] ──► training corpus + statistics
```
