---
title: Spans (NER) format
---

# Spans (NER) format

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`to_spans.py` converts every `ner/*` entry in the task registry into
GLiNER-style span training data, reading the extracted text plus its annotations
sidecar (which must provide a `spans` field). It emits
`tasks/ner/<dataset>/<split>.jsonl.gz`. The legacy
`--schema {gliner|conll|generic}` flag is gone — only the GLiNER schema is
produced.

**Entry point:** [`scripts/data/to_spans.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/data/to_spans.py)
**Config:** [`configs/data/tasks.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/tasks.yaml)
