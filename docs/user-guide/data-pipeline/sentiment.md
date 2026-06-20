---
title: Sentiment format
---

# Sentiment format

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`to_sentiment.py` converts every `sentiment/*` registry entry into
sentiment-classification data with normalized `{negative, neutral, positive}`
labels. It writes `tasks/sentiment/<dataset>/<split>.jsonl.gz`.

**Entry point:** [`scripts/data/to_sentiment.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/data/to_sentiment.py)
**Config:** [`configs/data/tasks.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/tasks.yaml)
