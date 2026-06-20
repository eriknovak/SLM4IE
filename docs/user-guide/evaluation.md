---
title: Evaluation
---

# Evaluation

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`scripts/evaluate.py` evaluates trained models on the Slovenian benchmarks (NER,
sentiment, SuperGLUE, etc.) registered in the task config.

**Entry point:** [`scripts/evaluate.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/evaluate.py)
**Config:** [`configs/data/tasks.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/tasks.yaml)
