---
title: Synthetic data
---

# Synthetic data

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`generate_synthetic.py` generates synthetic information-extraction training data
via LLM APIs, used to bootstrap low-resource tasks.

**Entry point:** [`scripts/data/generate_synthetic.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/data/generate_synthetic.py)
**Config:** [`configs/data/synthetic.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/synthetic.yaml)
