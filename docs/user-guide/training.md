---
title: Training
---

# Training

!!! warning "🚧 Not yet implemented"
    This workflow is **scaffolded but not yet implemented** — the entry point
    (`scripts/train.py`) is a stub that raises `NotImplementedError`. This page
    will fill in once training lands. Until then, treat the configs below as the
    intended shape.

`scripts/train.py` will pretrain or fine-tune small language models from YAML
configs under `configs/training/`, with MLflow experiment tracking.

**Entry point:** [`scripts/train.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/train.py)
**Config:** [`configs/training/pretrain.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/training/pretrain.yaml) (and [`configs/training/finetune_ner.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/training/finetune_ner.yaml))
