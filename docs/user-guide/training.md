---
title: Training
---

# Training

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`scripts/train.py` pretrains or fine-tunes small language models from YAML
configs under `configs/training/`, with MLflow experiment tracking.

**Entry point:** [`scripts/train.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/train.py)
**Config:** [`configs/training/pretrain.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/training/pretrain.yaml) (and [`configs/training/finetune_ner.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/training/finetune_ner.yaml))
