---
title: Training
---

# Training

Pretrain or fine-tune a model from a YAML config:

```bash
uv run python scripts/train.py
```

Configuration lives under
[`configs/training/`](https://github.com/eriknovak/SLM4IE/blob/main/configs/training/):

- `pretrain.yaml` — pretraining hyperparameters.
- `finetune_ner.yaml` — NER fine-tuning recipe.

Training is MLflow-tracked; see `slm4ie/utils/mlflow.py` for the
integration helpers.

For HPC execution, use the [SLURM batch scripts](slurm.md).
