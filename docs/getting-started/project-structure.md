---
title: Project Structure
---

# Project Structure

Top-level layout of the SLM4IE repository:

```text
SLM4IE/
├── configs/                # YAML configuration files
│   ├── data/                 # download, extract, pretrain, tasks, tokenization, synthetic
│   ├── models/               # model architecture configs
│   ├── tokenizers/           # tokenizer training configs
│   ├── training/             # pretrain.yaml, finetune_ner.yaml
│   └── experiments/          # end-to-end experiment configs
├── slm4ie/                 # Library source
│   ├── data/                 # download, extract, processing, schema, tasks, synthetic
│   ├── models/               # model components and registry
│   ├── tokenizers/           # tokenizer training and analysis
│   ├── training/             # trainer, callbacks, evaluation
│   └── utils/                # config, I/O, MLflow helpers
├── scripts/                # CLI entry points (thin wrappers around slm4ie/)
│   ├── data/                 # download.py, extract.py, to_pretrain.py, to_tokenization.py,
│   │                         #   to_spans.py, to_sentiment.py, to_superglue.py, generate_synthetic.py
│   ├── tokenizers/           # train.py, analyze.py, export.py
│   ├── train.py              # model pretraining/fine-tuning
│   └── evaluate.py           # benchmark evaluation
├── slurm/                  # SLURM batch scripts for HPC training
├── notebooks/              # exploratory Jupyter notebooks
├── tests/                  # pytest test suite
├── docs/                   # documentation source (this site)
├── pyproject.toml          # project metadata and dependencies
└── uv.lock                 # locked dependency versions
```

## Where things live

- **Library code** lives under `slm4ie/`. Modules are importable and have no
  CLI logic.
- **CLI scripts** live under `scripts/` and are thin wrappers — they parse
  arguments, load a YAML config, and dispatch into `slm4ie/`. Run them via
  `uv run python scripts/...`.
- **Configs** are YAML files under `configs/`, grouped by purpose.
  Hyperparameters, dataset URLs, and paths live here, never hardcoded in Python.
- **Data** is **outside the repo** at `/vault/data/SLM4IE/` (or the path
  configured in `configs/data/*.yaml`). The repo is data-free.
- **SLURM** batch scripts under `slurm/` exist for HPC execution; see
  [SLURM / HPC](../user-guide/slurm.md).
- **Tests** mirror the library layout under `tests/`.

## Data layout

Datasets live outside the repo at `/vault/data/SLM4IE/`, in a five-tier tree.
Each conversion route owns a disjoint subtree:

```text
raw/<key>/...                                      # download.py
extracted/                                         # extract.py — canonical form
  <key>.jsonl                                        text + metadata
  <key>.annotations.jsonl.gz                         per-document annotations
pretrain/                                          # to_pretrain.py
  00_convert/ … 05_2_dedup/                          datatrove stages
  06_statistics/                                     corpus stats
tasks/<task>/<dataset>/{train,val,test}.jsonl.gz   # to_spans / to_sentiment / to_superglue
tokenization/<dataset>.jsonl.gz                    # to_tokenization.py
```

See the [Data Pipeline](../user-guide/data-pipeline/index.md) overview for how
these outputs are produced and consumed.
