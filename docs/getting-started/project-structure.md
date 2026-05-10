---
title: Project Structure
---

# Project Structure

Top-level layout of the SLM4IE repository:

```text
SLM4IE/
├── configs/                # YAML configuration files
│   ├── data/                 # download, extract, processing, benchmarks, synthetic
│   ├── models/               # model architecture configs
│   ├── tokenizers/           # tokenizer training configs
│   ├── training/             # pretrain.yaml, finetune_ner.yaml
│   └── experiments/          # end-to-end experiment configs
├── slm4ie/                 # Library source
│   ├── data/                 # download, extract, processing, schema, synthetic
│   ├── models/               # model components and registry
│   ├── tokenizers/           # tokenizer training and analysis
│   ├── training/             # trainer, callbacks, evaluation
│   └── utils/                # config, I/O, MLflow helpers
├── scripts/                # CLI entry points (thin wrappers around slm4ie/)
│   ├── data/                 # download.py, extract.py, to_*.py, curate.py, ...
│   ├── tokenizers/           # train.py, analyze.py
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

- **Library code** lives under `slm4ie/`. Modules are importable and have
  no CLI logic.
- **CLI scripts** live under `scripts/` and are thin wrappers — they parse
  arguments, load a YAML config, and dispatch into `slm4ie/`. Run them
  via `uv run python scripts/...`.
- **Configs** are YAML files under `configs/`, grouped by purpose.
  Hyperparameters, dataset URLs, and paths live here, never hardcoded in
  Python.
- **Data** is **outside the repo** at `/vault/data/SLM4IE/` (or the path
  configured in `configs/data/*.yaml`). The repo is data-free.
- **SLURM** batch scripts under `slurm/` exist for HPC execution; see
  [SLURM / HPC](../user-guide/slurm.md).
- **Tests** mirror the library layout under `tests/`.

## Output paths

By convention, pipeline outputs are organized under the configured
`output_dir`:

```text
<output_dir>/
├── <key>.jsonl                     # per-dataset extracted text
├── <key>.annotations.jsonl.gz      # per-dataset annotations (when present)
├── datatrove/<key>.jsonl.gz        # pretraining route (Document shape)
├── spans/<schema>/<key>.jsonl.gz   # IE training route (gliner|conll|generic)
├── eval/sentiment/<key>.jsonl.gz   # SA benchmark conversion
├── eval/superglue_sl/.../*.jsonl.gz  # SuperGLUE per-task per-split files
└── statistics/                     # corpus statistics from curate.py
```

See the [Data Pipeline](../user-guide/data-pipeline/index.md) overview
for how these outputs are produced and consumed.
