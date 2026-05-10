---
title: User Guide
---

# User Guide

End-to-end walkthrough of the SLM4IE workflows. All scripts are CLI
wrappers around `slm4ie/` modules and read from YAML configs in
`configs/`. Run them via `uv run` (recommended) or after activating
`.venv/`.

## Workflows

- **[Data Pipeline](data-pipeline/index.md)** — download raw corpora,
  extract them to a unified JSONL shape, and convert to either the
  pretraining (datatrove) or IE training (spans, sentiment, SuperGLUE)
  format. Includes corpus curation.
- **[Tokenizers](tokenizers.md)** — train and analyze tokenizers from
  YAML configs.
- **[Training](training.md)** — pretrain or fine-tune models from
  `configs/training/*.yaml`.
- **[Evaluation](evaluation.md)** — run benchmark evaluation against
  produced models.
- **[SLURM / HPC](slurm.md)** — batch scripts for cluster execution.
- **[Tests](tests.md)** — running the pytest suite.

## Conventions

- Scripts default to processing every dataset declared in their config;
  pass `--datasets <key> [<key> ...]` to select a subset.
- Existing outputs are skipped unless `--force` is passed.
- Output paths come from the config's `output_dir`; CLI flags such as
  `--output-dir` override on a per-run basis.
