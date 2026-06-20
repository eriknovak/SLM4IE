---
title: User Guide
---

# User Guide

End-to-end walkthrough of the SLM4IE workflows. All scripts are CLI wrappers
around `slm4ie/` modules and read from YAML configs in `configs/`. Run them via
`uv run` (recommended) or after activating `.venv/`.

!!! info "Maturity varies"
    Pages for workflows that have not yet been run end-to-end carry a 🚧
    *In progress* banner. The [Data Pipeline](data-pipeline/index.md)'s
    download → extract → pretraining-corpus path is verified; tokenizer
    training, model training, and evaluation are implemented but unrun.

## Workflows

- **[Data Pipeline](data-pipeline/index.md)** — download raw corpora, extract
  them to a unified JSONL shape, and fork into one of three conversion routes
  (pretraining corpus, task datasets, tokenizer gold).
- **[Tokenizers](tokenizers.md)** — train and score six tokenizers across a
  vocab sweep. 🚧
- **[Training](training.md)** — pretrain or fine-tune models from
  `configs/training/*.yaml`. 🚧
- **[Evaluation](evaluation.md)** — run benchmark evaluation against produced
  models. 🚧
- **[SLURM / HPC](slurm.md)** — batch scripts for cluster execution. 🚧
- **[Tests](tests.md)** — running the pytest suite.

## Conventions

- **Explicit selection.** Pass one or more dataset keys as positional
  arguments, or `--all`. A bare invocation errors out — there is no implicit
  "process everything" default.
- **Idempotent.** Existing outputs are skipped unless `--force` is passed.
- **Configurable paths.** Output paths come from each config's `output_dir`;
  CLI flags such as `--output-dir` override on a per-run basis.
- **Parallelism.** Most converters accept `--max-workers` and write per-dataset
  logs under `logs/<script>/<UTC-timestamp>/`.
