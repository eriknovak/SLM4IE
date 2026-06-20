---
title: Changelog
---

# Changelog

Release notes for SLM4IE. The repository is in early development; versioned
releases will be added here as the project matures.

## Unreleased

- **Data pipeline:** download and extract stages, plus the eight-stage
  `to_pretrain.py` pretraining-corpus route (datatrove convert, language
  filtering, spam removal, Gopher quality/repetition, exact + sentence dedup,
  statistics) — run end-to-end. See [Corpus Statistics](datasets/corpus-statistics.md).
- **Task converters:** `to_spans` (NER), `to_sentiment`, and `to_superglue`
  driven by the `tasks.yaml` registry; `to_tokenization` for tokenizer gold —
  implemented, not yet run end-to-end.
- **Tokenizers:** six-backend training/analysis/export sweep across a vocab
  range with morphology-aware metrics — implemented, not yet run.
- **Training & evaluation:** scaffolding with MLflow tracking and SLURM batch
  scripts — implemented, not yet run.
- **Documentation:** this MkDocs Material site, with verified workflows
  documented in full and in-progress workflows flagged.
