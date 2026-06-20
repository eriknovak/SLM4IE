---
title: Tokenizer-quality data
---

# Tokenizer-quality data

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

`to_tokenization.py` materializes lexicon-derived datasets used only for
tokenizer / morphology evaluation — they never enter the pretraining corpus. It
currently covers Sloleks 3.1 (the Slovenian inflectional lexicon) and writes
`tokenization/<dataset>.jsonl.gz`.

**Entry point:** [`scripts/data/to_tokenization.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/data/to_tokenization.py)
**Config:** [`configs/data/tokenization.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/tokenization.yaml)
It is the prerequisite gold for the [Tokenizers](../tokenizers.md) comparison stage.
