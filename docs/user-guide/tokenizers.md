---
title: Tokenizers
---

# Tokenizers

!!! warning "🚧 In progress"
    This workflow is implemented but has **not yet been run end-to-end**, so
    this page is a stub. Detailed usage and results will land here once the
    pipeline is verified. Until then, treat the entry point and config below as
    the source of truth.

This stage trains six tokenizers (byte-level BPE, character-level charBPE, BERT
WordPiece, SentencePiece Unigram, MorphBPE, and MorphPiece) across a
16k/32k/64k vocab sweep, scores them with six metrics (Fertility, CTC
compression, Rényi efficiency, MorphScore, Morph-Edit-Distance, and
Morph-Consistency), and exports each as a HuggingFace tokenizer. It consumes the
deduplicated corpus (`pretrain/05_2_dedup/`) for training and the Sloleks gold
(`tokenization/sloleks.jsonl.gz`) for the morph metrics. Requires the `tokenize`
extra (`uv sync --extra tokenize`).

**Entry point:** [`scripts/tokenizers/train.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/tokenizers/train.py) (also [`scripts/tokenizers/analyze.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/tokenizers/analyze.py) and [`scripts/tokenizers/export.py`](https://github.com/eriknovak/SLM4IE/blob/main/scripts/tokenizers/export.py))
**Config:** [`configs/tokenizers/tokenizers.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/tokenizers/tokenizers.yaml)
