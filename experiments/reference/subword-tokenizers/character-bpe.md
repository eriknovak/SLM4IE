---
title: Character BPE
kind: algorithm
status: draft
summary: Joins the most frequent pair of adjacent characters, again and again, until the vocabulary is full.
---

## Description

Byte-pair encoding (BPE) on Unicode characters. It is the baseline that
MorphBPE changes in one line. The entry is a draft: its algorithm is not
written or checked yet.

## Facts

- **Base unit**: Unicode characters
- **Chunk while training**: whole word
- **Picks next token by**: highest pair count
- **Encoding**: standard BPE merges, no lexicon
- **Needs lexicon**: no

## Sources

- **Code**: `slm4ie/tokenizers/backends/char_bpe.py`
