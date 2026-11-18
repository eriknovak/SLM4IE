---
title: Byte-level and tokenizer-free models
kind: algorithm
status: draft
summary: Reads raw bytes, or learned groups of bytes, instead of pieces from a fixed vocabulary, so no tokenizer is trained at all.
---

## Description

A tokenizer-free model takes the bytes of the UTF-8 text as its input units.
ByT5 feeds every byte to the encoder. MEGABYTE and the Byte Latent Transformer
first group bytes into patches, fixed or chosen by how predictable the next
byte is, and run the large model over patches while a small model handles the
bytes inside each one. The sequence is longer than with subword pieces, and
the patching is what keeps the cost bearable.

## Why it matters here

This is the alternative to the whole tokenizer sweep: if a small Slovenian
model can read bytes at an acceptable cost, no morpheme-aware vocabulary is
needed. The entry states what that cost is, so the sweep's results can be
weighed against it.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: ByT5: Towards a token-free future with pre-trained byte-to-byte models, Xue et al., 2021, arXiv:2105.13626
- **Paper**: MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers, Yu et al., 2023, arXiv:2305.07185
- **Paper**: Byte Latent Transformer: Patches Scale Better Than Tokens, Pagnoni et al., 2024, arXiv:2412.09871

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
