---
title: RMSNorm and pre-normalisation
kind: component
status: draft
summary: Normalises each token's vector by its root mean square and does so before each part of the block rather than after, so deep models train without a warm-up.
couples_to: [encoder-transformers/bert, encoder-transformers/modernbert]
---

## Description

Layer normalisation rescales a token's vector to a fixed spread before it
enters a part of the block. RMSNorm drops the centring step and keeps only the
rescaling by the root mean square, which is cheaper and trains as well.
Placing the normalisation before the attention and feed-forward parts, rather
than after the residual addition as in the original Transformer, keeps the
gradient at the same scale through every layer, so a deep model trains stably
with less learning-rate warm-up.

## Why it matters here

It is the Block normalisation column of the encoder topic. BERT is post-norm,
ModernBERT is pre-norm, and recent decoders are pre-norm with RMSNorm. The
entry gives the reason behind the move so a small encoder can take the stable
variant.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Root Mean Square Layer Normalization, Zhang and Sennrich, 2019, arXiv:1910.07467
- **Paper**: On Layer Normalization in the Transformer Architecture, Xiong et al., 2020, arXiv:2002.04745

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
