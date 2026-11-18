---
title: Sliding-window and local attention
kind: component
status: draft
summary: Lets a token attend only to a fixed window of neighbours in most layers, so the cost of attention grows linearly with the length of the text.
couples_to: [encoder-transformers/modernbert]
---

## Description

Full attention compares every token with every other, so its cost grows with
the square of the text length. In sliding-window attention a token sees only a
fixed number of tokens to each side. Stacked layers still pass information
across the whole text, one window per layer. Longformer mixes local windows
with a few global tokens; Mistral uses a window in every layer; ModernBERT
alternates one full layer with two windowed layers.

## Why it matters here

It is the Attention span column of the encoder topic and the reason ModernBERT
reads 8,192 tokens at a cost a small model can afford. A Slovenian encoder
meant for whole documents takes the same choice.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Longformer: The Long-Document Transformer, Beltagy et al., 2020, arXiv:2004.05150
- **Paper**: Mistral 7B, Jiang et al., 2023, arXiv:2310.06825

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
