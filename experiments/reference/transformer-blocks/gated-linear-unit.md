---
title: "Gated linear units: SwiGLU and GeGLU"
kind: component
status: draft
summary: Replaces the feed-forward part of a block with two linear maps whose outputs are multiplied, one of them passed through an activation, which scores better per FLOP.
couples_to: [encoder-transformers/modernbert]
---

## Description

The feed-forward part of a block maps each token up to a wider vector, applies
an activation and maps it back down. A gated linear unit computes two wide
vectors, passes one through an activation such as Swish or GELU, and
multiplies them element by element before mapping down. For the same parameter
count it gives a lower loss than a single activation, and it is the feed-
forward part of LLaMA, PaLM, Gemma and ModernBERT.

## Why it matters here

It is the Activation column of the encoder topic. The entry says what the gate
does and what it costs, so an encoder experiment can name it as a fixed
setting or as the part it varies.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: GLU Variants Improve Transformer, Shazeer, 2020, arXiv:2002.05202

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
