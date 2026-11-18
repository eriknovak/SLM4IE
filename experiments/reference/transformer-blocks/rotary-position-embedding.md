---
title: Rotary position embedding
kind: component
status: draft
summary: Tells the model where a token is by rotating its query and key vectors by an angle that grows with position, so attention depends on distance rather than on place.
couples_to: [encoder-transformers/modernbert, pretraining-recipes/context-length-extension]
---

## Description

The original Transformer adds a position vector to each token at the input. A
rotary embedding instead rotates the query and key vectors inside every
attention layer, by an angle proportional to the token's position. The dot
product of two rotated vectors then depends only on how far apart the tokens
are, so distance rather than absolute place reaches every layer, and the model
can be stretched to longer texts than it was trained on.

## Why it matters here

ModernBERT and every recent decoder use it, and it is the Position encoding
column of the encoder topic. A small Slovenian encoder built on that recipe
inherits it, and the context-length extension entry depends on it.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al., 2021, arXiv:2104.09864

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
