---
title: Context-length extension
kind: algorithm
status: draft
summary: Stretches a model trained on short texts to longer ones by rescaling the rotary position frequencies and training briefly at the new length.
couples_to: [transformer-blocks/rotary-position-embedding, encoder-transformers/modernbert]
---

## Description

A model trained at a context of a few thousand tokens fails past it, because
it has never seen those positions. With rotary position embeddings the
positions are angles, and the frequencies can be rescaled so the new length
maps onto the range the model knows. YaRN rescales the low and high
frequencies differently and adds a temperature on the attention, so a short
training run at the new length is enough. Gemini 1.5 showed the practical end
of this road at millions of tokens.

## Why it matters here

ModernBERT reaches 8,192 tokens by exactly this staged route, and a Slovenian
encoder meant for whole reports would copy it. The entry says what the
rescaling is and how much training the extension needs.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: YaRN: Efficient Context Window Extension of Large Language Models, Peng et al., 2023, arXiv:2309.00071
- **Paper**: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, Gemini Team, Google, 2024, arXiv:2403.05530

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
