---
title: Selective state-space layers and hybrids
kind: component
status: draft
summary: Replaces attention with a recurrent layer whose state is updated by input-dependent gates, so cost is linear in the length; hybrids keep a few attention layers.
---

## Description

A state-space layer carries a fixed-size state along the text and updates it
at every token, like a recurrent network, but with a structure that can be
trained in parallel. Mamba makes the update gates depend on the input, which
lets the layer decide what to keep and what to forget. Cost and memory are
linear in the length. Hybrids such as Jamba interleave a few attention layers
so the model can still look back at an exact token.

## Why it matters here

The design competes with sliding-window attention as the way a small model
reads long documents. The entry is a draft until an experiment weighs it
against the windowed encoder; what it gives up is exact recall of a distant
token, which extraction over long texts needs.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces, Gu and Dao, 2023, arXiv:2312.00752
- **Paper**: Jamba: A Hybrid Transformer-Mamba Language Model, Lieber et al., 2024, arXiv:2403.19887

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
