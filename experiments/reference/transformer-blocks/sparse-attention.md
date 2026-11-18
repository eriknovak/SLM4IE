---
title: Native sparse attention
kind: component
status: draft
summary: Lets each query attend to a learned selection of past blocks, compressed summaries and a local window, with kernels that make the selection cheap on GPUs.
---

## Description

Sparse attention keeps the softmax but lets each token look at only some
of the others. Native sparse attention combines three views of the past:
compressed summaries of blocks of tokens, a few blocks selected by their
summary scores, and a window of the nearest tokens. The selection is
learned during pretraining rather than fixed by a pattern, and the kernels
are written so that selecting blocks, not single tokens, keeps the GPU
busy. DeepSeek-V3.2 uses a related design for its long-context serving.

## Why it matters here

It is the third candidate, with linear attention and selective state-space
layers, for an encoder that reads documents of tens of thousands of tokens
whole. Against those two it keeps exact attention over the blocks it
selects, so distant recall survives. The entry is a draft at the lowest
priority until such an experiment is planned.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention, Yuan et al., 2025, arXiv:2502.11089

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
