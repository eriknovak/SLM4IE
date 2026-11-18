---
title: FlashAttention
kind: algorithm
status: draft
summary: Computes exact attention in blocks that fit the GPU's on-chip memory, so the large attention matrix is never written out and long texts train faster.
couples_to: [encoder-transformers/modernbert]
---

## Description

Standard attention writes the full matrix of token-to-token scores to GPU
memory, reads it back for the softmax, and reads it again for the weighted
sum. FlashAttention splits queries, keys and values into blocks small enough
for the chip's fast on-chip memory, runs the softmax incrementally across
blocks with a running maximum and sum, and never materialises the matrix. The
result is the same attention, computed in less time and in memory that grows
linearly with the length.

## Why it matters here

It is a kernel, not a model change, but it decides whether a small model can
be pretrained at 8,192 tokens on the project's GPUs. The entry records what it
needs from the shapes and precision, so training configs can state whether it
is on.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Dao et al., 2022, arXiv:2205.14135

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
