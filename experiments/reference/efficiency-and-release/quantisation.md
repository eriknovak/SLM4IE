---
title: Post-training quantisation
kind: algorithm
status: draft
summary: Stores the trained weights in 8 or 4 bits instead of 16, with per-block scales and a correction for the rounding error, so the model fits smaller hardware with little loss.
---

## Description

After training, each weight matrix is rounded to a small integer grid with a
scale per row or per block. LLM.int8 keeps a few outlier columns in full
precision and shows that 8-bit holds across sizes. GPTQ rounds the weights one
column at a time and adjusts the remaining columns to cancel the error, which
makes 4-bit usable for large models. No retraining is needed; the quality cost
grows as the model gets smaller.

## Why it matters here

A small model is already cheap, so the gain here is running it on a laptop or
a CPU server. The entry records how the quality cost scales with size, which
is what a model of a few hundred million parameters has to check before it is
released in 4-bit.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale, Dettmers et al., 2022, arXiv:2208.07339
- **Paper**: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, Frantar et al., 2022, arXiv:2210.17323

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
