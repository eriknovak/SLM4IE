---
title: Mixed-precision training
kind: algorithm
status: draft
summary: Keeps the weights in full precision but runs the arithmetic in 16, 8 or 4 bits, with loss scaling or per-block scales so small gradients are not lost.
---

## Description

Each halving of the bits per number roughly doubles the arithmetic a GPU does
per second and halves the memory a tensor takes. Mixed precision runs the
forward and backward passes in the low format and keeps a full-precision copy
of the weights for the update. In 16-bit float the gradients must be scaled up
so they do not underflow; bfloat16 keeps the exponent range and needs no
scaling; 8-bit float needs a scale per tensor or per block, which DeepSeek-V3
showed works for a whole pretraining run.

## Why it matters here

The project's GPUs decide which format is free and which costs accuracy. The
entry records what each format needs so a SLURM training config can state its
precision and the reason.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Mixed Precision Training, Micikevicius et al., 2017, arXiv:1710.03740
- **Paper**: FP8 Formats for Deep Learning, Micikevicius et al., 2022, arXiv:2209.05433
- **Paper**: DeepSeek-V3 Technical Report, DeepSeek-AI, 2024, arXiv:2412.19437

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
