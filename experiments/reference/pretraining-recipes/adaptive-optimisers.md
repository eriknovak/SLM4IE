---
title: "Adaptive optimisers: Adam, Adafactor and Muon"
kind: algorithm
status: draft
summary: Scales each weight's update by a running estimate of its gradient's size; the variants differ in how much memory the estimate takes and how stable a long run is.
---

## Description

Adam keeps two running averages per weight, of the gradient and of its square,
and divides one by the root of the other, so every weight moves at its own
rate. AdamW applies weight decay separately from that step. Adafactor stores
the second average as a row and a column factor instead of a full matrix,
which cuts the optimiser's memory to a fraction. Muon replaces the per-weight
scaling with an orthogonalised update per matrix and was the optimiser of a
trillion-parameter run.

## Why it matters here

The optimiser is a dial of every pretraining run and the first thing to blame
when a run diverges. The entry fixes what each variant stores and what it was
shown to do, so a training config can name its choice.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Adam: A Method for Stochastic Optimization, Kingma and Ba, 2014, arXiv:1412.6980
- **Paper**: Decoupled Weight Decay Regularization, Loshchilov and Hutter, 2017, arXiv:1711.05101
- **Paper**: Adafactor: Adaptive Learning Rates with Sublinear Memory Cost, Shazeer and Stern, 2018, arXiv:1804.04235
- **Paper**: Kimi K2: Open Agentic Intelligence, Kimi Team, 2025, arXiv:2507.20534, the Muon optimiser at scale

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
