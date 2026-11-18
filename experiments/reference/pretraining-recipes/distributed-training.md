---
title: Data, tensor and pipeline parallelism
kind: algorithm
status: draft
summary: Splits a training run across several GPUs by copying the model, by cutting its matrices, or by assigning layers to devices, and shards the optimiser state so no device holds all of it.
---

## Description

Data parallelism gives every GPU a copy of the model and a different batch,
then averages the gradients. Tensor parallelism cuts each weight matrix across
GPUs so one layer runs on several devices. Pipeline parallelism assigns whole
layers to different devices and streams batches through them. ZeRO keeps data
parallelism but shards the optimiser state, gradients and weights across
devices, so the memory per GPU falls with the device count.

## Why it matters here

A small model fits on one GPU, so this entry matters only when the project
trains on several nodes of its cluster. It is the lowest-priority entry of the
topic and is filled when a multi-node run is planned.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism, Shoeybi et al., 2019, arXiv:1909.08053
- **Paper**: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, Rajbhandari et al., 2019, arXiv:1910.02054

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
