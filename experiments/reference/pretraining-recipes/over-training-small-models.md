---
title: Over-training small models
kind: benchmark
status: draft
summary: Trains a small model on far more tokens than the compute-optimal rule gives, because a model is run many more times than it is trained and a smaller one is cheaper to run.
variant_of: pretraining-recipes/compute-optimal-training
---

## Description

The compute-optimal rule minimises the training cost of reaching a loss. LLaMA
argued that the cost that matters is serving, so a smaller model trained for
longer is the better buy even though the training run is less efficient. Llama
3 trained an eight-billion-parameter model on fifteen trillion tokens, nearly
a hundred times the compute-optimal count, and its loss was still falling.

## Why it matters here

A small Slovenian model is in exactly this position: cheap to serve, and
limited by how much native text exists rather than by compute. The entry
states how far past twenty tokens per parameter the gains continued, which
bounds how much the data programme's corpus is worth.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: LLaMA: Open and Efficient Foundation Language Models, Touvron et al., 2023, arXiv:2302.13971
- **Paper**: The Llama 3 Herd of Models, Llama Team, Meta, 2024, arXiv:2407.21783

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
