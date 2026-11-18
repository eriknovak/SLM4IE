---
title: Compute-optimal training
kind: benchmark
status: draft
summary: For a fixed compute budget the loss is lowest when tokens grow as fast as parameters, about twenty tokens per parameter; most models of 2022 were far under-trained.
variant_of: pretraining-recipes/scaling-laws
---

## Description

Hoffmann et al. refit the scaling laws with the learning-rate schedule matched
to each run's length and found that parameters and tokens should grow at the
same rate. The seventy-billion-parameter Chinchilla, trained on four times the
data of a model four times its size, beat it at the same compute. The rule of
thumb is about twenty tokens per parameter.

## Why it matters here

It sets the lower bound on how much Slovenian text a model of a given size
needs, and so ties the data programme's token targets to the model sizes the
project can train. It is what the over-training entry departs from.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Training Compute-Optimal Large Language Models, Hoffmann et al., 2022, arXiv:2203.15556

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
