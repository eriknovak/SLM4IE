---
title: Direct preference optimisation
kind: algorithm
status: draft
summary: Learns from preference pairs with one classification-style loss on the language model itself, with no reward model and no reinforcement learning loop.
variant_of: post-training/rlhf
---

## Description

DPO rewrites the reinforcement-learning objective of RLHF so that the optimal
reward is a function of the language model's own probabilities. The preference
data can then be fit directly: raise the probability of the preferred answer
relative to the rejected one, weighted by how far the model has moved from its
reference. One loss, no sampling, no reward model.

## Why it matters here

If a small Slovenian decoder is ever tuned on which extraction is correct,
this is the recipe a small team can afford. The entry is a draft until such an
experiment is planned.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Direct Preference Optimization: Your Language Model is Secretly a Reward Model, Rafailov et al., 2023, arXiv:2305.18290

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
