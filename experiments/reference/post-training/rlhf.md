---
title: Reinforcement learning from human feedback
kind: algorithm
status: draft
summary: Trains a reward model on which of two answers people preferred, then optimises the language model against that reward while keeping it close to its starting point.
---

## Description

People compare pairs of answers to the same prompt. A reward model is trained
to predict their choice. The language model is then updated by reinforcement
learning to produce answers the reward model scores high, with a penalty for
drifting from the model it started as. InstructGPT applied the recipe to GPT-3
and the result was preferred to a model a hundred times its size.

## Why it matters here

A small extractor has no preferences to learn, so this entry serves the
comparison: it says what the large baselines went through and why their
outputs look the way they do. It is the base that direct preference
optimisation simplifies.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Deep Reinforcement Learning from Human Preferences, Christiano et al., 2017, arXiv:1706.03741
- **Paper**: Training language models to follow instructions with human feedback, Ouyang et al., 2022, arXiv:2203.02155

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
