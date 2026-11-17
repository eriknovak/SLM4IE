---
title: In-context learning
kind: algorithm
status: draft
summary: Shows a model a few solved examples inside the prompt and lets it continue the pattern, with no change to its weights.
---

## Description

A decoder large enough will, given a few input-output pairs in its prompt,
produce the output for a new input in the same form. GPT-3 showed the ability
grows with scale and named it few-shot learning. Nothing is trained; the
examples are read at every request and take up context.

## Why it matters here

Every large-model baseline for zero-shot extraction is prompted this way. The
entry records what the method needs and what it costs per request, so
baselines in a Slovenian extraction experiment are described the same way each
time.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Language Models are Few-Shot Learners, Brown et al., 2020, arXiv:2005.14165

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
