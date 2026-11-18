---
title: Scaling laws
kind: benchmark
status: draft
summary: Loss falls as a power law of parameters, tokens and compute, so the loss of a larger run can be predicted from small ones.
---

## Description

Kaplan et al. trained decoders across several orders of magnitude of size and
data and found that the test loss follows a power law in each of the three
quantities, with no sign of levelling off. The laws let a team predict the
loss of a run before paying for it and say how to split a budget between a
larger model and more data. The split they gave, which favoured parameters,
was later corrected by the compute-optimal entry.

## Why it matters here

A small-model project has to choose its size and its token count on purpose.
This entry is the base the next two entries change, and it gives the form of
the fit an experiment would use to extrapolate from scratch runs.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Scaling Laws for Neural Language Models, Kaplan et al., 2020, arXiv:2001.08361

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
