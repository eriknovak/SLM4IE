---
title: Chain-of-thought prompting
kind: algorithm
status: draft
summary: Asks the model to write its reasoning steps before the answer, which raises accuracy on multi-step tasks for models past a certain size.
---

## Description

A chain-of-thought prompt shows worked examples in which the steps are written
out before the final answer. The model then writes steps for the new input
too. The gain appears only in models of tens of billions of parameters and is
largest on arithmetic and logic. For extraction the steps are a place to list
candidate spans before committing to them.

## Why it matters here

The large-model baselines for zero-shot extraction can be prompted with or
without reasoning steps, and the choice changes their score and cost. The
entry fixes which variant a baseline uses so comparisons stay fair.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, Wei et al., 2022, arXiv:2201.11903

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
