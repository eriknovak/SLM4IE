---
title: Reinforcement learning for reasoning
kind: algorithm
status: draft
summary: Rewards a model for reaching a verifiable answer, so it learns to spend tokens on reasoning before answering; the reasoning models of 2024 and 2025 come from it.
---

## Description

When a task has a checkable answer, such as a maths problem or a unit test,
the model can be trained by reinforcement learning on the check alone. GRPO,
from DeepSeekMath, samples several answers per prompt and rewards each by how
it ranks within the group, which needs no value model. DeepSeek-R1 showed that
this alone, from a pretrained base, produces long reasoning chains and the
scores of OpenAI's o1.

## Why it matters here

Extraction against a gold set is a verifiable task, so a small decoder could
in principle be trained this way. The entry is a draft, lowest priority in the
topic, kept because the reasoning baselines the project compares against are
trained by it.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/), OpenAI, 2024
- **Paper**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, Shao et al., 2024, arXiv:2402.03300, the GRPO objective
- **Paper**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, DeepSeek-AI, 2025, arXiv:2501.12948

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
