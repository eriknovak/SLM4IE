---
title: Instruction tuning
kind: algorithm
status: draft
summary: Fine-tunes a model on many tasks written as natural-language instructions, so it follows an instruction for a task it never saw.
couples_to: [information-extraction/gliner]
---

## Description

FLAN and T0 took dozens of labelled datasets, wrote each example as an
instruction with its answer, and fine-tuned a pretrained model on the mixture.
Held-out tasks, never seen during tuning, were then solved from their
instruction alone. The method turns a next-token predictor into a model that
does what it is told, and is the first step of every chat model's post-
training.

## Why it matters here

Zero-shot extraction is a held-out task by definition. The span extractors in
the information-extraction topic are trained the same way, on many entity
types so that a new type works from its name. The entry links the two and says
what the recipe needs in tasks and in examples per task.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Finetuned Language Models Are Zero-Shot Learners, Wei et al., 2021, arXiv:2109.01652
- **Paper**: Multitask Prompted Training Enables Zero-Shot Task Generalization, Sanh et al., 2021, arXiv:2110.08207

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
