---
title: Knowledge distillation
kind: algorithm
status: draft
summary: Trains a small student model to match the output distribution of a larger teacher rather than the one-hot labels, so the student keeps most of the teacher's quality at a fraction of its size.
---

## Description

A teacher model's output is a probability over every token, which holds more
information than the one correct token. The student is trained to match that
distribution, with a softened temperature, alongside the usual loss.
DistilBERT cut BERT to sixty percent of its size and kept ninety-seven percent
of its score. Gemma 2 distilled its small models from a larger one during
pretraining itself, over the whole corpus.

## Why it matters here

Distillation is the most direct route to a small Slovenian model from a larger
multilingual one, and the span extractors in the project are themselves
trained on labels a large model produced. The entry says what the student
needs from the teacher and at which stage, so an experiment can choose between
distilling and training from scratch.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter, Sanh et al., 2019, arXiv:1910.01108
- **Paper**: Gemma 2: Improving Open Language Models at a Practical Size, Gemma Team, 2024, arXiv:2408.00118

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
