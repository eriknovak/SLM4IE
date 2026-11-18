---
title: Pretrain, then fine-tune
kind: algorithm
status: draft
summary: Learns general language from unlabelled text first and adapts to a task on a small labelled set second, which is the recipe every entry here assumes.
---

## Description

Before 2018 a model was trained from scratch on each task's labelled data.
ULMFiT and ELMo showed that a model first trained on a large unlabelled
corpus, then fine-tuned on a task, needs far fewer labels and scores higher.
ULMFiT also set the fine-tuning habits still used: a learning rate that
differs per layer, and layers unfrozen one group at a time.

## Why it matters here

Every experiment in the project runs this recipe: pretrain a Slovenian model,
then fine-tune an extraction head. The entry is background that the other
entries build on and is the lowest priority to fill.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Universal Language Model Fine-tuning for Text Classification, Howard and Ruder, 2018, arXiv:1801.06146
- **Paper**: Deep contextualized word representations, Peters et al., 2018, arXiv:1802.05365

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
