---
title: Encoder-only models
kind: component
status: draft
summary: Reads the whole text at once and gives every token a vector that reflects both sides of it; extraction tags those vectors.
couples_to: [encoder-transformers/bert, information-extraction/gliner]
---

## Description

An encoder keeps the reading half of the original Transformer. Every token
attends to every other, in both directions, and the model is pretrained by
hiding some tokens and predicting them from the rest. The output is one vector
per token. Information extraction reads those vectors directly: a named entity
is a span whose tokens are tagged, and the span extractors in the information-
extraction topic score pairs of them.

## Why it matters here

This is the family the project builds on; its lineage from BERT to ModernBERT
is the encoder topic. The entry here sits in the comparison with the other two
families and points to that topic for the details.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., 2018, arXiv:1810.04805

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
