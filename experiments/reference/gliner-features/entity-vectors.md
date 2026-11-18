---
title: Entity vectors
kind: component
status: from-paper
summary: The prediction call can return, for each found span, the span vector and its type's vector that the model scored, for linking, clustering or search downstream.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

Inside the model, every candidate span and every type name is a vector, and
the score is their dot product. A switch on the prediction call returns
those vectors with each found entity. Relations return their own vector, or
one for the head and one for the tail. The span vector has read the whole
text, so the same word gets different vectors in different contexts. The
vectors come back as plain arrays and can feed any later step, such as a
nearest-neighbour search.

## What the source adds

The library added the switches in release v0.2.29 for entities and
relations. They need native PyTorch, and a stateful streaming session
returns none.

- **In short**: span, type and relation vectors returned with each prediction, since v0.2.29

## Why it matters here

The medical data uses OMOP domains, but the OMOP vocabulary goes further
and names concepts, such as a specific drug or test. Entity vectors are a
first step towards linking a found span to its concept by nearest-neighbour
search over concept names. They also help analysis.

- **Medical data (#46)**: link found spans to OMOP concepts as a follow-up
  study, using the span vectors against encoded concept names.
- **Error analysis in any experiment**: cluster the false positives by
  their vectors to see which confusions repeat.

## Facts

- **Stage**: scoring — returned by the prediction call
- **Architectures**: any — on native PyTorch
- **Checkpoints**: any — the vectors are the model's own
- **Since**: v0.2.29 — span, type and relation vectors
- **Combines with**: nested spans, relation extraction, bi-encoder labels — a bi-encoder's cached type vectors live in the same space as its span vectors
- **Conflicts with**: ONNX export, streaming — exported runtimes raise an error and a stateful session returns none
- **Output**: float32 arrays — one per span, type or relation

## Sources

- **Release notes**: GLiNER v0.2.29, <https://github.com/urchade/GLiNER/releases/tag/v0.2.29>
- **Docs**: GLiNER usage, returning contextual vectors, <https://urchade.github.io/GLiNER/usage.html#returning-contextual-vectors>
- **Paper vs code**: no paper; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: score spans → decode → **return span and type vectors** → link, cluster or search

Linking to concepts is not a library feature; the project would write it.

## Limits

### Trained to separate types, not concepts

The vectors were trained to tell types apart, not to tell two drugs apart.
Two different drugs may sit close together.

- **Question**: does nearest-neighbour search with GLiNER span vectors find the right OMOP concept for Slovene drug and test mentions often enough to beat plain string matching?
