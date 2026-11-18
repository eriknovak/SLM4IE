---
title: Bi-encoder with cached labels
kind: component
status: from-paper
summary: Labels and text go through separate encoders, so label vectors are computed once and cached, and a call can match spans against thousands of types.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

The ordinary GLiNER model reads labels and text together, so its cost grows
with the number of labels and it fits a few dozen at most. A bi-encoder gives
labels their own encoder. Label vectors are computed once, cached, and
reused for every text. The text encoder then only builds span vectors, and
each span is scored against every cached label by a dot product. The price
is that a label's vector never sees the text, and a span's vector never
sees the labels.

## What the paper adds

The Million-Label NER paper scales GLiNER to thousands of entity types this
way. It reports up to 130 times the throughput of the uni-encoder at 1,024
labels, through the cached label vectors. The released base model pairs an
Ettin ModernBERT text encoder of 150M parameters with a BGE label encoder,
trained in two stages, and is English only.

- **In short**: separate label and text encoders with cached label vectors, for very large label sets

## Why it matters here

The project's label sets are small, so the main gain does not apply yet. It
would apply if medical spans were typed by OMOP concept class or concept
rather than by domain, which runs to thousands of labels.

- **Medical data (#46)**: a follow-up that types spans by fine-grained OMOP
  concept class would need this architecture.
- **Backbone comparison (#45)**: a Slovene ModernBERT trained in #48 could
  serve as the text encoder, which keeps the Slovene work reusable here.

## Facts

- **Stage**: architecture — chosen before training
- **Architectures**: bi-encoder span, bi-encoder token — two architectures
- **Checkpoints**: knowledgator/gliner-bi-small-v1.0 and gliner-bi-base-v2.0 — English; base uses Ettin-150m and BGE
- **Since**: before v0.2.26 — the paper is from February 2026
- **Combines with**: entity vectors, export — cached label vectors share the space of span vectors, and bi-encoders export to ONNX
- **Conflicts with**: label context — label vectors never read the text, which the uni-encoder relies on
- **Label scale**: thousands — the cache removes the per-call cost of labels

## Sources

- **Paper**: The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder, Stepanov et al., 2026, arXiv:2602.18487 — the architecture and throughput claim
- **Model card**: knowledgator/gliner-bi-base-v2.0, <https://huggingface.co/knowledgator/gliner-bi-base-v2.0> — encoders and label caching calls
- **Docs**: GLiNER architectures, <https://urchade.github.io/GLiNER/architectures.html> — the bi-encoder designs
- **Paper vs code**: this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: encode and cache the labels → **encode the text and score spans against cached labels** → decode

Prompt compression saves label work for a uni-encoder by fixing the labels;
this design saves it while keeping them open.

## Limits

### Labels without context

A uni-encoder lets the text shape each label's vector, which the original
GLiNER paper credits for part of its accuracy. The bi-encoder gives that up.

- **Question**: on the Slovene NER test sets with their few types, how much span F1 does a bi-encoder lose against a uni-encoder on the same Slovene backbone?
