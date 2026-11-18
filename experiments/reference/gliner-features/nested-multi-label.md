---
title: Nested and multi-label spans
kind: component
status: from-paper
summary: Two switches at inference let spans overlap or nest, and let one span carry several types, in place of the default greedy decoding that keeps only non-overlapping spans with one type each.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

By default a GLiNER model decodes greedily. It sorts the spans that pass the
threshold and keeps each one that does not overlap a span already kept, with
one type per span. Two switches change this. The first allows overlapping
spans, so "University of Ljubljana Medical Faculty" can hold both an
organisation and the city inside it. The second allows one span to carry
every type that passes the threshold, so a drug name can be both a
"substance" and a "treatment". The model's scores are the same either way;
only the decoding changes.

## What the source adds

Both switches are part of the ordinary prediction call and are older than
the releases this topic covers. The streaming model accepts them too.

- **In short**: inference switches for overlapping spans and several types per span, on any GLiNER model

## Why it matters here

Medical text nests entities often: a measurement holds a body part, a
procedure holds a device. The OMOP domains of the medical data are not
mutually exclusive in every case either. Whether the gold sets allow nesting
decides how the model must decode.

- **Medical data (#46)**: the annotation guide should state whether spans
  may nest and whether a span may take two domains. The decoding then
  follows the guide.
- **NER benchmarks**: the four Slovene NER test sets are flat, so the
  benchmark scores keep the default decoding and stay comparable with the
  reference scores.

## Facts

- **Stage**: scoring — two switches on the prediction call
- **Architectures**: any — the decoding runs after scoring on every architecture
- **Checkpoints**: any — no special training is needed to decode this way
- **Since**: before v0.2.26 — part of the prediction call from early releases
- **Combines with**: label descriptions, calibration — calibrated scores make one threshold mean the same across types
- **Conflicts with**: none known — streaming and export keep both switches
- **Default**: flat, one type — the setting the benchmark scores use

## Sources

- **Docs**: GLiNER usage, multi-label classification, <https://urchade.github.io/GLiNER/usage.html#multi-label-classification>
- **Paper vs code**: no paper on the switches; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: score every span against every type → **decode flat or nested, single or multi-label** → return spans

Scoring is the GLiNER head. Calibration, when fitted, changes the scores
before this step.

## Limits

### Trained flat, decoded nested

A model trained on flat data never saw a gold nested span. Its scores for
the inner span may be low because training marked it wrong.

- **Question**: on a nested medical gold set, does a model trained on flat Slovene NER find the inner spans, or must the training data itself be nested?
- **Issue**: #59
