---
title: Prompt compression
kind: component
status: from-paper
summary: For a fixed label set, the label vectors are averaged once over sample texts and stored, so later calls skip writing the labels into the prompt.
variant_of:
part_of: information-extraction/gliner
couples_to: [{entry: gliner-features/label-descriptions, as: excluded}]
---

## Description

A GLiNER model writes every label name into the prompt of every call, which
costs encoder work each time. When the label set never changes, that work
repeats for nothing. Prompt compression runs the model once over a set of
sample texts with the labels, averages each label's vector across them, and
stores the averages as a fixed matrix. Later calls read the matrix and send
only the text. An optional step then fine-tunes the model on its own
predictions, so it adjusts to the stored vectors.

## What the source adds

The library added compression in release v0.2.27 for uni-encoder span,
token and relation models. The stored matrix is saved with the model's
weights, and training can continue after it.

- **In short**: fixed label vectors stored once, so calls carry text only, since v0.2.27

## Why it matters here

The medical and scientific models will serve a fixed set of types once
released. Compression would make those models faster at the cost of their
zero-shot ability, which is the property the project studies.

- **Release (#50, #51)**: a compressed variant of a domain model, for users
  who need only its fixed types, beside the zero-shot one.
- **Not for the experiments**: any test of new labels needs the full prompt.

## Facts

- **Stage**: deployment — after training, for a fixed label set
- **Architectures**: uni-encoder span, token, relation — no bi-encoder or streaming support stated
- **Checkpoints**: any uni-encoder — the matrix is built from the user's own model
- **Since**: v0.2.27 — with an optional self-distillation step
- **Combines with**: export — a compressed model still exports
- **Conflicts with**: label descriptions, zero-shot use — the stored vectors are fixed per name and cannot take new labels or descriptions
- **Sample texts**: needed — the averages come from a forward pass over them

## Sources

- **Release notes**: GLiNER v0.2.27, <https://github.com/urchade/GLiNER/releases/tag/v0.2.27>
- **Docs**: GLiNER usage, prompt compression, <https://urchade.github.io/GLiNER/usage.html#prompt-compression-precomputed-prompt-embeddings>
- **Paper vs code**: no paper; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: train → **compress the fixed labels** → export → release

The bi-encoder entry reaches a similar saving by a different route: it
caches label vectors without fixing the label set.

## Limits

### Averaged context

A label's vector normally depends on the text it reads with. The stored
average is the same for every text, which can lose that context.

- **Question**: how much span F1 does a compressed Slovene domain model lose against its full-prompt form, before and after the self-distillation step?
