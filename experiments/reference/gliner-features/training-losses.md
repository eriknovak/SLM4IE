---
title: Training losses
kind: component
status: from-paper
summary: GLiNER training picks one loss over span-type scores from binary cross-entropy, focal, span-level Dice, cross-entropy over types and a calibrated cross-entropy.
variant_of:
part_of: information-extraction/gliner
couples_to: [{entry: gliner-features/confidence-calibration, as: loss term}]
---

## Description

Training scores every candidate span against every type and compares the
scores with the gold spans. Most pairs are negatives, since almost no span
of a text is an entity of a given type. The loss decides how much each
mistake costs. Binary cross-entropy treats every pair alike. Focal loss
lowers the weight of easy negatives. Span-level Dice loss scores the overlap
between predicted and gold spans as a whole, which suits rare types.
Cross-entropy over types makes the types of one span compete, and its
calibrated form adds a term that pulls the scores towards being calibrated.

## What the source adds

Binary cross-entropy and focal loss are older than the releases this topic
covers. Release v0.2.29 added span-level Dice loss with weighting by span
width. Release v0.2.30 added cross-entropy and its calibrated form, with a
weight for the calibration term.

- **In short**: a choice of five losses for GLiNER training, Dice since v0.2.29 and cross-entropy since v0.2.30

## Why it matters here

The loss is a training setting that changes every result after it. If the
backbone comparison and the domain fine-tuning use different losses, a gap
between them cannot be credited to the backbone or the data.

- **Backbone comparison (#45)**: fix one loss for every arm, and record it
  as a decision.
- **Domain fine-tuning (#51)**: the domain types are rarer than person or
  place, so Dice loss against focal loss is a cheap ablation.
- **Calibration**: the calibrated loss is the training-side half of a
  cascade to Claude.

## Facts

- **Stage**: training — one setting of the training call
- **Architectures**: any trainable GLiNER model — the loss acts on span-type scores
- **Checkpoints**: not applicable — chosen when training a new model
- **Since**: before v0.2.26 for cross-entropy and focal — Dice v0.2.29, cross-entropy over types and calibrated v0.2.30
- **Combines with**: calibration, nested spans — the calibrated loss feeds calibration; multi-label decoding relies on independent type scores
- **Conflicts with**: none — each run uses exactly one loss
- **Options**: binary cross-entropy, focal, Dice, cross-entropy, calibrated cross-entropy — five values

## Sources

- **Release notes**: GLiNER v0.2.29 and v0.2.30, <https://github.com/urchade/GLiNER/releases> — Dice loss, cross-entropy and calibrated loss
- **Docs**: GLiNER training guide, <https://urchade.github.io/GLiNER/training.html> — the loss settings
- **Paper vs code**: no paper on the losses; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: build candidate spans → score against types → **compute the loss** → update weights

The candidate spans and their scores come from the GLiNER head. The
calibration entry covers what happens to the scores after training.

## Limits

### Independent scores against competing types

Cross-entropy over types makes one span's types compete, which suits flat
data. Multi-label decoding wants each type scored on its own, which binary
cross-entropy and focal loss give.

- **Question**: on Slovene NER, which loss gives the highest strict span F1 at a fixed backbone, and does the choice change for the rarer domain types?
