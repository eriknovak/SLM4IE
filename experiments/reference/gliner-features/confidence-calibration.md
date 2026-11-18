---
title: Confidence calibration
kind: component
status: from-paper
summary: Fits a temperature and a bias, or a small learned head, on held-out data with the encoder frozen, so that a span's score reads as the chance it is correct.
variant_of:
part_of: information-extraction/gliner
couples_to: []
---

## Description

A GLiNER model keeps every span whose score passes a threshold, but the
score is not a probability. A span scored 0.8 may be right far less than 80%
of the time. Calibration rescales the scores after training so that they
match how often the spans are right. The library offers two ways. A global
temperature and an optional bias divide and shift every score, fitted on a
held-out set while the encoder stays frozen. A learned calibration head
instead adjusts the confidence of each candidate span from its own features.
Training can also add a calibration term to the loss, so the scores come out
closer to calibrated before any fitting.

## What the source adds

The library added both methods in release v0.2.30. The learned head is
adapted from GLiClass, the authors' zero-shot classifier. Neither a paper nor
a calibrated checkpoint ships with it. Calibration changes nothing until it is
fitted, and the fitted head is saved with the model.

- **In short**: post-hoc temperature scaling and a learned confidence head for any GLiNER model, since v0.2.30

## Why it matters here

The project's targets compare a small model's F1 to a frontier LLM's at a
fraction of the cost per document. Calibrated scores make a cascade possible.
The small model keeps the spans it is sure of and sends the uncertain ones to
Claude. Sweeping the confidence cut-off then traces F1 against the share of
documents Claude sees, which is the cost the targets care about.

- **Domain fine-tuning (#51)**: report each fine-tuned model with its
  cascade curve beside the frontier LLM's score.
- **Silver-label check (#43)**: calibrated scores flag which LLM-made silver
  spans a GLiNER model disagrees with, for a human to review first.
- **Backbone comparison (#45)**: calibration error, such as expected
  calibration error (the gap between confidence and accuracy, averaged over
  confidence bins), is a cheap second measure beside span F1.

## Facts

- **Stage**: after training — fitted on held-out data with the encoder frozen; a loss term also exists for training
- **Architectures**: any — works on every GLiNER architecture under native PyTorch
- **Checkpoints**: none calibrated — every model must be fitted on its own held-out data
- **Since**: v0.2.30 — temperature, bias and learned head arrived together
- **Combines with**: label descriptions, nested spans, training losses — the calibrated loss term is one of the new training losses
- **Conflicts with**: ONNX export, streaming — the learned head is not applied in ONNX or OpenVINO runtimes or in streaming models
- **Fitting data**: a held-out split with gold spans — the Slovene NER validation splits already exist in the task data

## Sources

- **Release notes**: GLiNER v0.2.30, <https://github.com/urchade/GLiNER/releases/tag/v0.2.30> — temperature and bias calibration, learned head, calibrated loss
- **Docs**: GLiNER training guide, <https://urchade.github.io/GLiNER/training.html> — the calibrated loss and its weight
- **Paper vs code**: no paper; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: train the model → **calibrate on held-out data** → pick a confidence cut-off → route uncertain spans to an LLM

Training is the backbone comparison's or the domain fine-tuning's step; the
training losses entry covers the loss term. Routing is not a library
feature: the project would write it.

## Limits

### Fitted per domain

A calibration fitted on news-like NER sets may not hold on medical or
scientific text, where the label set and the span lengths differ.

- **Question**: does a calibration fitted on the general Slovene NER sets keep its error on the medical and scientific gold sets, or must each domain be fitted again?

### No help after export

The deployed form is likely ONNX or quantized, and the learned head does not
run there. Only the global temperature could be folded into the exported
threshold by hand.

- **Question**: how much of the calibration gain does the global temperature alone keep, compared with the learned head?
- **Issue**: #61
