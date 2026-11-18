---
title: Export and quantization
kind: component
status: from-paper
summary: A trained GLiNER model is exported to ONNX or OpenVINO, or loaded in int8, fp16 or fp8 precision, so it runs faster and in less memory without retraining.
variant_of:
part_of: information-extraction/gliner
couples_to: [{entry: efficiency-and-release/quantisation, as: method}, {entry: gliner-features/confidence-calibration, as: excluded}, {entry: gliner-features/entity-vectors, as: excluded}]
---

## Description

A trained model can run outside PyTorch. Export writes it as an ONNX file
(Open Neural Network Exchange, a format many runtimes read) or as an OpenVINO
model (Intel's runtime for CPUs). Export can store the weights in int8, eight
bits per weight in place of 32. The model can also stay in PyTorch and be
loaded in int8, fp16 or fp8 precision, and compiled for the GPU. Each option
trades a little accuracy for speed and memory, and none retrains the model.

## What the source adds

ONNX export is older than the releases this topic covers. Release v0.2.27
added quantization and compilation on load. v0.2.29 added OpenVINO and
install options per runtime. v0.2.30 added fp8, which needs a GPU of the
Ada or Hopper generation (compute capability 8.9 or higher).

- **In short**: ONNX and OpenVINO export, and int8, fp16 or fp8 loading, for encoder-based GLiNER models

## Why it matters here

The project's targets are a model under 8 GB of GPU memory and a low cost
per document. Both are properties of the deployed form, so they must be
measured on the exported or quantized model, not on the training copy.

- **Domain fine-tuning (#51)**: report span F1, peak memory and documents
  per second for the full-precision and int8 forms of each final model.
- **Release (#50)**: ship the ONNX form beside the PyTorch weights, so the
  model runs on a CPU server.
- **Backbone comparison (#45)**: a decoder arm cannot be exported, which is
  a cost the comparison should state.

## Facts

- **Stage**: deployment — applied to a trained model
- **Architectures**: uni-encoder, bi-encoder, relation — decoder and streaming models need PyTorch
- **Checkpoints**: any exportable model — the export runs on the user's own checkpoint
- **Since**: before v0.2.26 for ONNX — quantization on load v0.2.27, OpenVINO v0.2.29, fp8 v0.2.30
- **Combines with**: prompt compression, bi-encoder labels — both cut work per call before export does
- **Conflicts with**: calibration head, entity vectors, streaming — the head is skipped, vectors raise an error, and streaming models cannot be exported
- **Needs training**: no — post-training only

## Sources

- **Release notes**: GLiNER releases v0.2.27 to v0.2.30, <https://github.com/urchade/GLiNER/releases> — quantization, OpenVINO, fp8
- **Docs**: GLiNER ONNX conversion, <https://urchade.github.io/GLiNER/convert_to_onnx.html> — export calls and supported architectures
- **Paper vs code**: no paper; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: train → calibrate → **export or quantize** → measure memory and speed → release

The quantisation entry covers the general method. Calibration's learned
head is lost at this step, which the calibration entry records.

## Limits

### Quality cost at small size

The quantisation entry notes that the cost of int8 grows as models get
smaller. A GLiNER model of a few hundred million parameters is small.

- **Question**: how much strict span F1 does int8 export cost a Slovene GLiNER model of each backbone size?
- **Issue**: #61
