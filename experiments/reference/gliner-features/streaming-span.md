---
title: Streaming span model
kind: component
status: from-paper
summary: A GLiNER head on a causal decoder LLM that caches its state, so text can arrive in chunks and recent span scores are revised as more context comes in.
variant_of:
part_of: information-extraction/gliner
couples_to: [{entry: model-families/decoder-only, as: backbone}]
---

## Description

The usual GLiNER model reads the whole text at once with a bidirectional
encoder. The streaming span model runs on a causal decoder instead, a model
that reads left to right like an LLM. It keeps the decoder's key-value cache
(the stored attention state of the text seen so far), so each new chunk costs
only its own tokens. Spans near the end of the text are scored again as more
right-hand context arrives. Each call returns the full list of spans found so
far. Labels stay fixed within a session unless the whole cache is rebuilt.

## What the source adds

The library added the architecture in release v0.2.28. The one released
checkpoint finds personal data such as names and identifiers, for masking
text in live conversations. It is a fine-tune of Qwen3-0.6B, tagged English
and multilingual. Its card says multilingual legal and administrative text
is its weakest domain.

- **In short**: the only documented GLiNER design on a decoder-only LLM, since v0.2.28

## Why it matters here

The streaming use is not this project's goal. What matters is the backbone.
The backbone comparison (#45) asks whether a small decoder LLM can carry a
Slovene GLiNER head. This architecture is the library's tested way to do
that, so it is the ready-made candidate for the decoder arm.

- **Backbone comparison (#45)**: train the streaming span head on a small
  decoder, such as Qwen3-0.6B or a Slovene-adapted decoder, on the same
  data as the encoder arms. Score it on whole texts so the arms compare.
- **Size target**: a 0.6B decoder fits under the project's one-billion
  parameter and 8 GB limits.

## Facts

- **Stage**: architecture — chosen before training; it fixes the backbone family
- **Architectures**: streaming span only — a causal decoder with a key-value cache
- **Checkpoints**: knowledgator/gliner-stream-pii-v1.0 — personal-data labels, Qwen3-0.6B, tagged multilingual
- **Since**: v0.2.28 — the card requires gliner 0.2.28 or later
- **Combines with**: nested spans, label descriptions — both pass through its inference call
- **Conflicts with**: ONNX export, entity vectors, calibration head — it runs only in PyTorch, returns no vectors in a stateful session, and skips the learned head
- **Training data**: ordinary span data — the same examples the encoder models train on

## Sources

- **Release notes**: GLiNER v0.2.28, <https://github.com/urchade/GLiNER/releases/tag/v0.2.28> — streaming inference
- **Docs**: GLiNER architectures, <https://urchade.github.io/GLiNER/architectures.html> — the streaming span design
- **Model card**: knowledgator/gliner-stream-pii-v1.0, <https://huggingface.co/knowledgator/gliner-stream-pii-v1.0> — backbone, languages, limits
- **Paper**: Qwen3 Technical Report, Qwen Team, 2025, arXiv:2505.09388 — the backbone
- **Paper vs code**: no paper on the architecture; this repository holds no GLiNER code yet, so no difference can be found

## Where it sits

- **Steps**: pick a backbone → **attach the streaming span head to a decoder** → train on span data → score

The encoder arms of the backbone comparison fill the same step with a
bidirectional encoder. The decoder-only entry covers the backbone family.

## Limits

### One direction of context

A causal decoder sees only the text to the left of a word. A span's end
word therefore knows nothing of what follows, unlike in an encoder. The
rescoring of recent spans only partly makes up for it.

- **Question**: on whole Slovene texts, does a 0.6B decoder with the streaming head reach the span F1 of an encoder of similar size?
- **Issue**: #60

### Untested as a general backbone

The library documents no way to put a decoder under the ordinary GLiNER head
outside this architecture. Whether one exists needs a read of the library
source.

- **Question**: can a Llama- or Qwen-style model serve as the encoder of an ordinary GLiNER model, and does it beat the streaming design?
