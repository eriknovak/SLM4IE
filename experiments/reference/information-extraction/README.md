---
title: Information Extraction
status: open
compare: [Tasks, Prompt tokens, Span candidates, Longest span, Scoring, Decoding, Backbone, Context length, Training data]
---

## Question

Which zero-shot extraction head should a small Slovenian encoder carry, and
which of its parts decide what the model can return and what it can never
return?

## Introduction

Zero-shot information extraction asks a model for entity types, labels or
relations it was never trained on, named at the moment of the request. The
entries here are one lineage of encoder-based extractors. Each one writes the
request into the input as a prompt, runs a bidirectional encoder once over
prompt and text, and matches the vectors of the request against the vectors of
the text. What changes from one generation to the next is the prompt grammar,
which text units are scored, how many of them, and what the decoder is allowed
to return. The Compared table puts those parts side by side so an experiment
can name the one it varies.
