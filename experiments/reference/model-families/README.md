---
title: Model families
status: open
compare: [Reads text, Pretraining objective, Output, Extraction route, Slovenian checkpoints]
---

## Question

Should a small Slovenian extractor be an encoder that tags, an encoder-decoder
that rewrites, or a decoder that generates, and what does each family give up?

## Introduction

Every Transformer language model belongs to one of three families, set by
which halves of the original encoder-decoder it keeps. An encoder reads the
whole text at once and gives each token a vector, a decoder reads left to right
and predicts the next token, and an encoder-decoder reads with the first and
writes with the second. The family decides how information extraction is done:
by tagging the vectors, by generating the answer as text, or by rewriting the
input into a structured form. The entries here are the three families, and the
encoder lineage has its own topic.
