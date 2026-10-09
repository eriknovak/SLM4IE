---
title: Encoder Transformers
status: open
compare: [Position encoding, Attention span, Block normalisation, Activation, Pretraining objective, Tokenizer, Context length]
---

## Question

Which encoder design should a small Slovenian model for information extraction be built on, and which of its parts can be swapped without changing the rest?

## Introduction

An encoder Transformer reads a whole text at once and gives every token one
vector that reflects the tokens on both sides of it. Information extraction
reads such vectors directly: a named entity is a span whose tokens are tagged,
and a relation is a pair of spans. The entries here are the lineage that
runs from BERT to the encoders released in 2024. Each one keeps most of its
predecessor and changes a few parts: how position is told to the model, how
far a token can attend, where the normalisation sits inside a block, which
activation the feed-forward part uses, and what the model is trained to
predict. The Compared table shows those parts side by side, so an experiment
can name the one it varies.
