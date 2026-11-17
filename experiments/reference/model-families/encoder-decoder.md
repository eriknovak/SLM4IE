---
title: Encoder-decoder models
kind: component
status: draft
summary: Reads with an encoder and writes with a decoder; every task is cast as turning one text into another, and mT5 covers Slovenian.
---

## Description

An encoder-decoder keeps both halves of the original Transformer. The encoder
reads the input in both directions and the decoder writes the output while
attending to the encoder's vectors. T5 pretrains it by removing spans of text
and writing them back; BART by corrupting the text in several ways and writing
the original. Any task becomes text in, text out, so extraction is a rewrite
of the input into a structured string.

## Why it matters here

mT5 is the one pretrained family of its size with Slovenian in the training
data, so it is a ready baseline. The entry records what the family costs
against an encoder of the same size and where its multilingual checkpoints
are.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, Raffel et al., 2019, arXiv:1910.10683
- **Paper**: BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension, Lewis et al., 2019, arXiv:1910.13461

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
