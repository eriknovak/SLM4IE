---
title: Decoder-only models
kind: component
status: draft
summary: Reads left to right and predicts the next token; extraction is done by generating the answer as text, which must then be parsed.
---

## Description

A decoder keeps the writing half of the original Transformer. Each token
attends only to the tokens before it, and the model is pretrained to predict
the next token, so the same model that reads can also write. Every chat model
and every model past a few billion parameters is a decoder. Information
extraction asks it to write the entities or relations out as text, often in a
fixed format, and parses what it wrote.

## Why it matters here

A small decoder is the generative alternative to the tagging encoder, and the
large decoders are the zero-shot baselines a Slovenian extractor is compared
against. The entry says what a generated answer costs in parsing, validity and
speed against a tagged one.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), Radford et al., 2018
- **Paper**: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford et al., 2019

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
