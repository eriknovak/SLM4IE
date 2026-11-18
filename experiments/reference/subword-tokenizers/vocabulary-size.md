---
title: Vocabulary size
kind: component
status: draft
summary: The number of pieces a tokenizer may use; larger vocabularies spend fewer tokens per text, especially outside English, at the price of a larger embedding table.
---

## Description

A tokenizer's vocabulary size is the number of distinct pieces it may emit.
Models of 2018 to 2022 used 30 to 50 thousand pieces; models of 2024 use 128
to 256 thousand. A larger vocabulary packs more characters into each token, so
a text is shorter in tokens, but every piece needs a row in the model's
embedding table and in its output layer, which for a small model is a large
share of all its parameters.

## Why it matters here

The sweep varies vocabulary size as its first dial. This entry gives the
reasons the dial matters in both directions for a small model, so the sweep's
finding on Slovenian fertility against size has a place to be read against.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/), Meta, 2024, the move from 32K to 128K pieces
- **Paper**: [tiktoken encodings cl100k_base and o200k_base](https://github.com/openai/tiktoken), OpenAI, 2023

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
