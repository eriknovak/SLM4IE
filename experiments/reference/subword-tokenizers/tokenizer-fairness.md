---
title: Tokenizer fairness across languages
kind: benchmark
status: draft
summary: Measures how many more tokens a tokenizer spends on the same text in one language than in another, and what that costs in context, price and accuracy.
---

## Description

The same sentence, translated into several languages, is cut into a different
number of tokens by one tokenizer in each language. The ratio of tokens per
word, called fertility, is the usual measure. A language with high fertility
gets a shorter effective context, pays more per request to a hosted model, and
tends to score lower on downstream tasks with a multilingual model than with a
tokenizer trained on that language alone.

## Why it matters here

Slovenian is one of the languages that pays this cost under English-centred
vocabularies. The tokenizer sweep reports fertility per tokenizer, and this
entry fixes what the number means, how it is computed, and what the two papers
found it predicts, so the sweep's metric has a source to cite.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Language Model Tokenizers Introduce Unfairness Between Languages, Petrov et al., 2023, arXiv:2305.15425
- **Paper**: How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models, Rust et al., 2021, arXiv:2012.15613

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
