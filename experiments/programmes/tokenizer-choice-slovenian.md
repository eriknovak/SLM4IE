---
title: Which tokenizer should a small Slovene language model use?
slug: tokenizer-choice-slovenian
short: Slovene tokenizer choice
status: open
members:
  - ../validation/morph-gold-slovenian/
planned:
  - slug: morph-metrics-slovenian
    short: New morph metrics
    title: whether paradigm consistency, stem and ending integrity, corpus-frequency weighting and held-out lexemes separate tokenizers that boundary F1 ranks alike
    builds_on: [morph-gold-slovenian]
  - slug: metric-redundancy-slovenian
    short: Metric redundancy
    title: which tokenizer metrics carry information the others do not, across the vocabulary sweep
    builds_on: [morph-metrics-slovenian]
  - slug: morph-table-slovenian
    short: Morph table retrain
    title: whether MorphBPE and MorphPiece trained on the Sloleks 3.2 split cut words closer to morphemes without losing compression
    builds_on: [morph-gold-slovenian]
  - slug: bpe-pruning-slovenian
    short: Intermediate token pruning
    title: whether dropping BPE tokens that only serve as steps to longer ones frees vocabulary for stems and endings at the same compression
    ticket: "#23"
    builds_on: [morph-metrics-slovenian]
  - slug: stem-protection-slovenian
    short: Stem protection
    title: whether keeping the stem one symbol during BPE training, with the ending free to merge, beats forbidding every merge across a morpheme boundary
    ticket: "#24"
    builds_on: [morph-table-slovenian]
  - slug: tokenizer-downstream-slovenian
    short: Tokenizer and model scores
    title: whether the tokenizer metrics predict how well a small Slovene model trained with each tokenizer does on Slovene benchmarks
    builds_on: [metric-redundancy-slovenian, bpe-pruning-slovenian, stem-protection-slovenian]
---

# Which tokenizer should a small Slovene language model use?

## Question

Slovene is morphologically rich: a noun inflects for six cases in three numbers, and the ending carries case, number and gender. A tokenizer that cuts words into pieces by frequency alone may split stems and endings at arbitrary points. The question is whether a small Slovene model needs a tokenizer changed for this, or whether an existing algorithm is already good enough or best. The project needs the answer before it pretrains its models, since the tokenizer is fixed for the life of a model. An answer names the algorithm and the vocabulary size, rests on metrics shown to measure what they claim, and is checked against the scores of small models trained with the candidates.

## What we now believe

- **Nothing is established yet**: the first experiment checks the morphological gold that every morph metric rests on.

## Threads

- **Trust in the morphological gold**: whether the gold that defines a correct morpheme cut changes which tokenizer looks best — [morph-gold-slovenian:H1], [morph-gold-slovenian:H2]
- **Vocabulary size**: which metrics still improve as the vocabulary grows, and where each levels off for Slovene — [morph-gold-slovenian:Q1]

## Open

- whether metrics beyond boundary F1 separate tokenizers that it ranks alike — answered by morph-metrics-slovenian
- which metrics are redundant — answered by metric-redundancy-slovenian
- whether the morphological algorithms improve once trained on a correct table — answered by morph-table-slovenian
- whether a changed BPE training rule helps Slovene — answered by bpe-pruning-slovenian and stem-protection-slovenian
- whether any tokenizer metric predicts model quality — answered by tokenizer-downstream-slovenian
- whether freezing the embeddings of some tokens, such as rare tokens or tokens with a single meaning, speeds up pretraining without losing model quality — needs a follow-up on the models of tokenizer-downstream-slovenian, not yet planned
