---
title: Subword tokenizers
status: open
compare: [Base unit, Chunk while training, Picks next token by, Encoding, Needs lexicon]
---

## Question

How should Slovenian words be cut into tokens, and at which step can a morpheme boundary be enforced?

## Introduction

A subword tokenizer cuts text into pieces from a fixed vocabulary, so that a
rare word is spelled from pieces the model has seen. A morpheme is the
smallest part of a word that carries meaning, such as a stem or an ending.
The entries here learn their vocabulary from the same corpus sample. They
differ in what they start from, how they pick the next token, and how they
cut a new word.
