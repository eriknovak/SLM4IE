---
title: Transformer blocks
status: open
compare: [Replaces, Where it sits, Cost in the length, Effect on training]
---

## Question

Which parts inside a Transformer block should a small Slovenian encoder take
from the recipes of 2019 to 2024, and what does each part cost or save?

## Introduction

A Transformer block is the unit a language model repeats: an attention part
that lets every token look at other tokens, a feed-forward part that
transforms each token on its own, and the normalisation and position signals
around them. The entries here are single parts, each of which replaced one
piece of the original 2017 block and was kept by nearly every model after it.
The encoder entries name these parts in their Facts; the entries here say what
each part replaces, where it sits in the block, how its cost grows with the
length of the text, and what it does to training.
