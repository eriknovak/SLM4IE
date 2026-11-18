---
title: Pretraining recipes
status: open
compare: [Fixes, Varies, Rule it gives, Scale shown at]
---

## Question

For a fixed compute budget, how much Slovenian text, at what precision, with
which optimiser and at what context length should a small model be pretrained?

## Introduction

Pretraining is the long first stage in which a model learns language from
unlabelled text. Its cost is set by three dials, the number of parameters, the
number of tokens read, and the precision of the arithmetic, and its outcome by
how the weights are updated and how long a text the model sees at once. The
entries here are the rules and techniques that fix those dials for a model
trained on a small budget. Each says what it holds fixed, what it varies, the
rule it gives, and the scale at which the rule was shown to hold.
