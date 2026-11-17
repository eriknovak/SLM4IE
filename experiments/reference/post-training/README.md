---
title: Post-training
status: open
compare: [Needs labels, Signal, Changes weights, Applies to encoders]
---

## Question

Once a small Slovenian model is pretrained, which post-training step turns it
into a zero-shot extractor, and which steps apply only to the large models it
is compared against?

## Introduction

Post-training is everything done to a pretrained model to make it follow a
request: showing examples in the prompt, fine-tuning on instructions, training
on human or model preferences, or rewarding correct reasoning. Some of these
change the weights and some do not, and some need labelled answers while others
need only a judgement of which of two answers is better. Most were made for
decoders, so the entries here say, for each step, what it needs, what signal it
learns from, whether the weights change, and whether it applies to an encoder
at all.
