---
title: Typed Decision Models
status: open
compare: [Question types, Backbone, Where an option is read, Options per question, Isolation between questions, Training objective, Calibration, Training data, Context length, Weights]
---

## Question

How is a model built that answers typed questions about a text with calibrated probabilities in one pass, and which of its parts could a small Slovenian encoder carry?

## Introduction

A typed decision model takes a piece of text or data, called the state, and
a set of questions whose answers are declared in advance: one option out of
several, a level on an ordered scale, or yes against no. It returns a
probability for every allowed answer and never a sentence, so the output is
valid by construction and software can branch on it. TypeSafe's Jev named the
class "System One models" in September 2026 and published neither weights nor
a recipe. The entries here are the closed model as reconstructed from its
behaviour, and three open models that each build the same interface in a
different way: a decoder with a pointer head, a bidirectional encoder with one
marker token per option, and a pair of contrastive encoders that score a state
against each option by a dot product. The Compared table puts the parts side
by side that decide where an option is read, how questions are kept apart,
what the training objective is, and how the probabilities are calibrated.
