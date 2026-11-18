---
title: GLiNER features
status: open
compare: [Stage, Architectures, Checkpoints, Since, Combines with, Conflicts with]
---

## Question

Which features of the GLiNER library should a Slovene GLiNER model be trained, scored and deployed with, and which of them work together?

## Introduction

The GLiNER library around the original model has grown features that act at different stages: some are fixed before training (the architecture), some during it (the loss), some when the model scores text (the decoding switches) and some at deployment (export and compression). Each entry here is one such feature, released between v0.2.26 and v0.2.30, with the planned experiments it could serve. Features at one stage can conflict with features at another, such as export dropping the learned calibration head, so the Combines with and Conflicts with rows are where to look before stacking them. No feature here ships with a Slovene checkpoint, so each one needs a model trained in this project before it applies.
