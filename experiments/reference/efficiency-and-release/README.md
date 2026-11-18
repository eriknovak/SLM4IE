---
title: Efficiency and release
status: open
compare: [Applied when, Size change, Quality cost, Needs training]
---

## Question

How does a small Slovenian model get smaller and cheaper to run after it is
trained, and in what form are its weights released?

## Introduction

A trained model is rarely run as it was trained. It can be distilled into a
smaller student, stored in fewer bits per weight, and then released under a
licence that says who may use it. The entries here are those steps. Each says
when it is applied, how much smaller or faster the model becomes, what quality
it costs, and whether it needs a training run of its own.
