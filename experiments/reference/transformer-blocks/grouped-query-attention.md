---
title: Multi-query and grouped-query attention
kind: component
status: draft
summary: Lets several attention heads share one set of keys and values, so the memory kept while generating shrinks several-fold at a small quality cost.
---

## Description

Multi-head attention keeps a separate key and value vector per head for every
past token, and a decoder must store all of them while it generates. Multi-
query attention gives all heads one shared key and value. Grouped query
attention is the middle ground: heads are split into a few groups, each with
its own key and value. The stored memory falls by the group count, and a
trained multi-head model can be converted with a short extra training run.

## Why it matters here

A small decoder used as a generative extractor spends most of its serving cost
on this memory. The entry states the trade, so a decoder experiment can name
how many groups it uses. An encoder does not generate and gains little from
it.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Fast Transformer Decoding: One Write-Head is All You Need, Shazeer, 2019, arXiv:1911.02150
- **Paper**: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Ainslie et al., 2023, arXiv:2305.13245

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
