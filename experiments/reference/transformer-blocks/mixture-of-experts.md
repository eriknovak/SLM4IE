---
title: Mixture of experts
kind: component
status: draft
summary: Replaces one feed-forward part with several and routes each token to a few of them, so the model holds more parameters than it uses for any one token.
---

## Description

In a mixture-of-experts block the feed-forward part is copied several times,
and a small router sends each token to one or two of the copies. The parameter
count grows with the number of experts while the compute per token stays that
of one or two. Training needs a balancing term so the router does not send
everything to the same expert, and serving needs all experts in memory even
though few are used.

## Why it matters here

Small open models with a few experts, such as the 26-billion-parameter Gemma
4, suggest the design is no longer only for the largest models. The entry is a
draft until an experiment considers an expert layer for a Slovenian model; its
memory cost at serving time is the first thing it would have to answer.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, Shazeer et al., 2017, arXiv:1701.06538
- **Paper**: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding, Lepikhin et al., 2020, arXiv:2006.16668
- **Paper**: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, Fedus et al., 2021, arXiv:2101.03961
- **Paper**: Mixtral of Experts, Jiang et al., 2024, arXiv:2401.04088
- **Paper**: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models, Dai et al., 2024, arXiv:2401.06066

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
