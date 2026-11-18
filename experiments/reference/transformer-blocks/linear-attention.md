---
title: Linear attention
kind: component
status: draft
summary: Drops the softmax so attention becomes two matrix products whose cost is linear in the length; most layers use it and a few keep softmax attention for exact recall.
---

## Description

Softmax attention scores every pair of tokens, so its cost grows with the
square of the text length. Linear attention replaces the softmax with a
feature map applied to queries and keys separately. The keys and values
can then be summed into one fixed-size matrix first, and each query reads
from that matrix, so the cost is linear in the length and the model keeps
a fixed-size state while generating. MiniMax-01 runs this "lightning"
form in seven of every eight layers and keeps softmax attention in the
eighth, since the summed state cannot recall one distant token exactly.

## Why it matters here

It competes with sliding-window attention and selective state-space layers
as the way a small model reads a whole long document. The entry is a draft
at the lowest priority: it matters only if an experiment treats scientific
or clinical documents of tens of thousands of tokens whole, where the
windowed encoder stops being enough. What it gives up, exact recall of a
distant token, is what extraction over long texts needs most.

## Facts

<One `- **Label**: value` row per label in the topic's `compare` list, then any others.>

## Sources

- **Paper**: MiniMax-01: Scaling Foundation Models with Lightning Attention, MiniMax, 2025, arXiv:2501.08313

## Algorithm

<One fenced `algorithm` block per algorithm, each followed by its Reading.>
