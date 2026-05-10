---
title: Evaluation
---

# Evaluation

Evaluate trained models on the configured benchmarks:

```bash
uv run python scripts/evaluate.py
```

Benchmark configs are declared in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml)
with `benchmark: true` and a `tasks:` list. Convert each benchmark to
its evaluation-ready shape with the relevant pipeline script first:

- [Sentiment format](data-pipeline/sentiment.md) — for SA benchmarks.
- [SuperGLUE format](data-pipeline/superglue.md) — for SuperGLUE-SL.
- [Spans format](data-pipeline/spans.md) — for span-level IE tasks.
- [Tokenizer eval format](data-pipeline/tokenizer-eval.md) — for
  tokenizer / morphology evaluation.

The full benchmark catalog is documented under
[Datasets → Benchmarks](../datasets/benchmarks.md).
