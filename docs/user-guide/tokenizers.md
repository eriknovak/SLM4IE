---
title: Tokenizers
---

# Tokenizers

Train tokenizers and compare them across configurations.

## Train

```bash
uv run python scripts/tokenizers/train.py
```

Configuration lives under
[`configs/tokenizers/`](https://github.com/eriknovak/SLM4IE/blob/main/configs/tokenizers/).

## Analyze

Compare trained tokenizers against each other and against gold-standard
morphology:

```bash
uv run python scripts/tokenizers/analyze.py
```

For tokenizer evaluation against the Sloleks lexicon, first prepare the
evaluation file with
[Tokenizer eval format](data-pipeline/tokenizer-eval.md).
