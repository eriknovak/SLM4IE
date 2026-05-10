---
title: Tokenizer evaluation format
---

# Tokenizer evaluation format

`scripts/data/to_tokenizer_eval.py` converts a Slovenian inflectional
lexicon (currently [Sloleks 3.1](../../datasets/benchmarks.md)) into a
flat lemma/word-form JSONL used to evaluate tokenizer behaviour against
gold-standard morphology.

This format is **only used for tokenizer / morphology evaluation** —
it never enters the pretraining corpus, and the dataset is intentionally
absent from `extract.yaml`.

```bash
uv run python scripts/data/to_tokenizer_eval.py sloleks
```

## Output

```text
<raw-dir>/eval/tokenizer/<key>.jsonl.gz
```

Each record contains a lemma plus its inflected word forms with MULTEXT-East
V6 / JOS MSDs.

Existing outputs are skipped unless `--force` is passed.
