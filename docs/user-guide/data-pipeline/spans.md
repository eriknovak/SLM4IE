---
title: Spans format
---

# Spans format (IE training route)

`scripts/data/to_spans.py` converts the per-dataset JSONL into span-level
IE training files (GLiNER / CoNLL / generic) for fine-tuning encoder
models on entity-style tasks.

```bash
# GLiNER training shape
uv run python scripts/data/to_spans.py kzb --schema gliner

# Every dataset, lossless generic shape
uv run python scripts/data/to_spans.py --all --schema generic
```

## Schemas

| Schema    | Purpose                                                        |
|-----------|----------------------------------------------------------------|
| `gliner`  | GLiNER training shape (token-level spans + entity types)       |
| `conll`   | CoNLL BIO sequence labels                                      |
| `generic` | Lossless `(text, spans)` pairs for custom downstream pipelines |

## Output

```text
<output_dir>/spans/<schema>/<key>.jsonl.gz
```

Existing outputs are skipped unless `--force` is passed.

## Requirements

The converter expects each annotations payload to carry a `spans`
field — `[start, end, label]` triples or `{start, end, label}` dicts
with end-exclusive token indices. **Records without spans are skipped
with a warning.**

For datasets where extraction does not yet emit `spans`, add the field
in the corresponding extractor under `slm4ie/data/extractors/`.
