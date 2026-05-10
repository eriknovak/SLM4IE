---
title: Sentiment format
---

# Sentiment format (SA evaluation)

`scripts/data/to_sentiment.py` converts raw SA benchmark downloads (e.g.
`sentinews`) into evaluation-ready JSONL with normalized
`{negative, neutral, positive}` labels.

The converter reads **directly from the raw download tree** — it bypasses
the [Extract](extract.md) step — and emits one record per item with
`id`, `text`, `label`, `label_id`, `level`, and `metadata`:

```bash
# Convert SentiNews (all annotation levels present in the download)
uv run python scripts/data/to_sentiment.py sentinews

# Restrict to document-level sentiment
uv run python scripts/data/to_sentiment.py sentinews --levels document

# Convert every SA-tagged benchmark dataset declared in download.yaml
uv run python scripts/data/to_sentiment.py --all
```

## Output

```text
<raw-dir>/eval/sentiment/<key>.jsonl.gz
<raw-dir>/eval/sentiment/<key>.label_map.json
```

Override the location with `--output-dir`. The accompanying
`label_map.json` records the integer `label_id` encoding so it remains
traceable.

Existing outputs are skipped unless `--force` is passed.
