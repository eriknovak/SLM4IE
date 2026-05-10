---
title: Datatrove format
---

# Datatrove format (pretraining route)

`scripts/data/to_datatrove.py` converts the per-dataset JSONL into the
[datatrove](https://github.com/huggingface/datatrove) `Document` shape
(`text` / `id` / `metadata`) so the corpus can be filtered, deduped,
and sharded with datatrove pipelines.

```bash
# Convert one dataset
uv run python scripts/data/to_datatrove.py kzb

# Convert every dataset in extract.yaml
uv run python scripts/data/to_datatrove.py --all

# Re-convert every dataset in extract.yaml
uv run python scripts/data/to_datatrove.py --all --force
```

## Output

```text
<output_dir>/datatrove/<key>.jsonl.gz
```

Override the location with `--output-dir`. Existing outputs are skipped
unless `--force` is passed.

Every record carries `dataset` and `domain` at the top level so downstream
filters and source-weighted sampling can use them via
`document.metadata`. `JsonlReader("…/datatrove/*.jsonl.gz")` ingests the
whole corpus in one go.

## Next step

For the **final pretraining corpus** (deduped, language-verified, profiled),
run [Curation](curate.md) on the datatrove shards.
