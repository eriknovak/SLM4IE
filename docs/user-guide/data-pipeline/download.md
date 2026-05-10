---
title: Download
---

# Download raw corpora

`scripts/data/download.py` fetches the raw archives declared in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml)
to the configured data directory.

```bash
# Download all enabled datasets
uv run python scripts/data/download.py

# Download specific datasets
uv run python scripts/data/download.py --datasets fineweb2 cc100

# Force re-download with custom output directory
uv run python scripts/data/download.py --output-dir /path/to/data --force

# Download only evaluation benchmarks (datasets marked `benchmark: true`)
uv run python scripts/data/download.py --only-benchmarks

# Download only pretraining corpora (skip benchmarks)
uv run python scripts/data/download.py --exclude-benchmarks
```

## Catalog

For the full catalog of datasets the script can fetch, see:

- [Pretraining Corpora](../../datasets/pretraining-corpora.md) — CLARIN.SI,
  HuggingFace, and direct HTTP sources.
- [Benchmarks](../../datasets/benchmarks.md) — evaluation datasets used
  for downstream IE tasks.

!!! note "Disk space"
    Pretraining corpora total tens of GB. Set `--output-dir` to a volume
    with adequate capacity, or rely on the default `/vault/data/SLM4IE/`
    location configured in `download.yaml`.

## Authentication

Some HuggingFace datasets are gated and require a HuggingFace token. See
[HuggingFace Authentication](../../getting-started/huggingface-auth.md).

## Next step

Once the raw archives are downloaded, run [Extract](extract.md) to
convert them to the unified JSONL shape.
