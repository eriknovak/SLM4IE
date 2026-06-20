---
title: Download
---

# Download raw corpora

!!! success "Verified end-to-end"
    This stage has been run end-to-end to assemble the project corpus.

`scripts/data/download.py` fetches the raw archives declared in
[`configs/data/download.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/download.yaml)
into the data store (`/vault/data/SLM4IE/raw/` by default).

Selection is **explicit**: pass one or more dataset keys as positional
arguments, or pass `--all`. A bare invocation errors out.

```bash
# Download every enabled dataset in the config
uv run python scripts/data/download.py --all

# Download specific datasets (positional, mutually exclusive with --all)
uv run python scripts/data/download.py fineweb2 cc100

# Force re-download into a custom output directory
uv run python scripts/data/download.py --all --output-dir /path/to/data --force

# Only evaluation benchmarks (datasets marked `benchmark: true`)
uv run python scripts/data/download.py --all --only-benchmarks

# Only pretraining corpora (skip benchmarks)
uv run python scripts/data/download.py --all --exclude-benchmarks

# Read a different YAML in configs/data/ (without the .yaml suffix)
uv run python scripts/data/download.py --all --config-name benchmarks

# Four datasets in parallel (thread pool; default cap is 4)
uv run python scripts/data/download.py fineweb2 cc100 mc4 hplt --max-workers 4
```

## Parallelism and logs

`download.py` processes datasets concurrently with `--max-workers`:

- `--max-workers 0` (default) — auto: `min(cpu_count // 2, n_datasets)`, capped
  at 4 for downloads to stay polite to remote servers.
- `--max-workers 1` — serial; tracebacks are unwrapped and the per-dataset
  progress bar is shown.
- `--max-workers N` — that many workers, capped at the number of selected
  datasets.

Per-dataset logs are always written to
`logs/download/<UTC-timestamp>/<key>.log`; the log directory is printed to
stderr at startup. In parallel mode the console prints only a periodic summary
line (`running=R done=D skipped=S failed=F waiting=W`).

## Catalog

For the full catalog of datasets the script can fetch, see:

- [Pretraining Corpora](../../datasets/pretraining-corpora.md) — CLARIN.SI,
  HuggingFace, and direct HTTP sources.
- [Benchmarks](../../datasets/benchmarks.md) — evaluation datasets for
  downstream IE tasks.

!!! note "Disk space"
    Pretraining corpora total tens of GB. Point `--output-dir` at a volume with
    adequate capacity, or rely on the default `/vault/data/SLM4IE/` location
    configured in `download.yaml`.

## Authentication

Some HuggingFace datasets are gated and require a token. See
[HuggingFace Authentication](../../getting-started/huggingface-auth.md).

## Next step

Once the raw archives are downloaded, run [Extract](extract.md) to convert
them to the unified JSONL shape.
