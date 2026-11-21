# Pretokenized corpus

Model training runs in other projects, starting with
[ModernSloBERT](https://github.com/eriknovak/ModernSloBERT). Each training
project reads the pretraining corpus in its own layout, tokenized in advance.
`scripts/pretokenize_corpus.py` tokenizes a stage of the curated corpus once
with an exported tokenizer and writes every layout an entry asks for from the
same samples, split and chunking. Install its dependencies with
`uv sync --group pretokenize`, which also installs alongside the `tokenizers`
group.

## Registry

Each entry of [`configs/data/pretokenize.yaml`](../configs/data/pretokenize.yaml)
is one output. Its key is the output path under the `pretokenized/` tier,
`<corpus>/<tokenizer>-<length>`:

```yaml
entries:
  sentence-dedup/bpe-32000-modernbert-1024:
    corpus: 06_sentence_dedup          # pretrain/<stage>/<dataset>/*.jsonl.gz
    tokenizer: bpe-32000/hf-modernbert # tokenizers/<run>/hf-<target>/
    max_seq_len: 1024                  # special tokens included
```

The tokenizer is an export of the tokenizer sweep (see
[tokenizer-sweep.md](tokenizer-sweep.md#using-a-tokenizer-downstream)); its
export target sets the special tokens that wrap every sample. The `defaults`
section sets the rest for every entry: `formats` (`mds` and `parquet`),
`min_chunk_tokens` (8), `validation_share` (0.1%), `min_validation_docs`
(10,000), `seed` and `shard_size_mb` (64).

## How a document becomes samples

1. The document is tokenized once, without special tokens.
2. Its token ids are cut where a new line starts; the corpus holds one sentence
   per line, so each piece is a sentence.
3. Whole sentences are packed into chunks of at most `max_seq_len` tokens minus
   the template's special tokens. A sentence that alone is too long is cut at
   the token level. No token is dropped.
4. Each chunk is wrapped by the tokenizer's template, for the `modernbert`
   target `<s> ... </s>`. A chunk shorter than `min_chunk_tokens` is dropped and
   counted in the manifest.
5. A seeded hash of the document id sends the whole document to `train` or
   `validation`. The share is the larger of `validation_share` and
   `min_validation_docs` divided by the number of documents, the same in every
   source. The minimum is met on average, so the drawn count can land a little
   below it; the manifest records the count drawn.

Each sample holds `input_ids` (the wrapped ids), `len` (their count) and `id`
(`<document id>#<chunk index>`).

## Output formats

| Format          | Reader                                                      | `input_ids` column                          |
| --------------- | ----------------------------------------------------------- | ------------------------------------------- |
| `mds`           | ModernBERT's non-streaming loader (`streaming: false`), the one with sequence packing | numpy array, uint16 (uint32 above 65,536 tokens) |
| `mds-streaming` | ModernBERT's streaming loader (`streaming: true`)           | bytes of an int64 array, with `len`         |
| `parquet`       | HuggingFace `datasets.load_dataset("parquet", ...)`         | list of uint16 (uint32 above 65,536 tokens) |

MDS shards are written uncompressed by mosaicml-streaming 0.10, whose format the
0.13 reader that ModernBERT pins also reads. A new layout is a new writer
registered with `@register_format` in `slm4ie/data/pretokenize/formats/`, not a
new script.

## Layout

```text
pretokenized/<corpus>/<tokenizer>-<length>/
  manifest.json                    written last; marks the output complete
  mds/{train,validation}/          index.json + one folder of MDS shards per input shard
  parquet/{train,validation}/      one <part>.parquet per input shard
  _parts/<part>.json               a finished input shard and its counts
  _settings.json  _inputs.json     the settings and input listing, for resume
```

Parts are numbered in input-shard order (`00000`, `00001`, ...), and every
format holds the same samples in the same order. The manifest records the
tokenizer path and digest, the lengths, seed and formats, the validation share
used, the input shards with their digest and document counts, and per source
and split the counts of documents, samples, tokens and dropped chunks and
tokens. `last_run` holds the latest run's time and throughput, which the run also
writes to `logs/pretokenize/<timestamp>/summary.log` beside the per-shard logs,
with the number of chunks dropped.

## Commands

```bash
uv run --group pretokenize python scripts/pretokenize_corpus.py run   sentence-dedup/bpe-32000-modernbert-1024
uv run --group pretokenize python scripts/pretokenize_corpus.py check sentence-dedup/bpe-32000-modernbert-1024
```

`run` converts input shards in parallel (`--max-workers`, default sized by
cores and memory, see
[Worker count by memory](data-pipeline.md#worker-count-by-memory)) and writes
each through `_partial/`, so an interrupted run resumes
without rewriting finished shards. It refuses to resume parts built from other
settings or another corpus. An output whose manifest matches its settings and
inputs is skipped; `--force` converts from scratch. Each written or skipped
output is logged to MLflow under `slm4ie/data/pretokenize` with the manifest as
artifact, keyed by its content digest, and linked to the curation run of the
corpus stage it read.

Size the workers by memory, not cores. Each worker tokenizes its shard in
batches capped at 1,000 documents and 4 million characters, so one worker peaks
at about 4 GB on the long theses of `kas` and 3 GB on the largest web shard
(measured 2026-11-21 with `bpe-32000/hf-modernbert`). The registry declares
4 GB per worker in `defaults.workers`, so the automatic count follows the free
memory. Under a cap, allow twice the peak per worker (10 workers under a 90 GB
cap), so an overrun stops only the run, not every process on the machine:

```bash
systemd-run --user --scope --unit=pretokenize -p MemoryMax=90G -p MemorySwapMax=0 \
  -p CollectMode=inactive-or-failed \
  uv run --group pretokenize python scripts/pretokenize_corpus.py run <entry> --max-workers 10
```

`check` reads every sample of every format back and fails when a format's token
or sample totals differ from the manifest, a stored `len` differs from its ids,
the formats differ, or a document has samples in both splits. A validation
split below `min_validation_docs` is a warning.

## Pointing a trainer at it

- **ModernBERT (ModernSloBERT)**: copy or link one output's `mds/` folder as the
  data root, with `streaming: false` and the `train` and `validation` splits.
  Set the special-token ids from the tokenizer export's summary
  (`tokenizers/_reports/export-summary.json`). A MorphPiece tokenizer needs
  `trust_remote_code=True` where the trainer loads it.
- **ModernBERT, streaming loader**: add `mds-streaming` to the entry's
  `formats`, then point the loader at the output's `mds-streaming/` folder with
  `streaming: true`. Sequence packing is not available in this mode.
- **HuggingFace**: `datasets.load_dataset("parquet", data_files={"train":
  "<output>/parquet/train/*.parquet", "validation":
  "<output>/parquet/validation/*.parquet"})`. Sort the files by name to keep
  the order of the MDS layout.
