---
title: Extraction Formats
---

# Raw extraction formats

Reference for the **extraction stage** — how each raw source under
`raw/<key>/` is read and mapped onto the unified `Document` schema in
`extracted/`. Use this when auditing an extractor or adding a new one.

For the commands and the rationale behind the text/annotations split, see
[Extract](../user-guide/data-pipeline/extract.md). This page documents the
*formats*, not the CLI.

## The stage in one pass

`scripts/data/extract.py` is a thin wrapper around
[`slm4ie.data.processing.extract_datasets`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/processing.py).
Every dataset key in
[`configs/data/extract.yaml`](https://github.com/eriknovak/SLM4IE/blob/main/configs/data/extract.yaml)
declares two things:

- `extractor` — the registry name of the reader (the eight documented below).
- `domain` — a free-text provenance tag (`web`, `wiki`, `parliamentary`,
  `academic`, `legal`, `medical`, `scientific`, `forum`, `blog`, `news`,
  `student`, `mixed`) that drives source-weighted sampling downstream.

Per dataset the orchestrator then:

1. Decompresses any archive in `raw/<key>/` — suffixes `.gz`, `.xz`, `.zip`,
   `.tgz`, `.tar.gz`, `.tar.zst`, `.tar.zstd`.
2. Dispatches to the registered extractor, which yields `Document` objects.
3. Writes **two artifacts, never merged**:
    - `extracted/<key>.jsonl` — `text`, `source`, `domain`, `doc_id`, `uid`,
      and `metadata`.
    - `extracted/<key>.annotations.jsonl.gz` — gzipped parallel arrays
      (`forms`, `lemmas`, `upos`, `feats`, `sentences`).

The annotations sidecar is written **only if at least one document in the
dataset carries real annotations**. In a mixed dataset, unannotated documents
get a **stub line** (`doc_id` + `uid` only) so the two files stay
line-aligned; readers detect a stub by the absence of the parallel-array
fields. Stubs seen before the first real annotation are buffered and flushed
once one appears — so a fully unannotated dataset produces no sidecar at all.

Both files are written to `.partial` and promoted with `os.replace`, so a
crashed run never leaves a half-written output in place. Outputs are skipped
unless `--force` is passed.

### The unified schema

Defined in
[`slm4ie/data/schema.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/schema.py):

```python
Document(text, source, domain, doc_id, metadata, annotations)
Annotations(tokens=[Token(form, lemma, upos, feats)], sentences=[[start, end], ...])
```

`uid` is a derived property — `"{source}:{doc_id}"` — so documents from
different corpora can never collide on a reused internal id. Sentence spans in
`Annotations.sentences` are **inclusive** `[start, end]` token-index pairs over
the flat token list: a two-sentence document with 5 + 7 tokens has
`sentences == [[0, 4], [5, 11]]`.

### Parallelism

Datasets are always processed **sequentially**; `--max-workers` sets shard
workers used *within* one dataset. Sharding only engages for extractors that
subclass `FileBasedExtractor` **and** have at least 8 input files **and**
more than one worker. Shards are parsed into temp files and concatenated in
order, so the merged output is byte-identical to the serial writer's.

| Extractor | Datasets | Annotations? | Shardable |
| --- | --- | --- | --- |
| [`jsonl`](#jsonl) | `classla_web_sl`, `legal_mc4`, `slovenian_news` | Only with `paragraphs` | No |
| [`json`](#json) | `povejmo_vemo_med` | No | No |
| [`text`](#text) | `cc100` | No | Yes |
| [`conllu`](#conllu) | `classlawiki_sl`, `oss`, `kzb`, `solar`, `suk`, `ssj500k` | Yes | Yes |
| [`tei`](#tei) | `parlamint_si`, `kas`, `siparl`, `janes_forum`, `janes_blog`, `janes_news`, `gigafida` | Yes, on the annotated paths | Yes |
| [`macocu`](#macocu) | `macocu_sl` | No | Yes |
| [`coleslaw`](#coleslaw) | `coleslaw` | No | No |
| [`huggingface`](#huggingface) | `finepdf`, `fineweb2`, `culturax`, `c4`, `hplt` | No | No |

All extractors discover their input files **recursively** (`rglob`) in sorted
order. Registry and base classes live in
[`slm4ie/data/extractors/__init__.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/__init__.py).

## jsonl

[`slm4ie/data/extractors/jsonl.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/jsonl.py)
— line-delimited JSON (`*.jsonl`), one object per line. Blank and malformed
lines are skipped with a warning.

```json
{"doc_id": "d1", "text": "Dober dan.", "url": "https://example.com",
 "paragraphs": [{"sentences": [{"tokens": [
   {"form": "Dober", "lemma": "dober", "upos": "ADJ", "feats": "Case=Nom"},
   {"form": "dan", "lemma": "dan", "upos": "NOUN", "feats": "Case=Nom"},
   {"form": ".", "lemma": ".", "upos": "PUNCT", "feats": null}]}]}]}
```

| Raw | → | `Document` |
| --- | --- | --- |
| **`text`** (configurable) | → | `text` — records with empty/missing text are skipped |
| **`doc_id`** (configurable) | → | `doc_id` |
| **`paragraphs`** | → | `annotations`, flattened across paragraphs → sentences → tokens |
| all other fields | → | `metadata` |

The field names are configurable through the `metadata:` block, so a feed that
names its fields differently needs no bespoke extractor. `slovenian_news` uses
this:

```yaml
slovenian_news:
  extractor: jsonl
  domain: news
  metadata:
    text_field: body            # default: "text"
    id_field: uri               # default: "doc_id"
    metadata_fields: [url, title, dateTime, source]
```

When `metadata_fields` is omitted every record field is kept as metadata except
the text field, the id field, and the structural fields `paragraphs` and
`conll`. When given, only the listed keys are kept.

Annotations are produced **only** when a `paragraphs` field is present and
yields at least one token — `classla_web_sl` carries them, `legal_mc4` and
`slovenian_news` do not.

## json

[`slm4ie/data/extractors/json.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/json.py)
— `*.json` holding a top-level array. A single top-level object is accepted and
treated as a one-record array; anything else is skipped with a warning.

```json
[
  {"doc_id": "vemo.1", "text": "Bolnik je prišel z bolečinami.",
   "specialty": "interna", "year": 2023},
  {"doc_id": "vemo.2", "text": "Drugi opis primera."}
]
```

| Raw | → | `Document` |
| --- | --- | --- |
| **`text`** (configurable) | → | `text` — empty/missing skipped |
| **`doc_id`** (configurable) | → | `doc_id` |
| every other non-null field | → | `metadata` |

Text-only; no annotations. The field names are configurable through the
`metadata:` block with the same knobs as `jsonl` — `text_field` (default
`text`), `id_field` (default `doc_id`), and `metadata_fields` (whitelist of
record fields to keep; when omitted, every other field is kept). Fields with a
null value are always dropped from metadata.

## text

[`slm4ie/data/extractors/text.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/text.py)
— `*.txt` streamed line-by-line so multi-GB inputs never need to fit in memory.
A **blank line is a document boundary** (the CC100 convention).

```text
Prvi dokument, prva vrstica.
Prvi dokument, druga vrstica.

Drugi dokument, ena sama vrstica.

Tretji dokument.
```

`text` is the block's non-empty lines joined with newlines and stripped. That is
the only field produced: no `doc_id`, no `metadata`, no annotations. See
[Documents without an id](#documents-without-an-id) for what the orchestrator
does about the missing id.

## conllu

[`slm4ie/data/extractors/conllu.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/conllu.py)
— `*.conllu` and `*.conll`. Ten tab-separated columns, `#` comment lines, blank
line = sentence boundary, `_` = missing value.

```text
# newdoc id = doc1
# sent_id = doc1.s1
# text = Predsednik je odprl sejo.
1	Predsednik	predsednik	NOUN	Ncmsn	Case=Nom	3	nsubj	_	NER=O
2	je	biti	AUX	Va-r3s-n	Tense=Pres	3	aux	_	NER=O
3	odprl	odpreti	VERB	Vmep-sm	VerbForm=Part	0	root	_	NER=O
4	sejo	seja	NOUN	Ncfsa	Case=Acc	3	obj	_	NER=O
5	.	.	PUNCT	Z	_	3	punct	_	SpaceAfter=No
```

Of the ten columns, only four become `Token` fields:

| Column | | → | `Token` |
| --- | --- | --- | --- |
| 2 | **FORM** | → | `form` |
| 3 | **LEMMA** | → | `lemma` |
| 4 | **UPOS** | → | `upos` |
| 6 | **FEATS** | → | `feats` |
| 10 | **MISC** | → | read only for `SpaceAfter=No` |

ID, XPOS, HEAD, DEPREL, and DEPS are not carried into the schema. Rows whose ID
marks a multiword token (`1-2`) or an empty node (`1.1`) are skipped.

**Document boundaries** are detected in priority order:

1. `# newdoc id = ...` markers — the standard CoNLL-U signal.
2. A change in the leading component of a hierarchical `# sent_id`
   (`solar1.1.1` → `solar2.1.1` opens a document with `doc_id = "solar2"`).
   Sources like Solar pack thousands of essays into one file without ever
   emitting `# newdoc id`. This heuristic is disabled for the rest of the file
   as soon as any `# newdoc id` is seen.
3. One document per file, `doc_id` = filename stem.

**Text** prefers each sentence's `# text = ` comment; when absent it is
reconstructed from the FORM column, honouring `SpaceAfter=No`. Sentence strings
are then joined with newlines. `oss`, `kzb`, and `kas` pull extra per-document
fields from a [metadata sidecar](#external-tsv-metadata).

## tei

[`slm4ie/data/extractors/tei.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/tei.py)
— `*.xml` in the TEI namespace `http://www.tei-c.org/ns/1.0`. Three structural
paths are **auto-detected** from the tree itself:

| Structure | Detected by | Unit |
| --- | --- | --- |
| Annotated with utterances | has `<w>`, has `<u>` | one document per `<u>` |
| Annotated without utterances | has `<w>`, no `<u>` | one document per file |
| Plain | no `<w>` | one document per `<p>` |

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body>
    <u xml:id="u1" who="#chair">
      <s xml:id="u1.s1">
        <w lemma="dober" msd="UPosTag=ADJ|Case=Nom">Dober</w>
        <w lemma="dan" msd="UPosTag=NOUN|Case=Nom">dan</w>
        <pc msd="UPosTag=PUNCT">.</pc>
      </s>
    </u>
  </body></text>
</TEI>
```

| Raw | → | `Document` |
| --- | --- | --- |
| **`<w>` / `<pc>`** element text | → | `Token.form` |
| **`lemma`** attribute | → | `Token.lemma` |
| **`msd`** or **`ana`** attribute | → | `Token.upos` + `Token.feats` |
| **`xml:id`** on `<u>` (or `<p>`) | → | `doc_id`; falls back to the filename stem on the per-file path |
| **`who`** / **`ana`** on `<u>` | → | `metadata`, layered over any sidecar fields |

Tokens are collected from `<w>` and `<pc>` children of each `<s>`, including
those wrapped in a `<name>` element. Morphology is read from `msd` first
(`UPosTag=X|Key=Val` → the `UPosTag` part becomes `upos`, the rest is rejoined
with `|` as `feats`); when absent, `ana` is parsed for an `mte:` MULTEXT-East v6
compact code, which is mapped to UPOS by its category character (refined by the
second character for NOUN/PROPN, VERB/AUX, CCONJ/SCONJ) and preserved verbatim
as `feats="MTE=<code>"`.

On the **plain** path, `<p>` scanning is scoped to `<body>` so that `<p>`
elements in the `<teiHeader>` (copyright notices, catalog codes) are never
emitted as documents; text comes from `.itertext()`.

!!! warning "TEI text loses original spacing"
    Reconstructed text on the annotated paths space-joins token forms
    unconditionally — TEI carries no `SpaceAfter` equivalent that this extractor
    reads. Punctuation therefore renders as `Dober dan .`, not `Dober dan.`. The
    `conllu` extractor *does* honour `SpaceAfter=No`, so reconstructed-text
    fidelity is not uniform across the two annotated routes.

Files larger than 64 MB are parsed with `iterparse(huge_tree=True)` instead of a
full DOM — GigaFida ships segments whose DOM would be 10–15× the file size and
OOM the extractor under sharding. The streaming path only handles the
per-file annotated structure; utterance- and plain-structured files fall back to
the full-DOM path.

## macocu

[`slm4ie/data/extractors/macocu.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/macocu.py)
— the MaCoCu monolingual DTD (`*.xml`), streamed with lxml `iterparse` filtered
on `<doc>` so large corpora never build a full tree.

```xml
<corpus id="MaCoCu-sl-2.0">
  <doc id="macocu.sl.1" title="Page One" url="https://example.com/1"
       crawl_date="2022-07-01" lm_score="0.95">
    <p id="macocu.sl.1.1" lang="sl">Dober dan.</p>
    <p id="macocu.sl.1.2" lang="sl">Kako ste?</p>
  </doc>
</corpus>
```

| Raw | → | `Document` |
| --- | --- | --- |
| **`<p>`** contents | → | `text`, joined with newlines; empty `<doc>` skipped |
| **`id`** attribute on `<doc>` | → | `doc_id` |
| **`title`**, **`crawl_date`**, **`lang_distr`**, **`url`**, **`domain`**, **`file_type`**, **`lm_score`** on `<doc>` | → | `metadata` (only those present and non-empty) |

Text-only; no annotations. `<doc>` attributes outside that whitelist are
dropped.

## coleslaw

[`slm4ie/data/extractors/coleslaw.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/coleslaw.py)
— `*.jsonl` across four legal subcorpora that ship **different schemas**,
detected per record by field presence.

```json
{"id": 1, "text": "Zakon o nečem.", "title": "Zakon"}
{"id": "Up-1", "fullText": "Sklep ustavnega sodišča."}
{"id": "c1", "jedro": "Bistvo.", "izrek": "Razveljavi se.", "obrazlozitev": "Obrazložitev sledi."}
{"id": "750", "skodni_dogodek": "Prometna nesreča.", "poskodba": "Zvin vratu."}
```

`text` is the first non-empty of, in order:

1. **`text`** — PISRS, UradniList.
2. **`fullText`** — USRS (Constitutional Court).
3. **`jedro`**, **`izrek`**, **`obrazlozitev`** joined with blank lines —
   SodnaPraksa `sp_courts`. The order mirrors the structure of Slovenian court
   decisions: essence → operative part → reasoning.
4. Six `sp_claims` prose fields (`skodni_dogodek`, `poskodba`,
   `telesne_bolecine`, `strah`, `zmanjsanje_zivljenjske_aktivnosti`,
   `dodatne_informacije`) joined in reading order — personal-injury summaries
   have no unified text body.

`doc_id` is **`doc_id`** if present, else **`id`** coerced to a string.
`metadata` is `{"subcorpus": <parent directory name>}` — `PISRS`,
`UradniList`, `SodnaPraksa`, `USRS` — plus every remaining non-null field that
was not consumed for text. Note that `id` is *not* reserved, so it survives into
`metadata` even when it supplied the `doc_id`. Text-only; no annotations.

## huggingface

[`slm4ie/data/extractors/huggingface.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/extractors/huggingface.py)
— each **immediate subdirectory** of `raw/<key>/` is an Arrow dataset written by
`save_to_disk()` and loaded with `load_from_disk()`. Both `Dataset` and
`DatasetDict` are handled; for a `DatasetDict` every split is iterated. A config
dir that fails to load is logged and skipped.

```text
raw/c4/
  sl/                       # config 1 — Slovene
    dataset_info.json
    state.json
    data-00000-of-00001.arrow
  hr/                       # config 2 — Croatian
    ...
```

A row, e.g. for AllenAI C4:

```python
{"text": "Dober dan, kako ste?",
 "timestamp": datetime(2019, 4, 25, 12, 34, 56),
 "url": "https://example.com/page"}
```

| Raw | → | `Document` |
| --- | --- | --- |
| **`text`** column | → | `text` — empty/missing rows skipped |
| every other non-null column | → | `metadata` |

`datetime` and `date` **values** (matched by Python type, not column name) are
converted to ISO-8601 strings so the row stays JSON-serializable. No `doc_id`
and no annotations — see [Documents without an id](#documents-without-an-id).

## External-TSV metadata

`conllu` and `tei` can merge per-document fields from a flat TSV shipped
alongside the text files, implemented in
[`slm4ie/data/metadata_sidecar.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/metadata_sidecar.py).
Used by `oss`, `kzb`, and `kas`.

The lookup key is derived from the **filename stem**, optionally narrowed by a
`key_pattern` regex whose first capture group becomes the actual key. The
matched row is projected through `fields` (which also renames columns) and
`splits` (which explodes separator-joined columns into JSON arrays). Values of
`""` or `"-"` are dropped as NA. The resulting dict is applied to **every**
document produced from that file, since rows are keyed by filename rather than
by document.

Given `oss-10000.conllu` and this config:

```yaml
oss:
  extractor: conllu
  domain: scientific
  metadata:
    path: OSS.CoNLL-U/OSS-metadata.tsv
    key_column: id
    key_from: filename_stem
    key_pattern: '^oss-(\d+)$'    # stem "oss-10000" → key "10000"
    fields:
      cerif: cerif                # tsv column → metadata field
      udc: udc
      type: doctype               # renamed on the way in
    splits:
      cerif: "|"                  # explode into a list
```

and this TSV:

```text
id	cerif	udc	type
10000	P000|T270	502(043)	Diplomsko delo
```

every document from that file gets:

```json
{"cerif": ["P000", "T270"], "udc": "502(043)", "doctype": "Diplomsko delo"}
```

`key_from` currently accepts only `filename_stem`. On the TEI utterance path,
`who` / `ana` take precedence over sidecar fields on a key collision, being the
more specific source.

## Notes

Cross-cutting behaviour worth knowing before extending the stage.

### Only three routes produce annotations

`conllu`, `tei` (on its two annotated paths), and `jsonl` (when a record
carries `paragraphs`). The other five — `json`, `text`, `macocu`, `coleslaw`,
`huggingface` — are text-only, and their datasets get no
`.annotations.jsonl.gz` at all.

### Documents without an id

`text` and `huggingface` never set a `doc_id`, so the orchestrator assigns a
positional fallback — `idx-{index:014d}` on the serial path, or a
shard-namespaced `idx-{shard:05d}-{local:010d}` when sharded. The fallback is
assigned *before* serialization, so `uid` in the output is never null; it is
simply derived from a positional id (`cc100:idx-00000000000042`) rather than a
source-stable one. Such ids are **not stable across re-extraction** if input
ordering changes. In practice only the serial fallback is reachable: `cc100` is
a single file and never shards, and `huggingface` is not a `FileBasedExtractor`.

### `json` ignores the `metadata:` block

Unlike `jsonl`, it hardcodes `text` and `doc_id`. A `metadata:` block on a
`json` entry would be silently inert — use the `jsonl` extractor if you need
configurable field names.

### Sharding is narrower than it looks

Only `text`, `macocu`, `conllu`, and
`tei` subclass `FileBasedExtractor`. `jsonl`, `json`, `coleslaw`, and
`huggingface` are plain `BaseExtractor`s and always run single-pass, regardless
of `--max-workers`.

## Adding an extractor

Subclass `BaseExtractor` (or `FileBasedExtractor`, to get sharding for free by
splitting enumeration from parsing), then call `register_extractor("<name>",
<Class>)` at module scope and import the module in
[`slm4ie/data/processing.py`](https://github.com/eriknovak/SLM4IE/blob/main/slm4ie/data/processing.py)
so registration fires. Point a new `extract.yaml` entry at the registry name.
