# SLM4IE

The shared vocabulary of this repository: the nouns the pipelines, configs and
experiment records use for their own concepts. Metrics and reader-facing
abbreviations live in [experiments/GLOSSARY.md](experiments/GLOSSARY.md)
instead.

## Language

### Pipeline shape

**Conversion route**:
One of the three ways an extracted dataset is consumed downstream — pretraining
curation, task conversion, tokenizer quality. Each owns a disjoint output tree.
_Avoid_: pipeline, path, fork

**Step package**:
One package under `slm4ie/data/` per pipeline step, every one of the same
shape: `config.py` loads the step's registry, `run.py` is its run module,
`tracking.py` logs to MLflow, and a backend-registry subpackage holds its
backends.
_Avoid_: route package, stage package, module group

**Run module**:
The argv-free entry point of a step package, `run.py`, called by the script
subcommand of the same name; a stage module exposes `run` for the same reason.
_Avoid_: driver, runner, main

**Data tier**:
One of the seven top-level folders under the data root, each holding the output
of one stretch of the pipeline. A tier is larger than a stage: all eight
curation stages write inside `pretrain/`.
_Avoid_: layer, level, stage

**Shared data**:
Data any experiment may read, built once at a tier above `experiments/` rather
than rebuilt per experiment.
_Avoid_: shared asset, common data, global data

**Dataset key**:
The short identifier a dataset is declared under in a registry, carried through
every tier as `<key>` and fixed in the filenames it produces.
_Avoid_: dataset id, dataset name, source id

**Digest**:
The content hash identifying one build of an output tree. It changes only when
the set of output shards changes, so a consumer can name the exact corpus it
read.
_Avoid_: checksum, fingerprint, version

### Extraction

**Extractor**:
The backend that reads one input format — CoNLL-U, TEI, JSONL and the rest —
into documents.
_Avoid_: parser, loader, reader

**Download source**:
The backend a raw dataset arrives through, either HTTP or HuggingFace.
_Avoid_: source, provider, origin

**Extracted dataset**:
The normalized form of one downloaded dataset: a text file and, when the source
carries annotations, its sidecar.
_Avoid_: extraction, extracted format, processed dataset

**Document**:
One text record with its provenance and metadata, the unit every route moves.
Curation reshapes it into datatrove's own document shape.
_Avoid_: row, sample, entry

**Source field**:
The provenance field on a document, naming the dataset it came from.
_Avoid_: source, origin, dataset

**Document id**:
The `doc_id` of a document: unique within its dataset and stable across
re-extraction, so that `<dataset key>:<document id>` (the `uid`) is the key
every consumer joins on. It is the native id when the source has one that is
unique within the dataset, else a positional id. Extraction asserts the
uniqueness: a repeat with identical text is dropped, a repeat with different
text fails the build.
_Avoid_: id, key, identifier, document key

**Native id**:
The identifier the raw source itself carries for a document, kept verbatim as
`native_id` for tracing it back; it may repeat and may be absent.
_Avoid_: original id, source id, raw id, external id

**Positional id**:
The document id `<unit>:<ordinal>` minted when a source has no usable native
id: the raw file's path under the dataset dir without its suffix, then the
document's zero-padded position in that file.
_Avoid_: synthetic id, fallback id, line id, index

**Domain**:
The content label a dataset carries onto its documents — web, news, legal,
parliamentary, scientific and so on. Corpus sampling and statistics group by it.
_Avoid_: genre, category, topic

**Annotations sidecar**:
The gzipped per-document annotation file beside an extracted dataset's text,
joined on the fly and never merged into it.
_Avoid_: sidecar, annotation file, labels file

**Stub**:
An annotations line carrying only identifiers, written for a document that has
none so the sidecar stays aligned with the text line for line.
_Avoid_: placeholder, empty record, null row

**Metadata table**:
The flat per-document TSV or CSV a few datasets ship beside their text, merged
into document metadata during extraction.
_Avoid_: metadata sidecar, sidecar, lookup table

### Pretraining curation

**Stage**:
One of the eight ordered steps of pretraining curation, each writing its own
durable folder.
_Avoid_: step, phase, pass, tier

**Scoped stage**:
A stage that processes one dataset at a time and can run over a subset of them.
_Avoid_: per-dataset stage, partial stage

**Corpus stage**:
A stage that reads every dataset at once and therefore runs only over the full
roster.
_Avoid_: global stage, full-corpus stage

**Shard**:
One gzipped JSONL piece of a stage's output for one dataset.
_Avoid_: chunk, part, batch

**Roster**:
The set of datasets one curation run covers. Adding or removing a dataset
invalidates every corpus stage.
_Avoid_: dataset list, selection, manifest

**Override**:
A per-dataset block that deep-merges onto a scoped stage's defaults, so one
dataset can differ without forking the config. Corpus stages reject them.
_Avoid_: exception, patch, custom config

**Config bucket**:
The datasets of a scoped stage that resolve to the same effective config and
so run together in one executor, named by that config's hash. Each bucket
keeps its own log folder.
_Avoid_: group, batch, partition

**Unit**:
One stage for one dataset (a scoped stage) or one corpus stage: the smallest
piece of the pipeline that is built, versioned and rebuilt on its own.
_Avoid_: job, step, task

**Sentinel**:
The completion marker a stage writes, carrying the config hash, stage version
and input digest that produced it plus the document digest of what it wrote.
Any mismatch invalidates that stage and every stage after it.
_Avoid_: checkpoint, marker, lock, cache

**Stage version**:
A hash of the code that runs a stage, computed automatically. Any edit to that
code — a fix, a refactor, a comment — changes it and reruns the stage.
_Avoid_: revision, version number

**Document digest**:
An order-independent hash over the documents a stage wrote for one dataset. It
ignores shard layout, worker count, compression and timestamps, so it changes
only when the documents do; a rerun that reproduces it leaves downstream
stages current.
_Avoid_: checksum, fingerprint, version

**Input digest**:
What a unit was built from: the upstream units' document digests, or for
`convert` the size and content hash of the extracted source file.
_Avoid_: input fingerprint, upstream hash

**Lock file**:
The committed record of every sentinel's hashes and digests, so each commit
names the corpus it expects and `status` can report where the disk differs.
_Avoid_: manifest, snapshot, dvc.lock

### Task conversion

**Entry**:
One `<task>/<dataset>` row in the task registry, the unit that names a source, a
converter and a role.
_Avoid_: task, dataset, benchmark

**Entry source**:
The block on an entry naming which keys it reads and where from.
_Avoid_: source, input

**Source kind**:
Whether an entry reads the extracted tier or goes straight to raw, which
task-native bundles do.
_Avoid_: source type, mode

**Task family**:
A group of entries sharing one output schema: spans, sentiment, and the
SuperGLUE subtasks.
_Avoid_: task type, benchmark family

**Converter**:
The backend that turns one task family's entries into its schema.
_Avoid_: adapter, parser, transformer

**Role**:
The field on an entry that fixes train/test isolation: `finetune_and_eval` keeps
every split, `held_out` drops train.
_Avoid_: split policy, usage, mode

**Split policy**:
How a converter's yielded key becomes an output split: hashed across the target
splits, or kept verbatim from the source.
_Avoid_: split strategy, split rule

**Variant**:
Which translation of SuperGLUE-SL an entry reads, human-translated or
machine-translated.
_Avoid_: version, edition, flavor

### Configs

**Registry**:
A declarative config the repo owns and many experiments read: the dataset
catalogs and the task registry. The in-process name-to-class map is a backend
registry, never bare.
_Avoid_: global config, settings

**Backend**:
One pluggable implementation registered under a name and selected by config.
Always qualified: converter backend, download-source backend, tokenizer backend.
_Avoid_: plugin, driver, engine

**Dials**:
The configs one experiment owns, holding the values it varies.
_Avoid_: experiment config, settings, options

**Local overlay**:
A gitignored `*.local.yaml` sibling deep-merged over a committed config, holding
secrets and machine-specific values.
_Avoid_: override file, secrets file, env file

### Experiments

**Experiment**:
One hypothesis under test, with its own slug, branch, dials and record.
_Avoid_: study, trial, MLflow experiment

**Slug**:
The two to four kebab-case words naming what an experiment puts under test.
Every other name for that experiment derives from it.
_Avoid_: id, name, key

**Category**:
Which of `data`, `methods` or `validation` an experiment belongs to, chosen by
what its hypothesis is about.
_Avoid_: group, area, folder

**Record**:
The one file per experiment carrying its hypothesis, decisions, findings and
verdict.
_Avoid_: report, writeup, log

**Findings book**:
The index of every experiment and what it showed.
_Avoid_: index, summary, changelog

**MLflow run**:
One tracked execution, logged with its params, metrics and artifacts.
_Avoid_: run, job, trial

### Tokenizers

**Sweep**:
The full grid of tokenizer backends by vocabulary size, trained and scored
together.
_Avoid_: ablation, grid search, benchmark

**Sweep run**:
One backend at one vocabulary size, with its own artifacts folder under the
tokenizers tier.
_Avoid_: run, model, variant

**Morpheme lexicon**:
The form-keyed segmentations and morpheme inventory derived from Sloleks,
feeding both the morph-aware backends and the morph metrics.
_Avoid_: dictionary, vocabulary, gold set

**Silver gold**:
The morpheme segmentation the morph metrics score against. It is inflectional
rather than true morphology, so what it feeds are relative comparators.
_Avoid_: gold standard, ground truth, reference segmentation
