# Experiment tracking

SLM4IE tracks its pipelines in [MLflow](https://mlflow.org/). The write side is
`slm4ie.utils.mlflow` (thin, no-op-safe helpers); the read side is
`slm4ie.utils.mlflow_report` (report-ready tables). Marimo notebooks under
`notebooks/` render the reports from the store.

The tracking URI resolves from a config `tracking_uri`, else
`$MLFLOW_TRACKING_URI`, else a local `sqlite:///mlflow.db` fallback. Every
producer logs **one run per build**, keyed by a content digest and upserted:
re-running an unchanged build is a no-op; `--force` replaces the run.

## Experiments

Experiments follow `slm4ie/<workstream>/<dataset-or-model>`.

| Experiment | Producer | What it records |
|---|---|---|
| `slm4ie/data/extract` | `extract.py --mlflow` | per-source row counts + quality profile |
| `slm4ie/data/pretrain` | `to_pretrain.py --all --mlflow` | step-indexed stage funnel + mixture shares |
| `slm4ie/data/tasks` | task converters `--mlflow` | per-split counts + label distribution |
| `slm4ie/tokenization/slovenian` | tokenizer sweep | per-tokenizer metrics + report |

The data-pipeline experiments are disabled by default (`mlflow.enabled: false`
in each config); the loggers are post-hoc, so enabling them on an already-built
tree logs from existing artifacts without reprocessing.

## Reserved conventions (forthcoming workstreams)

Model pretraining and downstream evaluation are not built yet. Their tracking is
reserved here so runs slot into a consistent scheme when the scripts
(`scripts/train.py`, `scripts/evaluate.py`) grow real bodies.

- **`slm4ie/pretrain/<model>`** — one run per pretraining run of `<model>`.
- **`slm4ie/eval/<benchmark>`** — one run per evaluation on `<benchmark>`
  (e.g. `slm4ie/eval/ner`); the task-native experiments (`slm4ie/ner/<dataset>`
  etc.) remain reserved for per-dataset eval breakdowns.

### Tag vocabulary

Every run carries `run_type` and `git_commit`; producers additionally tag:

| Tag | Meaning |
|---|---|
| `run_type` | `data_pipeline`, `sweep`, `pretrain`, or `eval` |
| `pipeline` / `phase` | producer stage (`extract`/`pretrain`/`tasks`, or `train`/`eval`) |
| `corpus_digest` | content digest keying the build (upsert key) |
| `config_hash` | hash of the run's output-affecting config |
| `git_commit` | HEAD at build time |
| `model_type` / `model_version` | model family + variant |
| `task` / `dataset` / `role` | task identity (for the task/eval experiments) |

### Lineage contract

Producers declare their output as a **`produced`** dataset input
(`log_dataset_input(name, digest, source, context="produced")`); consumers
declare the same `name` + `digest` with a consumption context, so the MLflow UI
links a consumer back to the exact artifact it read. The spine is
**corpus → tokenizer → model → eval**:

- The **corpus build** (`to_pretrain`) produces `pretrain/05_2_dedup`.
- The **tokenizer sweep** produces `tokenizer/<key>` (already wired — the
  train child run logs a digest over each exported tokenizer's artifacts).
- A **pretraining run** logs `pretrain/05_2_dedup` and the chosen
  `tokenizer/<key>` as `training` inputs, and produces its model.
- An **eval run** logs the model and the task-split digest
  (`tasks/<task>/<dataset>`) as `eval` inputs.

Digests come from `slm4ie.data.curate.corpus_digest` (a stat-based manifest
hash), so producer and consumer resolve to the same entity as long as both hash
the same directory.

!!! note "Optuna is deferred"
    Hyperparameter / ablation search over pretraining (Optuna) builds on this
    MLflow backbone but is not set up yet. It is additive — trials nest under
    the reserved `slm4ie/pretrain/*` experiments — so it can be added later
    without reworking the conventions above.
