# 2. `role: held_out` gates out the train split

Status: Accepted

## Context

The three task converters (`to_spans`, `to_sentiment`, `to_superglue`) read a
flat `<task>/<dataset>` registry (`configs/data/tasks.yaml`) in which every entry
carries a `role` of `finetune_and_eval` or `held_out`. The registry, its docs,
and `README` all described `role` as what "enforces train/test isolation
regardless of directory placement".

In practice it enforced nothing. `role` was validated in `tasks.py` but never
read by any converter: a `held_out` entry flowed through the exact same write
path as a `finetune_and_eval` one, and every `held_out` entry declared a `train`
split it happily wrote. A held-out dataset is meant to be untouched by
fine-tuning, so emitting a `train.jsonl.gz` for it silently contradicts the
isolation the field was supposed to guarantee.

This landed alongside #38, which folded the three converters' triplicated CLI and
orchestration into one shared driver + `TaskConverter` registry in
`slm4ie/data/`. With the write path unified, the role gate has one place to live.

## Decision

A `held_out` entry no longer writes a `train` split. The shared driver
(`slm4ie/data/task_converter.py`) computes the target split set once per entry:
the declared splits for a `finetune_and_eval` entry, and the declared splits
*minus* `train` for a `held_out` entry. Only those splits' output files are
opened, so a `held_out` entry never produces a `train.jsonl.gz`.

Nothing is dropped — the population is re-bucketed, not truncated:

- **Hash-policy families** (`to_spans`, `to_sentiment`) hash each record's stable
  id into the target splits. When `train` is absent the whole population is
  spread ~50/50 across the remaining `val`/`test` splits instead of collapsing
  the train bucket into one split. A held-out hash entry therefore writes the
  same number of records as it would as a finetune entry, just across two splits
  instead of three.
- **Source-policy family** (`to_superglue`) keeps each record in its source
  split and simply never reads the `train.jsonl` source file for a `held_out`
  entry.

`train` — not `test` — is the split dropped. Held-out evaluation uses the
labeled `val` split, whereas SuperGLUE-SL `test` sets are blind and unlabeled;
collapsing to `test` would destroy the only labeled eval split.

The `splits:` of every `held_out` entry in `tasks.yaml` was trimmed to drop
`train`, so the config reflects reality and no empty file is implied. The gate in
the driver is independent of the config: it drops `train` for any `held_out`
entry even if a `train` split is still declared, so the behavior cannot silently
regress by re-adding a `train:` line.

## Consequences

- `role` now enforces train/test isolation instead of merely annotating it: a
  held-out dataset can no longer leak a fine-tuning split.
- The output tree changes for held-out datasets: `ner/suk` and every SuperGLUE-SL
  subtask now write `val`/`test` only. Any consumer that assumed a `train.jsonl.gz`
  for those datasets must stop doing so — there was never labeled held-out train
  data to rely on.
- Held-out hash datasets re-bucket ~50/50 val/test rather than 80/10/10
  train/val/test. `ner/suk` is the only held-out hash entry today; the ratio is
  documented in the driver and unit-tested.
- The behavior is covered by a role-gating acceptance test that converts the same
  records as both roles and asserts the held-out run drops `train` while
  preserving the total record count.
