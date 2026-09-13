# Tokenizer sweep re-baseline + derivational morph gold (Slovenian)

**Date:** 2026-06-23
**Commits:** derivational gold (#18), download URL (#19), auto-unzip (#20),
statistical parity for `*_deriv` (this branch).
**Experiment:** `slm4ie/archive/tokenizer-sweep-slovenian-2026-06` (MLflow, localhost:5555)
**Report:** `/vault/data/SLM4IE/tokenizers/_reports/report.{md,json}`

## What changed since 2026-06-21

Added a **derivational** morph-gold axis from the Sloleks 2.0 word relations
(CLARIN 11356/1986; 42,623 derivationally-decomposed lemmas, ~3.3k linguist-
verified). Three effects:

1. The derivational morphemes are **unioned into the backend morpheme table**
   (`morphbpe`/`morphpiece`), so those backends train on derivational pieces too.
2. The analysis emits **`*_deriv` columns** scored against the derivational gold.
3. The two derivational boundary metrics (`morph_score_f1_deriv`,
   `morph_edit_distance_deriv`) get **full statistical parity** with the
   inflectional track: bootstrap CIs + Holm-corrected paired significance,
   resampled over the derivational gold forms (`morph_consistency_deriv` stays a
   point estimate, like `morph_consistency`).

All 18 runs (6 backends x 16k/32k/64k) were retrained from scratch with the
union table, replacing the 2026-06-21 numbers.

## Sanity check: inflectional axis is stable

Inflectional `morph_score_f1` (vocab 64k) is unchanged vs 2026-06-21, so the
re-baseline did not disturb the existing axis:

| backend | Jun-21 | Jun-23 |
| --- | --- | --- |
| bpe | 0.0258 | 0.0262 |
| morphbpe | 0.0627 | 0.0638 |
| morphpiece | 0.0472 | 0.0464 |

(Cross-run `morph_consistency` shifted uniformly downward, including for the
unchanged non-morph backends, so it reflects a metric/sample change between the
two runs rather than a tokenizer effect — do not cross-compare it across sweeps.)

## New: derivational axis (`morph_score_f1_deriv`, vocab 64k)

With bootstrap CIs and compact-letter significance groups (a shared letter =
not significantly different; here every backend is in its own group, so all
pairwise differences are significant):

| backend | f1_deriv | 95% CI | group |
| --- | --- | --- | --- |
| morphpiece | 0.7592 | [0.7554, 0.7630] | a |
| unigram | 0.3263 | [0.3237, 0.3289] | b |
| morphbpe | 0.2706 | [0.2680, 0.2733] | c |
| wordpiece | 0.1872 | [0.1848, 0.1895] | d |
| charbpe | 0.1551 | [0.1528, 0.1575] | e |
| bpe | 0.1102 | [0.1081, 0.1123] | f |

### ⚠️ morphpiece's derivational lead is largely circular — do not over-read it

`morphpiece` tokenizes a known word by **direct MorphTable lookup** (emits the
table's morphemes verbatim). The table is built from the same union lexicon the
derivational gold comes from: **67.3% of the gold lemmas are in morphpiece's
table with their gold decomposition**, so it reproduces those by construction —
its 0.76 mostly measures *table coverage of the gold*, not generalization. (Not
1.0 because the other ~33% fall back to byte-level BPE.)

`morphbpe` uses the table only to **constrain training merges** (inference is
standard BPE, no lookup), so its 0.27 is a far fairer derivational signal.

**Honest reading** (excluding morphpiece's leakage): among backends that do not
memorize the table, `unigram` (0.33) leads, then `morphbpe` (0.27) >
`wordpiece` (0.19) > `charbpe` (0.16) > `bpe` (0.11) — all significantly
separated. The derivational ranking differs from the inflectional one
(`morphbpe` is no longer the clear winner once derivation is the target).

## Compression vs morphology trade-off (unchanged)

Morph-aware backends still pay in compression: at 64k, `morphbpe` fertility
1.338 / `morphpiece` 1.562 vs `charbpe` 1.170 (best). Compression leaders for
LM pretraining remain `charbpe`/`bpe`-64k; the morph-aligned pick is
`morphbpe`-64k.

## Cost / operational notes

- **First retrain OOM'd** at 18-wide concurrency (`train.py --all` default = all
  cores): a `unigram` worker was killed → `BrokenProcessPool` cascade. Fixed by
  `--max-workers 4`. **Always cap concurrency** on this box (125 GB / 40 cores).
- `morphbpe-64000` took ~13.5 h (single-threaded merge loop, ~11 GB RSS, stable)
  — matches the prior sweep's 14.6 h. It is the intrinsic long pole of the sweep,
  **not** a regression from the union table.
- Re-running just `analyze --all --force` (after the parity code change)
  regenerates `eval_units.npz` with the derivational per-form arrays and the
  report with deriv CIs — no retrain needed.

## Caveats (per CLAUDE.md)

Both morph golds are **silver** (inflectional = heuristic alignment;
derivational = rule-generated, ~8% linguist-verified). All morph metrics are
**relative comparators**, not absolute morphology. The derivational metric is
only meaningful as a generalization test for backends that do not embed the gold
in a lookup table (i.e. everything except `morphpiece`).

## Follow-ups

- **Done (this branch):** full statistical parity for the derivational boundary
  metrics (`analysis.py` generalized 1→N golds).
- Consider a **verified-only** derivational slice (the ~3.3k linguist-scored
  lemmas) to reduce silver noise.
- Consider annotating/excluding table-lookup backends (`morphpiece`) from the
  derivational leaderboard given the circularity above.
- Backlog: replace the inflectional heuristic gold with eLex-2023 BSSJB true
  gold (small, needs author contact).
