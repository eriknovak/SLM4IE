# Tokenizer sweep — Slovenian (2026-06-21)

End-to-end run of the tokenizer comparison sweep: 6 backends × 3 vocab sizes =
**18 runs**, trained on a 5 GB seeded random sample of the deduplicated corpus
(`pretrain/06_sentence_dedup`, seed 13), evaluated on a held-out 200 MB sample
(seed 99, 26.4M word tokens) plus the Sloleks-derived morpheme gold
(3.0M forms).

- MLflow experiment: `slm4ie/archive/tokenizer-sweep-slovenian-2026-06` (id 10) at `localhost:5555`
  — parent `sweep-train` + `sweep-eval`, 18 child runs each.
- Report source: `/vault/data/SLM4IE/tokenizers/_reports/report.{md,json}`.
- Main repo commit at run time: `25cb7f0`.

This is the **first completed run** of this sweep, so there is no prior baseline
to diff against; the numbers below are the baseline for future comparisons.

## Caveats

- **Morph metrics are relative comparators**, not absolute morphology: the gold
  is Sloleks *inflectional* silver-gold. `morph_score_f1` is low in absolute
  terms for every backend — read it as a ranking, not an accuracy.
- The eval is a held-out sample, not the whole 32 GB corpus (by design).

## Headline metrics (vocab = 64000, the best compression tier)

| backend | fertility ↓ | chars/token ↑ | rényi ↑ | morph_f1 ↑ | morph_edit ↓ | morph_consist ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| bpe | 1.173 | 4.625 | 0.478 | 0.0258 | 3.048 | 0.213 |
| charbpe | **1.170** | **4.637** | 0.479 | 0.0283 | 2.990 | 0.218 |
| unigram | 1.200 | 4.523 | 0.478 | 0.0476 | 3.169 | 0.218 |
| wordpiece | 1.198 | 4.530 | 0.490 | 0.0428 | 3.064 | 0.211 |
| morphbpe | 1.315 | 4.125 | 0.486 | **0.0627** | **2.940** | 0.160 |
| morphpiece | 1.428 | 3.800 | 0.494 | 0.0472 | 3.276 | **0.370** |

(Full 18-row table in `report.md`.)

## Findings

**1. Vocab size: bigger = better compression, monotonically.** Across every
backend, going 16k→64k lowers fertility and tokens-per-byte and raises
chars-per-token. Rényi efficiency *drops* with vocab size (larger vocab → more
rare pieces → less uniform usage), so it trades off against raw compression.

**2. Compression winners: `charbpe` / `bpe`.** At 64k they reach the lowest
fertility (~1.17) and highest chars/token (~4.63). The byte-level vs char-level
ablation is essentially a wash on compression (charbpe marginally ahead).

**3. Morphology splits into two different "best" backends:**
   - **Boundary alignment** (`morph_score_f1` + `morph_edit_distance`):
     **`morphbpe`** wins — highest F1 (~0.063, ~2.4× the plain BPE backends) and
     lowest edit distance (2.94 at 64k). This is the constrained-training
     backend doing exactly what it's designed for.
   - **Segmentation consistency** (`morph_consistency`): **`morphpiece`**
     dominates by a wide margin (0.36–0.37 vs ≤0.22 for everything else) — its
     MorphTable produces consistent splits of shared stems. Notably `morphbpe`
     is *lowest* on consistency (0.12–0.16), so the two morph backends optimize
     different morphological properties.

**4. The compression↔morphology trade-off is real.** The morph-aware backends
pay for their morphology: `morphbpe`/`morphpiece` have the highest fertility
(1.32 / 1.43 at 64k vs 1.17 for charbpe) — i.e. they fragment more. Standard
backends compress better but align worse to morpheme boundaries.

**5. `wordpiece-16000` is the outlier to avoid** — worst fertility (1.99) and
chars/token (2.72) by a large margin, though it recovers by 64k.

## Recommendation for LM pretraining

No single winner — it depends on the objective:

- **Pure efficiency / smallest sequences:** `charbpe-64000` (or `bpe-64000`).
- **Morphological awareness** (relevant to this project's Slovenian IE focus):
  `morphbpe-64000` — best boundary alignment, at ~12% higher fertility than
  charbpe. If consistency of stem splits matters more than boundary F1,
  `morphpiece-64000` instead.
- **64k is the sweet spot** across backends for compression and edit distance;
  16k only competes on Rényi efficiency.

## Operational notes (what this run cost)

- **Training:** 2 workers (after an 18-wide run OOM-crashed tmux). `morphbpe`
  constrained training dominates wall-clock and scales ~quadratically in vocab:
  `morphbpe-32000` took 4.4 h, `morphbpe-64000` **14.6 h** — the long pole of
  the whole sweep. `morphpiece` was cheap (~8 min each).
- **Analysis:** originally hung (`morph_consistency` had a non-terminating
  sampler — fixed in PR #13). Post-fix the full 18-run analysis completes in
  **~22 min** (8-wide process pool) instead of never.

## Follow-ups

- For faster reruns, training time is dominated by `morphbpe` at high vocab —
  consider a smaller sample just for that backend, or a vocab cap.
- Analysis is now dominated by the corpus pass over 26.4M eval words (~4.4
  min/task). Shrinking the eval sample (corpus stats stabilize well below
  200 MB) would bring the sweep to single-digit minutes — deferred from PR #13.
- `morph_consistency`'s per-group sample cap (200) can be revisited now that the
  metric is fast, for a lower-variance estimate.
