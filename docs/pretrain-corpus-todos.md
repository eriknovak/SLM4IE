# Pretraining Corpus — Follow-up TODOs

Action items from the data-analysis review of the `to_pretrain.py --all` run that
finished 2026-06-07 (final corpus: 27,182,636 docs / ~7.87B words across 20
datasets, 9 domains). Ordered by priority. Severity: 🔴 blocker · 🟠 important ·
🟡 nice-to-have.

## ✅ 1. Adult / SEO spam contamination — RESOLVED (closed 2026-06-10)

The global top-200 word table was polluted with escort/porn/dating vocabulary at
extreme ranks — `prostitutke` was the ~10th most frequent content word (10.9M),
plus `porno`, `seks`, `kurba`/`kurbe`, `zmenke`, `masaža`, and supporting
`ženske`/`telo`/`posnetki`/`brezplačno`. This is grammatical Slovenian, so
neither the language filter nor Gopher heuristics removed it — a blocker for a
general/professional IE model.

**Fix:** new `spam` stage at `02_spam` (after language, before quality), using a
per-language lexicon + URL/domain blocklist + optional pluggable model hook. See
`docs/superpowers/specs/2026-06-08-spam-adult-filter-design.md`.

- [x] URL/domain blocklist filter (adult/escort/SEO domains), language-agnostic,
      matched against `metadata.url` (`slm4ie/data/spam/domains.txt`, 67 domains).
- [x] Per-language lexicon filter — curated `sl` + `en` adult/spam lists under
      `slm4ie/data/spam/<code>/`; LDNOOBW auto-loaded for other languages. Lists
      expanded 2026-06-08 (sl/adult 93, sl/spam 32, en/adult 53, en/spam 29).
- [x] Pluggable model hook (`spam.model`), off by default; lexicon + URL shipped.
- [x] Configurable, sentinel-tracked via `configs/data/pretrain.yaml::spam`;
      stage folders renumbered; full test suite green.
- [x] Full pipeline re-run from `02_spam` (finished 2026-06-10). Spam stage
      dropped **1,109,079 docs (1.89%)** (adult_lexicon 662,987 · spam_lexicon
      427,716 · url 827 · both 17,549).
- [x] **Verified outcome:** every tracked offender is gone from the new top-200
      (`prostitutke` 10.9M→below #200; `porno`/`seks`/`kurba`/`zmenke`/`masaža`
      all gone), the new top-25 is clean, and the domain mix is undistorted
      (web 82.3%→83.8%). New corpus: 26,502,111 docs / ~6.95B words at
      `06_statistics/`.

Remaining cleanup (not blockers):

- [x] Delete orphaned old-numbered dirs (`02_quality…05_statistics`, ~370 GB)
      (done 2026-06-10).
- [ ] (Optional) per-source spam quantification; merge the UT1 adult domain list
      into the blocklist at runtime.

## 🟠 2. Heavy web skew (82% of words)

Web sources contribute 82.3% of words; curated edited Slovenian
(gigafida + kas + wiki) is only ~13% combined. Fine for breadth, but risks
under-using high-quality text for zero-shot IE.

- [ ] Configure source-weighted sampling at training time using the existing
      `dataset` / `domain` metadata (upweight gigafida/kas/wiki/parliamentary,
      downweight raw CommonCrawl).
- [ ] Decide target domain mixture and document it in the training config.

## 🟡 3. Residual foreign-language leakage — ACCEPTED AS-IS (closed 2026-06-08)

English/foreign tokens appear in the top 200 (`the`, `of`, `and`, `more`, `de`;
<1% of mass). Diagnosed with `scripts/analysis/diagnose_language_leakage.py`
(read-only, 1,772 docs across all 20 datasets sampled from the existing corpus):

- **Whole foreign docs slipping past the doc-level filter: 0.11%** (2/1772) —
  the lingua doc-level filter is working as intended.
- **Embedded foreign text: 28% of docs** carry some non-`sl` paragraph, but only
  **3.4% of character mass** is foreign, and **~88% of the leakage tokens
  (`the`/`of`/`and`) sit inside paragraphs lingua labels Slovenian** — i.e. they
  are isolated English words, brand names, and loanwords within Slovenian
  sentences (e.g. "Chang-shu purple čaj", "Victoria's Secret", "SKF TMSP 1"),
  not coherent foreign blocks.

**Decision:** embedded foreign text inside Slovenian documents is acceptable; the
problem is not whole-doc leakage (which is ~0). No scrubber and no config change.
Even a paragraph-level scrubber would only touch ~12% of the leakage tokens.

- [x] Diagnose embedded-vs-whole-doc leakage (read-only script; verdict above).
- [x] Confirm doc-level filter is effective (0.11% whole-foreign) — no config
      tightening needed.
- [ ] (Descoped) sub-document scrubber — not worth it for embedded loanwords.

## 🟡 4. Under-represented domains

Medical (0.009% of words) and student (0.01%) are negligible.

- [ ] Confirm whether medical/student IE are in scope.
- [ ] If yes, source additional in-domain Slovenian data; otherwise leave as-is.

## 🟡 5. Scale / token budget

~7.87B words ≈ ~11–15B tokens — on the low side for SLM pretraining.

- [ ] Decide epoch count / whether to accept the Slovenian-focused budget.
- [ ] Re-estimate token count with the actual trained tokenizer (current figure
      is a rough word-to-subword extrapolation).
