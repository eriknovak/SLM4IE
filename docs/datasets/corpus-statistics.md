---
title: Corpus Statistics
---

# Corpus Statistics

These numbers describe the **final pretraining corpus** — the output of the
verified [Download → Extract → Pretraining corpus](../user-guide/data-pipeline/pretrain.md)
pipeline, after language filtering, spam removal, Gopher quality/repetition
heuristics, and exact + sentence deduplication. They are read straight from
`pretrain/06_statistics/aggregate.json`.

!!! success "Verified result"
    This is the one workflow that has been run end-to-end. The figures below are
    the actual computed corpus statistics, not estimates.

## Corpus totals

| Metric | Value |
|--------|-------|
| Documents | **28,282,149** |
| Words | **7,230,628,501** (≈ 7.23B) |
| Source datasets | 21 |
| Domains | 10 |

## By domain

Word counts and their share of the corpus, by domain (sorted by size). The
corpus is web-dominated, as expected for a Slovenian pretraining mix, with
substantial academic, news, and legal tails.

| Domain | Documents | Words | Avg words/doc | Share |
|--------|-----------|-------|---------------|-------|
| web | 24,899,774 | 5,831,914,720 | 234 | 80.66% |
| academic | 51,765 | 449,531,892 | 8,684 | 6.22% |
| news | 1,779,521 | 284,960,644 | 160 | 3.94% |
| mixed | 615,828 | 257,934,856 | 419 | 3.57% |
| legal | 227,269 | 160,839,713 | 708 | 2.22% |
| scientific | 19,319 | 129,364,916 | 6,696 | 1.79% |
| parliamentary | 637,491 | 106,350,318 | 167 | 1.47% |
| wiki | 44,749 | 8,192,383 | 183 | 0.11% |
| student | 6,082 | 908,858 | 149 | 0.01% |
| medical | 351 | 630,201 | 1,795 | 0.01% |

## By dataset

Per-source contribution to the deduplicated corpus (sorted by word count).

| Dataset | Documents | Words | Avg words/doc | Share |
|---------|-----------|-------|---------------|-------|
| `c4` | 3,985,511 | 1,453,563,777 | 365 | 20.10% |
| `fineweb2` | 6,749,899 | 1,096,911,118 | 163 | 15.17% |
| `culturax` | 3,464,458 | 968,928,477 | 280 | 13.40% |
| `classla_web_sl` | 3,602,316 | 930,039,661 | 258 | 12.86% |
| `finepdf` | 309,749 | 473,190,787 | 1,528 | 6.54% |
| `kas` | 51,765 | 449,531,892 | 8,684 | 6.22% |
| `cc100` | 2,518,393 | 341,388,936 | 136 | 4.72% |
| `hplt` | 1,883,163 | 292,452,352 | 155 | 4.04% |
| `slovenian_news` | 1,779,521 | 284,960,644 | 160 | 3.94% |
| `macocu_sl` | 2,386,285 | 275,439,612 | 115 | 3.81% |
| `gigafida` | 613,814 | 257,615,039 | 420 | 3.56% |
| `coleslaw` | 227,269 | 160,839,713 | 708 | 2.22% |
| `oss` | 18,836 | 124,274,154 | 6,598 | 1.72% |
| `siparl` | 523,707 | 80,330,961 | 153 | 1.11% |
| `parlamint_si` | 113,784 | 26,019,357 | 229 | 0.36% |
| `classlawiki_sl` | 44,749 | 8,192,383 | 183 | 0.11% |
| `kzb` | 483 | 5,090,762 | 10,540 | 0.07% |
| `solar` | 6,082 | 908,858 | 149 | 0.01% |
| `povejmo_vemo_med` | 351 | 630,201 | 1,795 | 0.01% |
| `suk` | 2,011 | 315,177 | 157 | 0.00% |
| `ssj500k` | 3 | 4,640 | 1,547 | 0.00% |

## Most frequent words

The top of the corpus-wide word-frequency table (the stats stage records the
top 5,000 words plus bigrams/trigrams). The leaders are the expected Slovenian
function words and connectives:

| Rank | Word | Count |
|------|------|-------|
| 1 | še | 38,931,957 |
| 2 | tem | 23,917,296 |
| 3 | ter | 23,397,049 |
| 4 | zaradi | 17,173,488 |
| 5 | zato | 14,690,121 |
| 6 | leta | 13,654,083 |
| 7 | ima | 12,459,576 |
| 8 | glede | 10,559,559 |
| 9 | strani | 10,016,763 |
| 10 | svoje | 9,264,696 |

## How these are produced

The `stats` stage (`06_statistics/`) runs single-process over the deduplicated
corpus and writes two artifacts:

- `aggregate.json` — corpus-wide totals, the `by_domain` and `by_dataset`
  breakdowns shown above, and the top-200 word-frequency table.
- `per_dataset/<key>.json` — the same document/word breakdown per source.

To regenerate them, rerun the stats stage:

```bash
uv run python scripts/data/to_pretrain.py --all --stage stats
```
