---
title: KPI coverage of the curated Slovene corpus under source and document-level domain labels
slug: kpi-coverage-slovenian
short: KPI coverage check
category: data
branch: exp/kpi-coverage-slovenian
base_commit: bb8c5ad
status: draft
ticket: "#3"
pr:
mlflow:
builds_on:
  - ../../data/curation-thresholds-slovenian/
  - ../../data/curation-quality-slovenian/
  - ../../data/data-landscape-slovenian/
varies: []
uses: []
tests: []
evidence: {}
types: [assessment]
concluded:
---

# KPI coverage of the curated Slovene corpus under source and document-level domain labels

## TL;DR

- **Hypothesis**: open — whether the curated Slovene corpus meets the four data targets the project committed to, once each document is labelled with its own domain rather than its source's.
- **Next**: prepare the two corpus builds, the judged gold and the catalogue as this experiment's inputs, then build the domain classifier.

## Hypothesis

- **Builds on**: the curation-thresholds experiment, which found that per-source filter settings keep about a quarter more good text than the shared pipeline and left the per-domain volumes of that loosened build untabled. It also answers what the data-landscape survey, which found open native Slovene text covers science on both targets and medicine only on tokens, left open. That survey counted what exists; this experiment reads the corpus the project actually built, and asks whether medical text hidden inside the web crawls changes the medical verdict.
- **Statement**: the curated Slovene corpus meets all four data targets, and medicine clears the examples target only once each document carries its own domain label rather than its source's.
- **Rationale**:
  - **Every document currently carries its source's domain, so medical text inside the web crawls is invisible.** The web crawls hold more than three quarters of the corpus. In the judged sample of the curation-quality experiment, four in a hundred kept web documents were read as medical by the judge, which projects to about a million web documents on the baseline build. Even at a precision of 0.8 that is far past the ten thousand documents the examples target asks for.
  - **Under source labels the targets split cleanly.** The statistics stage already shows science past both targets and medicine past the token target but far short of the document target on both builds, so only the medical document count can change its verdict.
  - **A shortfall has two possible causes that call for different follow-ups.** A domain that had the material before curation and lost it points at a filter to tune. A domain that never had it points at the open datasets the landscape survey catalogued, or at building examples.
  - **The data can carry the bars, by the counts on disk.** Both builds, baseline and loosened, have a finished statistics stage with per-domain documents and words, and clear the corpus-wide and per-domain token targets even with the four gated sources removed. The judged sample holds 637 documents the judge called medical, 196 of them outside the medical source, and the judge agrees with a person on medical-or-not in 142 of 145 hand-labelled documents. The classifier pass over the roughly 41 million documents of the loosened build is one tracked run of a few CPU hours. Counts in [the per-domain totals of both builds](tables/scoping-domains-by-build.csv) and [the judged gold per domain](tables/scoping-gold-by-domain.csv).
- **Predictions**: every clause is read on the final corpus of the validated build ([D1]), with a token counted as a word times two, a document as one example, and a domain assigned by source label or by the document-level classifier ([D2]); a pass is a total above its target.
  - **H1 — The corpus holds enough Slovene text as a whole**: confirmed if the final corpus exceeds five billion estimated tokens both with every source and with the gated sources removed; refuted if either count falls short — the public release would then need more open text before any domain question matters.
  - **H2 — Science and medicine each hold enough text to pretrain on**: confirmed if the scientific-plus-academic domain and the medical domain each exceed half a million estimated tokens under source labels; refuted if either falls short — that domain's text would then be a source gap or a curation loss to attribute.
  - **H3 — Medicine reaches the examples target only when documents are labelled individually**: confirmed if medical documents stay below ten thousand under source labels and exceed ten thousand under document labels; refuted if the document-level count also stays below ten thousand — medical examples would then have to be built, not found, even inside the web crawls.
  - **H4 — Science reaches the examples target under both labelling rules**: confirmed if scientific-plus-academic documents exceed ten thousand under source labels and under document labels; refuted if either count falls short — the per-document reading would then show that the source label over-credits science.
  - **Q1 — Where does each shortfall arise, before curation or at a filter stage?**: read on the table of documents per domain at every curation stage of both builds ([M1], [M6]), where a domain that had the material at the converted stage and lost it is a curation loss naming the stage, and one that never had it is a source gap; robust if the stage or gap named is the same on the loosened and the baseline build — the answer feeds the follow-up, a filter to tune or the catalogue datasets that would close the gap. Added 2026-10-20 at the data phase.
  - **Verdict rule**: confirmed when all four bars hold; refuted when any bar is refuted; inconclusive when the classifier fails its quality gate ([D3]) so that H3 and H4 cannot be read under document labels. Q1 is read beside that outcome and does not change it.
  - **Read beside the clauses, not as clauses**: the classifier's macro-F1 and per-domain precision on the judged gold; the second reading of the examples target, annotated extraction examples per domain from the landscape catalogue; the same per-domain totals on the baseline build.
- **Outcome**: open

## Design

- **Inputs**: the loosened and the baseline corpus builds with their statistics stage, the judged sample of the curation-quality experiment as domain gold, and the landscape survey's catalogue of open Slovene datasets. All four are read in place from the earlier experiments and described in `## Datasets`.
- **Method / factors**: one pass from source-labelled counts to a per-target verdict.
  - Count documents and words per domain and source at every curation stage of both builds, from the statistics stage and the stage folders ([M1]).
  - Score every document against one lexicon per domain and assign the highest-scoring domain when it clears a threshold and leads the runner-up by a margin, else the source's own label ([M2]).
  - Measure that classifier on the judged gold, macro-F1 over the thirteen labels and precision per domain, and pass or fail the gate ([M3]).
  - Label every document of the final corpus of both builds and total documents and tokens per domain under the document-level labels ([M4]).
  - Read each target under each counting rule, source versus document labels, all sources versus gated removed, and write the pass/fail table ([M5]).
  - Attribute each shortfall to the stage that lost the material or to a missing source, and list the catalogue datasets that would close it ([M6]).
  - Factors: the labelling rule (source label, document-level classifier); the source set (all, gated removed); the build (loosened, baseline).
- **Metrics**: primary pass or fail per target and counting rule; secondary macro-F1 and medical precision of the classifier on the judged gold, estimated tokens and documents per domain — each defined in `experiments/GLOSSARY.md`.
- **Protocol**: the classifier pass over a full build is a tracked run, one per build, with per-domain totals logged to MLflow. The counts, the pass/fail table, the gap attribution and the catalogue join are recomputed by `analysis.py` from the statistics files and the run outputs.
- **Compute**: local CPU; one pass over about 41 million documents and 11 billion words per build, a few hours each.
- **People**: none; the gold already exists. A few hours of hand labelling only if the medical confusions on the gold need adjudicating.
- **Data access**: the four gated sources (culturax, gigafida, kas, solar) are on disk and are reported separately, since a public release cannot carry them.
- **External dependency**: none.

## Datasets

- **Loosened build**: the curated corpus the curation-thresholds experiment rebuilt with its per-route loosened filter settings, read in place from that experiment's data folder (`pretrain-loosened/`). One item is one document of the final stage, after both dedup stages. It holds 18 Slovene sources and about 41 million documents. Its last three stages, the two dedup stages and the statistics, were built 2026-10-16 to 2026-10-19 at commit 8e21f67; the earlier stages are reused from that experiment's candidate builds of 2026-10-05 onward. [Documents, words and estimated tokens per source of both builds](tables/dataset-builds-statistics.csv)
- **Baseline build**: the shared pretraining corpus under `data/pretrain/`, built by the pipeline on main from the same 18 sources. It holds about 30 million final-stage documents, built 2026-10-05 to 2026-10-08 at commit 2ece205. Its rows sit in the same table as the loosened build's.
- **Judged gold**: the 8,930 documents of the curation-quality experiment's judged sample, each joined on its id to the domain its judge assigned. The sample was drawn in September from an earlier build, up to 40 kept and 40 dropped documents per source, stage and decision, so it over-represents dropped text and small sources. It spans 20 sources, two of them (parlamint_si, legal_mc4) in neither build here. Its kept flag is the drawn stage's decision then, not membership in either final corpus. About half the documents were cut to 2,000 characters before judging. Sample and verdicts are hash-checked, then halved into a tune and a test side ([D3]) under `final/gold-tune-test-split/`. [The judged gold per judge domain](tables/dataset-gold-statistics.csv), [drawn as a chart](figures/dataset-gold-documents-by-domain.svg)
- **Landscape catalogue**: the data-landscape survey's 295 rows for Slovene, one per dataset with its domains, access, size and annotation, searched 2026-09-17. A dataset listing several domains counts once under each, and the survey's domain names are renamed to the judge's labels ([D2]). [Catalogue datasets per domain and access](tables/dataset-catalogue-statistics.csv)

## Methods

### M1 — Count each source's documents after every curation stage of both builds

- **Input**: each build's curation output folder. One item is one source at one stage, such as `kas` after the quality filter, with the documents that stage read and wrote. The counts come from the small completion file each per-source stage writes beside its output, from the exact-dedup stage's own files, and from the statistics stage's file per source.
- **Output**: one row per build and source in `tables/funnel-documents-by-source.csv`. It carries the source's domain label and access from the dataset registry, its documents after each of the seven stages, its final words, estimated tokens and the share of its documents kept. The same counts summed over the sources sharing a domain label, with how many sources each sum spans, are in `tables/funnel-documents-by-domain.csv`, which Q1 and [M6] read.
- **How**: documents are counted at every stage; words only at the last, since counting them at every stage means splitting every stage's text into words again.
  1. For the five stages that run per source (convert, language, spam, quality, repetition), read each source's documents read and written from its completion file. Stop if a stage read other than what the stage before it wrote, the sign of a stage reused from another build.
  2. Exact dedup runs over the whole corpus and records no per-source count, so count the documents in each source's files once. Cache the counts under this experiment's `interim/` with the stage's document digest, a hash of its output, and reuse them while the digest holds.
  3. Take each source's documents and words after sentence dedup from the statistics stage; a source it has no file for lost its last document earlier and counts zero there. Its words leave out stopwords and punctuation. Estimate tokens as $t = 2w$, where $w$ is that word count, the conversion every KPI reading in the project uses.
  4. Stop unless the per-source sums equal the counts both dedup stages recorded for the whole corpus.
  5. Compute the share kept, $r = n_{\text{final}} / n_{\text{convert}}$, where $n_{\text{convert}}$ is the source's documents after conversion and $n_{\text{final}}$ after sentence dedup. Sum every count per build and domain label, and recompute $r$ on the sums.
- **Code**: `analysis.py::funnel_source_rows`, summed by `analysis.py::funnel_domain_rows`
- **Settings**: none; the stages counted are the pipeline's own

## Decisions

### D1 — Read the targets on the loosened build, with the baseline beside it

- **Decision**: the validated corpus is the final stage of the loosened build the curation-thresholds experiment named its adoption candidate, as it stands in that experiment's data folder. The baseline build on the shared path is tabled beside it under every counting rule, never as the verdict.
- **Why**: the loosened settings are the ones the shared corpus will carry once adopted, so a verdict on the baseline would be stale on arrival. The baseline beside it shows what the filter change gained per domain.
- **Alternatives**:
  - Wait for the shared corpus to be rebuilt with the loosened settings — the same documents, weeks later.
  - Read the baseline alone — answers a question about a build the project is leaving behind.
- **History**:
  - 2026-10-20 set at record creation; the issue's ordering comment asked the verdicts to wait for frozen thresholds, which the loosened build now carries.

### D2 — The judge's document-level domains are the gold, and its thirteen labels are the label set

- **Decision**: the classifier predicts the thirteen domain labels of the curation-quality judge's rubric: medical, scientific, legal, news, parliamentary, academic, wiki, forum, blog, student, web, finance, other. It is scored against that judge's label on the 8,930 judged documents. Headline science is scientific plus academic. The issue's names map onto these as encyclopedic = wiki, forum/social = forum, general-web = web.
- **Why**: it is the only document-level domain gold that exists, it spans every source, and the judge agrees with a person on medical-or-not in 142 of 145 hand-labelled documents. A label set that differs from the gold's cannot be scored on it.
- **Alternatives**:
  - Source labels as gold — blind to exactly the web documents the classifier exists to find.
  - A fresh hand-labelled set — days of labelling for a question the judge's labels already answer on medicine.
- **History**:
  - 2026-10-20 set at record creation.

### D3 — The classifier must pass a quality gate on the gold before any count under document labels is read

- **Decision**: the judged gold is halved under a fixed seed into a tune side and a test side, copies of one text on one side, each domain split evenly, frozen under `final/`. Threshold and margin are picked on the tune side. The classifier passes if its test-side macro-F1 is at least 0.7 and its medical precision at least 0.8. Below either floor, a small supervised classifier is tried the same way; below both floors there too, H3 and H4 are inconclusive under document labels.
- **Why**: a classifier that mislabels one web page in five would clear the ten thousand bar on noise alone. A gate read on the tuning documents measures fit, not that error rate. The gold holds dozens of groups of copied text, which a random split would put on both sides. The gate is a build check: it decides whether a clause can be read, never how.
- **History**:
  - 2026-10-20 set at record creation, from the issue's requirement 4.
  - 2026-10-20 tune/test split added at the data phase, after the review gate flagged tuning and scoring on one set.

## Findings

## Verdict

## Reproduce
