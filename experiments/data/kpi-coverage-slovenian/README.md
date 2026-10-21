---
title: KPI coverage of the curated Slovene corpus under source and document-level domain labels
slug: kpi-coverage-slovenian
short: KPI coverage check
category: data
branch: exp/kpi-coverage-slovenian
base_commit: bb8c5ad
status: running
ticket: "#3"
pr:
mlflow: slm4ie/data/kpi-coverage-slovenian
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

- **Hypothesis**: leaning confirmed — the curated Slovene corpus clears all four bars on the loosened build, corpus tokens, domain tokens, and science and medical documents, and medicine clears the examples target only once each document is labelled with its own domain rather than its source's.
- **F1 — Every data target clears under document-level labels, and under source labels only the medical examples target misses.** Medical documents miss the examples target under source labels and clear it a hundredfold under document labels, while every other count clears under both.
- **F2 — The medical shortfall under source labels is a source gap, since the medical source never held enough distinct texts.** The medical source falls below the examples target at exact dedup, which removes only copies, so only a new dataset can close the gap.
- **Caveat**: the classifier clears its medical precision floor by a single document, but the medical count would stay above target at a precision a hundred times lower.
- **Next**: conclude, noting that the medical source's converted count holds about three copies per distinct leaflet, so its source-label counts overstate distinct texts.

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
  - **Q1 — Where does each shortfall arise, before curation or at a filter stage?**: read on the table of documents per domain at every curation stage of both builds ([M1], [M7]), where a domain that had the material at the converted stage and lost it is a curation loss naming the stage, and one that never had it is a source gap; robust if the stage or gap named is the same on the loosened and the baseline build — the answer feeds the follow-up, a filter to tune or the catalogue datasets that would close the gap. Added 2026-10-20 at the data phase.
  - **Verdict rule**: confirmed when all four bars hold; refuted when any bar is refuted; inconclusive when the classifier fails its quality gate ([D3]) so that H3 and H4 cannot be read under document labels. Q1 is read beside that outcome and does not change it.
  - **Read beside the clauses, not as clauses**: the classifier's macro-F1 and per-domain precision on the judged gold; the second reading of the examples target, annotated extraction examples per domain from the landscape catalogue; the same per-domain totals on the baseline build.
- **Outcome**: open

## Design

- **Inputs**: the loosened and the baseline corpus builds with their statistics stage, the judged sample of the curation-quality experiment as domain gold, and the landscape survey's catalogue of open Slovene datasets. All four are read in place from the earlier experiments and described in `## Datasets`.
- **Method / factors**: one pass from source-labelled counts to a per-target verdict.
  - Count documents and words per domain and source at every curation stage of both builds, from the statistics stage and the stage folders ([M1]).
  - Score every document against one lexicon per domain and assign the highest-scoring domain when it clears a threshold and leads the runner-up by a margin, else the source's own label ([M2]).
  - Measure that classifier on the judged gold, macro-F1 over the twelve scored labels and precision per domain, and pass or fail the gate ([M3]).
  - Where it fails, fit a supervised classifier on the tune side and read the same gate ([M4]).
  - Label every document of the final corpus of both builds and total documents and tokens per domain under the document-level labels ([M5]).
  - Read each target under each counting rule, source versus document labels, all sources versus gated removed, and write the pass/fail table ([M6]).
  - Attribute each shortfall to the stage that lost the material or to a missing source, and list the catalogue datasets that would close it ([M7]).
  - Factors: the labelling rule (source label, document-level classifier); the source set (all, gated removed); the build (loosened, baseline).
- **Metrics**: primary pass or fail per target and counting rule; secondary macro-F1 and medical precision of the classifier on the judged gold, estimated tokens and documents per domain — each defined in `experiments/GLOSSARY.md`.
- **Protocol**: the classifier pass over a full build is a tracked run, one per build, with per-domain totals logged to MLflow. The counts, the pass/fail table, the gap attribution and the catalogue join are recomputed by `analysis.py` from the statistics files and the run outputs.
- **Compute**: local CPU for the counts and the TF-IDF pass, chunked over several workers, over about 41 million documents and 11 billion words per build. The local GPU for the embeddings candidate, which would need days on a CPU.
- **People**: none; the gold already exists. A few hours of hand labelling only if the medical confusions on the gold need adjudicating.
- **Data access**: the four gated sources (culturax, gigafida, kas, solar) are on disk and are reported separately, since a public release cannot carry them.
- **External dependency**: the multilingual-e5-small sentence encoder, downloaded once from the Hugging Face hub for the embeddings candidate of [M4]; nothing else.

## Datasets

- **Loosened build**: the curated corpus the curation-thresholds experiment rebuilt with its per-route loosened filter settings, read in place from that experiment's data folder (`pretrain-loosened/`). One item is one document of the final stage, after both dedup stages. It holds 18 Slovene sources and about 41 million documents. Its last three stages, the two dedup stages and the statistics, were built 2026-10-16 to 2026-10-19 at commit 8e21f67; the earlier stages are reused from that experiment's candidate builds of 2026-10-05 onward. [Documents, words and estimated tokens per source of both builds](tables/dataset-builds-statistics.csv)
- **Baseline build**: the shared pretraining corpus under `data/pretrain/`, built by the pipeline on main from the same 18 sources. It holds about 30 million final-stage documents, built 2026-10-05 to 2026-10-08 at commit 2ece205. Its rows sit in the same table as the loosened build's.
- **Judged gold**: the 8,930 documents of the curation-quality experiment's judged sample, each joined on its id to the domain its judge assigned. The sample was drawn in September from an earlier build, up to 40 kept and 40 dropped documents per source, stage and decision, so it over-represents dropped text and small sources. It spans 20 sources, two of them (parlamint_si, legal_mc4) in neither build here. Its kept flag is the drawn stage's decision then, not membership in either final corpus. About half the documents were cut to 2,000 characters before judging. Sample and verdicts are hash-checked, then halved into a tune and a test side ([D3]) under `final/gold-tune-test-split/`. [The judged gold per judge domain](tables/dataset-gold-statistics.csv), [drawn as a chart](figures/dataset-gold-documents-by-domain.svg)
- **Landscape catalogue**: the data-landscape survey's 295 rows for Slovene, one per dataset with its domains, access, size and annotation, searched 2026-09-17. A dataset listing several domains counts once under each, and the survey's domain names are renamed to the judge's labels ([D2]). [Catalogue datasets per domain and access](tables/dataset-catalogue-statistics.csv)

## Methods

### M1 — Count each source's documents after every curation stage of both builds

- **Input**: each build's curation output folder. One item is one source at one stage, such as `kas` after the quality filter, with the documents that stage read and wrote. The counts come from the small completion file each per-source stage writes beside its output, from the exact-dedup stage's own files, and from the statistics stage's file per source.
- **Output**: one row per build and source in `tables/funnel-documents-by-source.csv`. It carries the source's domain label and access from the dataset registry, its documents after each of the seven stages, its final words, estimated tokens and the share of its documents kept. The same counts summed over the sources sharing a domain label, with how many sources each sum spans, are in `tables/funnel-documents-by-domain.csv`, which Q1 and [M7] read.
- **How**: documents are counted at every stage; words only at the last, since counting them at every stage means splitting every stage's text into words again.
  1. For the five stages that run per source (convert, language, spam, quality, repetition), read each source's documents read and written from its completion file. Stop if a stage read other than what the stage before it wrote, the sign of a stage reused from another build.
  2. Exact dedup runs over the whole corpus and records no per-source count, so count the documents in each source's files once. Cache the counts under this experiment's `interim/` with the stage's document digest, a hash of its output, and reuse them while the digest holds.
  3. Take each source's documents and words after sentence dedup from the statistics stage; a source it has no file for lost its last document earlier and counts zero there. Its words leave out stopwords and punctuation. Estimate tokens as $t = 2w$, where $w$ is that word count, the conversion every KPI reading in the project uses.
  4. Stop unless the per-source sums equal the counts both dedup stages recorded for the whole corpus.
  5. Compute the share kept, $r = n_{\text{final}} / n_{\text{convert}}$, where $n_{\text{convert}}$ is the source's documents after conversion and $n_{\text{final}}$ after sentence dedup. Sum every count per build and domain label, and recompute $r$ on the sums.
- **Code**: `analysis.py::funnel_source_rows`, summed by `analysis.py::funnel_domain_rows`
- **Settings**: none; the stages counted are the pipeline's own

### M2 — Label each document with the domain its own words point to

- **Input**: one document of a corpus build, its text and its source's domain label. The thirteen marker-word lists, one per judge label ([D2]), hand-curated in [`configs/classifier-lexicon.yaml`](configs/classifier-lexicon.yaml). An entry is a whole word (`dne`) or a prefix ending in `*` (`zdravil*`), which matches every word starting with it, since Slovene inflects heavily. Each entry sits in a tier, `strong` for words that almost only occur in that domain and `weak` for words common there but seen elsewhere too. The lists were seeded from the words most distinctive of each judge domain on the tune half of the gold, ranked by log-odds against the other domains, in [the seed table](tables/classifier-lexicon-seed-words-by-domain.csv). They were then pruned by hand to words that name the domain's subject or genre, dropping words that mark one source or one text, so the gate measures the domain and not the source.
- **Output**: one label of the thirteen per document.
- **How**:
  1. Split the text into words, runs of letters and digits, lowercased. Let $n$ be their count.
  2. For every word and every domain, take the weight of the strongest entry of that domain the word matches, $w_s = 3$ for a strong entry, $w_w = 1$ for a weak one, zero when none matches. Sum over the words into $h_d$, the weighted hits of domain $d$.
  3. Score each domain as hits per thousand words, $s_d = 1000 \, h_d / n$; an empty text scores zero everywhere.
  4. Rank the domains by score, ties towards the earlier list. Assign the leader when the text has at least $m$ words, $s_{(1)} \ge \tau$ and $s_{(1)} - s_{(2)} \ge \mu$, where $s_{(1)}$ and $s_{(2)}$ are the leading and runner-up scores, $\tau$ the threshold and $\mu$ the margin. The length floor $m$ stops one strong word deciding a short text on its own.
  5. Otherwise keep the source's label, mapped through the fallback table when it is not one of the thirteen; the mixed-genre reference corpus falls back to `other`.
- **Code**: `slm4ie/data/curate/inspect/domains.py::label_document`
- **Settings**: `rule.threshold` ($\tau$, the lowest leading score that assigns a domain), `rule.margin` ($\mu$, the smallest lead over the runner-up), `rule.min_words` ($m$, the shortest text the rule is read on), `weights` (the hit weight of each tier), `fallback` (source labels outside the thirteen and the label they count as), `lexicons` (the entries per domain and tier). Threshold and margin are picked on the tune half of the gold ([D3], [M3]).

### M3 — Score the lexicon classifier on the judged gold and read its gate

- **Input**: the judged gold, halved into its tune and test side ([D3]). One item is one judged document with its text, the judge's domain and its source's label. The lexicons and the length floor of [M2].
- **Output**: four tables. [The tune-side grid](tables/classifier-lexicon-gate-by-rule.csv): one row per threshold and margin pair with its macro-F1 and medical precision, the pick marked. [The gate](tables/classifier-lexicon-gate.csv): one row per side with its documents, the rule read, macro-F1 and medical precision, the same precision and recall over the documents of other sources, and whether both floors hold. Only the test row is the gate; the tune row shows the fit. [Precision, recall and F1 per domain and side](tables/classifier-lexicon-gate-by-domain.csv), each also read off the domain's own source. [The confusion counts](tables/classifier-lexicon-confusion-by-domain.csv): documents per judge domain and assigned label, per side.
- **How**:
  1. Score every judged document once against the lexicons ([M2], steps 1 to 3), so every rule reads the same scores. Labels are read with scientific and academic folded into science ([D2]).
  2. On the tune side, read the rule ([M2], step 4) at every threshold $\tau \in \{0, 5, \ldots, 100\}$ and margin $\mu \in \{0, 5, \ldots, 40\}$, the length floor fixed at the config's $m$. For each label $d$, let $g_d$ be its gold documents, $a_d$ the documents assigned it and $c_d$ those that are both. Precision is $P_d = c_d / a_d$, recall $R_d = c_d / g_d$ and $F_d = 2 P_d R_d / (P_d + R_d)$, zero where undefined. Macro-F1 is the mean of $F_d$ over the twelve labels, each counting once.
  3. Pick the pair with the highest macro-F1 at four decimals; ties go to the higher medical precision, then the lower threshold, then the lower margin. The pick is written into the config; the analysis stops when the config's rule differs from it.
  4. On the test side, read the config's rule and compute the same metrics. The gate holds when macro-F1 is at least 0.7 and medical precision off the medical source, step 5, at least 0.8 ([D3]). A hash of the lexicons is frozen at that read; a changed lexicon stops the analysis, so the test side is read once per lexicon.
  5. Read each domain's precision and recall again on the documents whose fallback label is not that domain, where only the lexicons can assign it. H3 rests on this read: the medical source's leaflets keep their label without any classifier.
- **Code**: `analysis.py::score_gold` (step 1), `analysis.py::rule_grid_rows` and `analysis.py::check_rule_tuned` (steps 2 and 3), `analysis.py::gate_rows` and `analysis.py::check_lexicons_frozen` (step 4), `analysis.py::gate_domain_rows` and `analysis.py::confusion_rows` (step 5 and the per-domain tables), all over `analysis.py::label_metrics`
- **Settings**: in `analysis.py`, `GRID_THRESHOLDS` (the thresholds searched), `GRID_MARGINS` (the margins searched), `GATE_MACRO_F1` (the macro-F1 floor of [D3]), `GATE_PRECISION` (the precision floor of [D3]), `GATE_DOMAIN` (the domain the precision floor reads, medical), `LABEL_MERGE` (the judge labels folded into one, scientific and academic into science), `DECIMALS` (the rounding of every metric, and of the pick); in `configs/classifier-lexicon.yaml`, `rule.min_words` (the length floor, fixed before the search), `rule.threshold` (the picked threshold) and `rule.margin` (the picked margin)

### M4 — Fit a supervised classifier on the tune side and read the same gate

- **Input**: the tune side of the judged gold ([D3]); one item is one judged document's text and judge domain. The test side is read once per candidate. Two candidates in [`configs/classifier-supervised.yaml`](configs/classifier-supervised.yaml), each a logistic regression over the twelve scored labels ([D2]) with its own features. `tfidf` uses word and character n-grams weighted by TF-IDF, a term's count in the document discounted by how many documents carry it. `embeddings` uses one vector per document from the multilingual-e5-small sentence encoder, which reads the first 512 tokens.
- **Output**: per candidate, the gate, per-domain and confusion tables of [M3] under `tables/classifier-<candidate>-…`. For tfidf: [gate](tables/classifier-tfidf-gate.csv), [per domain](tables/classifier-tfidf-gate-by-domain.csv), [confusion](tables/classifier-tfidf-confusion-by-domain.csv). For embeddings: [gate](tables/classifier-embeddings-gate.csv), [per domain](tables/classifier-embeddings-gate-by-domain.csv), [confusion](tables/classifier-embeddings-confusion-by-domain.csv). The regularisation grid: [tfidf](tables/classifier-tfidf-gate-by-regularisation.csv) and [embeddings](tables/classifier-embeddings-gate-by-regularisation.csv). The fit itself is kept under `final/classifier-supervised/`.
- **How**:
  1. Cut every text to its first 2,000 characters, the length the gold was cut to before judging, so the gate measures the classifier on the text it will read in the corpus.
  2. Split the tune side into $k$ stratified folds that keep copies of one text together. For each regularisation strength $C$ in the config's list, fit the features and a logistic regression inside each fold, every class weighted to the same total, and score the out-of-fold predictions by the macro-F1 of [M3]. Pick the highest, ties to the smaller $C$, the stronger regularisation. Refit the pick on the whole tune side and save it.
  3. Label both sides with the saved fit and read the gate of [M3], steps 4 and 5. A document's fallback label still comes from its source, so the off-source read is the same read.
  4. Reuse a saved fit while the config, the tune documents and the library version hash to what it was fitted from; a change stops the analysis, so each candidate's test side is read once.
- **Code**: `slm4ie/data/curate/inspect/supervised.py::fit_classifier` (steps 1 and 2), `analysis.py::fit_or_load_candidate` (step 4), `analysis.py::write_gate_tables` with `analysis.py::label_predicted` (step 3), over the texts of `analysis.py::gold_texts`
- **Settings**: `max_chars` (characters read per text), `regularisation` (the $C$ values tried), `folds` ($k$), `workers` (folds fitted at once), `seed` (the fold assignment), `max_iterations` (solver iterations before a fit is declared not converged); per candidate `features` (tfidf or embeddings), for tfidf `word_ngrams` (shortest and longest word n-gram), `char_ngrams` (shortest and longest character n-gram), `min_df` (documents an n-gram must occur in), `max_features` (n-grams kept per kind); for embeddings `model` (the encoder), `revision` (the hub commit loaded), `prefix` (text the encoder expects before each document), `batch_size` (documents encoded at once), `max_length` (tokens read per document); in `analysis.py`, `SUPERVISED_CANDIDATES` (the candidates read, in order)

### M5 — Label every final-corpus document of both builds with the passing classifier

- **Input**: the final stage of one build, after both dedup stages ([D1]); one item is one document, its text and its source. The tfidf fit of [M4] that passed the gate, saved under `final/classifier-supervised/tfidf.pkl`. The ids of the judged gold.
- **Output**: per build, one JSON file under `final/` with the documents and words per source and assigned label, and the gold ids the build still holds, read by [M6] and [M7]. The same totals summed over sources, documents and words per label, are logged as one MLflow run per build.
- **How**:
  1. Read every shard of the stage, one process per shard, and label each document with the saved fit, which reads its first 2,000 characters ([M4], step 1).
  2. Count each document under its source and its assigned label. Count its words as the statistics stage counts them, word tokens with at least one letter and two characters, stopwords left out, so tokens estimated as $t = 2w$ match [M1].
  3. Note every gold id the stage still holds, so a gold document's true kept flag is known for this build.
  4. Sum the shards and stop unless the documents counted equal those the stage's completion record holds, so no shard is missed. Each finished shard is saved as it ends, so a pass that stops part way resumes from where it was. It resumes only while the stage, fit, stopwords and gold ids are the ones it started with.
- **Code**: `slm4ie/data/curate/inspect/supervised.py::label_corpus` (steps 1 to 4), `slm4ie/data/curate/inspect/supervised.py::track_corpus_labels` (the JSON and the MLflow run, with the hashes of the fit and the gold file), run by `scripts/curate_pretraining_corpus.py classify`
- **Settings**: on the command line, `--classifier` (the saved fit), `--gold` (the judged sample whose ids are looked for), `--checkpoint` (the file of finished shards), `--max-workers` (shards read at once), `--shards-per-source` (the first shards read per source, for a smoke run only); in the build's curation config, `stopwords` (the words left out of every word count)

### M6 — Read every target under every counting rule on both builds

- **Input**: the two files of [M5], one per build, with documents and words per source and assigned label. The per-source funnel of [M1], to check them against. Each source's domain label and access from the dataset registry. The gate verdict of [M3] and [M4].
- **Output**: [the target table](tables/kpi-pass-by-rule.csv), one row per build, domain, measure, labelling rule and source set. Each row holds the count, the target and whether the count is above it. It also holds the share of the count that must truly belong to the domain for it to stay above. The totals behind it: [documents and estimated tokens per domain](tables/corpus-documents-by-domain.csv) under each labelling rule, with the open sources beside, and [the same per source and assigned label](tables/corpus-documents-by-source.csv). [The gate read again](tables/classifier-tfidf-gate-by-final-corpus.csv) on the test-side gold documents each final corpus still holds.
- **How**:
  1. Stop unless the saved fit hashes to what both corpus passes logged, and unless every source's documents and words in a pass equal what the statistics stage counted for it.
  2. Sum documents and words per domain under each labelling rule. Under source labels a document counts under its source's registry label; under document labels, under the label the classifier assigned it. Scientific and academic both count as science ([D2]). Sum once with every source and once without the four gated ones.
  3. Read the corpus in estimated tokens $t = 2w$, where $w$ is its word count, and in words, each word read as one token, against $5 \times 10^9$. Read science and medicine in estimated tokens and in words against $5 \times 10^5$, and in documents against $10^4$.
  4. A count passes when it is above its target. A count under document labels is not read when the classifier that labelled the corpus failed its gate ([D3]).
  5. For each count $n$ with target $T$, the share needed is $T / n$: the lowest precision at which the domain's true documents or tokens would still clear the target.
  6. Read the gate's macro-F1 and medical precision of [M3] again on the test-side documents whose ids [M5] found in each final corpus, beside the gate and never as it.
- **Code**: `analysis.py::read_corpus_labels` (step 1), `analysis.py::domain_totals` (step 2), `analysis.py::kpi_rows` with `analysis.py::kpi_row` (steps 3 to 5), `analysis.py::final_corpus_gate_rows` (step 6)
- **Settings**: in `analysis.py`, `CORPUS_TOKEN_TARGET` (estimated tokens the corpus must hold), `DOMAIN_TOKEN_TARGET` (estimated tokens per domain), `DOMAIN_DOCUMENT_TARGET` (documents per domain), `TARGET_DOMAINS` (the domains read, science and medical), `TOKENS_PER_WORD` (the word-to-token estimate), `CORPUS_CLASSIFIER` (the candidate whose fit labelled the corpora), `CORPUS_CLASSIFIER_SHA256` (the hash of that fit both passes logged)

### M7 — Attribute each missed target to a missing source or to the stage that lost it

- **Input**: the rows of [M6] whose count is not above its target. The per-source funnel of [M1], documents after every stage of both builds. The landscape catalogue.
- **Output**: [the attribution](tables/gap-attribution-by-stage.csv), one row per missed target with the documents its sources held after conversion and at the end, the cause and the stage named. [The open catalogue datasets](tables/gap-catalogue-datasets.csv) of each domain with a missed target that the corpus does not yet hold. [The catalogue datasets annotated with extraction spans](tables/catalogue-ie-examples-by-domain.csv) per domain, the second reading of the examples target.
- **How**:
  1. Sum, per stage, the documents of the sources whose label folds into the missed domain, leaving out the gated sources where the row does.
  2. If the sources held no more than the target after conversion, the cause is a source gap. Otherwise it is a curation loss, and the stage named is the first after which they hold no more than the target.
  3. A missed target under document labels or in words names no cause, since those counts exist only at the final stage.
  4. List the catalogue datasets of each missed domain that are openly downloadable and not in the dataset registry, largest first. Catalogue domains are renamed to the judge's labels, science folded ([D2]).
  5. List, per domain, the catalogue datasets annotated with extraction spans, whatever their access.
- **Code**: `analysis.py::gap_rows` with `analysis.py::domain_stage_counts` (steps 1 to 3), `analysis.py::gap_catalogue_rows` (step 4), `analysis.py::ie_example_rows` (step 5)
- **Settings**: in `analysis.py`, `IE_ANNOTATION` (the catalogue's annotation value for extraction spans), `CATALOGUE_DOMAINS` (the catalogue's domain names renamed to the judge's labels)

## Decisions

### D1 — Read the targets on the loosened build, with the baseline beside it

- **Decision**: the validated corpus is the final stage of the loosened build the curation-thresholds experiment named its adoption candidate, as it stands in that experiment's data folder. The baseline build on the shared path is tabled beside it under every counting rule, never as the verdict.
- **Why**: the loosened settings are the ones the shared corpus will carry once adopted, so a verdict on the baseline would be stale on arrival. The baseline beside it shows what the filter change gained per domain.
- **Alternatives**:
  - Wait for the shared corpus to be rebuilt with the loosened settings — the same documents, weeks later.
  - Read the baseline alone — answers a question about a build the project is leaving behind.
- **History**:
  - 2026-10-20 set at record creation; the issue's ordering comment asked the verdicts to wait for frozen thresholds, which the loosened build now carries.

### D2 — The judge's document-level domains are the gold, and its labels, science merged, are the label set

- **Decision**: the classifier predicts the curation-quality judge's domain labels: medical, scientific, legal, news, parliamentary, academic, wiki, forum, blog, student, web, finance, other. Scientific and academic are folded into one label, science, so twelve are scored and labelled. Scored on the judge's 8,930 documents. The issue's names map onto these as encyclopedic = wiki, forum/social = forum, general-web = web.
- **Why**: it is the only document-level domain gold that exists, it spans every source, and the judge agrees with a person on medical-or-not in 142 of 145 hand-labelled documents. A label set that differs from the gold's cannot be scored on it.
- **Alternatives**:
  - Source labels as gold — blind to exactly the web documents the classifier exists to find.
  - A fresh hand-labelled set — days of labelling for a question the judge's labels already answer on medicine.
- **History**:
  - 2026-10-20 set at record creation.
  - 2026-10-20 scientific and academic folded into science for scoring and labelling, at the code phase: both are science writing, the judge's line between them was every classifier's largest confusion in [M3] and [M4], and the headline already summed them. The lexicon rule still reads its margin on the two scores before the fold; folding them first would move its test macro-F1 by 0.0002.

### D3 — The classifier must pass a quality gate on the gold before any count under document labels is read

- **Decision**: the judged gold is halved under a fixed seed into a tune and a test side, copies of one text together and each domain split evenly. Settings are picked on the tune side. A classifier passes with test macro-F1 at least 0.7 and off-source medical precision at least 0.8. Failing that, a supervised classifier is tried the same way.
- **Why**: a classifier that mislabels one web page in five would clear the ten thousand bar on noise alone. A gate read on the tuning documents measures fit, not that error rate. The gold holds dozens of groups of copied text, which a random split would put on both sides. The gate decides whether a clause can be read, never how.
- **History**:
  - 2026-10-20 set at record creation, from the issue's requirement 4; a supervised classifier that fails too leaves H3 and H4 inconclusive, as the Verdict rule says.
  - 2026-10-20 tune/test split added at the data phase, after the code review of the experiment flagged tuning and scoring on one set.
  - 2026-10-20 the precision floor moved from every test document to the documents outside the medical source, after the review gate of [M3] showed the medical source's leaflets inflating it (0.715 over all documents, 0.381 off the source); H3 counts the medical documents inside the web crawls, which is the off-source read.
  - 2026-10-20 the test side was read a second time for all three classifiers after [D2] merged scientific and academic, the one re-read this gate allows. Before the merge, test macro-F1 and off-source medical precision were 0.540 and 0.381 for the lexicons, 0.670 and 0.820 for tfidf, 0.682 and 0.650 for embeddings. After it, tfidf passes at 0.702 and exactly 0.800, 48 of 60 assigned documents, so one more misassigned document would fail it.

## Findings

### F1 — Every data target clears under document-level labels, and under source labels only the medical examples target misses · key

- **Summary**: Medical documents miss the examples target under source labels and clear it a hundredfold under document labels, while every other count clears under both.
- **Runs**: c90d350793274d64b766a2cf089bc49b, the classifier pass over the loosened build, and 355409bfde9e42029c96f3d2be95b6a1, the same pass over the baseline build, both under `slm4ie/data/kpi-coverage-slovenian` ([M5]); `analysis.py` at 024330f over their totals and the statistics stage of both builds ([M6]).
- **Result**: ![Each data target's count as a multiple of its target, on a log scale, under source labels and under document-level labels, per build. The vertical line is the target itself, a multiple of one. On the loosened build the medical document count sits at 0.59 of its target under source labels and at 106 times it under document labels, and science documents move from 17 to 96 times. Every token count sits past a thousand times its target except medical tokens under source labels, at 44 times, and the whole corpus, at 4.5 times its five billion with every source and 3.5 times with the gated sources removed. The baseline build shows the same shape with the medical source count at 0.04 of the target. The target table linked from the Reading holds twenty-eight counts per build: the corpus in tokens and in words, with and without the gated sources, and science and medicine in tokens, words and documents under both labelling rules and both source sets.](figures/kpi-target-multiple-by-rule.svg)
- **Reading**:
  - **All but four of the fifty-six counts sit above their target, and the four misses are the medical document count under source labels.** The misses fall on both builds, with and without the gated sources. Under source labels the medical count is the one medical leaflet source after both dedup stages. Under document labels it is what the TF-IDF classifier that passed the gate ([D3]) assigned medical, mostly from the web crawls ([F3]). [The table of every count against its target](tables/kpi-pass-by-rule.csv) also reads each target counting one token per word instead of two; no pass changes.
  - **[H1], [H2], [H3] and [H4] are each confirmed on the loosened build.** The corpus holds over four times the tokens H1 asks for, over three times without the gated sources, and science and medicine each far exceed H2's tokens. Medicine misses the examples target under source labels and clears it under document labels, as H3 predicts; science clears it under both, as H4 predicts. The medical count under document labels stays above its target as long as at least one in a hundred documents labelled medical truly is medical. The gate measured 0.80 off the medical source. The baseline build gives the same four verdicts.
  - **A classifier whose precision collapsed on the corpus would overturn the document-label reads.** The gate is read on the judged gold, which over-represents dropped text, not on the corpus itself.
- **Implication**: The verdict can be written on the loosened build, since all four bars hold. What remains is where the source-label medical shortfall comes from ([F2]), which decides whether the follow-up is a filter to tune or a dataset to add.
- **History**:
  - 2026-10-21 first result, from the two classifier passes and the statistics stage of both builds, read on the twelve merged labels ([D2]) with the TF-IDF classifier that passed the gate ([D3])

### F2 — The medical shortfall under source labels is a source gap, since the medical source never held enough distinct texts · key

- **Summary**: The medical source falls below the examples target at exact dedup, which removes only copies, so only a new dataset can close the gap.
- **Runs**: no run; `analysis.py` at 024330f over the per-stage document counts of both builds ([M1], [M7]) and the landscape catalogue's row for the medical source.
- **Result**: ![The medical source's documents after every curation stage of both builds, on a log scale, with the horizontal line at the examples target of ten thousand. Both builds start from 23,542 converted documents. The loosened build keeps 23,181 through the four content filters, then exact dedup removes two thirds and sentence dedup a quarter of what is left, down to 5,941. The baseline build's quality filter cuts the source to 3,599 and the stages after it to 367. The converted count is about three times the source's 7,924 distinct texts, since the release ships each leaflet in two files and repeats it per medicine. The loosened final count sits a hundred above the publisher's 5,841 deduplicated texts, because that file rewrote quote marks and its near-copies escape exact dedup. The attribution rows are in the table linked from the Reading.](figures/funnel-medical-documents-by-stage.svg)
- **Reading**:
  - **Both builds start with more than twice the target and end below it, at different stages.** The attribution rule ([M7]) names the first stage after which the medical source holds no more than the target. That is exact dedup on the loosened build and the quality filter on the baseline, a rule-based filter built for English web text. [The attribution table](tables/gap-attribution-by-stage.csv) reads the same without the gated sources, since the medical source is open.
  - **[Q1] is partial: each build's shortfall is attributed, but the builds name different stages, and the loosened build's stage removes copies, not text.** The source's release ships every leaflet twice, as an original and a publisher-deduplicated file. The original also repeats a leaflet for every medicine it covers. The curation-thresholds experiment, which tuned the filters per source, traced every medical drop at exact dedup to such a copy. The publisher's deduplicated file holds 5,841 texts, so the source never had ten thousand distinct documents. The rule calls this a curation loss at exact dedup, but that stage removes only copies, so the cause is a source gap.
  - **A medical source with more than ten thousand distinct texts before curation, lost at a filter, would overturn this reading.** The loosened build keeps the source through the quality filter, and its distinct texts still fall short.
- **Implication**: Closing the medical examples gap under source labels means adding a dataset, not tuning a filter. The [open catalogue datasets in the medical domain](tables/gap-catalogue-datasets.csv) show the native ones are small, each medical journal archive under three thousand documents, and the large ones translated or machine-written.
- **History**:
  - 2026-10-21 first result, from the per-stage counts of both builds ([M1]) and the attribution rule ([M7]); the copy origin follows finding F6 of the curation-thresholds experiment, which traced the medical source's exact-dedup drops to copies inside the source

### F3 — Medical documents under document labels come almost entirely from the web crawls · supporting F1

- **Summary**: The web crawls supply nineteen in twenty of the documents the classifier labels medical, and the medical leaflet source fewer than one in a hundred.
- **Runs**: c90d350793274d64b766a2cf089bc49b and 355409bfde9e42029c96f3d2be95b6a1, the classifier passes over the loosened and the baseline build ([M5]); `analysis.py` at 024330f over their per-source totals ([M6]).
- **Result**: ![Documents the classifier labels medical in each source, per build, the largest source at the top. On the loosened build the five largest are web crawls: culturax with 276,533, fineweb2 with 272,639, hplt with 145,553, c4 with 129,554 and classla_web_sl with 73,341. The medical leaflet source povejmo_vemo_med keeps 5,919 of its 5,941 documents under the medical label, the news source holds 26,836 and the gated academic theses 1,926. The per-source rows are in the source table linked from the Reading.](figures/corpus-medical-documents-by-source.svg)
- **Reading**:
  - **The medical count is spread over eight web crawls rather than resting on one.** The largest single source, the gated culturax crawl, holds about a quarter of the medical documents, and removing all four gated sources leaves three quarters, as the open-source rows of the [per-source table](tables/corpus-documents-by-source.csv) show. The medical leaflet source keeps nearly every one of its documents under the medical label, so the classifier agrees with the source label where one exists.
  - **This is the mechanism the pass under document labels ([F1]) rests on: the medical examples sit inside general web text, not in a medical dataset.** It also means the count's precision is the classifier's precision on web documents, the off-source read the gate measures ([D3]), and not its precision on leaflets, which is near one.
  - **Finding that the web documents labelled medical are mostly one site's boilerplate or copies of one leaflet would overturn the reading that these are about a million distinct medical examples.** The classifier reads each document's first 2,000 characters alone, and the pass counts documents, not distinct texts.
- **History**:
  - 2026-10-21 first result, from the two classifier passes

### F4 — The classifier's medical precision holds on the gold documents each final corpus still keeps · supporting F1

- **Summary**: On the test gold each final corpus still holds, off-source medical precision stays above its floor on both builds while macro-F1 slips below 0.7.
- **Runs**: c90d350793274d64b766a2cf089bc49b and 355409bfde9e42029c96f3d2be95b6a1, which recorded the gold ids each final corpus holds ([M5]); `analysis.py` at 024330f over the frozen TF-IDF fit ([M4], [M6]).
- **Result**: [The gate's metrics read again on the test-side gold documents found in each final corpus: documents, macro-F1, medical precision over all of them and off the medical source, and medical recall off the source.](tables/classifier-tfidf-gate-by-final-corpus.csv)
- **Reading**:
  - **The documents the corpus keeps are easier for the classifier on medicine and harder on average.** The gold was drawn with up to forty kept and forty dropped documents per source and stage, so the test side holds many documents the pipeline removed. Just over half of it survives in the loosened final corpus and about a third in the baseline's. On the survivors, off-source medical precision is 0.82 on the loosened build and 0.85 on the baseline, and macro-F1 falls about two hundredths below its floor on each.
  - **This supports the medical count under document labels ([F1]) and leaves the gate's verdict as it stands.** The gate is read once on the whole test side ([D3]), where the classifier passed. This second read describes the corpus the count was taken on, where the medical precision is at least as high. The lower macro-F1 falls on the other eleven labels, whose counts bear on no target except science, which clears its document target with ninety times the margin.
  - **An off-source medical precision below the floor on the kept documents would overturn the support.** The read rests on a few dozen medical assignments off the source, so a handful of mislabels would move it.
- **History**:
  - 2026-10-21 first result, from the gold ids the two classifier passes found

### F5 — Annotated extraction examples exist for science but not for medicine in native Slovene · minor

- **Summary**: The catalogue holds one native science dataset with extraction spans above ten thousand documents and no native medical one above a hundred.
- **Runs**: no run; `analysis.py` at 024330f over the landscape catalogue ([M7]).
- **Result**: [Catalogue datasets annotated with extraction spans, per domain, with their access, provenance, document count and whether the count clears the examples target.](tables/catalogue-ie-examples-by-domain.csv)
- **Reading**: Read as annotated examples, the examples target holds for science on the native KAS-term dataset alone and for medicine only through the machine-translated HUMADEX dataset, so native medical extraction examples must still be built.
- **History**:
  - 2026-10-21 first result, from the landscape catalogue

## Verdict

## Reproduce
