# Glossary

Terms and abbreviations the records use that a reader arriving cold may not
know, and every metric they report. One line each, `- **Term** (alias, alias):
definition` — the report links the first mention in each block to this page
and shows the definition on hover. Words match in any case; short or
symbolic forms (F1, κ, q05, P@k) match as written. Extend the list whenever a
finding introduces a term; never remove one a record still uses.

The repository's own vocabulary — tier, stage, entry, slug, backend and the rest
— lives in [CONTEXT.md](../CONTEXT.md) instead.

## Metrics

- **Precision**: of the items the system returned, the share that are correct. 1 means nothing wrong was returned.
- **Recall**: of the correct items, the share the system returned. 1 means nothing was missed.
- **F1** (F1 score): harmonic mean of precision and recall; 1 is perfect, 0 means nothing right. Macro F1 averages F1 over classes or units with equal weight; micro F1 pools all decisions first.
- **Macro-F1**: the F1 score computed per class and averaged with equal weight per class, so a rare class such as medical counts as much as a frequent one.
- **Precision@k** (P@k): precision over the top k ranked items only.
- **Recall@k** (R@k): the share of correct items found within the top k.
- **Accuracy**: the share of all decisions that are correct; misleading when classes are imbalanced.
- **Cohen's κ** (κ, kappa): agreement between two raters beyond what chance would give; 1 is full agreement, 0 chance level, below 0 worse than chance.
- **Quadratic-weighted κ** (weighted kappa): Cohen's κ for an ordered scale such as 1-5, where a disagreement costs the square of its distance, so a one-point miss counts far less than a three-point one.
- **Wilson interval** (Wilson score interval): a confidence interval for a share that, unlike the textbook one, never runs below 0 or above 1, which matters for shares near either end.
- **Drop precision**: of the documents a curation stage dropped, the share judged bad text; high means the stage removes what it should.
- **Target precision**: of the documents a content filter dropped, the share judged to carry what that filter is built to catch — not Slovene for the language filter, spam for the spam filter, ill-formed text for the quality filter, repeated furniture for the repetition filter.
- **Residual bad rate**: of the documents a curation stage kept, the share judged bad text; what the stage failed to catch.
- **Kept win rate**: of the pairs of one kept and one dropped document from the same source and stage, the share where the judge prefers the kept one in both presentation orders.
- **Order disagreement**: the share of pairs whose answer changes when the two documents swap places; it measures the judge's position bias.
- **Side-with-judge share**: of the cases where the judge and the pipeline conflict, the share where a person labelling blind agrees with the judge.
- **Twin match rate**: of a dedup stage's dropped documents, the share whose content survives in the finished corpus as another copy — byte-identical for exact dedup, sharing at least half its three-sentence windows for sentence dedup.
- **Window coverage**: the share of a document's three-sentence windows that another document also contains.
- **Retention**: the share of a source's converted documents that reach the finished corpus.
- **Type-token ratio** (TTR): distinct tokens divided by tokens; it falls as a sample grows, so it is only compared at one fixed token budget.
- **Out-of-vocabulary rate** (OOV rate): the share of alphabetic tokens a lexicon does not list; names, foreign words, code and encoding damage all count as out of vocabulary.
- **CPU hours**: processor time a step consumed, summed over its tasks; comparable between steps and machines.
- **Wall hours** (wall clock): elapsed time a step took on the machine that ran it, at the worker count it was given.
- **Agresti–Coull interval** (Agresti–Coull adjustment): a normal-approximation confidence interval for a share that first adds about two successes and two failures, so a share of 0 or 1 still gets a width; summed over independent shares, it gives the interval on a total such as bad text gained.
- **Share needed**: the target divided by the count, the lowest share of a count that must truly belong to its domain for the count to stay above the target; a share above 1 means the count falls short even if every item is right.
- **Kendall's τ** (Kendall, Kendall tau, rank agreement): agreement between two rankings of the same items, the share of item pairs ordered alike minus the share ordered oppositely; 1 is the same order, 0 unrelated, −1 reversed.
- **Bootstrap** (bootstrap interval): a confidence interval found by redrawing the scored items with replacement many times and recomputing the statistic on each draw; the interval spans the middle 95% of those values.
- **Morpheme-boundary F1** (morph boundary F1, MorphScore): F1 of the cuts a tokenizer makes inside a word against the cuts a morphological gold makes there, pooled over all scored forms.
- **Morph-Edit-Distance**: the number of token insertions, deletions and substitutions that turn a word's tokens into its gold morphemes, averaged over forms; lower is closer to the gold.
- **Morph-Consistency**: F1 of whether two words that share a gold morpheme also share a token, over sampled word pairs; high means a morpheme is spelled by the same token wherever it appears. Pairs are drawn per shared morpheme, at most 200 each, so on a stem-ending gold the many stems outweigh the few endings.
- **Fertility**: tokens per word over a held-out text sample; 1 means every word is one token, and higher means words are cut into more pieces.
- **Compression** (characters per token, bytes per token, tokens per byte): how much text one token covers on a held-out sample; higher means shorter token sequences.
- **Rényi efficiency**: Rényi entropy of the token frequencies (order 2.5) divided by its maximum for the vocabulary size; it is high when tokens are used evenly, low when a few tokens dominate.
- **Quarter rule** (levelled off, still improving, no early gain): the levelling rule of the vocabulary sweep. A metric has levelled off for an algorithm when its gain from 64k to 128k is under a quarter of its gain from 16k to 32k; it falls when that late gain is negative, and the rule does not apply when there was no early gain.
- **Off-source precision** (precision off the source, off-source read): a domain's precision read only on documents whose source carries a different label, so the classifier has to find the domain from the text rather than inherit it from the source.

## Terms

- **CJVT**: the Centre for Language Resources and Technologies of the University of Ljubljana, which maintains Sloleks.
- **q05, q50, q95** (q05, q50, q95): the 5th, 50th (median) and 95th percentiles of a distribution.
- **Curation pipeline**: the eight stages that turn extracted text into the pretraining corpus, in order convert, language, spam, quality, repetition, exact dedup, sentence dedup and statistics; the six from language to sentence dedup each keep or drop every document.
- **Content filter**: one of the four curation stages that judge a document's content — the language, spam, quality and repetition filters — as opposed to the two dedup stages, which remove copies.
- **Gopher heuristics** (Gopher-style, quality filter, repetition filter): the rule-based quality and repetition filters published with DeepMind's Gopher model — thresholds on document length, word length, symbol ratios and repeated n-grams, tuned on English web text.
- **Letter rule** (letter share): the Gopher quality rule that most words contain a letter, meant to catch tables of numbers and symbol junk; the letter share is the share of a document's words that contain a letter, and the rule drops the document when it falls under 0.8.
- **Route** (curation route, raw-crawl route, curated-prose route, medical-and-academic route): a named set of content-filter settings that every source of one kind shares in the curation pipeline, so a web crawl and an edited corpus are filtered differently. Not the repository's conversion routes, which are the three output trees that fork after extraction.
- **Raw web crawl** (web crawl, web crawls): a source scraped from the open web with little editing — here c4, cc100, culturax, fineweb2, hplt, macocu_sl, classla_web_sl and finepdf. In the curation-thresholds experiment finepdf, a crawl of PDF documents, takes the curated-prose route.
- **Curated source**: a source assembled or edited by people for a purpose — here the parliamentary, medical, legal, scientific, news, wiki and reference corpora. In the curation-thresholds experiment the medical and academic ones take a route of their own.
- **Sloleks**: the reference lexicon of Slovene word forms, used here to check whether a token is an ordinary Slovene word.
- **LLM judge** (judge): a language model given a fixed rubric and asked to label documents, standing in for a human annotator once its agreement with one is measured.
- **Bad text** (lenient bar, strict bar): at the lenient bar, a document the judge scores coherence 2 or lower, labels garbage or boilerplate, or flags as adult or spam; the strict bar also counts coherence 3.
- **Coherence**: the rubric's 1-5 score of whether a document reads as connected, sensible text, from unreadable (1) to fluent, well-formed prose (5).
- **Boilerplate**: text furniture rather than written content — menus, footers, cookie notices, navigation, templated disclaimers.
- **Calibration set**: documents labelled by both a person and the judge, so their agreement is known before the judge's labels are used as evidence.
- **Adjudication set**: documents drawn where the judge and the pipeline flatly disagree, labelled by a person to settle which of the two is right.
- **Stratified sample**: a fixed number of documents from every source, stage and decision, so small sources are represented; its rates describe each cell, not the corpus as a whole.
- **Pairwise comparison**: the judge is shown two documents and asked which is better, or whether they tie; each pair is asked twice with the documents swapped.
- **Position bias**: a judge's tendency to prefer whichever document it is shown first (or second), regardless of content.
- **Grill** (grill round): a structured design interview that questions a plan one decision at a time; a history line's `Q15` names its fifteenth question.
- **SLING**: the Slovenian national supercomputing network, whose clusters the project can use for jobs that do not fit on one machine.
- **Gated source**: a dataset whose licence restricts redistribution, so it is counted separately from what a public release could hold.
- **KPI** (KPI 2, KPI 3, KPI 4): a key performance indicator the ARIS proposal (Z2-70067) commits the project to. KPI 2 asks for more than 10,000 examples in each of at least two domains, medicine and science; KPI 3 for more than 5B tokens per language; KPI 4 for more than 500,000 tokens per domain.
- **Access filter** (open, open+login, all): which datasets a total counts. `open` is what anyone can download, `open+login` adds what a registered user can, `all` adds gated and not-downloadable sources — what exists rather than what is reachable.
- **Provenance** (native, translated, generated): how a dataset's Slovene text came to exist. `native` is written in Slovene, `translated` is carried into Slovene from another language, `generated` is machine-written or automatically annotated.
- **Provenance class** (human-translated, machine-translated, synthetic, bilingual resource, mixed): the finer reading of a checked row. Human-translated text was translated by people; machine-translated by a translation system or a language model; synthetic text was written by a language model; a bilingual resource is a dictionary or parallel corpus whose translation direction is unknown; mixed combines several.
- **Machine translation** (MT): translating text automatically, with a dedicated system such as Google Translate or DeepL or with a language model prompted to translate.
- **Estimated tokens** (tokens_estimated): a reported word count multiplied by two, the working subword-tokens-per-word factor until the project tokenizer exists. Empty when the source reported no word count.
- **Unknown** (as a KPI verdict): the total falls short of the threshold but datasets in that domain reported no size, so the shortfall may be an absence of evidence rather than an absence of text.
- **CLARIN.SI** (CLARIN, clarin.si): the Slovenian node of CLARIN, the European research infrastructure for language resources; the main publisher of Slovene corpora, which it addresses by `11356/<n>` handles.
- **LINDAT** (LINDAT/CLARIAH-CZ): the Czech CLARIN node, which re-publishes some Slovene holdings under `11234/<n>` handles.
- **ELG** (European Language Grid): an EU platform cataloguing language resources and services; for Slovene it largely mirrors CLARIN.SI rather than publishing its own.
- **NER** (named-entity recognition): labelling spans of text with the kind of entity they name — a person, a place, a drug, a diagnosis. One of the information-extraction tasks the project targets.
- **Entity span** (span, IE spans, extraction spans): one named entity recorded as the character range of its surface text in a document, `[start, end, label]` with `end` exclusive, plus an upper-case label such as `PER`, `LOC`, `ORG`, `MISC` or `DERIV-PER` (a word derived from a person's name, like "Freudovo"). Decoded from the IOB tags of the source, where `B-` opens an entity, `I-` continues it and `O` is outside any entity.
- **IE** (information extraction): pulling structured facts — entities, relations, attributes — out of running text. The project's target task family.
- **PDF** (Portable Document Format): the page-layout file most Slovene journals and institutions publish articles and guidelines in. Its text has to be extracted before it can be counted, and a scanned PDF yields none.
- **Tokenizer**: the component that splits text into the subword units a language model reads; its vocabulary decides how many tokens a word becomes.
- **OAI-PMH** (Open Archives Initiative Protocol for Metadata Harvesting): the standard interface journals and repositories expose for listing every record they hold, so a whole archive can be enumerated without scraping.
- **Instruction set** (instruction sets): prompt-and-answer pairs written, often by another model, to train a chat model to follow requests; machine output rather than native prose.
- **MLflow**: the experiment-tracking server the project logs every run to, with its parameters, metrics and output files.
- **Lineage run** (lineage): an MLflow run that trains or scores nothing and only records which data and code produced a set of outputs, keeping them as attached files.
- **Download registry** (registry): the project's list of datasets it downloads and builds its corpus from, kept in `configs/data/download.yaml`.
- **Agreement gate** (gate): the rule that a judge run's verdicts count as findings only when the judge agrees with the person on bad text at a Cohen's κ of at least 0.6 on the calibration set judged in that same run.
- **Fresh draw** (third draw): a stratified sample drawn after the settings were frozen, held back from tuning; the third draw is a later one from the loosened build, limited to the filters and sources whose settings changed.
- **Frozen build** (baseline build, loosened build): the baseline build is the shared corpus rebuilt at the pipeline's settings on main, with no per-source routes; the frozen build is the candidate corpus built with the route settings frozen before the fresh draw; the loosened build relaxes its quality and repetition settings per route and is built into its own folder beside it.
- **Volume-weighted** (weighted by volume): a rate over several sources in which each source counts by the documents it holds in the build, not by how many of its documents were judged, so the rate describes the corpus rather than the average source.
- **Kept corpus** (kept documents): the documents that leave the repetition filter, the last content filter; the dedup stages that follow remove copies but judge no content, so a build's bad-text share is read there.
- **1:1 rate** (exchange rate): the rule that a filter is worth keeping on a source only where at least half of what it drops is bad text, so one bad document removed is worth one good document lost.
- **Document-level domain** (document labels, document-level labels): the domain a classifier or judge assigns to one document from its own text, as opposed to the source label every document of a dataset inherits.
- **Lexicon scoring** (lexicon classifier): labelling a document by counting, per 1,000 words, its weighted hits on a word list per domain and taking the highest-scoring domain when it clears a threshold and leads the runner-up by a margin.
- **Source gap**: a shortfall against a target in a domain that never had the material before curation, so only new sources can close it.
- **Curation loss**: a shortfall against a target in a domain that had the material before curation and lost it at a named filter stage.
- **Source label** (source labels): the domain every document of a dataset inherits from the dataset's entry in the download registry, as opposed to a document-level domain read from the text.
- **Data target** (examples target, token target, corpus target): the three KPI thresholds read as pass-or-fail bars — ten thousand documents per domain (the examples target), half a million estimated tokens per domain (the token target) and five billion estimated tokens per language (the corpus target); read for science and medicine, they give the four bars H1 to H4 of the KPI-coverage experiment.
- **Judged gold** (gold, tune side, test side): the curation-quality judge's sample of documents with the domain the judge assigned each, halved into a tune side for picking a classifier's settings and a test side read once to pass or fail it.
- **Classifier gate** (quality gate): the rule that a domain classifier's labels count only when its test-side macro-F1 is at least 0.7 and its off-source medical precision at least 0.8.
- **Log-odds** (log-odds ratio): the logarithm of how many times more often a word occurs in one domain's documents than in the others', used to rank the words most distinctive of a domain.
- **TF-IDF** (tfidf): term frequency times inverse document frequency; a word or character n-gram's count in a document, discounted by how many documents carry it, so n-grams that occur everywhere weigh little.
- **Regularisation** (C, regularisation strength): a penalty on large weights in a fitted model that keeps it from memorising the training documents; C is the inverse strength, so a smaller C regularises more.
- **Logistic regression**: a linear classifier that turns a weighted sum of features into a probability per class; with class weights, every class counts the same whatever its size.
- **Sentence encoder** (embeddings, multilingual-e5-small): a neural model that turns a text into one fixed-length vector so that texts about the same thing lie close; multilingual-e5-small is the small multilingual encoder tried here.
- **Exact dedup** (exact deduplication): the curation stage that drops every document whose text is byte-identical to one already kept, wherever in the corpus the copy sits.
- **Converted documents** (convert stage): a source's documents after the convert stage, before any filter runs; the starting count every retention and loss share is taken over.
- **Funnel** (source funnel, domain funnel): a source's or domain's document count after every curation stage in order, read to see at which stage text is lost.
- **Medical leaflet source** (medical source, povejmo_vemo_med): the one dataset in the download registry labelled medical, PoVeJMo-VeMo-Med 1.0, drug package leaflets from the Slovene central drug database; its release ships the same texts twice, so its converted count is about three times its distinct texts.
- **KAS-term**: a terminology-identification dataset built from KAS, the corpus of Slovene academic theses, with the spans of technical terms marked in each text.
- **HUMADEX**: a medical named-entity dataset machine-translated into Slovene, with entity spans; large, but not native Slovene text.
- **BPE** (byte-pair encoding, byte-level BPE): a tokenizer trained by repeatedly merging the most frequent adjacent pair of symbols into one token; byte-level BPE starts from the 256 byte values, so it can encode any string.
- **Character BPE** (charBPE, character-level BPE): BPE started from the characters of the training text rather than from bytes, so a character outside its alphabet becomes an unknown token.
- **WordPiece**: the tokenizer of BERT, which merges the symbol pair that most raises the likelihood of the training text and marks word-internal pieces with a prefix.
- **Unigram** (SentencePiece Unigram): a tokenizer that starts from a large candidate vocabulary and prunes the pieces whose removal least lowers the likelihood of the training text; a word's tokens are its most likely segmentation.
- **MorphBPE**: BPE whose training forbids merges across the morpheme boundaries of a morpheme table, so a token never straddles a morpheme; at inference it is standard BPE.
- **MorphPiece**: byte-level BPE applied after the words listed in a morpheme table are first split by the table.
- **Morpheme table** (morph table, morpheme lexicon): the list of words with their morpheme cuts that MorphBPE and MorphPiece read while training.
- **Unknown token** (`<unk>`): the token a tokenizer emits for a character its alphabet does not hold.
- **Lexeme** (paradigm): a dictionary entry with its whole set of inflected forms; the paradigm is that set of forms.
- **Stem-ending split** (stem, ending): a cut of an inflected form into the part shared by its whole paradigm, the stem, and the part that carries case, number, person or gender, the ending.
- **Manual forms** (manually checked forms, automatic rows): in Sloleks 3.2, the forms of lexemes whose cuts the lexicon's authors checked by hand; the rest are generated by rules and called automatic.
- **Heuristic gold** (lemma-alignment gold, corrected heuristic): the project's own morphological gold, which cuts an inflected form where it first differs from its lemma, with rules for the usual stem changes.
- **Sloleks gold** (Sloleks 3.2 gold, Sloleks 3.2 split): the morphological gold cut from the Sloleks 3.2 stem-ending split, one cut per form.
- **NFC**: Unicode normalisation form C, which composes an accented letter and its accent into one code point so that the same letter is counted once.
- **CC BY-SA** (CC, Creative Commons): the Creative Commons attribution, share-alike licence, under which a dataset may be reused and redistributed when its authors are credited and derived data carries the same licence.
- **JSONL** (JSON lines): a text file with one JSON object per line, the shape of every converted dataset in the project.
- **NUL** (NUL character): the byte value zero, which never belongs in text; in the corpus it stands where a letter was lost upstream.
- **SSJ** (ssj500k, UD Slovenian SSJ): the manually annotated Slovene reference corpus, whose Universal Dependencies treebank the MorphScore v2 Slovene items are drawn from.
- **UD** (Universal Dependencies): a cross-lingual scheme of part-of-speech tags, morphological features and syntactic dependencies, with one annotated treebank per language.
