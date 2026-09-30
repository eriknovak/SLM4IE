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
- **Precision@k** (P@k): precision over the top k ranked items only.
- **Recall@k** (R@k): the share of correct items found within the top k.
- **Accuracy**: the share of all decisions that are correct; misleading when classes are imbalanced.
- **Cohen's κ** (κ, kappa): agreement between two raters beyond what chance would give; 1 is full agreement, 0 chance level, below 0 worse than chance.
- **Quadratic-weighted κ** (weighted kappa): Cohen's κ for an ordered scale such as 1-5, where a disagreement costs the square of its distance, so a one-point miss counts far less than a three-point one.
- **Wilson interval** (Wilson score interval): a confidence interval for a share that, unlike the textbook one, never runs below 0 or above 1, which matters for shares near either end.
- **Drop precision**: of the documents a curation stage dropped, the share judged bad text; high means the stage removes what it should.
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

## Terms

- **q05, q50, q95** (q05, q50, q95): the 5th, 50th (median) and 95th percentiles of a distribution.
- **LLM judge** (judge): a language model given a fixed rubric and asked to label documents, standing in for a human annotator once its agreement with one is measured.
- **Bad text**: a document the judge scores coherence 2 or lower, labels garbage or boilerplate, or flags as adult or spam; the stricter bar also counts coherence 3.
- **Coherence**: the rubric's 1-5 score of whether a document reads as connected, sensible text, from unreadable (1) to fluent, well-formed prose (5).
- **Boilerplate**: text furniture rather than written content — menus, footers, cookie notices, navigation, templated disclaimers.
- **Calibration set**: documents labelled by both a person and the judge, so their agreement is known before the judge's labels are used as evidence.
- **Adjudication set**: documents drawn where the judge and the pipeline flatly disagree, labelled by a person to settle which of the two is right.
- **Stratified sample**: a fixed number of documents from every source, stage and decision, so small sources are represented; its rates describe each cell, not the corpus as a whole.
- **Pairwise comparison**: the judge is shown two documents and asked which is better, or whether they tie; each pair is asked twice with the documents swapped.
- **Position bias**: a judge's tendency to prefer whichever document it is shown first (or second), regardless of content.
- **Gated source**: a dataset whose licence restricts redistribution, so it is counted separately from what a public release could hold.
