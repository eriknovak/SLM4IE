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

## Terms

- **q05, q50, q95** (q05, q50, q95): the 5th, 50th (median) and 95th percentiles of a distribution.
