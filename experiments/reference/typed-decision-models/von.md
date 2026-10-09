---
title: Von
kind: algorithm
status: from-paper
summary: An open decision model on a ModernBERT-large encoder that packs question, state and options into one sequence with a mask token before each option, scores each mask token's state with a small head while each option attends only to the state and itself, trains with cross-entropy plus a Brier term, and reports confidence through a temperature that depends on the request.
variant_of:
part_of:
couples_to: []
---

## Description

Von answers the same typed questions as Jev with a bidirectional encoder of
395 million parameters instead of a decoder. The question's instructions and
the state form a prefix. After a separator, every option is written out behind
a mask token, the token a masked language model was trained to fill. One
encoder pass gives each mask token a hidden state that has read the prefix and
its own option, and a two-layer head turns that state into one score. A
softmax over the scores is the question's distribution.

Three choices distinguish it from a decoder with a pointer head.

- **Options cannot see one another.** From version 1.2 the attention mask lets
  an option token attend to the prefix and to its own option span only, and
  every option span restarts at the same position. An option's score is
  therefore a function of the prefix and that option alone, and reordering
  the options cannot change any score.
- **Digits are single tokens.** Every run of digits is split into single
  characters before tokenisation, because the encoder's vocabulary cuts
  numbers inconsistently and the model could not learn arithmetic over them.
- **Confidence tracks the request.** The softmax temperature is not a
  constant but a bounded linear function of the answer's entropy, the length
  of the state and the number of options, fitted on labelled items.

A yes-or-no question is two described options. When the caller gives no
descriptions, the model also scores the same question against an empty state
and subtracts part of that context-free preference, because the model prefers
yes with no state at all.

## What the source adds

The model card and repository publish the corpus composition, the loss, the
calibration map with the items it was fitted on, the order-invariance test
across orderings, a threshold sweep that says which confidence cutoff keeps
which accuracy, and the probes that found each weakness. Corpus builders and
the training script are in the repository, so the recipe can be rerun on
another encoder or another language.

## Why it matters here

Von is the open model in this topic whose backbone is closest to what this
project trains: a masked-language encoder of the ModernBERT generation. Its
head and packing need nothing from the encoder beyond a mask token and
hidden states, so the same design would sit on a Slovenian encoder. The
repository's fine-tuning guide names a fresh head on another encoder for
another language as a supported route, with the English corpus as the base
and the language's own labelled rows added on top.

## Facts

- **Question types**: choice over described options; Noul as two described options, true first, with a context-free bias subtracted when no descriptions are given; score over levels, returning the expected level
- **Backbone**: ModernBERT-large, 395 million parameters, every weight fine-tuned; a multilingual sibling, Laya, uses mmBERT-base, and the repository's guide trains a fresh head on other masked-language encoders
- **Where an option is read**: the hidden state of the mask token placed before the option's text, through a normalised two-layer head to one logit
- **Options per question**: bounded by the 8,192-token sequence; the authors recommend few
- **Isolation between questions**: complete, one encoder pass per question; within a question an option attends only to the prefix and itself, with its positions reset, so the scores are order-invariant
- **Training objective**: cross-entropy plus half the Brier score against the target distribution over the options, accepting a soft target
- **Calibration**: temperature $T = \theta_0 + \theta_1 \hat{H} + \theta_2 \log_{10}(\text{state tokens})/4 + \theta_3 K/8$, with $\hat{H}$ the normalised entropy of the raw distribution and $K$ the option count, clamped to $[0.3, 12]$ and fitted by likelihood on labelled items; confidence as the vendor adapter's formula; a yes-or-no probability is pushed out of a band around one half before it is returned
- **Training data**: about 290,000 balanced decision items across operational, security, moderation, semantic, triage and adversarial families, the semantic part including the ANLI and WANLI entailment sets as yes-or-no questions; 40,000 long-context items and 40,000 synthetic two-hop and numeric items; no item of the JevBench benchmark
- **Context length**: 8,192 tokens; a longer state is cut in the middle
- **Weights**: Apache-2.0 on Hugging Face as wfzyx/von, about three gigabytes

## Sources

- **Model card**: Von 1.3, <https://huggingface.co/wfzyx/von>, which carries the corpus breakdown, the calibration numbers and the benchmark record
- **Repository**: github.com/wfzyx/von, commit 9319ced read for this entry
- **Code**: none in this repository; in the Von repository, src/von/models/option_marker.py packs the sequence, builds the masks and positions and holds the head; src/von/backends/option_marker_backend.py evaluates each question type, resolves the temperature and the confidence and applies the yes-or-no band; training/train_option_marker.py holds the loss; src/von/calibrate.py fits the temperature map
- **Paper**: JevOut: Natural Context Can Flip Decision Models, 2026, arXiv:2609.30243, which evaluates Von 1.0 beside Jev
- **Example**: `experiments/reference/typed-decision-models/examples.py`, which runs the released temperature map on toy logits
- **Paper vs code**: no paper. Against the model card, the code computes the confidence with the option-count formula of the vendor adapter and not a top-two margin; the card does not mention the empty-state pass that debiases a yes-or-no question without descriptions, nor the band that moves a returned yes-or-no probability away from one half, both of which the code applies by default

## Algorithm

```algorithm
\caption{Von: option-marker scoring in one encoder pass}
\Require state $x$; question $t$; options $o_1, \dots, o_K$ with descriptions; encoder $E$; head $g$; temperature map $\theta$
\Ensure a distribution $p$ over the options and a confidence $c$
\State $s \gets t\ x$ \texttt{[SEP]} \texttt{[MASK]} $o_1$ \texttt{[MASK]} $o_2 \cdots$ \texttt{[MASK]} $o_K$, with every digit its own token and any marker string inside user text broken
\State mask: a prefix token attends to the prefix; an option token attends to the prefix and to its own span
\State positions: every option span restarts at the prefix length; the local-attention window is measured on these positions
\State $H \gets E(s)$ under the mask and positions
\State $z_k \gets g(H[\texttt{[MASK]}_k])$ for $k = 1, \dots, K$
\State $T \gets \mathrm{clamp}(\theta_0 + \theta_1 \hat{H}(z) + \theta_2 \log_{10}|x| / 4 + \theta_3 K / 8,\ 0.3,\ 12)$
\State $p \gets \mathrm{softmax}(z / T)$ and $c \gets (K \max p - 1)/(K - 1)$
\Return $\arg\max z$, $p$, $c$; for a score question also $\sum_k (k - 1)\, p_k$
```

- **1**: The whole question is one string. The prefix is instructions then
  state, and each option follows a mask token. Digits are split so that the
  same number tokenises the same way wherever it appears. A literal mask or
  separator string in user text is broken, so a document cannot add a
  phantom option.
- **2** and **3**: The mask and the positions together make each option's
  computation independent of the other options. Blocking attention alone is
  not enough, because rotary positions encode distance and an option's
  distance from the prefix would still depend on how many option tokens sit
  before it.
- **4** and **5**: One pass gives every token a state. The head reads the
  mask token in front of each option, the position a masked language model
  learned to fill from context, and returns one score per option.
- **6**: The temperature is computed from the request itself. A flat raw
  distribution, a long state and many options each raise it, so the model
  claims less than its raw scores suggest where it tends to be wrong.
- **7**: The scaled scores become probabilities and the confidence is the
  top probability rescaled by the option count. The temperature is monotone,
  so the chosen option is the argmax of the raw scores either way.
- **8**: The chosen option, the distribution and the confidence are returned.
  A score question's answer is the expected level.

```algorithm
\caption{Von training: cross-entropy plus Brier over the option markers}
\Require labelled items $(x, t, O, y)$, some with a target distribution $d$; encoder and head initialised from the previous release; Brier weight $0.5$
\Ensure fine-tuned encoder and head, then a fitted map $\theta$
\ForAll{batches}
  \State $z \gets$ lines 1 to 5 of Algorithm 1; $p \gets \mathrm{softmax}(z)$; $d \gets$ one-hot $y$ unless the item carries a target distribution
  \State $\ell \gets \mathrm{mean}\big(-\sum_k d_k \log p_k\big) + 0.5\, \mathrm{mean}\big(\sum_k (p_k - d_k)^2\big)$
  \State update every weight of the encoder and the head
\EndFor
\State $\theta \gets \arg\min_\theta \sum -\log \mathrm{softmax}(z / T_\theta)[y]$ over labelled items, with the weights frozen
\Return encoder, head, $\theta$
```

- **1** and **2**: Training packs and scores exactly as serving does, at a
  temperature of one. The target is the labelled option, or a whole
  distribution when the item was built with one.
- **3**: The loss adds a Brier term, the squared distance between the
  predicted and the target distribution, to the cross-entropy. The Brier
  score is a proper scoring rule, which rewards reporting the frequency one
  actually observes, and it penalises a confident wrong answer more than the
  cross-entropy alone does near certainty.
- **4**: The encoder is not frozen. The released weights are a full retrain
  from the previous release with the independent-options mask on.
- **5**: The temperature map is fitted afterwards on labelled items, by the
  likelihood of the correct option, with a scalar temperature fitted beside it
  and the better one under cross-validation kept.
- **6**: The checkpoint holds the encoder, the head and the map, and the map
  can be refitted on a deployer's labels without touching the weights.

## Worked example

The example script runs the released calibration module with the map shipped
in the Von 1.2 checkpoint on one set of raw scores for three options, with
the state at three lengths.

![Temperature and confidence for the same raw scores at three state lengths](tables/von-temperature.csv)

The raw scores do not change, so the chosen option does not change. The
temperature rises with the length of the state, from about three at forty
tokens to about nine at four thousand, and the reported confidence falls from
0.21 to 0.08. The model says less about a long state because, on the items
the map was fitted on, it was right less often there.

## Where it sits

Von serves the same endpoint as Jev and ships as a Python and TypeScript
package. Inside, it is a stock ModernBERT encoder with a head of a few hundred
thousand parameters, run once per question on CPU or GPU. A tiered
confidence gate in the package splits answers into those taken automatically
and those escalated, at a cutoff the repository measured on held-out items.

## Limits

### English only

The encoder and the corpus are English. Other languages get token matching
with a confidence the map never saw, and the multilingual sibling Laya on
mmBERT-base is a separate model with its own weights.

### The map is fitted on one distribution

The temperature map is fitted on benchmark items. The repository's own
threshold sweep finds that on out-of-domain probes the ranking by confidence
breaks and no cutoff keeps accuracy, so a deployer refits the map on labels.

### A yes-or-no question without descriptions is weak

With no descriptions, the two options are generic and surface cues win. The
empty-state debiasing and the band around one half are repairs on top of
that, and the band means the returned probability is no longer the
calibrated one.

### An option is judged alone

Blocking cross-option attention buys order invariance and loses the ability
to judge an option relative to the others. A question whose answer is "the
better of these two" is read as two separate judgements.
