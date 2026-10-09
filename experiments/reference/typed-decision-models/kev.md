---
title: Kev
kind: algorithm
status: from-paper
summary: An open Jev-like family on Qwen bases that packs the state and every question's option list into one causal sequence under a block mask, scores each option's closing token against the question's decide token with a pointer head, and is trained with cross-entropy through a low-rank adapter.
variant_of: typed-decision-models/jev
part_of:
couples_to: []
---

## Description

Kev is a family of decision models built by Jared Palmer on Qwen3.5 and
Qwen3.8 bases, in four sizes from 0.8 to 27 billion parameters, that answer
the same typed questions as Jev. The design follows the reconstruction of
Jev's architecture: the state is written once at the front of the sequence,
each question follows as its own branch holding the instructions and every
option between marker tokens, and an attention mask lets a branch read the
state and itself but never another branch. A small pointer head then compares
the hidden state of each option's closing marker with the hidden state of a
decide marker at the end of the branch, and a softmax over those comparisons
is the question's distribution.

Three choices keep the recipe cheap and open to inspection.

- **The base stays frozen.** Only a low-rank adapter of rank 16 on the
  attention, feed-forward and recurrent projections, and the pointer head,
  are trained. The 27 billion model is the exception and fine-tunes every
  weight.
- **Options are text, not slots.** The head has no fixed output layer, so a
  question may bring any number of options and any wording. A yes-or-no
  question is two options, false and true; a score question lists its levels
  as options.
- **Training and serving share one text format.** A training example is an
  API request with a label on every question, so the model never sees a
  layout at training time that it does not see when served.

## What the source adds

The repository is the record of the method rather than a paper. It fixes
frozen evaluation suites with checksums, keeps a plan file of what was tried
and what did not help, and ships every number in the README with the report
it came from, checked in continuous integration. The releases state that no
Jev outputs were used for training.

## Why it matters here

Kev is the cheapest documented way to turn an existing small decoder into a
decision model: a short adapter run on one GPU, with the recipe, the data
builders and the evaluation suites public. The option text is read in
context, so nothing in the head is tied to English, and a Slovenian base or a
Slovenian fine-tune would reuse the whole pipeline. The published per-stage
data and learning rates give a starting point for a fine-tune on Slovenian
labelled decisions.

## Facts

- **Question types**: choice over named options; Noul as the two options false and true; score with each level an option, the answer being the expected level
- **Backbone**: Qwen3.5 base at 0.8, 4 and 9 billion parameters, and the post-trained Qwen3.8 at 27 billion; a low-rank adapter of rank 16 and scale 32 on the attention, feed-forward and recurrent projections, with the base frozen except for the 27 billion model
- **Where an option is read**: the hidden state of the option's closing marker token, scored against the hidden state of the question's decide token by a bilinear pointer head of width 256
- **Options per question**: as many as fit the question's token budget of 8,192 tokens; the served protocol allows 255
- **Isolation between questions**: exact; a block-causal mask on attention-only bases, and one row per question with the state's cache reused on bases with recurrent layers, with each question's positions restarting right after the state
- **Training objective**: cross-entropy on the correct option, with the adapter and head trained together; optional Brier, ordinal and label-smoothing terms exist in the code and the released models do not use them
- **Calibration**: one scalar temperature per checkpoint, fitted on held-out labelled rows and dividing the logits at inference; confidence as the vendor adapter's formula
- **Training data**: a base set of 10,000 examples from ten public datasets, 896 generated policy examples and 1,680 examples from 60 generated rule structures, two epochs; then short follow-up stages from the released checkpoint on generated cases with day counts or removed evidence, 5,219 real consumer-finance complaints and generated tool-use data; no Jev outputs
- **Context length**: states of up to 65,536 tokens served, plus 8,192 per question; the base stage trains on states of at most 384 tokens and the document stages on at most 7,552, and the 27 billion model on up to 32,768
- **Weights**: Apache-2.0 adapters and heads on Hugging Face under jaredpalmer, with the 27 billion model as full weights

## Sources

- **Essay**: Jev's Architecture Unmasked, Archer Hume, 17 September 2026, the reconstruction this design implements, <https://archerhume.com/posts/jevs-architecture-unmasked>
- **Repository**: github.com/jaredpalmer/kev, commit 5e42a7a read for this entry; the model cards under docs/model-cards list every training stage with its data and cost
- **Code**: none in this repository; in the Kev repository, kev/model.py packs a record, builds the block-causal mask and holds the pointer head; kev/train.py holds the question loss; kev/api.py holds the confidence formulas; kev/metrics.py fits the temperature
- **Weights**: <https://huggingface.co/jaredpalmer/kev-4b> and its siblings
- **Example**: `experiments/reference/typed-decision-models/examples.py`, which ports the packing rule to a toy record
- **Paper vs code**: no paper. Against the README, the code also offers an option-isolation mode in which an option span sees only the state, the instructions and itself and all option spans share one position range, which makes the scores order-invariant; the released models do not use it. The README says training is plain cross-entropy; the loss function also accepts a soft target distribution, used for generated examples that carry one

## Difference from predecessor

- **Line**: 8
- **Predecessor**: $z_k \gets W h_k + b$, one score per answer read from the branch's final state
- **Here**: $z_{k,o} \gets (W_k h_{k,o})^\top (W_q h_k) / \sqrt{d_p}$, the option's closing token matched against the decide token
- **Change**: the score of an option is a match between two token states, not a row of a fixed output layer.
- **Effect**: Any number of options with any wording fits the head, and an option's score depends on its own text.

## Algorithm

```algorithm
\caption{Kev: one pass over the state and every question, read by a pointer head}
\Require state $x$; questions $q_1, \dots, q_Q$ with instructions $t_k$ and options $O_k$; adapted decoder $f$; pointer head $W_q, W_k$ of width $d_p$; temperature $T$
\Ensure a distribution $p_k$ over $O_k$ for every question
\State $S \gets$ \texttt{<state>} and the tokens of $x$, cut to the state limit; segment $0$, positions $0, \dots, |S|-1$
\ForAll{questions $q_k$}
  \State $B_k \gets$ \texttt{<q>} $t_k$, then \texttt{<opt>} $o$ \texttt{</opt>} for each $o \in O_k$, then \texttt{<decide>}; segment $k$, positions restarting at $|S|$
\EndFor
\State mask: token $i$ may attend to token $j \le i$ when $j$ is in the state or in $i$'s own segment
\State $H \gets f(S B_1 \cdots B_Q)$ under the mask; on a base with recurrent layers, each $S B_k$ runs as its own row with the state's cache reused
\ForAll{questions $q_k$}
  \State $h_k \gets H[\texttt{<decide>}_k]$ and $h_{k,o} \gets H[\texttt{</opt>}_{k,o}]$ for each option
  \State $z_{k,o} \gets (W_k h_{k,o})^\top (W_q h_k) / \sqrt{d_p}$
  \State $p_k \gets \mathrm{softmax}(z_k / T)$
\EndFor
\Return $p_1, \dots, p_Q$
```

- **1**: The state opens the sequence behind one marker token. A state longer
  than the limit is cut, or refused when the caller asks for strictness.
- **2** and **3**: Each question is appended as a branch. The branch's
  positions start where the state ended, so every question sits at the same
  distance from the state whatever came before it in the sequence.
- **4**: The mask is causal within the state and within a branch, and a
  branch may read the state. A branch never reads another branch, which is
  what makes the questions independent of one another.
- **5**: One forward pass gives every token a hidden state. The recurrent
  layers of the newer Qwen bases ignore attention masks, so there each
  question is its own row, state first, and the state's cache is computed
  once and reused. The two forms agree to rounding.
- **6** to **8**: The pointer head projects the decide token as a query and
  every closing option marker as a key, and their scaled product is the
  option's score. The decide token comes last in its branch, so it has read
  every option before it is compared with each.
- **9**: A softmax over the question's scores is its distribution. The
  temperature is one during training and a fitted constant when served, so
  the argmax never changes.
- **10**: Every question's distribution is returned in one response. A score
  question's answer is the expected level under its distribution, and a
  yes-or-no question returns the probability of true.

```algorithm
\caption{Kev training: cross-entropy through the adapter and the head}
\Require records of labelled questions; frozen base; adapter of rank $16$; two epochs; held-out labelled rows
\Ensure adapter, pointer head and temperature $T$
\ForAll{batches of records}
  \State $z_k \gets$ the scores of Algorithm 1 with $T = 1$ for every question $k$
  \State $\ell \gets \sum_k -\log \mathrm{softmax}(z_k)[y_k]$, or $-\sum_o d_{k,o} \log \mathrm{softmax}(z_k)[o]$ when a question carries a target distribution $d_k$
  \State update the adapter and the head by AdamW on $\ell$, clipping the gradient norm
\EndFor
\State $T \gets \arg\min_T \sum -\log \mathrm{softmax}(z/T)[y]$ over the held-out rows
\Return adapter, head, $T$
```

- **1** and **2**: Training runs the same packing and readout as serving,
  with the temperature at one so that the fitted value keeps its meaning.
- **3**: The loss is the negative log probability of the labelled option.
  Generated examples may carry a whole target distribution instead of one
  label, and then the loss is the cross-entropy against it.
- **4**: Only the adapter and the head move. The 27 billion model trains
  every weight at a much lower learning rate instead.
- **5**: After training, one scalar temperature is fitted by likelihood on
  labelled rows the model did not train on. It changes how sure the model
  says it is and never which option it picks.
- **6**: The checkpoint is the adapter, the head and the temperature; the
  base is downloaded separately.

## Worked example

The example script ports the packing rule to a record whose tokens are words:
a three-word state and two questions, one with two named options and one
yes-or-no question.

![The packed sequence, with segment, position and what the last decide token sees](tables/kev-packing.csv)

Both questions' positions restart at four, right after the state, so each
question is laid out as if it were the only one. The decide token of the
second question sees the four state tokens and its own ten tokens, and none
of the first question's. The pointer head compares it with the two closing
option markers of its own question only.

## Where it sits

Kev serves the same endpoint as Jev, so the official TypeSafe client runs
against it unchanged. Inside, it is a stock Qwen decoder with an adapter and
a head of a few million parameters, so any tooling that serves Qwen serves
Kev. The repository ships a fine-tuning path that starts from a released
checkpoint, fits the temperature on a held-out slice of the user's own
labels, and scores the result against the untouched model.

## Limits

### Options still see one another

Isolation is exact between questions and absent within one. The decide token
attends to the whole option list, so reordering options can change their
scores, and the released models do not use the order-invariant mode.

### The temperature is fitted elsewhere

One scalar temperature fitted on the author's held-out suites does not
transfer to a deployer's workload, and the repository documents a case where
the served confidence sits well above the accuracy on a new dataset. The
repository therefore ships a per-workload refit.

### English and short states

The released models are trained on English and on states of at most a few
thousand tokens. Longer states are served, but the model cards validate
accuracy only up to 8,192 tokens for the three smaller sizes.

### Day arithmetic and removed evidence

The models answer poorly when a decision needs the number of days between
two dates, which a preprocessor can add, and when the deciding evidence is
absent from the state, which the follow-up stages train against but do not
remove.
