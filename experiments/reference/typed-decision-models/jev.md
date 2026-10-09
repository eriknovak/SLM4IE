---
title: Jev
kind: algorithm
status: from-paper
summary: TypeSafe's closed decision model, reconstructed from how its API behaves, encodes the state once, gives every question an isolated branch that reads the whole option list, and returns a probability per allowed answer with no generated text.
variant_of:
part_of:
couples_to: []
---

## Description

Jev is a hosted model that answers typed questions about a piece of text or
data, called the state. Each question declares its allowed answers before the
call: a choice among named options, a score on an ordered list of levels, or a
yes against a no, which the vendor calls a Noul. The model returns a
probability for every allowed answer and a confidence per question. It never
writes a sentence, so there is nothing to parse and no answer outside the
declared set. All questions of one request are answered in a single pass, and
one question's text has no effect on another's answer.

TypeSafe released Jev on 15 September 2026 as the first of what it calls
System One models, after Kahneman's name for fast, automatic judgement. The
vendor describes a new architecture, a parallel sampler and a training method
it calls Reinforcement Learning for Calibrated Decisions. It has published no
paper, no weights and no recipe. What is written here as the algorithm is a
reconstruction from the model's behaviour at the API, by Archer Hume, and each
line says whether it is observed, published or inferred.

## What the source adds

Hume's essay reconstructs the architecture by probing the API rather than the
model. Four kinds of evidence carry the argument.

- **Token accounting.** The input tokens billed for a request are a shared
  part for the state plus one part per question, exactly additive. The state
  is therefore encoded once, not once per question.
- **Latency sweeps.** Time grows with the length of the state but stays flat
  up to about a hundred questions on a short state, which fits one encoding
  of the state followed by question branches run in parallel.
- **An isolation test.** A code word placed inside a sibling question is
  never chosen; the same code word moved into the state is chosen nine times
  in ten. The boundary between state and question is a hard boundary in the
  model, not only in the API.
- **An option-interaction test.** Adding an irrelevant fifth option to a
  four-way question shifts the odds between the existing options. Options
  are therefore read together, not scored one at a time with a fixed
  denominator.

## Why it matters here

Jev defines the interface the open models in this topic reproduce: state in,
typed probabilities out, one pass. A small Slovenian encoder would need the
same interface to be compared against them. The reconstruction also lists the
parts any such model must decide on, which the Compared table of this topic
puts side by side: where an option is read, how questions are kept apart, and
how the probabilities are calibrated.

## Facts

- **Question types**: choice over named options, score over ordered levels, Noul as a yes-or-no probability
- **Backbone**: unpublished; a sparse mixture-of-experts decoder is inferred from latency, and the tokenizer behind the token counts matches none of 192 public tokenizers tried
- **Where an option is read**: unpublished; options are processed jointly, since adding an option shifts the odds between the others
- **Options per question**: up to 255 named options
- **Isolation between questions**: complete; text in a sibling question has no effect on an answer
- **Training objective**: unpublished; the vendor names reinforcement learning for calibrated decisions, which the essay reads as training against a proper scoring rule on outcomes
- **Calibration**: the published adapter reports confidence as $(p_{\max} - 1/K)/(1 - 1/K)$ for $K$ options, which is zero at chance and one at certainty, not a probability of being right
- **Training data**: unpublished; the vendor speaks of synthetic data
- **Context length**: 32,768 tokens of text; no images
- **Weights**: closed, served through an API

## Sources

- **Blog**: Introducing System One Models and Jev, TypeSafe AI, 15 September 2026, <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- **Essay**: Jev's Architecture Unmasked, Archer Hume, 17 September 2026, <https://archerhume.com/posts/jevs-architecture-unmasked>
- **Paper**: Type-Safe Is Not Error-Free: Typed Decision Models Follow the Option Name, Not the Definition Bound to It, Sun, Xu, Shi and Yang, 2026, arXiv:2609.26758
- **Paper**: TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models, Sharma, Duncan, Marceau Caron and Vollmer, 2026, arXiv:2610.11392
- **Code**: none in this repository and none released by the vendor; the confidence formula is in TypeSafe's published Python adapter, system-one-adapter 0.2.1, in its confidence_metrics module, and the released CLM package ports the same formula
- **Example**: `experiments/reference/typed-decision-models/examples.py`, which runs the ported confidence formula on toy distributions
- **Paper vs code**: no paper and no code; each line of the algorithm is marked observed, published or inferred

## Algorithm

```algorithm
\caption{Jev as reconstructed from its API behaviour}
\Require state $x$; questions $q_1, \dots, q_Q$, each with instructions and an allowed answer set $O_k$
\Ensure a probability $p_k(o)$ for every allowed answer and a confidence $c_k$ per question
\State encode $x$ once into hidden states $H_x$ \Comment{observed: tokens are billed once for the state; time grows with the state, not with the number of questions}
\ForAll{questions $q_k$, in parallel}
  \State encode $q_k$ and its answer list $O_k$ after $H_x$; the branch reads $H_x$ and itself and never another branch \Comment{observed: a fact placed in a sibling question is invisible}
  \State $z_k \gets W h_k + b$, one score per answer in $O_k$, read from the branch's final state \Comment{inferred: the readout is not observable}
  \State $p_k \gets \mathrm{softmax}(z_k)$ \Comment{observed: adding an answer shifts the odds between the others, so the answers share one normalisation}
  \State $c_k \gets (\max_o p_k(o) - 1/|O_k|)/(1 - 1/|O_k|)$ \Comment{published: the vendor's adapter}
\EndFor
\Return $p_1, \dots, p_Q$ and $c_1, \dots, c_Q$, with no generated text
```

- **1**: The state is the expensive part of a request and is read once. The
  billed tokens of a request are the state's count plus a small count per
  question, which is what a shared prefix produces.
- **2** and **3**: Every question becomes a branch on top of the same state.
  A branch cannot see another branch, which the isolation test shows: moving
  a code word from a sibling question into the state raises its probability
  from zero to about nine in ten.
- **4**: Some head turns the branch into one score per allowed answer. The
  essay cannot tell a classifier over a fixed final position from a pointer
  that compares the answer's own representation with the question's, and
  neither can the probes. The open entries of this topic each pick one.
- **5**: The scores are normalised together. If each answer had its own
  independent score, adding an answer would leave the odds between the old
  ones unchanged. They change, so the answers are read as one list.
- **6**: The confidence is a rescaled top probability. It says how far above
  chance the top answer sits, and the worked example shows that the same top
  probability earns more confidence when more options share the rest.
- **7**: Nothing is generated. The returned object is the set of
  distributions, and an answer outside the declared set cannot occur.

## Worked example

The example script runs the confidence formula, as the released CLM package
ports it, on a top probability shared out evenly over the remaining options.

![Confidence for one top probability across option counts](tables/confidence-by-options.csv)

A top probability of one half is chance with two options and earns no
confidence. With ten options the same one half is far above the chance level
of one tenth, and the formula reports 0.44. The confidence therefore measures
distance from chance, not the chance of being right, and the number is not
comparable across questions with different option counts.

## Where it sits

Jev is served behind one endpoint, `/v1/systemone`, whose request carries the
state and a map of typed questions and whose response carries the typed
answers. The open models of this topic implement the same wire shape, so a
client written for Jev talks to any of them by changing the base URL. The
type-safe decoders of the information-extraction topic give the same guarantee
by a different route: there the scores come from an extraction encoder and a
search enforces declared rules among them.

## Limits

### The recipe is unpublished

Architecture, size, data and the training objective are vendor statements or
inferences. The essay's author marks the sparse backbone as the least certain
part, and the probes cannot distinguish a causal decoder from a bidirectional
encoder.

### Well-typed is not correct

The guarantee covers the shape of the output only. The 2026 study of typed
decision models reassigns option names to rubrics while keeping everything
else fixed and finds that decisions follow the option name rather than the
rubric bound to it, with the type-error rate at zero throughout.

### Calibration holds on the tested distribution

The essay measures a small calibration error on a multiple-choice academic
benchmark and a confidence that drops with task difficulty on generated
arithmetic. The TypedBench benchmark finds the hosted model wording-sensitive
and systematically underconfident on its policy-labelled suites. Neither
result transfers to a workflow that was not tested.

### Confidence is not a probability

The reported confidence is the top probability rescaled by the option count.
Two questions with different option counts report different confidences for
the same top probability, so one threshold does not mean one error rate.
