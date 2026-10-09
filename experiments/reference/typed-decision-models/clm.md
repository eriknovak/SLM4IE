---
title: CLM
kind: algorithm
status: from-paper
summary: The Contrastive Language Model answers a typed question as a lookup, with a frozen embedding model and two small projection heads trained by a two-way InfoNCE loss so that the softmax over the scaled cosines between the state and each option is the question's distribution, and every option is embedded once and cached.
variant_of:
part_of:
couples_to: []
---

## Description

CLM, from Stanford and NVIDIA, builds the Jev interface out of two encoders
rather than one sequence. A frozen language model turns a text into one
vector. A state head projects the vector of the state together with the
question, and an action head projects the vector of each option's
description. The score of an option is the cosine between the two projected
vectors times a learned scale, and a softmax over a question's scores is its
distribution. The two heads are trained contrastively: a state is pulled
towards the option that was taken and pushed away from the others in the
batch.

Two consequences follow from scoring by similarity.

- **Options are embedded independently of the state.** An option's vector is
  computed once and cached, so a question with many candidates, or many
  states against the same candidates, costs one embedding per new text and a
  dot product per pair.
- **The same call ranks anything.** A typed question is a state plus a closed
  set of candidate texts, so the engine that answers it also ranks free-form
  candidates: best-of-N answers, tool names, next moves. The authors use it
  as a verifier that picks the best of several candidate solutions.

The released model, CLM-8B, uses Qwen3-8B as the frozen embedder with
last-token pooling and two projection heads of about twenty million
parameters each.

## What the source adds

The source is a blog post with a repository, models and data rather than a
paper. It publishes the three-stage data recipe with the size of each stage,
a comparison of training with hard negatives from the start against adding
them later, and a scaling study in which the held-out contrastive loss falls
as a power law in compute, data, head size and encoder size, with encoder
size the strongest lever and the best head size growing linearly with the
training tokens.

## Why it matters here

CLM separates the two parts a typed decision model needs: a representation of
text, and a scoring rule over declared options. The heads are tiny and the
embedder is any model that gives one vector per text, so the recipe applies
to a small Slovenian encoder with a pooling layer as readily as to an
8-billion-parameter decoder. The published stage sizes also say how much
question-answer and hard-negative data the contrastive route consumed, which
sets expectations for a smaller language.

## Facts

- **Question types**: choice over described options; Noul as the two candidates false and true, with the statement itself as the default description; score over levels, returning the expected level; and free ranking of candidate texts
- **Backbone**: a frozen Qwen3-8B with last-token pooling, giving a 4,096-dimensional vector, and two trainable projection heads of about twenty million parameters each, mapping to 512 dimensions
- **Where an option is read**: as its own text through the action head, never in the same sequence as the state; the score is the learned scale times the cosine between the projected state and the projected option
- **Options per question**: unbounded; each candidate is embedded once and cached
- **Isolation between questions**: complete; each question embeds the state followed by a blank line and its instructions as one text, and no option ever sees the state
- **Training objective**: InfoNCE in both directions over a batch of matched state and action pairs, states against actions and actions against states, with pairs that share the same action masked out; hard negatives join the denominator in the second stage
- **Calibration**: none beyond the learned scale, capped at 100; the server accepts a temperature; confidence as the vendor adapter's formula
- **Training data**: about 60 million question-answer pairs from Nemotron DQA, each question a state and its answer the action; then about 30 million synthetic hard negatives written by Gemini 2.5 Flash-Lite as plausible wrong answers to the same questions; then about one million agent trajectory steps from the Agent Data Protocol set and two terminal-trace sets, mixed with 40 percent replay of the first stage
- **Context length**: 2,048 tokens per text by default, longer when the embedding server and the engine are both raised
- **Weights**: Apache-2.0 heads on Hugging Face as Contrastive-LM/CLM-v0.1-8B, 75 megabytes; the embedder is the public Qwen3-8B

## Sources

- **Blog**: Contrastive Language Models, Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré and Mirhoseini, 2026, <https://contrastive-lm.notion.site>, which carries the scaling-law fits and figures
- **Repository**: github.com/Contrastive-LM/CLM, commit d5f9ef0 read for this entry; its README states the loss, the data recipe and the scaling results
- **Code**: none in this repository; in the CLM repository, src/clm/schema.py builds the state text and the candidate texts and turns scores into typed answers; src/clm/heads.py defines the projection heads and the scoring; src/clm/engine.py answers a request with the embedding cache; train/finetune.py holds the two-way InfoNCE loss and the fine-tuning paths
- **Weights and data**: <https://huggingface.co/Contrastive-LM>
- **Example**: `experiments/reference/typed-decision-models/examples.py`, which runs the released readout on toy vectors and ports the loss to a toy batch
- **Paper vs code**: no paper. Against the blog, the code scales the cosines by a learned logit scale capped at 100 rather than dividing by a fixed temperature, and it masks out in-batch pairs whose actions are the same text, so a repeated answer is not counted as a negative of itself. The blog calls the state the question; the served state text is the state, a blank line and the question's instructions

## Algorithm

```algorithm
\caption{CLM: a typed question answered as a contrastive lookup}
\Require state $x$; questions with instructions $t_k$ and options $O_k$ with descriptions; frozen embedder $e$; heads $f_s, f_a$; scale $\sigma$; temperature $\tau$
\Ensure a distribution $p_k$ over $O_k$ for every question
\ForAll{questions $q_k$}
  \State $u_k \gets f_s(e(x \mathbin{\|} t_k)) / \lVert \cdot \rVert$, the state and the instructions as one text
  \ForAll{options $o \in O_k$}
    \State $v_o \gets f_a(e(\text{description of } o)) / \lVert \cdot \rVert$, taken from the cache when the text was seen before
  \EndFor
  \State $z_{k,o} \gets \sigma\, u_k^\top v_o$
  \State $p_k \gets \mathrm{softmax}(z_k / \tau)$
\EndFor
\Return $p_1, \dots, p_Q$; a Noul answer is $p_k(\text{true})$ and a score answer is $\sum_j (j - 1)\, p_{k,j}$
```

- **1** and **2**: Each question gets its own state text, the state followed
  by the instructions, embedded by the frozen model and projected by the
  state head to a unit vector. Putting the instructions in the state text is
  the layout the heads were trained on.
- **3** and **4**: Every option's description is embedded and projected by
  the action head on its own. An option never sees the state, so its vector
  is reused across states and across questions.
- **5**: The score is the cosine between the two unit vectors times the
  learned scale. This is the same quantity the training loss normalises
  over, so the softmax at serving time is the trained distribution.
- **6**: A softmax over the question's options gives its distribution. The
  served temperature defaults to one.
- **7**: The typed answer is read off the distribution: the argmax for a
  choice, the probability of true for a yes-or-no question, the expected
  level for a score.

```algorithm
\caption{CLM training: two-way InfoNCE in three stages}
\Require matched pairs $(s_i, a_i)$: question–answer pairs, then the same with hard negatives $h_{ik}$, then agent steps with replay; frozen embedder $e$
\Ensure heads $f_s, f_a$ and scale $\sigma$
\ForAll{batches of $B$ pairs}
  \State $u_i \gets f_s(e(s_i))$ and $v_j \gets f_a(e(a_j))$, both unit length
  \State $Z_{ij} \gets \sigma\, u_i^\top v_j$; set $Z_{ij} \gets -\infty$ where $a_j$ is the same text as $a_i$ and $j \neq i$
  \State in the second stage, append $\sigma\, u_i^\top f_a(e(h_{ik}))$ to row $i$ for every hard negative $h_{ik}$
  \State $\ell \gets \frac{1}{2B} \sum_i \big[ -\log \mathrm{softmax}(Z_{i,\cdot})_i - \log \mathrm{softmax}(Z_{\cdot,i})_i \big]$
  \State update $f_s$, $f_a$ and $\sigma$
\EndFor
\Return $f_s$, $f_a$, $\sigma$
```

- **1** and **2**: The embedder is run once per text and frozen; only the
  two heads and the scale learn. In practice the embeddings are computed
  ahead of training and the heads train on cached vectors.
- **3**: Every state is scored against every action in the batch. Two
  states that happen to share an answer would wrongly push each other away,
  so such off-diagonal pairs are removed from the denominator.
- **4**: The second stage adds, for each state, wrong answers written to
  look right. The source reports that adding them after a plain first stage
  beats training with them from the start at the same budget, because they
  refine a representation rather than form it.
- **5**: The loss is the mean of two cross-entropies, one picking the right
  action for each state and one picking the right state for each action.
  The row direction is what serving uses; the column direction keeps the
  action space from collapsing.
- **6**: The third stage trains on agent steps, each step's context as the
  state and the action taken as the answer, with two fifths of each batch
  replayed from the first stage so the general representation survives.
- **7**: The checkpoint is the two heads and the scale, a file of tens of
  megabytes that only makes sense with the embedder and pooling it was
  trained against.

## Worked example

The example script runs the released readout on toy geometry: unit vectors in
a plane, the state at angle zero and each option at an angle that stands for
how well its description matches, scaled by the shipped factor of one
hundred.

![Three typed questions answered from toy state and option vectors](tables/clm-answers.csv)

A five-degree difference between two options, a cosine gap of about six
hundredths, becomes a scaled gap of about six and a probability near one for
the closer option. The yes-or-no question returns the probability of true,
and the score question returns an expected level between its two closest
levels.

The second table ports the two-way loss to a batch of three matched pairs at
three temperatures, with every state nearest its own action.

![Two-way InfoNCE loss of a toy batch at three temperatures](tables/clm-infonce.csv)

At a temperature of one the cosines are all close to one and the loss sits
near chance for three candidates. Dividing by a tenth, which is what a learned
scale of ten does, separates the pairs, and a scale of a hundred drives the
loss to zero. The learned scale is what turns small cosine gaps into decisive
probabilities.

## Where it sits

CLM serves the same endpoint as Jev, with the embedder behind a separate
embedding server and the heads in a small CPU process that also holds the
vector cache. The repository ships fine-tuning for a new head on cached
embeddings, including the task-disjoint heads it used as verifiers on coding
benchmarks, and a plain classification fine-tune on the typed-decisions
dataset.

## Limits

### An option cannot be judged against the others

The score of an option depends on the state and that option only. A
question that asks which of two options is better is answered by two
independent similarities, and a question whose right answer is "none of the
above" needs that answer written out as a candidate.

### The embedder is large

The released model embeds with an 8-billion-parameter decoder and needs a GPU
for the embedding server. The heads are small, but the latency and cost
figures the source reports are those of that embedder.

### No calibration step

The distribution is the softmax of scaled cosines with a scale learned under
the contrastive loss, and nothing fits it to observed accuracy afterwards. A
deployer who needs a threshold with a known error rate fits a temperature on
labels.

### Long states are cut

Each text is embedded within the embedder's context limit, 2,048 tokens by
default, and a longer state loses its tail. Raising the limit costs memory on
the embedding server.
