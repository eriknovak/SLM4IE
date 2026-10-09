---
title: Type-safe encoder models
kind: component
status: from-paper
summary: The decoders behind GLiNER2.5 that take every score of the encoder and return only an assignment the declared schema allows, an exact search with pruning for label rules and a beam search over typed entity and relation candidates, so a returned output is well-typed by construction.
variant_of:
part_of: information-extraction/gliner2-5
couples_to: []
---

## Description

A type-safe encoder model is an extractor whose output is guaranteed to fit
the schema the user declared, not merely likely to. The encoder itself does
not change: GLiNER2.5 scores every label, span and relation independently, as
its predecessors did. What changes is that the scores are not thresholded one
by one. The schema is compiled into a set of constraints, and a decoder
searches for the assignment with the highest total score among those that
satisfy every constraint. If no assignment satisfies them, the decoder says
so instead of returning something invalid.

Two decoders exist, one per kind of output.

- **Constrained classification.** A classification schema declares tasks,
  each single-label, multi-label or ordinal, and rules between them written as
  logic: a label implies another, two labels exclude each other, a task
  selects at least or at most so many labels, an ordinal task sits at or above
  a level. Every rule is a three-valued expression, true, false or not yet
  decided, so a partial assignment can be pruned as soon as a rule is false.
  The decoder enumerates each task's feasible label sets, orders tasks by how
  many rules touch them, and runs a depth-first search with a bound that
  skips any branch that cannot beat the best complete assignment.
- **Joint entity and relation extraction.** A joint schema declares entity
  types, relation types with the entity types allowed as head and tail, and
  rules such as at most one relation per head, no self loops, no cycles, or
  that a relation is symmetric. Entity and relation candidates become nodes
  and edges with scores. A beam search adds edges in score order, each time
  keeping both the state with and the state without the edge, admitting an
  edge only if its endpoints exist, their types are allowed and every rule
  still holds. The best feasible finished state is returned.

"Type-safe" names the guarantee: a relation's head is always an entity of a
declared head type, a label combination the rules forbid is never returned,
and a field declared as a choice holds one of its choices. It does not name
accuracy. A well-typed output can still be wrong.

## Why it matters here

An extraction benchmark here will be scored by exact match of structured
output, and an invalid structure, such as a relation whose head is not an
entity, is a miss that no amount of model quality fixes. The constrained
decoders are pure Python with no dependence on the encoder, so they can sit
on any head this project trains, including a GLiNER head on a Slovenian
backbone, and the gain they give can be measured on its own by decoding the
same scores with and without the rules.

## Facts

- **Tasks**: constrained text classification; joint entity and relation extraction with typed endpoints
- **Prompt tokens**: none of its own; it reads the scores GLiNER2.5 produces
- **Span candidates**: entity spans with probability at least 0.05, at most 32 per type; relation pairs from the 12 best head and tail role spans per count slot, at most 128 pairs per slot and 256 edges per type
- **Longest span**: inherited from GLiNER2.5
- **Scoring**: a candidate's utility is its logit centred on the decision threshold, so a probability of 0.5 scores zero; a solution's score is the sum of its chosen utilities
- **Decoding**: classification by exact depth-first search with a utility bound, 200,000 nodes at most, falling back to a beam of 16; joint extraction by a beam of 32 over edges in score order, checked against the greedy solution; an infeasible schema raises an error or, when asked, widens retention and minimises violations
- **Backbone**: none; pure Python over scores
- **Context length**: inherited; long texts are chunked and the chunk results merged
- **Training data**: none; the rules are declared, not learned
- **Constraints, classification**: label references, and, or, not, implies, iff, excludes, exactly one of, at least, at most, exactly, at level, min level, max level, between levels, any selected, any other selected, is default
- **Constraints, joint**: typed endpoints, no self loops, unique pair, unique slot, at most so many per head or per tail, overlap policy for entities, symmetric, inverse, acyclic

## Sources

- **Paper**: none; the vendor describes the guarantee in its release notes for GLiNER2.5 and GLiNER2.5-Decide, Fastino, 2026
- **Paper**: Type-Safe Is Not Error-Free: Typed Decision Models Follow the Option Name, Not the Definition Bound to It, Sun, Xu, Shi and Yang, 2026, arXiv:2609.26758, which measures what the guarantee does not cover
- **Code**: none in this repository; the reference implementation is the gliner2 package of the fastino-ai/GLiNER2 repository, version 2.1.0 read for this entry: classification/schema.py and classification/constraints.py declare the tasks and rules, classification/compiler.py checks them, classification/candidates.py enumerates a task's label sets, classification/decoding/exact.py and classification/decoding/beam.py search; joint_ie/schema.py and joint_ie/constraints.py declare the typed schema, joint_ie/candidates.py builds nodes and edges, joint_ie/optimizers/beam.py and joint_ie/optimizers/greedy.py search
- **Example**: `experiments/reference/information-extraction/examples.py`, which runs the released solver and beam search on toy scores
- **Paper vs code**: no paper. Against the release note, the code adds a rescue rule: a label a constraint names is kept as a candidate even below the retention floor, so a rule can force a low-scoring label, and an entity span a relation needs is rescued into the node set even when its own score fell below the entity threshold. The note says invalid assignments raise an error; the code also offers a relaxed mode that returns the assignment with the fewest violations.

## Algorithm

```algorithm
\caption{Constrained classification: exact search over feasible label sets}
\Require tasks $t_1, \dots, t_T$ with labels and cardinality bounds, rules $\mathcal{C}$, logits $z_{t,l}$, retention floor $0.5$, cap $64$
\Ensure the feasible assignment with the highest utility, or none
\State compile: every rule names declared tasks and labels; the rules are satisfiable on the declared label sets; a task with a default label gets the rule that the default holds iff no other label does
\ForAll{tasks $t$}
  \State $u_{t,l} \gets z_{t,l} - \mathrm{logit}(\tau_t)$; keep labels with $\sigma(z_{t,l}) \ge 0.5$, at most $64$, plus every label a rule names
  \State $L_t \gets$ the feasible label sets of $t$: one per label for a single-label task, every subset within the bounds when a rule reads the whole selection, else subsets of the rule-named labels completed with the positive free labels; each with utility $\sum u$, sorted descending
\EndFor
\State order tasks by the number of rules touching them, descending; $\mathrm{bound}_i \gets \sum_{k \ge i} \max L_{t_k}$; $u^\star \gets -\infty$
\ForAll{sets $l \in L_{t_i}$ in order, at depth $i$ of the search}
  \If{$u + u_l + \mathrm{bound}_{i+1} \le u^\star$}
    \State stop this task: every later set scores lower
  \EndIf
  \If{every rule touching $t_i$ is true or undecided under the partial assignment}
    \State descend to $t_{i+1}$ with $u + u_l$; at the last task set $u^\star \gets u + u_l$ and record the assignment
  \EndIf
\EndFor
\Return the recorded assignment, or none when no branch reached the last task
```

- **1**: Compilation rejects a schema that can never be satisfied, before any text is scored. A default label becomes an ordinary rule so the search needs no special case for it.
- **2** and **3**: A label's utility is its logit minus the threshold's logit, so the sum of utilities is the log-odds of the whole assignment against the thresholds. Labels below the floor are dropped, except those a rule names: a rule may force a label the model scored low, and dropping it would make the schema infeasible for a reason the model never saw.
- **4**: Rather than search over labels, the search runs over each task's feasible label sets. For a single-label task that is one set per label. For a multi-label task the free labels, which no rule mentions, are always taken when positive, so only the rule-named labels are enumerated.
- **5**: Tasks with many rules go first because their choice prunes the most. The bound at each depth is what the remaining tasks could add at best.
- **6** to **8**: Sets are visited in utility order, so the first set that cannot beat the best complete assignment ends the task's loop: every later set is worse.
- **9** and **10**: A rule is evaluated in three values. False prunes the branch; undecided, because a task the rule reads is not yet assigned, lets the search go on. At the last task the assignment is complete and feasible.
- **11**: None means no feasible assignment exists among the retained labels. The caller then raises an infeasibility error, or in relaxed mode widens retention to every label and, failing that, returns the assignment with the fewest violations.

```algorithm
\caption{Joint entity and relation decoding: beam search over typed candidates}
\Require entity candidates $V$ with type, span and utility; relation candidates $A$ with head, tail, slot and utility; compiled rules $\mathcal{C}$; beam width $32$
\Ensure the feasible set of entities and relations with the highest score
\State compile: each relation gets typed endpoints, no self loops, a unique pair, a unique slot, and its declared per-head and per-tail limits; entities get the overlap policy
\State $V \gets$ spans with probability $\ge 0.05$, at most $32$ per type; $A \gets$ pairs of the $12$ best head and tail role spans per slot, joined to nodes of the allowed types, rescued into $V$ if missing, at most $128$ per slot
\State order $A$ by edge utility plus both endpoint utilities, descending; $B \gets \{\emptyset\}$
\ForAll{edges $a \in A$ in order}
  \ForAll{states $s \in B$}
    \State keep $s$ unchanged, the skip
    \If{$a$'s slot is free in $s$, its new endpoints pass the overlap policy, every rule allows $a$ after $s$'s edges, and the gain $u_a + \sum u_{\text{new nodes}} \ge 0$}
      \State add $s \cup \{a\}$ with score $+$ gain
    \EndIf
  \EndFor
  \State merge states with equal nodes, edges and slots; keep the $32$ best
\EndFor
\State finish every state with each positive node the rules allow; add the greedy solution as one more candidate
\State drop every candidate that fails a full re-check, including symmetric and inverse completeness
\Return the surviving candidate with the highest score, else the empty set marked infeasible
```

- **1**: The schema is lowered to rule objects. Typed endpoints come from the relation's declared head and tail types; the per-slot rule keeps each relation instance to one pair.
- **2**: Candidates are capped so the search is bounded. A span the relation scorer wants as an endpoint but the entity scorer dropped is rescued, because a relation cannot exist without its nodes.
- **3**: Edges are ranked by what taking them is worth, endpoints included, so a strong relation can bring in a weak entity.
- **4** to **8**: Each edge is offered to every state in the beam, and each state spawns a copy without it and, if allowed, a copy with it. The checks are the schema's rules, evaluated against the edges already in the state, so a second relation on a head limited to one is refused.
- **9**: Equal states are merged and the beam cut, which keeps the search linear in the number of edges.
- **10**: Entities need no relation to be returned, so every finished state is topped up with the positive nodes the rules admit. The greedy solution is added as a candidate so the beam never does worse than greedy.
- **11**: Symmetric and inverse rules are whole-result properties and cannot be checked edge by edge; every candidate is re-validated in full.
- **12**: The best feasible candidate wins. When none is feasible the empty set is returned with an infeasibility flag, which is the guarantee: never an invalid structure.

## Worked example

The example script runs the released solver and beam search on toy scores.

The classification schema has a single-label task "safety" with labels safe
and unsafe, a multi-label task "harm" with labels violence and fraud, and one
rule: any selected harm implies unsafe. The logits make the text read as safe
on its own while violence scores high.

![Independent and constrained decoding of the same logits](tables/constrained-classification.csv)

Independent decoding returns safe with violence, which the rule forbids.
The constrained decoder returns unsafe with violence: the rule leaves two
feasible assignments, safe alone or unsafe with violence, and the second has
the higher utility. The utility falls from 3.5 to 2.1, which is the price of
feasibility.

The joint schema has types person and organisation and a relation works_for
from a person to an organisation, limited to one per head and one per tail.
The toy sentence "Ana Novak vodi Krko, Lek pa vodi Maja Kos." gives four
entity candidates, one spurious organisation "Novak" inside the person span,
and four relation candidates, one of them from an organisation to an
organisation.

![Greedy and beam decoding of the same candidates](tables/joint-decoding.csv)

Greedy takes the strongest relation first, Ana Novak to Lek, which then
blocks both Ana Novak to Krko by the per-head limit and Maja Kos to Lek by
the per-tail limit, for a total of 2.0. The beam keeps the state that skipped
that edge and ends with the two weaker relations, for 2.8. Both decoders drop
the organisation-to-organisation edge by its endpoint types and the nested
"Novak" by the overlap policy.

## Where it sits

The decoders are the last stage of GLiNER2.5. The encoder and its heads
produce logits for every label, span and relation candidate; the candidate
builder turns them into utilities and caps them; the decoder picks the
assignment; a result builder maps spans back to character offsets. The same
decoders serve GLiNER2.5-Decide, the vendor's decision model, where each
question is a classification task and the rules connect answers.

## Limits

### Well-typed is not correct

The guarantee covers the structure of the output and nothing about its
content. The 2026 study of typed decision models finds that renaming the
options of a question flips up to 70 points of decisions while the type-error
rate stays at zero: the models follow the option's name, not its definition.

### The rules are the user's

Nothing is learned about which combinations are valid. A rule left out is not
enforced, and a rule written wrong makes the schema infeasible or forces a
wrong label with full confidence.

### Infeasibility is a hard stop

When the rules admit no assignment over the retained candidates, the default
is an error, not a best effort. The relaxed mode returns the assignment with
the fewest violations, which is then not well-typed.

### The search is bounded

Exact search stops at 200,000 nodes and falls back to a beam; the joint beam
holds 32 states. A schema with many interacting rules or many relation slots
may return a feasible but not optimal assignment.
