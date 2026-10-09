"""Generate the worked-example tables of the information-extraction topic.

Two kinds of table. The prompt layouts and the candidate counts follow the
rules the released GLiNER and GLiNER2 code applies, ported line for line,
because both packages import PyTorch at module level and the project does not
carry it. The two decoding tables run GLiNER2's own constraint solver and joint
beam search: those modules are pure Python, so the script loads them from the
installed package without executing its torch-bound package roots.

    uv run --with gliner2 python experiments/reference/information-extraction/examples.py
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Tuple

TABLES = Path(__file__).resolve().parent / "tables"

#: Toy input: two people, two companies, one employment relation each.
TOKENS: List[str] = ["Ana", "Novak", "vodi", "Krko", ",", "Lek", "pa", "vodi", "Maja", "Kos", "."]
ENTITY_TYPES: List[str] = ["person", "organisation"]
#: GLiNER: the `ent_token`, `sep_token` and `max_width` of `urchade/gliner_multi-v2.1`.
GLINER_ENT = "<<ENT>>"
GLINER_SEP = "<<SEP>>"
GLINER_MAX_WIDTH = 12
#: GLiNER2 span model: the `max_width` of `fastino/gliner2-base-v1`.
GLINER2_MAX_WIDTH = 8
#: GLiNER2.5 boundary head of `fastino/gliner2.5-multi-v1`: `start_top_k`, `boundary_top_k_alpha`,
#: `boundary_top_k_max`, `boundary_top_k_bucket`, `ends_per_start` and `candidate_budget`.
BOUNDARY_TOP_K = 24
BOUNDARY_ALPHA = 0.08
BOUNDARY_K_MAX = 128
BOUNDARY_BUCKET = 8
PAIRS_PER_BOUNDARY = 12
CANDIDATE_BUDGET = 192
#: Text lengths, in words, the candidate counts are computed for.
LENGTHS: List[int] = [8, 32, 128, 512, 2048, 4096]
#: Toy classifier logits: the text reads as safe on its own, yet a harm type scores high.
CLASSIFICATION_LOGITS: Dict[str, Dict[str, float]] = {
    "safety": {"safe": 1.5, "unsafe": 0.1},
    "harm": {"violence": 2.0, "fraud": -2.0},
}
#: Toy joint candidates: (type, start, end, score) for entities, half-open token spans.
NODES: List[Tuple[str, int, int, float]] = [
    ("person", 0, 2, 0.2),
    ("organisation", 1, 2, 0.1),
    ("organisation", 3, 4, 0.2),
    ("organisation", 5, 6, 0.2),
    ("person", 8, 10, 0.2),
]
#: Toy relation candidates: (head node, tail node, score), by index into NODES.
EDGES: List[Tuple[int, int, float]] = [(0, 3, 1.2), (0, 2, 1.0), (4, 3, 1.0), (2, 3, 0.6)]


#: GLiNER2 packages whose `__init__` imports torch; each is replaced by an empty package with the same path.
TORCH_BOUND_PACKAGES: List[str] = [
    "gliner2",
    "gliner2.classification",
    "gliner2.classification.decoding",
    "gliner2.joint_ie",
    "gliner2.joint_ie.optimizers",
]


def torch_free(name: str) -> ModuleType:
    """Import one GLiNER2 module without running the package roots, which import torch.

    Args:
        name: Dotted module name inside the installed `gliner2` package.

    Returns:
        ModuleType: The imported module.

    Raises:
        ModuleNotFoundError: If `gliner2` is not installed.
    """
    if "gliner2" not in sys.modules:
        root = importlib.util.find_spec("gliner2")
        if root is None or not root.submodule_search_locations:
            raise ModuleNotFoundError("gliner2 is not installed; run with `uv run --with gliner2`")
        for package in TORCH_BOUND_PACKAGES:
            stub = ModuleType(package)
            stub.__path__ = [str(Path(p).joinpath(*package.split(".")[1:])) for p in root.submodule_search_locations]
            sys.modules[package] = stub
    return importlib.import_module(name)


def write_csv(name: str, header: List[str], rows: List[List[object]]) -> None:
    """Write one table under `tables/`.

    Args:
        name: File stem.
        header: Column names.
        rows: Table rows.
    """
    TABLES.mkdir(exist_ok=True)
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def span_grid_count(length: int, max_width: int) -> int:
    """Spans of at most `max_width` words in a text of `length` words, as `prepare_span_idx` enumerates them.

    Args:
        length: Words in the text.
        max_width: Longest span, in words.

    Returns:
        int: Spans with a start in the text and an end before its last word.
    """
    return sum(min(max_width, length - start) for start in range(length))


def boundary_budget(n_boundaries: int) -> int:
    """Starts, or ends, GLiNER2.5 keeps per query, as `resolve_boundary_budget` computes it.

    Args:
        n_boundaries: Token boundaries in the text, one more than its words.

    Returns:
        int: The budget, grown with the length and rounded up to a bucket.
    """
    requested = int(math.ceil(BOUNDARY_ALPHA * max(n_boundaries - 1, 0)))
    requested = max(BOUNDARY_TOP_K, min(requested, BOUNDARY_K_MAX))
    return min(BOUNDARY_K_MAX, int(math.ceil(requested / BOUNDARY_BUCKET) * BOUNDARY_BUCKET))


def prompt_layouts() -> None:
    """Write the input sequence each generation builds for the toy text."""
    gliner = [token for name in ENTITY_TYPES for token in (GLINER_ENT, name)] + [GLINER_SEP] + TOKENS
    text = [token.lower() for token in TOKENS]
    entities = ["(", "[P]", "entities", "("] + [t for n in ENTITY_TYPES for t in ("[E]", n)] + [")", ")"]
    labels = ["(", "[P]", "sentiment", "(", "[L]", "positive", "[L]", "negative", ")", ")"]
    fields = ["(", "[P]", "employment", "(", "[C]", "employee", "[C]", "employer", ")", ")"]
    relation = ["(", "[P]", "works_for", "(", "[R]", "head", "[R]", "tail", ")", ")"]
    composed = entities + ["[SEP_STRUCT]"] + labels + ["[SEP_STRUCT]"] + relation + ["[SEP_TEXT]"] + text
    write_csv(
        "prompt-layouts",
        ["Model", "Task", "Input sequence"],
        [
            ["GLiNER", "entities", " ".join(gliner)],
            ["GLiNER2", "entities", " ".join(entities + ["[SEP_TEXT]"] + text)],
            ["GLiNER2", "classification", " ".join(labels + ["[SEP_TEXT]"] + text)],
            ["GLiNER2", "structure", " ".join(fields + ["[SEP_TEXT]"] + text)],
            ["GLiNER2.5", "relation", " ".join(relation + ["[SEP_TEXT]"] + text)],
            ["GLiNER2.5", "entities + classification + relation", " ".join(composed)],
        ],
    )


def candidate_counts() -> None:
    """Write how many spans each generation scores per entity type, by text length."""
    rows = []
    for length in LENGTHS:
        budget = boundary_budget(length + 1)
        proposed = 2 * budget * PAIRS_PER_BOUNDARY
        rows.append(
            [
                length,
                span_grid_count(length, GLINER_MAX_WIDTH),
                span_grid_count(length, GLINER2_MAX_WIDTH),
                budget,
                proposed,
                min(proposed, CANDIDATE_BUDGET),
            ]
        )
    write_csv(
        "candidate-counts",
        [
            "Words",
            "GLiNER spans (width ≤ 12)",
            "GLiNER2 spans (width ≤ 8)",
            "GLiNER2.5 starts kept",
            "GLiNER2.5 pairs proposed",
            "GLiNER2.5 candidates scored",
        ],
        rows,
    )


def constrained_classification() -> None:
    """Decode the toy classifier logits with and without the declared constraint."""
    schema_module = torch_free("gliner2.classification.schema")
    constraints = torch_free("gliner2.classification.constraints")
    compiler = torch_free("gliner2.classification.compiler")
    base = torch_free("gliner2.classification.decoding.base")
    exact = torch_free("gliner2.classification.decoding.exact")
    independent = torch_free("gliner2.classification.decoding.independent")

    schema = schema_module.ClassificationSchema()
    schema.single("safety", ["safe", "unsafe"])
    schema.multi("harm", ["violence", "fraud"])
    schema.constrain(constraints.implies(constraints.any_selected("harm"), constraints.label("safety", "unsafe")))
    compiled = compiler.compile_schema(schema)
    scores = SimpleNamespace(tasks=CLASSIFICATION_LOGITS)
    config = SimpleNamespace(candidate_threshold=0.5, max_candidates_per_task=64)
    problem = base.build_problem(compiled, scores, config)
    free = independent.IndependentDecoder().decode(problem)
    bound = exact.ExactDecoder().decode(problem)

    def mark(solution: object, task: str, label: str) -> str:
        return "selected" if label in solution.assignments[task].labels else ""

    rows = []
    for task, logits in CLASSIFICATION_LOGITS.items():
        for label, logit in logits.items():
            probability = 1 / (1 + math.exp(-logit))
            rows.append([task, label, f"{probability:.2f}", mark(free, task, label), mark(bound, task, label)])
    rows.append(["utility", "", "", f"{free.score:.1f}", f"{bound.score:.1f}"])
    rows.append(["violations", "", "", len(free.violations), len(bound.violations)])
    write_csv("constrained-classification", ["Task", "Label", "Probability", "Independent", "Constrained"], rows)


def joint_decoding() -> None:
    """Decode the toy entity and relation candidates greedily and with the beam."""
    schema_module = torch_free("gliner2.joint_ie.schema")
    compiler = torch_free("gliner2.joint_ie.compiler")
    candidates = torch_free("gliner2.joint_ie.candidates")
    greedy = torch_free("gliner2.joint_ie.optimizers.greedy")
    beam = torch_free("gliner2.joint_ie.optimizers.beam")

    schema = schema_module.JointSchema()
    for name in ENTITY_TYPES:
        schema.entity(name)
    schema.relation("works_for", head="person", tail="organisation", max_per_head=1, max_per_tail=1)
    compiled = compiler.compile_schema(schema)
    nodes = [candidates.NodeCandidate(kind, start, end, score) for kind, start, end, score in NODES]
    # One count slot per candidate, as the candidate builder assigns from the relation lattice.
    edges = [
        candidates.EdgeCandidate(
            "works_for", nodes[head].candidate_id, nodes[tail].candidate_id, score, slot=slot, hypothesis="works_for"
        )
        for slot, (head, tail, score) in enumerate(EDGES)
    ]
    problem = candidates.JointProblem(tuple(nodes), tuple(edges), compiled.constraints)
    greedy_solution = greedy.GreedyOptimizer().optimize(problem)
    beam_solution = beam.BeamOptimizer(beam_width=32).optimize(problem)

    def text(node: object) -> str:
        return " ".join(TOKENS[node.start : node.end])

    def kept(solution: object, item: object) -> str:
        if hasattr(item, "entity_type"):
            return "kept" if item.candidate_id in solution.node_ids else ""
        return "kept" if any(edge.candidate_id == item.candidate_id for edge in solution.edges) else ""

    by_id = problem.node_by_id
    rows = [[text(n), n.entity_type, f"{n.score:.1f}", kept(greedy_solution, n), kept(beam_solution, n)] for n in nodes]
    for edge in edges:
        label = f"{text(by_id[edge.head])} → {text(by_id[edge.tail])}"
        rows.append(
            [label, edge.relation_type, f"{edge.score:.1f}", kept(greedy_solution, edge), kept(beam_solution, edge)]
        )
    rows.append(["total", "", "", f"{greedy_solution.score:.1f}", f"{beam_solution.score:.1f}"])
    write_csv("joint-decoding", ["Candidate", "Type", "Score", "Greedy", "Beam"], rows)


def main() -> None:
    """Write every table of the topic."""
    prompt_layouts()
    candidate_counts()
    constrained_classification()
    joint_decoding()


if __name__ == "__main__":
    main()
