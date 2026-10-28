"""Generate the worked-example tables of the typed-decision-models topic.

Four tables. The confidence table and the CLM answers run the released CLM
schema module, which is pure Python and holds the readout the server uses.
The Von temperature table runs the released Von calibration module with the
map shipped in the checkpoint's `marker_calibration.json`. The Kev packing
table and the CLM loss table are ported rules: Kev's `encode` needs a Qwen
tokenizer and the CLM loss needs PyTorch, which the project does not carry.

    uv run --with contrastive-lm --with von-sdk python experiments/reference/typed-decision-models/examples.py
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

# ===============================
# Defaults
# ===============================

TABLES = Path(__file__).resolve().parent / "tables"

#: Top probabilities and option counts the confidence formula is shown for.
TOP_PROBABILITIES: list[float] = [0.5, 0.8, 0.95]
OPTION_COUNTS: list[int] = [2, 3, 5, 10]

#: Von 1.2 calibration map, from `marker_calibration.json` of the `wfzyx/von` checkpoint.
VON_MAP: dict[str, float] = {
    "bias": 2.0151,
    "entropy": -3.2369,
    "log_tokens": 11.4916,
    "n_options": -3.5574,
    "lo": 0.3,
    "hi": 12.0,
}
#: Toy requests: raw option logits, state length in tokens. The same logits at three lengths.
VON_LOGITS: list[float] = [2.0, 0.5, -0.5]
VON_STATE_TOKENS: list[int] = [40, 400, 4000]
#: Von feature scaling: `log10(tokens)` divided by this, and the option count divided by that.
VON_LOG_TOKENS_SCALE = 4.0
VON_OPTIONS_SCALE = 8.0

#: Toy CLM geometry: unit vectors in the plane, given by angle in degrees, and the shipped scale 100.
CLM_SCALE = 100.0
CLM_STATE_ANGLE = 0.0
CLM_QUESTIONS: dict[str, tuple[dict, dict[str, float]]] = {
    "department": (
        {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "", "technical": ""},
        },
        {"billing": 5.0, "technical": 20.0},
    ),
    "urgent": ({"type": "noul", "instructions": "Is this urgent?"}, {"false": 14.0, "true": 10.0}),
    "frustration": (
        {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Angry"],
        },
        {"0": 25.0, "1": 9.0, "2": 7.0},
    ),
}
#: Toy InfoNCE batch: three states and their matched actions, by angle in degrees.
CLM_BATCH_STATES: list[float] = [0.0, 40.0, 90.0]
CLM_BATCH_ACTIONS: list[float] = [5.0, 30.0, 100.0]
CLM_TEMPERATURES: list[float] = [1.0, 0.1, 0.01]

#: Toy Kev record: words stand for tokens.
KEV_STATE: list[str] = ["Disk", "at", "98%"]
KEV_QUESTIONS: list[tuple[list[str], list[list[str]]]] = [
    (["Who", "owns", "this?"], [["sre"], ["app"]]),
    (["Act", "now?"], [["false"], ["true"]]),
]

# ===============================
# Helper functions
# ===============================


def write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
    """Write one table under `tables/`.

    Args:
        name: File stem.
        header: Column names.
        rows: Table rows.
    """
    # write the header and the rows
    TABLES.mkdir(exist_ok=True)
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def confidence_by_options() -> None:
    """Write the adapter's confidence for one top probability spread over more and more options."""
    from clm.schema import confidence

    # build one row per top probability
    rows = []
    for top in TOP_PROBABILITIES:
        row: list[object] = [f"{top:.2f}"]
        for count in OPTION_COUNTS:
            rest = (1 - top) / (count - 1)
            row.append(f"{confidence([top] + [rest] * (count - 1)):.2f}")
        rows.append(row)

    # write the table
    write_csv("confidence-by-options", ["Top probability"] + [f"{k} options" for k in OPTION_COUNTS], rows)


def unit(angle: float) -> tuple[float, float]:
    """Unit vector at an angle.

    Args:
        angle: The angle, in degrees.

    Returns:
        The cosine and sine of the angle.
    """
    return math.cos(math.radians(angle)), math.sin(math.radians(angle))


def cosine(a: float, b: float) -> float:
    """Cosine between the unit vectors at two angles.

    Args:
        a: The first angle, in degrees.
        b: The second angle, in degrees.

    Returns:
        The cosine of the angle between them.
    """
    return math.cos(math.radians(a - b))


def clm_answers() -> None:
    """Answer three typed questions from toy state and option vectors with CLM's own readout."""
    from clm.schema import answer_from_logits, probabilities_of

    # answer each question and list its options
    rows = []
    for qid, (question, angles) in CLM_QUESTIONS.items():
        keys = list(angles)
        logits = [CLM_SCALE * cosine(CLM_STATE_ANGLE, angles[k]) for k in keys]
        answer = answer_from_logits(question, keys, logits)
        probabilities = probabilities_of(answer)
        value = answer.get("choice", answer.get("score", answer.get("noul")))
        for key, logit in zip(keys, logits):
            rows.append(
                [
                    qid,
                    question["type"],
                    key,
                    f"{logit:.1f}",
                    f"{probabilities[key]:.3f}",
                    f"{value:.3f}" if isinstance(value, float) else value,
                    f"{answer['confidence']:.3f}" if "confidence" in answer else "",
                ]
            )

    # write the table
    write_csv(
        "clm-answers", ["Question", "Type", "Option", "Scaled cosine", "Probability", "Answer", "Confidence"], rows
    )


def log_softmax(values: Sequence[float]) -> list[float]:
    """Log of the softmax of a vector.

    Args:
        values: The logits.

    Returns:
        The log probabilities, one per logit.
    """
    top = max(values)
    total = top + math.log(sum(math.exp(v - top) for v in values))
    return [v - total for v in values]


def positive(value: float) -> float:
    """Clip a loss at zero, so a rounding error below zero prints as zero and not as a negative zero.

    Args:
        value: The loss.

    Returns:
        The loss, or zero when it is not above zero.
    """
    return value if value > 0 else 0.0


def clm_infonce() -> None:
    """Write the bidirectional InfoNCE loss of the toy batch at three temperatures, as `_clm_loss` computes it."""
    # score the batch at each temperature
    n = len(CLM_BATCH_STATES)
    rows = []
    for temperature in CLM_TEMPERATURES:
        logits = [[cosine(s, a) / temperature for a in CLM_BATCH_ACTIONS] for s in CLM_BATCH_STATES]
        forward = [-log_softmax(logits[i])[i] for i in range(n)]
        backward = [-log_softmax([logits[j][i] for j in range(n)])[i] for i in range(n)]
        loss = (sum(forward) + sum(backward)) / (2 * n)
        rows.append([f"{temperature:g}"] + [f"{positive(v):.3f}" for v in forward] + [f"{positive(loss):.3f}"])

    # write the table
    write_csv(
        "clm-infonce",
        ["Temperature"] + [f"State {i + 1} → action loss" for i in range(n)] + ["Batch loss"],
        rows,
    )


def von_temperature() -> None:
    """Write Von's request-dependent temperature, and the confidence it yields, for one set of logits at three state lengths."""
    from von.calibrate import _temp

    # normalised entropy of the raw option distribution
    n_options = len(VON_LOGITS)
    raw = [math.exp(v) for v in log_softmax(VON_LOGITS)]
    entropy = -sum(p * math.log(p) for p in raw) / math.log(n_options)
    # calibrate the temperature at each state length
    rows = []
    for tokens in VON_STATE_TOKENS:
        features = {
            "bias": 1.0,
            "entropy": entropy,
            "log_tokens": math.log10(tokens) / VON_LOG_TOKENS_SCALE,
            "n_options": n_options / VON_OPTIONS_SCALE,
        }
        temperature = _temp(VON_MAP, features)
        probabilities = [math.exp(v) for v in log_softmax([z / temperature for z in VON_LOGITS])]
        confidence = (n_options * max(probabilities) - 1) / (n_options - 1)
        rows.append([tokens, f"{entropy:.2f}", f"{temperature:.2f}", f"{max(probabilities):.3f}", f"{confidence:.3f}"])

    # write the table
    write_csv(
        "von-temperature",
        ["State tokens", "Normalised entropy", "Temperature", "Top probability", "Confidence"],
        rows,
    )


def kev_packing() -> None:
    """Write the packed sequence Kev builds for the toy record, with segment, position and what `<decide>` sees."""
    # start with the state segment
    tokens: list[str] = ["<state>"] + KEV_STATE
    segment = [0] * len(tokens)
    position = list(range(len(tokens)))
    state_length = len(tokens)
    decide_at: list[int] = []

    # append one branch per question
    for k, (instructions, options) in enumerate(KEV_QUESTIONS, start=1):
        branch = ["<q>"] + instructions
        for option in options:
            branch += ["<opt>"] + option + ["</opt>"]
        branch.append("<decide>")
        tokens += branch
        segment += [k] * len(branch)
        position += list(range(state_length, state_length + len(branch)))
        decide_at.append(len(tokens) - 1)

    # mark what the last decide token sees and scores
    last_decide = decide_at[-1]
    rows = []
    for index, (token, seg, pos) in enumerate(zip(tokens, segment, position)):
        sees = index <= last_decide and (seg == 0 or seg == segment[last_decide])
        scored = token == "</opt>" and seg == segment[last_decide]
        rows.append([index, token, seg, pos, "yes" if sees else "", "yes" if scored else ""])

    # write the table
    write_csv(
        "kev-packing",
        ["Index", "Token", "Segment", "Position", "Seen by the last <decide>", "Scored for the last <decide>"],
        rows,
    )


# ===============================
# Main function
# ===============================


def main() -> None:
    """Write every table of the topic."""
    confidence_by_options()
    clm_answers()
    clm_infonce()
    von_temperature()
    kev_packing()


if __name__ == "__main__":
    main()
