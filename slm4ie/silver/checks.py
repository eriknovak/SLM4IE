"""Rule checks on made examples that need no judge.

Each stored record from `slm4ie.silver.make` is checked by rules a script can
decide: the reply parsed, every mention matches the text, every label is in the
domain's set, and per job, the labelled text is the passage itself or the
written text covers its brief. A failed or unparsed call fails every check, so
it counts as wrong in every share.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Any

from slm4ie.silver.make import ANNOTATE, WRITE, Item

# ===============================
# Defaults
# ===============================

# a written text may be this share shorter or longer than its brief's target
LENGTH_TOLERANCE = 0.3

# ===============================
# Results
# ===============================


@dataclass(frozen=True)
class CheckResult:
    """The rule checks of one output; a check that does not apply to the job is `None`.

    Attributes:
        parsed: The call succeeded and its reply parsed against the schema.
        aligned: Every mention matched the text, so none is left unaligned.
        labels_known: Every mention's label is in the domain's label set.
        text_matches: `annotate` only: the example text equals the passage text exactly.
        labels_covered: `write` only: every label the brief requires has at least one span.
        length_ok: `write` only: the written text's word count is within `LENGTH_TOLERANCE` of the target.
    """

    parsed: bool
    aligned: bool
    labels_known: bool
    text_matches: bool | None
    labels_covered: bool | None
    length_ok: bool | None

    @property
    def passes(self) -> bool:
        """Return True when every check that applies passes."""
        return all(value is not False for value in self.checks().values())

    def checks(self) -> dict[str, bool | None]:
        """Return the checks by name, `None` for those that do not apply."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


# ===============================
# Checks
# ===============================


def check_output(record: Mapping[str, Any], labels: Iterable[str], item: Item) -> CheckResult:
    """Check one stored output by the rules that need no judge.

    A record with an error or no parse fails every check that applies to its job.

    Args:
        record: One stored record of `make_examples`.
        labels: The domain's label names.
        item: The item the record answers: its passage text for `annotate`, its
            required labels and target words for `write`.

    Returns:
        One boolean per check, `None` for the checks of the other job.
    """
    is_annotate = item.job == ANNOTATE
    is_write = item.job == WRITE

    # a failed or unparsed call fails every check
    if record["error"] is not None or record["parsed"] is None or record["example"] is None:
        return CheckResult(
            parsed=False,
            aligned=False,
            labels_known=False,
            text_matches=False if is_annotate else None,
            labels_covered=False if is_write else None,
            length_ok=False if is_write else None,
        )

    # check the mentions against the text and the label set
    allowed = set(labels)
    example = record["example"]
    labels_known = all(mention["label"] in allowed for mention in record["parsed"]["mentions"])
    aligned = not record["unaligned"]

    # check what the job asks of the text
    text_matches = labels_covered = length_ok = None
    if is_annotate:
        text_matches = example["text"] == item.text
    if is_write:
        spanned = {span["label"] for span in example["spans"]}
        labels_covered = all(label in spanned for label in item.required_labels)
        length_ok = abs(len(example["text"].split()) - item.words) <= LENGTH_TOLERANCE * item.words

    return CheckResult(True, aligned, labels_known, text_matches, labels_covered, length_ok)


def check_shares(results: Iterable[CheckResult]) -> dict[str, float | None]:
    """Compute the share of outputs passing each check, for a table.

    A check's share counts only the outputs it applies to.

    Args:
        results: The check results of a group of outputs, such as one model's.

    Returns:
        Check name to its passing share, `None` when it applies to no output, plus `passes`, the share passing all.
    """
    results = list(results)
    shares: dict[str, float | None] = {}
    for field in fields(CheckResult):
        applied = [getattr(result, field.name) for result in results if getattr(result, field.name) is not None]
        shares[field.name] = sum(applied) / len(applied) if applied else None
    shares["passes"] = sum(result.passes for result in results) / len(results) if results else None
    return shares
