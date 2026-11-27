"""Tests for slm4ie.silver.checks."""

import pytest

from slm4ie.silver.checks import CheckResult, check_output, check_shares
from slm4ie.silver.make import Item

# ===============================
# Test Data
# ===============================

LABELS = ["Drug", "Condition"]
PASSAGE = "Bolnik z glavobolom je prejel paracetamol."
ANNOTATE_ITEM = Item("annotate", "p1", "medical", text=PASSAGE, words=6)
WRITE_ITEM = Item("write", "b1", "medical", words=4, required_labels=("Drug", "Condition"))
WRITTEN = "Glavobol zdravimo s paracetamolom."


def record(text=PASSAGE, mentions=(("paracetamol", "Drug"),), spans=None, unaligned=(), error=None):
    """Build a stored record whose mentions are all aligned unless `unaligned` is given."""
    mentions = [{"text": surface, "label": label} for surface, label in mentions]
    if spans is None:
        spans = [{"start": text.find(m["text"]), "end": text.find(m["text"]) + len(m["text"]), "label": m["label"]}
                 for m in mentions]  # fmt: skip
    return {
        "error": error,
        "parsed": {"mentions": mentions},
        "example": {"id": "m/x", "text": text, "spans": spans},
        "unaligned": list(unaligned),
    }


# ===============================
# check_output
# ===============================


def test_check_output_passes_clean_annotation():
    """An aligned annotation of the passage with known labels passes every check."""
    result = check_output(record(), LABELS, ANNOTATE_ITEM)

    assert result.passes and (result.labels_covered, result.length_ok) == (None, None)


def test_check_output_fails_every_check_of_failed_call():
    """A call that raised fails every check of its job."""
    failed = {"error": "RuntimeError: 503", "parsed": None, "example": None, "unaligned": None}

    result = check_output(failed, LABELS, WRITE_ITEM)

    assert result == CheckResult(False, False, False, None, False, False)


def test_check_output_flags_unparsed_reply():
    """A reply with no parse fails the parse check."""
    unparsed = {"error": None, "parsed": None, "example": None, "unaligned": None}

    assert not check_output(unparsed, LABELS, ANNOTATE_ITEM).parsed


def test_check_output_flags_mention_that_misses_text():
    """A mention left unaligned fails the alignment check."""
    missed = record(unaligned=[{"text": "aspirin", "label": "Drug"}])

    assert not check_output(missed, LABELS, ANNOTATE_ITEM).aligned


def test_check_output_flags_unknown_label():
    """A mention whose label is outside the domain's set fails the label check."""
    unknown = record(mentions=[("paracetamol", "Medicine")])

    assert not check_output(unknown, LABELS, ANNOTATE_ITEM).labels_known


def test_check_output_flags_annotated_text_that_differs_from_passage():
    """An example whose text is not the passage exactly fails the text check."""
    changed = record(text=PASSAGE.lower())

    assert not check_output(changed, LABELS, ANNOTATE_ITEM).text_matches


def test_check_output_flags_written_text_missing_required_label():
    """A written text with no span of a required label fails the coverage check."""
    uncovered = record(text=WRITTEN, mentions=[("paracetamolom", "Drug")])

    assert not check_output(uncovered, LABELS, WRITE_ITEM).labels_covered


@pytest.mark.parametrize(
    ("target", "expected"),
    [(4, True), (5, True), (6, False), (3, False)],
    ids=["exact", "within-30-percent", "too-short", "too-long"],
)
def test_check_output_length_within_30_percent_of_target(target, expected):
    """A four-word text fits a target of 4 or 5 words, but not of 6 or 3."""
    written = record(text=WRITTEN, mentions=[("Glavobol", "Condition"), ("paracetamolom", "Drug")])
    item = Item("write", "b1", "medical", words=target, required_labels=("Drug",))

    assert check_output(written, LABELS, item).length_ok is expected


# ===============================
# check_shares
# ===============================


def test_check_shares_counts_only_outputs_a_check_applies_to():
    """A write-only check is shared over write outputs; `passes` over every output."""
    failed = {"error": "boom", "parsed": None, "example": None, "unaligned": None}
    results = [check_output(record(), LABELS, ANNOTATE_ITEM), check_output(failed, LABELS, WRITE_ITEM)]

    shares = check_shares(results)

    assert (shares["parsed"], shares["text_matches"], shares["length_ok"], shares["passes"]) == (0.5, 1.0, 0.0, 0.5)
