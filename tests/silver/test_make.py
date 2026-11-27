"""Tests for slm4ie.silver.make."""

import json

import pytest

from slm4ie.evaluation.backends import ChatBackend, ChatResult
from slm4ie.silver.make import Item, TokenEstimate, make_examples, model_settings, project_spend

# ===============================
# Test Data
# ===============================

LABELS = {
    "medical": {
        "Drug": {"description": "A medicine.", "examples": ["paracetamol"]},
        "Condition": {"description": "A disease.", "examples": ["glavobol"]},
    }
}
PROMPTS = {
    "annotate": {"system": "Annotate.", "user": "{labels}\nLabels: {label_names}\nText:\n{text}"},
    "write": {"system": "Write.", "user": "{labels}\n{text_type}; {topic}; {words}; {required_labels}"},
}
SPEC = {"model": "toy/model", "usd_per_million_input": 1.0, "usd_per_million_output": 2.0}
PASSAGE = "Bolnik z glavobolom je prejel paracetamol."
ANNOTATE_ITEM = Item("annotate", "p1", "medical", text=PASSAGE, words=6)
WRITE_ITEM = Item("write", "b1", "medical", text_type="leaflet", topic="pain", words=5, required_labels=("Drug",))
ESTIMATES = {"annotate": TokenEstimate(1_000, 500), "write": TokenEstimate(2_000, 1_000)}
# the reply cap: an annotate call's worst case is 1,000 x 1 + 1,000 x 2 per million = 0.003
MAX_TOKENS = 1_000

ANNOTATION = json.dumps({"mentions": [{"text": "paracetamol", "label": "Drug"}]})
WRITTEN = json.dumps({"text": "Vzemite paracetamol.", "mentions": [{"text": "paracetamol", "label": "Drug"}]})


class ScriptedBackend(ChatBackend):
    """A backend that answers with a function of the user turn and counts its calls."""

    def __init__(self, reply, cost=0.01, model="toy/model", seed=13):
        """Store the reply function and the cost each call reports."""
        super().__init__(model, seed)
        self.reply = reply
        self.cost = cost
        self.calls = 0

    def _complete(self, messages, schema):
        self.calls += 1
        return ChatResult(self.reply(messages[1]["content"]), 100, 10, 0.0, cost_usd=self.cost)


def by_job(user):
    """Answer an annotate prompt with an annotation and any other prompt with a written passage."""
    return ANNOTATION if "Text:" in user else WRITTEN


def run(tmp_path, backends, items=(ANNOTATE_ITEM, WRITE_ITEM), limit_usd=10.0, **options):
    """Run `make_examples` with one scripted backend per variant."""
    return make_examples(
        {variant: SPEC for variant in backends},
        items,
        LABELS,
        PROMPTS,
        tmp_path,
        limit_usd,
        backend_factory=lambda variant, spec: backends[variant],
        max_tokens=options.pop("max_tokens", MAX_TOKENS),
        token_estimates=ESTIMATES,
        **options,
    )


def records(tmp_path, variant):
    """Read the stored records of one variant."""
    return [json.loads(line) for line in (tmp_path / f"{variant}.jsonl").read_text().splitlines()]


# ===============================
# make_examples
# ===============================


def test_make_examples_stores_one_record_per_call_with_its_example(tmp_path):
    """Each call is one record, holding the reply, its parse and the aligned GLiNER example."""
    run(tmp_path, {"m": ScriptedBackend(by_job)})

    annotated, written = records(tmp_path, "m")
    assert annotated["example"] == {"id": "m/p1", "text": PASSAGE, "spans": [{"start": 30, "end": 41, "label": "Drug"}]}
    assert written["example"]["spans"] == [{"start": 8, "end": 19, "label": "Drug"}]
    assert (annotated["model"], annotated["cost_usd"], annotated["seed"], len(annotated["prompt_hash"])) == (
        "toy/model",
        0.01,
        13,
        16,
    )


def test_make_examples_stores_unparsed_reply_with_parsed_null(tmp_path):
    """A reply off the schema is stored raw, with no parse and no example."""
    run(tmp_path, {"m": ScriptedBackend(lambda user: "no JSON here")}, items=[ANNOTATE_ITEM])

    (record,) = records(tmp_path, "m")
    assert (record["raw"], record["parsed"], record["example"], record["error"]) == ("no JSON here", None, None, None)


def test_make_examples_stores_error_of_raising_backend_and_continues(tmp_path):
    """A call that raises is stored with its message and the next call is still made."""

    def fail_on_annotate(user):
        if "Text:" in user:
            raise RuntimeError("upstream 503")
        return WRITTEN

    summary = run(tmp_path, {"m": ScriptedBackend(fail_on_annotate)})

    failed, written = records(tmp_path, "m")
    assert "upstream 503" in failed["error"] and written["error"] is None
    assert (summary.calls, summary.failed) == (2, 1)


def test_make_examples_rerun_skips_stored_calls(tmp_path):
    """A second run over the same items makes no call and counts them as skipped."""
    run(tmp_path, {"m": ScriptedBackend(by_job)})
    backend = ScriptedBackend(by_job)

    summary = run(tmp_path, {"m": backend})

    assert (backend.calls, summary.skipped, len(records(tmp_path, "m"))) == (0, 2, 2)


def test_make_examples_stops_before_call_that_would_pass_cap(tmp_path):
    """With 0.20 spent and a 0.30 cap, two 0.04 calls fit and the third would pass the cap."""
    items = [Item("annotate", f"p{index}", "medical", text=PASSAGE, words=6) for index in range(4)]
    backend = ScriptedBackend(by_job, cost=0.04)

    # the first call's guard is its worst case, 0.003; later ones the costliest call, 0.04
    summary = run(tmp_path, {"m": backend}, items=items, limit_usd=0.30, spent_usd=0.20)

    assert (backend.calls, summary.capped, summary.spent_usd) == (2, True, pytest.approx(0.28))


def test_make_examples_guard_prices_a_reply_of_the_full_output_cap(tmp_path):
    """With 0.08 spent and a 0.10 cap, a call whose full reply could cost 0.033 is never sent.

    The estimate of 500 output tokens prices the call at 0.002, so a guard from the estimate
    would send it; this call then costs 0.03 and the total would reach 0.11, past the cap.
    """
    backend = ScriptedBackend(by_job, cost=0.03)

    # worst case: 1,000 x 1 + 16,000 x 2 per million = 0.033
    summary = run(tmp_path, {"m": backend}, items=[ANNOTATE_ITEM], limit_usd=0.10, spent_usd=0.08, max_tokens=16_000)

    assert (backend.calls, summary.capped) == (0, True)


def test_project_spend_matches_list_price_arithmetic():
    """Two variants on one item of each job are priced from the token estimates and list prices."""
    cheap = {**SPEC, "usd_per_million_input": 0.5, "usd_per_million_output": 1.0}

    projection = project_spend({"a": SPEC, "b": cheap}, [ANNOTATE_ITEM, WRITE_ITEM], ESTIMATES)

    # a: (1,000 x 1 + 500 x 2) + (2,000 x 1 + 1,000 x 2) = 6,000 per million; b is half of it
    assert projection.per_variant == pytest.approx({"a": 0.006, "b": 0.003})
    assert (projection.calls, projection.total_usd) == (4, pytest.approx(0.009))


SAMPLING = {"seed": 13, "temperature": 0.0, "reasoning_effort": "low", "max_tokens": 16000}


def test_model_settings_keep_fixed_settings_for_a_plain_spec():
    """A spec with no overrides gets the fixed seed, temperature, effort and length."""
    assert model_settings(SAMPLING, {"model": "m"}) == SAMPLING


def test_model_settings_drop_unaccepted_settings_and_take_spec_effort():
    """A model that accepts no seed or temperature gets None for both, and its own effort when named."""
    spec = {"model": "m", "accepts_seed": False, "accepts_temperature": False, "reasoning_effort": "high"}

    assert model_settings(SAMPLING, spec) == {
        "seed": None,
        "temperature": None,
        "reasoning_effort": "high",
        "max_tokens": 16000,
    }
