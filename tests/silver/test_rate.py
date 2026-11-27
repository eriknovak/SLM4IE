"""Tests for slm4ie.silver.rate."""

import json

import pytest

from slm4ie.evaluation.backends import ChatBackend, ChatResult
from slm4ie.silver.checks import CheckResult
from slm4ie.silver.make import Item, TokenEstimate
from slm4ie.silver.rate import (
    KeyEntry,
    RatingUnit,
    agreement,
    draw_person_sample,
    rate_outputs,
    rating_pool,
    read_rating_sheet,
    write_pool,
    write_rating_sheet,
)

# ===============================
# Test Data
# ===============================

LABELS = {
    "medical": {
        "Drug": {"description": "A medicine.", "examples": ["paracetamol"]},
        "Condition": {"description": "A disease.", "examples": ["glavobol"]},
    },
    "scientific": {"Method": {"description": "A method.", "examples": ["regresija"]}},
}
JUDGE_USER = (
    "{labels}\n{label_names}\nText:\n{text}\nMentions:\n{mentions}\n"
    "Give two judgements.\n1. naturalness: how natural it reads, from 1 to 5.\n2. realism: how plausible it is.\n"
)
PROMPTS = {"judge": {"system": "Judge.", "user": JUDGE_USER}}
JUDGE_SPEC = {"model": "toy/judge", "usd_per_million_input": 1.0, "usd_per_million_output": 2.0}
ESTIMATE = TokenEstimate(1_000, 500)
# the reply cap: a judge call's worst case is 1,000 x 1 + 1,000 x 2 per million = 0.003
MAX_TOKENS = 1_000
PASSAGE = "Bolnik z glavobolom je prejel paracetamol."
PASSED = CheckResult(True, True, True, True, None, None)
FAILED = CheckResult(False, False, False, False, None, None)

UNIT = RatingUnit("u1", "medical", PASSAGE, (("glavobolom", "Condition"), ("paracetamol", "Drug")))


def record(variant, item_id, job="annotate", domain="medical", text=PASSAGE):
    """Build a stored make record with two parsed mentions."""
    mentions = [{"text": "glavobolom", "label": "Condition"}, {"text": "paracetamol", "label": "Drug"}]
    return {
        "variant": variant,
        "job": job,
        "item_id": item_id,
        "domain": domain,
        "parsed": {"mentions": mentions},
        "example": {"id": f"{variant}/{item_id}", "text": text, "spans": []},
    }


def answer(naturalness=4, numbers=(1, 2)):
    """Build a judge answer for a unit with the given mention numbers."""
    return json.dumps(
        {
            "errors": [{"phrase": "je prejel", "correction": "je dobil"}],
            "naturalness": naturalness,
            "realism": 5,
            "mentions": [{"number": n, "label_correct": True, "span_correct": True} for n in numbers],
            "missed": [],
        }
    )


class ScriptedBackend(ChatBackend):
    """A backend that answers with a fixed reply and counts its calls."""

    def __init__(self, reply, cost=0.01, model="toy/judge", seed=13):
        """Store the reply and the cost each call reports."""
        super().__init__(model, seed)
        self.reply = reply
        self.cost = cost
        self.calls = 0

    def _complete(self, messages, schema):
        self.calls += 1
        return ChatResult(self.reply, 100, 10, 0.0, cost_usd=self.cost)


def rate(tmp_path, backend, units=(UNIT,), limit_usd=10.0, **options):
    """Rate the units with one scripted judge `j`."""
    return rate_outputs(
        units,
        {"j": JUDGE_SPEC},
        LABELS,
        PROMPTS,
        tmp_path,
        limit_usd,
        backend_factory=lambda name, spec: backend,
        seed=13,
        max_tokens=options.pop("max_tokens", MAX_TOKENS),
        token_estimate=ESTIMATE,
        **options,
    )


def stored(tmp_path):
    """Read the stored ratings of judge `j`."""
    return [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines()]


def pool_units(count_per_stratum=3):
    """Build a pool over two jobs, two domains and two families plus real text, with its key."""
    units, key = [], {}
    strata = [(job, domain, family) for job in ("annotate", "write") for domain in LABELS for family in ("a", "b")]
    strata += [("real", domain, "real") for domain in LABELS]
    for job, domain, family in strata:
        for index in range(count_per_stratum):
            unit_id = f"{job}-{domain}-{family}-{index}"
            units.append(RatingUnit(unit_id, domain, PASSAGE, ()))
            key[unit_id] = KeyEntry(f"{family}-model", family, job, f"i{index}")
    return units, key


def filled_rating(naturalness=4):
    """The rating a person gives the unit `UNIT` on a filled sheet."""
    return {
        "errors": [{"phrase": "je prejel", "correction": "je dobil"}],
        "naturalness": naturalness,
        "realism": 3,
        "mentions": [
            {"number": 1, "label_correct": True, "span_correct": False},
            {"number": 2, "label_correct": False, "span_correct": True},
        ],
        "missed": [{"text": "Bolnik", "label": "Condition"}],
    }


def fill_sheet(path, naturalness="4"):
    """Fill in the blank fields of a written sheet holding only `UNIT`, as a person would."""
    sheet = path.read_text(encoding="utf-8")
    unit_part = sheet[sheet.index("## Unit u1") :]
    filled = (
        unit_part.replace("label correct (y/n):", "label correct (y/n): y", 1)
        .replace("span correct (y/n):", "span correct (y/n): n", 1)
        .replace("label correct (y/n):\n", "label correct (y/n): n\n")
        .replace("span correct (y/n):\n", "span correct (y/n): y\n")
        .replace("Naturalness (1-5):", f"Naturalness (1-5): {naturalness}")
        .replace("Realism (1-5):", "Realism (1-5): 3")
    )
    errors_head, _, rest = filled.partition("- \n")
    filled = errors_head + "- je prejel => je dobil\n" + rest
    missed_head, _, rest = filled.rpartition("- \n")
    filled = missed_head + "- Bolnik => Condition\n" + rest
    path.write_text(sheet[: sheet.index("## Unit u1")] + filled, encoding="utf-8")


# ===============================
# rating_pool
# ===============================


def test_rating_pool_hides_maker_and_includes_real_passages(tmp_path):
    """Units show no maker, job or item id; the key holds them, and each real passage is a bare unit."""
    checked = [(record("secret-model", "p1"), PASSED), (record("secret-model", "p2"), FAILED)]
    passages = [Item("annotate", "p1", "medical", text=PASSAGE, words=6)]

    pool = rating_pool(checked, passages, {"secret-model": "acme"})
    write_pool(pool, tmp_path)

    seen = (tmp_path / "units.jsonl").read_text(encoding="utf-8")
    assert "secret-model" not in seen and "annotate" not in seen and "p1" not in seen
    makers = sorted((pool.key[u.unit_id].maker, pool.key[u.unit_id].family, len(u.mentions)) for u in pool.units)
    assert makers == [("real-text", "real", 0), ("secret-model", "acme", 2)]


# ===============================
# rate_outputs
# ===============================


def test_rate_outputs_drops_rating_off_the_scale(tmp_path):
    """A naturalness of 6 is stored raw with a null rating and counted as dropped."""
    summary = rate(tmp_path, ScriptedBackend(answer(naturalness=6)))

    (rating,) = stored(tmp_path)
    assert (rating["rating"], rating["error"], summary.dropped) == (None, None, 1)


def test_rate_outputs_drops_rating_with_mismatched_mention_numbers(tmp_path):
    """A rating that covers mention 1 only, of the unit's two, is dropped."""
    rate(tmp_path, ScriptedBackend(answer(numbers=(1,))))

    (rating,) = stored(tmp_path)
    assert rating["rating"] is None


def test_rate_outputs_rerun_skips_stored_ratings(tmp_path):
    """A second run over the same units makes no call and counts the stored pairs as skipped."""
    rate(tmp_path, ScriptedBackend(answer()))
    backend = ScriptedBackend(answer())

    summary = rate(tmp_path, backend)

    assert (backend.calls, summary.skipped, len(stored(tmp_path))) == (0, 1, 1)


def test_rate_outputs_stops_before_call_that_would_pass_cap(tmp_path):
    """With 0.20 spent and a 0.30 cap, two 0.04 ratings fit and the third would pass the cap."""
    units = [RatingUnit(f"u{index}", "medical", PASSAGE, ()) for index in range(4)]
    backend = ScriptedBackend(answer(numbers=()), cost=0.04)

    summary = rate(tmp_path, backend, units=units, limit_usd=0.30, spent_usd=0.20)

    assert (backend.calls, summary.capped, summary.spent_usd) == (2, True, pytest.approx(0.28))


def test_rate_outputs_guard_prices_a_reply_of_the_full_output_cap(tmp_path):
    """With 0.08 spent and a 0.10 cap, a rating whose full reply could cost 0.033 is never asked for.

    The estimate of 500 output tokens prices the call at 0.002, so a guard from the estimate
    would send it; this call then costs 0.03 and the total would reach 0.11, past the cap.
    """
    backend = ScriptedBackend(answer(), cost=0.03)

    # worst case: 1,000 x 1 + 16,000 x 2 per million = 0.033
    summary = rate(tmp_path, backend, limit_usd=0.10, spent_usd=0.08, max_tokens=16_000)

    assert (backend.calls, summary.capped) == (0, True)


# ===============================
# The person's sample and sheet
# ===============================


def test_draw_person_sample_is_balanced_and_repeatable(tmp_path):
    """Ten strata of three units and a size of 15 give two units per stratum, the same for one seed."""
    units, key = pool_units()

    sample = draw_person_sample(units, key, size=15, seed=7)

    strata = [(key[u.unit_id].job, u.domain, key[u.unit_id].family) for u in sample]
    assert sorted(strata.count(stratum) for stratum in set(strata)) == [2] * 10
    assert sample == draw_person_sample(units, key, size=15, seed=7)


def test_write_rating_sheet_carries_the_judge_prompt_judgements(tmp_path):
    """The sheet lists the numbered judgements in the judge prompt's own words, so the two cannot drift."""
    path = tmp_path / "sheet.md"
    edited = {"judge": {**PROMPTS["judge"], "user": JUDGE_USER.replace("how plausible it is", "how real it seems")}}

    write_rating_sheet([UNIT], LABELS, edited, path)

    sheet = path.read_text(encoding="utf-8")
    assert "1. naturalness: how natural it reads, from 1 to 5." in sheet and "2. realism: how real it seems." in sheet


def test_read_rating_sheet_returns_unfilled_units_as_unrated(tmp_path):
    """A sheet read back as written holds every unit with no rating."""
    path = tmp_path / "sheet.md"
    write_rating_sheet([UNIT, RatingUnit("u2", "scientific", "Uporabili smo regresijo.", ())], LABELS, PROMPTS, path)

    assert read_rating_sheet(path) == {"u1": None, "u2": None}


def test_read_rating_sheet_round_trips_filled_values(tmp_path):
    """Values a person writes into the blank fields come back in the judges' rating shape."""
    path = tmp_path / "sheet.md"
    write_rating_sheet([UNIT], LABELS, PROMPTS, path)
    fill_sheet(path)

    assert read_rating_sheet(path) == {"u1": filled_rating()}


def test_read_rating_sheet_raises_on_off_scale_value(tmp_path):
    """A naturalness of 7 raises an error naming the unit and the value."""
    path = tmp_path / "sheet.md"
    write_rating_sheet([UNIT], LABELS, PROMPTS, path)
    fill_sheet(path, naturalness="7")

    with pytest.raises(ValueError, match=r"u1.*'7'"):
        read_rating_sheet(path)


# ===============================
# agreement
# ===============================


def test_agreement_kappa_matches_hand_computed_value():
    """Naturalness 1,2,3,3 against 1,3,3,2: mean gap 0.5 over a chance gap of 14/16 gives kappa 3/7."""
    ratings_a = {f"u{i}": filled_rating(value) for i, value in enumerate([1, 2, 3, 3])}
    ratings_b = {f"u{i}": filled_rating(value) for i, value in enumerate([1, 3, 3, 2])}

    result = agreement(ratings_a, ratings_b, "naturalness")

    assert (result.n, result.raw, result.within_one, result.kappa) == (4, 0.5, 1.0, pytest.approx(3 / 7))


def test_agreement_kappa_is_none_for_constant_rater():
    """A rater who gives every unit the same naturalness leaves kappa undefined."""
    ratings_a = {f"u{i}": filled_rating(4) for i in range(3)}
    ratings_b = {f"u{i}": filled_rating(value) for i, value in enumerate([3, 4, 5])}

    assert agreement(ratings_a, ratings_b, "naturalness").kappa is None
