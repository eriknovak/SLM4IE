"""Tests for slm4ie.evaluation.prompts."""

import json
from pathlib import Path

import pytest

from slm4ie.evaluation.prompts import (
    Window,
    answer_schema,
    build_messages,
    load_label_descriptions,
    parse_answer,
    sentence_windows,
)
from slm4ie.evaluation.scoring import align_mentions
from slm4ie.utils.config import load_yaml

# ===============================
# Test Data
# ===============================

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "configs" / "evaluation" / "ner-labels.yaml"
TASKS_PATH = REPO_ROOT / "configs" / "data" / "tasks.yaml"

DESCRIPTIONS = {"PER": "A person's name.", "LOC": "A place name."}
LABELS = list(DESCRIPTIONS)


def answer(*mentions: tuple[str, str]) -> str:
    """Serialise mentions as a model answer in the schema's shape."""
    return json.dumps({"mentions": [{"text": text, "label": label} for text, label in mentions]})


def write_labels(tmp_path: Path, content: str) -> Path:
    """Write a label description file into the test's temporary directory."""
    path = tmp_path / "ner-labels.yaml"
    path.write_text(content, encoding="utf-8")
    return path


# ===============================
# sentence_windows
# ===============================


def test_short_text_is_one_window_at_offset_zero():
    """A text under the cap is read in one window."""
    text = "Janez živi v Kranju. Rad ima Ljubljano."
    assert sentence_windows(text, max_chars=2_000) == [Window(start=0, text=text)]


def test_windows_cut_only_between_sentences():
    """Every window is a run of whole sentences."""
    text = "Prvi stavek je tu. Drugi stavek je tu. Tretji stavek je tu."
    windows = sentence_windows(text, max_chars=40)
    assert [w.text for w in windows] == ["Prvi stavek je tu. Drugi stavek je tu.", "Tretji stavek je tu."]


def test_window_text_is_the_document_slice_at_its_start():
    """Adding a window's start to an offset in it gives the document offset."""
    text = "Ena.\n\nDva je tu.\nTri je tu in še malo."
    for window in sentence_windows(text, max_chars=12):
        assert text[window.start : window.start + len(window.text)] == window.text


def test_newline_ends_a_sentence():
    """A line break is a sentence boundary, as in the CoNLL-U task texts."""
    text = "brez pike na koncu\nin nova vrstica"
    assert [w.text for w in sentence_windows(text, max_chars=20)] == ["brez pike na koncu", "in nova vrstica"]


def test_abbreviation_and_initial_do_not_end_a_sentence():
    """A title, an initial or an ordinal before a capital stays inside the sentence."""
    text = "Govoril je dr. Novak. Prišel je J. Kos. Tekel je na 3. Ljubljanskem maratonu."
    windows = sentence_windows(text, max_chars=10)
    assert [w.text for w in windows] == [
        "Govoril je dr. Novak.",
        "Prišel je J. Kos.",
        "Tekel je na 3. Ljubljanskem maratonu.",
    ]


def test_sentence_longer_than_the_cap_is_its_own_window():
    """A sentence is never cut, so one over the cap is read whole."""
    long_sentence = "Dolg " + "dolg " * 9 + "konec."
    windows = sentence_windows(f"Kratek. {long_sentence} Spet kratek.", max_chars=20)
    assert [w.text for w in windows] == ["Kratek.", long_sentence, "Spet kratek."]


def test_blank_text_has_no_window():
    """A text of only whitespace gives the model nothing to read."""
    assert sentence_windows(" \n ", max_chars=2_000) == []


def test_mentions_aligned_in_a_window_shift_to_document_offsets():
    """A span aligned in a window maps back by the window's start."""
    text = "Janez je doma. Kranj je mesto, kjer živi Janez."
    window = sentence_windows(text, max_chars=20)[1]
    spans = align_mentions(window.text, [{"text": "Janez", "label": "PER"}]).spans
    assert text[window.start + spans[0]["start"] : window.start + spans[0]["end"]] == "Janez"
    assert window.start + spans[0]["start"] == text.rindex("Janez")


# ===============================
# build_messages
# ===============================


def test_messages_are_a_system_and_a_user_turn():
    """The prompt is one system turn and one user turn."""
    messages = build_messages("Janez živi v Kranju.", DESCRIPTIONS)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_user_turn_holds_the_window_text_and_every_label_description():
    """The model sees the text and each label with its description."""
    user = build_messages("Janez živi v Kranju.", DESCRIPTIONS)[1]["content"]
    assert "Janez živi v Kranju." in user
    assert "PER: A person's name." in user
    assert "LOC: A place name." in user


def test_user_turn_asks_for_copied_text_every_occurrence_in_reading_order():
    """The instructions match what align_mentions expects of an answer."""
    user = build_messages("Janez.", DESCRIPTIONS)[1]["content"].lower()
    assert "exactly as" in user
    assert "every occurrence" in user
    assert "order" in user


# ===============================
# answer_schema
# ===============================


def test_schema_restricts_labels_to_the_given_set():
    """A label outside the test set's own labels does not validate."""
    item = answer_schema(LABELS)["properties"]["mentions"]["items"]
    assert item["properties"]["label"]["enum"] == LABELS
    assert item["required"] == ["text", "label"]


def test_schema_root_is_an_object_with_a_required_mentions_list():
    """Structured-output backends need an object at the root."""
    schema = answer_schema(LABELS)
    assert schema["type"] == "object"
    assert schema["required"] == ["mentions"]
    assert schema["properties"]["mentions"]["type"] == "array"


# ===============================
# parse_answer
# ===============================


def test_valid_answer_returns_its_mentions_in_order():
    """A schema-shaped answer yields its mentions as returned."""
    mentions = parse_answer(answer(("Janez", "PER"), ("Kranju", "LOC")), LABELS)
    assert mentions == [{"text": "Janez", "label": "PER"}, {"text": "Kranju", "label": "LOC"}]


def test_empty_mentions_list_is_a_valid_answer():
    """A window without entities parses to no mentions, not a miss."""
    assert parse_answer(answer(), LABELS) == []


def test_answer_in_a_markdown_code_fence_parses():
    """One surrounding code fence is stripped before parsing."""
    fenced = "```json\n" + answer(("Janez", "PER")) + "\n```"
    assert parse_answer(fenced, LABELS) == [{"text": "Janez", "label": "PER"}]


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        '{"mentions": [{"text": "Janez", "label": "PER"}',
        '[{"text": "Janez", "label": "PER"}]',
        '{"entities": []}',
        '{"mentions": [{"text": "Janez"}]}',
        '{"mentions": [{"text": "", "label": "PER"}]}',
        '{"mentions": [{"text": 3, "label": "PER"}]}',
        '{"mentions": ["Janez"]}',
        '{"mentions": [{"text": "Janez", "label": ["PER"]}]}',
        '{"mentions": [{"text": "Janez", "label": {"name": "PER"}}]}',
    ],
    ids=[
        "prose",
        "truncated",
        "bare-list",
        "wrong-key",
        "no-label",
        "empty-text",
        "number-text",
        "string-item",
        "list-label",
        "object-label",
    ],
)
def test_answer_off_the_schema_is_a_miss(raw):
    """An answer the schema rejects returns None, which scores as no mentions."""
    assert parse_answer(raw, LABELS) is None


def test_answer_with_a_label_outside_the_set_is_a_miss():
    """An unknown label fails the schema rather than entering macro F1."""
    assert parse_answer(answer(("Janez", "PER"), ("EU", "ORG")), LABELS) is None


# ===============================
# load_label_descriptions
# ===============================


TOY_LABELS = """\
label_sets:
  toy:
    entries: [ner/toy]
    labels:
      PER:
        description: A person.
        examples: {unseen: [Ana], common: [Janez, Novak]}
      LOC:
        description: A place.
        examples: {unseen: [Kamnik], common: [Ljubljana]}
"""


def test_label_descriptions_load_for_an_entry_in_label_order(tmp_path):
    """An entry gets its label set's descriptions in file order, with the unseen examples by default."""
    path = write_labels(tmp_path, TOY_LABELS)
    assert load_label_descriptions(path, "ner/toy") == {
        "PER": "A person. Examples: Ana.",
        "LOC": "A place. Examples: Kamnik.",
    }


def test_label_descriptions_take_the_named_example_variant(tmp_path):
    """The common variant swaps in its own example names."""
    path = write_labels(tmp_path, TOY_LABELS)
    assert load_label_descriptions(path, "ner/toy", examples="common")["PER"] == "A person. Examples: Janez, Novak."


def test_unknown_example_variant_raises_key_error_naming_it(tmp_path):
    """A variant no label carries stops the run before any model call."""
    path = write_labels(tmp_path, TOY_LABELS)
    with pytest.raises(KeyError, match="rare"):
        load_label_descriptions(path, "ner/toy", examples="rare")


def test_unknown_entry_raises_key_error_naming_it(tmp_path):
    """An entry no label set lists stops the run before any model call."""
    path = write_labels(tmp_path, TOY_LABELS)
    with pytest.raises(KeyError, match="ner/other"):
        load_label_descriptions(path, "ner/other")


def test_entry_listed_in_two_label_sets_raises_value_error(tmp_path):
    """An entry must have exactly one set of descriptions."""
    path = write_labels(tmp_path, TOY_LABELS + TOY_LABELS.replace("label_sets:\n", "").replace("toy:", "twin:", 1))
    with pytest.raises(ValueError, match="ner/toy"):
        load_label_descriptions(path, "ner/toy")


@pytest.mark.parametrize("examples", ["unseen", "common"])
@pytest.mark.parametrize("entry", ["ner/ssj500k", "ner/suk", "ner/janes_tag", "ner/slavicner"])
def test_committed_descriptions_cover_the_task_registry_labels(entry, examples):
    """Each benchmark entry's descriptions name exactly its registry labels, in order, under both variants."""
    labels = load_yaml(TASKS_PATH)["entries"][entry]["labels"]
    descriptions = load_label_descriptions(LABELS_PATH, entry, examples=examples)
    assert list(descriptions) == labels
    assert all("Examples: " in text for text in descriptions.values())
