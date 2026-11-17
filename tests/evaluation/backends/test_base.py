"""Tests for slm4ie.evaluation.backends.base and the backend registry."""

import pytest

from slm4ie.evaluation.backends import (
    ChatBackend,
    ChatResult,
    available_backends,
    fold_system_turn,
    get_backend,
    strip_think,
)

# ===============================
# Test Data
# ===============================


class FakeBackend(ChatBackend):
    """A backend that returns a fixed reply."""

    def _complete(self, messages, schema):
        return ChatResult("<think>hmm</think>\n{}", 7, 3, 0.0)


# ===============================
# strip_think
# ===============================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>a\nb</think>\n\n{}", "{}"),
        ("reasoning only</think>{}", "{}"),
        ('  {"mentions": []} ', '{"mentions": []}'),
        ("<think>never closed", ""),
        ("<think>a</think>x<think>b</think>{}", "{}"),
    ],
)
def test_strip_think_keeps_only_the_answer(raw, expected):
    """Strip think keeps only the answer."""
    assert strip_think(raw) == expected


# ===============================
# ChatBackend and the registry
# ===============================


def test_chat_strips_reasoning_and_reports_counts():
    """Chat strips reasoning and reports counts."""
    result = FakeBackend("m").chat([{"role": "user", "content": "x"}], {})

    assert result.text == "{}"
    assert (result.input_tokens, result.output_tokens) == (7, 3)
    assert result.seconds >= 0
    assert (result.cache_write_tokens, result.cache_read_tokens, result.cost_usd) == (0, 0, None)


def test_peak_gpu_memory_is_none_by_default():
    """Peak gpu memory is none by default."""
    assert FakeBackend("m").peak_gpu_memory_bytes() is None


def test_registry_lists_the_three_backends():
    """Registry lists the three backends."""
    assert available_backends() == ["claude", "transformers", "vllm"]


def test_get_backend_builds_by_name_with_options():
    """Get backend builds by name with options."""
    backend = get_backend("vllm", "gemma4-12b", url="http://host:1/")

    assert backend.model == "gemma4-12b"
    assert backend.url == "http://host:1"


def test_get_backend_rejects_an_unknown_name():
    """Get backend rejects an unknown name."""
    with pytest.raises(KeyError, match="nope"):
        get_backend("nope", "m")


# ===============================
# fold_system_turn
# ===============================

SYSTEM_AND_USER = [{"role": "system", "content": "Answer in JSON."}, {"role": "user", "content": "Text: Ana."}]


class RecordingBackend(ChatBackend):
    """A backend that keeps the turns it was given."""

    def _complete(self, messages, schema):
        self.seen = messages
        return ChatResult("{}", 1, 1, 0.0)


def test_fold_system_turn_opens_the_user_turn_with_the_system_text():
    """The system text and a blank line open the single user turn, as Gemma 3's template folds it."""
    assert fold_system_turn(SYSTEM_AND_USER) == [{"role": "user", "content": "Answer in JSON.\n\nText: Ana."}]


def test_fold_system_turn_leaves_a_chat_without_system_turn_unchanged():
    """A chat that opens with a user turn passes through."""
    assert fold_system_turn(SYSTEM_AND_USER[1:]) == SYSTEM_AND_USER[1:]


def test_chat_folds_the_system_turn_only_when_asked():
    """A backend built with fold_system sends one user turn; by default it sends both."""
    folded, plain = RecordingBackend("m", fold_system=True), RecordingBackend("m")
    folded.chat(SYSTEM_AND_USER, {})
    plain.chat(SYSTEM_AND_USER, {})

    assert [m["role"] for m in folded.seen] == ["user"]
    assert plain.seen == SYSTEM_AND_USER
