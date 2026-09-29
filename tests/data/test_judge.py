"""Tests for the command-line document judge."""

import json
import os
import stat
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from slm4ie.data.judge import (
    DOCUMENTS_PLACEHOLDER,
    PAIRWISE_STAGES,
    PAIRWISE_TASK,
    VERDICT_SCHEMA,
    draw_calibration_set,
    draw_pairs,
    judge_documents,
    judge_pairs,
    judged_ids,
    parse_verdicts,
    read_documents,
    render_prompt,
    submit_batch,
    validate_batch,
)

_RUBRIC = f"Label every document.\n\n{DOCUMENTS_PLACEHOLDER}\n"


def _verdict(document_id: str, **overrides: Any) -> Dict[str, Any]:
    """Build a valid verdict for a document.

    Args:
        document_id: The document the verdict belongs to.
        overrides: Fields to replace in the default verdict.

    Returns:
        A verdict dict the validator accepts unless an override breaks it.
    """
    verdict = {
        "id": document_id,
        "language": "sl",
        "text_type": "prose",
        "adult_or_spam": False,
        "coherence": 4,
        "machine_translated": False,
        "pii": False,
        "domain": "news",
        "note": "",
    }
    verdict.update(overrides)
    return verdict


def _sample(path: Path, count: int) -> List[Dict[str, Any]]:
    """Write a sample file of documents to judge.

    Args:
        path: File to write.
        count: How many documents to write.

    Returns:
        The documents written.
    """
    documents = [{"id": f"demo:{index:03d}", "text": f"besedilo {index}"} for index in range(count)]
    path.write_text("\n".join(json.dumps(doc) for doc in documents) + "\n", encoding="utf-8")
    return documents


def _stub_judge(path: Path, body: str) -> str:
    """Write an executable stub standing in for the judge command.

    Args:
        path: File to write the stub to.
        body: Python source the stub runs; it receives the prompt on stdin.

    Returns:
        The stub's path, for use as the `command` argument.
    """
    path.write_text(f"#!/usr/bin/env python3\nimport json, sys\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


#: A stub that answers every batch with a valid verdict per document.
_ANSWER_EVERY_DOCUMENT = """
prompt = sys.stdin.read()
documents = json.loads(prompt[prompt.index("["):prompt.rindex("]") + 1])
print(json.dumps([{
    "id": d["id"], "language": "sl", "text_type": "prose", "adult_or_spam": False,
    "coherence": 4, "machine_translated": False, "pii": False, "domain": "news", "note": "",
} for d in documents]))
"""


class TestReadingAndResuming:
    """The judge reads its input and never re-asks about a document it has judged."""

    def test_documents_are_read_in_order(self, tmp_path: Path) -> None:
        """Every line of the sample becomes a document to judge."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 3)

        assert [doc["id"] for doc in read_documents(source)] == ["demo:000", "demo:001", "demo:002"]

    def test_a_document_without_text_is_refused(self, tmp_path: Path) -> None:
        """A row with no text cannot be judged, so the run stops rather than guessing."""
        source = tmp_path / "sample.jsonl"
        source.write_text(json.dumps({"id": "demo:000", "text": ""}) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="has no id or no text"):
            read_documents(source)

    def test_a_missing_verdict_file_means_nothing_is_done(self, tmp_path: Path) -> None:
        """A first run has no verdicts to skip."""
        assert judged_ids(tmp_path / "verdicts.jsonl") == set()

    def test_existing_verdicts_are_skipped(self, tmp_path: Path) -> None:
        """A rerun judges only what the earlier run did not finish."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 4)
        destination = tmp_path / "verdicts.jsonl"
        destination.write_text(json.dumps(_verdict("demo:000")) + "\n", encoding="utf-8")
        command = _stub_judge(tmp_path / "judge.py", _ANSWER_EVERY_DOCUMENT)

        counts = judge_documents(
            source, destination, _rubric(tmp_path), backend="cli", batch_size=3, concurrency=1, command=command
        )

        assert counts == {"documents": 4, "skipped": 1, "batches": 1, "judged": 3, "failed": 0}
        assert judged_ids(destination) == {"demo:000", "demo:001", "demo:002", "demo:003"}

    def test_a_finished_run_does_nothing(self, tmp_path: Path) -> None:
        """Rerunning a completed judge run sends no batches at all."""
        source = tmp_path / "sample.jsonl"
        documents = _sample(source, 2)
        destination = tmp_path / "verdicts.jsonl"
        destination.write_text("".join(json.dumps(_verdict(d["id"])) + "\n" for d in documents), encoding="utf-8")
        command = _stub_judge(tmp_path / "judge.py", "raise SystemExit('the judge should not be called')")

        counts = judge_documents(source, destination, _rubric(tmp_path), backend="cli", concurrency=1, command=command)

        assert counts["batches"] == 0
        assert counts["judged"] == 0


def _rubric(tmp_path: Path) -> Path:
    """Write the rubric prompt used by the judging tests.

    Args:
        tmp_path: Test-local directory.

    Returns:
        Path to the rubric file.
    """
    path = tmp_path / "rubric.md"
    path.write_text(_RUBRIC, encoding="utf-8")
    return path


class TestCalibrationSet:
    """The hand-labelled set spans the strata instead of pooling in one corner."""

    def _sample_rows(self, per_cell: int) -> List[Dict[str, Any]]:
        """Build sample rows across two sources, two stages and both decisions.

        Args:
            per_cell: Documents in each cell.

        Returns:
            Rows shaped like the sampler's output.
        """
        rows = []
        for dataset in ("alpha", "beta"):
            for stage in ("quality", "spam"):
                for decision in ("kept", "dropped"):
                    for index in range(per_cell):
                        rows.append(
                            {
                                "id": f"{dataset}:{stage}:{decision}:{index:03d}",
                                "dataset": dataset,
                                "cells": [{"stage": stage, "decision": decision, "shard": "00000.jsonl.gz"}],
                                "text": "besedilo",
                            }
                        )
        return rows

    def test_every_cell_is_represented(self) -> None:
        """Eight cells and sixteen documents means two from each cell."""
        chosen = draw_calibration_set(self._sample_rows(40), 16)

        cells = Counter((row["dataset"], row["cells"][0]["stage"], row["cells"][0]["decision"]) for row in chosen)
        assert len(chosen) == 16
        assert set(cells.values()) == {2}

    def test_a_smaller_sample_is_taken_whole(self) -> None:
        """Asking for more than exists returns everything, not an error."""
        rows = self._sample_rows(1)

        assert len(draw_calibration_set(rows, 500)) == len(rows)

    def test_the_same_seed_chooses_the_same_documents(self) -> None:
        """The calibration set is reproducible, and a new seed changes it."""
        rows = self._sample_rows(40)

        first = [row["id"] for row in draw_calibration_set(rows, 16, seed=1)]
        again = [row["id"] for row in draw_calibration_set(rows, 16, seed=1)]
        other = [row["id"] for row in draw_calibration_set(rows, 16, seed=2)]

        assert first == again
        assert first != other

    def test_a_document_in_several_cells_is_counted_once(self) -> None:
        """A document kept by two stages does not take two places in the set."""
        rows = [
            {
                "id": "alpha:000",
                "dataset": "alpha",
                "cells": [
                    {"stage": "quality", "decision": "kept", "shard": "00000.jsonl.gz"},
                    {"stage": "spam", "decision": "kept", "shard": "00000.jsonl.gz"},
                ],
                "text": "besedilo",
            }
        ]

        chosen = draw_calibration_set(rows, 10)

        assert [row["id"] for row in chosen] == ["alpha:000"]


class TestPrompt:
    """Each call carries the rubric and the batch's documents, nothing else."""

    def test_documents_replace_the_placeholder(self, tmp_path: Path) -> None:
        """The batch is substituted into the rubric as JSON."""
        prompt = render_prompt(_RUBRIC, [{"id": "demo:000", "text": "besedilo"}], 2000)

        assert DOCUMENTS_PLACEHOLDER not in prompt
        assert '"id": "demo:000"' in prompt
        assert "Label every document." in prompt

    def test_text_is_cut_to_the_limit(self, tmp_path: Path) -> None:
        """Only the first `max_chars` characters of a document are sent."""
        prompt = render_prompt(_RUBRIC, [{"id": "demo:000", "text": "a" * 100}], 10)

        assert '"text": "aaaaaaaaaa"' in prompt

    def test_the_stage_and_decision_are_never_sent(self, tmp_path: Path) -> None:
        """The judge is blind: it sees the id and text only."""
        document = {"id": "demo:000", "text": "besedilo", "cells": [{"stage": "quality", "decision": "dropped"}]}

        prompt = render_prompt(_RUBRIC, [document], 2000)

        assert "quality" not in prompt
        assert "dropped" not in prompt

    def test_a_rubric_without_a_placeholder_is_refused(self) -> None:
        """A rubric that cannot carry documents is a mistake, not a silent no-op."""
        with pytest.raises(ValueError, match="placeholder"):
            render_prompt("no placeholder here", [{"id": "demo:000", "text": "x"}], 2000)


class TestParsing:
    """The judge's reply is prose until the array in it has been found and checked."""

    def test_a_bare_array_is_read(self) -> None:
        """A reply that is only the array parses."""
        assert parse_verdicts(json.dumps([_verdict("demo:000")]))[0]["id"] == "demo:000"

    def test_a_fenced_array_is_read(self) -> None:
        """A reply wrapped in a code fence parses."""
        reply = f"Here you go:\n```json\n{json.dumps([_verdict('demo:000')])}\n```\n"

        assert parse_verdicts(reply)[0]["domain"] == "news"

    def test_a_reply_without_an_array_is_refused(self) -> None:
        """A reply carrying no array cannot be salvaged."""
        with pytest.raises(ValueError, match="no JSON array"):
            parse_verdicts("I could not judge these documents.")


class TestValidation:
    """A batch is written whole or not at all."""

    def test_valid_verdicts_are_narrowed_to_the_rubric(self) -> None:
        """Extra fields the judge invents are dropped."""
        batch = [{"id": "demo:000", "text": "x"}]
        verdict = _verdict("demo:000", confidence=0.9)

        checked = validate_batch([verdict], batch)

        assert "confidence" not in checked[0]
        assert set(checked[0]) == {
            "id",
            "language",
            "text_type",
            "adult_or_spam",
            "coherence",
            "machine_translated",
            "pii",
            "domain",
            "note",
        }

    def test_an_id_that_was_never_sent_is_refused(self) -> None:
        """A verdict for a document the judge invented is not written."""
        with pytest.raises(ValueError, match="not sent"):
            validate_batch([_verdict("demo:999")], [{"id": "demo:000", "text": "x"}])

    def test_a_repeated_id_is_refused(self) -> None:
        """Judging the same document twice in one reply is a fault, not a partial."""
        batch = [{"id": "demo:000", "text": "x"}, {"id": "demo:001", "text": "y"}]

        with pytest.raises(ValueError, match="same id twice"):
            validate_batch([_verdict("demo:000"), _verdict("demo:000")], batch)

    def test_a_short_reply_keeps_what_came_back(self) -> None:
        """A judge that skips a document costs that document, not the whole group."""
        batch = [{"id": "demo:000", "text": "x"}, {"id": "demo:001", "text": "y"}]

        checked = validate_batch([_verdict("demo:000")], batch)

        assert [row["id"] for row in checked] == ["demo:000"]

    @pytest.mark.parametrize(
        "field,value",
        [
            ("text_type", "essay"),
            ("domain", "sport"),
            ("coherence", 0),
            ("coherence", 6),
            ("coherence", "high"),
            ("adult_or_spam", "no"),
            ("language", "Slovenian"),
        ],
    )
    def test_values_outside_the_rubric_are_refused(self, field: str, value: Any) -> None:
        """Every label must be one of the values the rubric offers."""
        with pytest.raises(ValueError, match=field):
            validate_batch([_verdict("demo:000", **{field: value})], [{"id": "demo:000", "text": "x"}])

    def test_a_missing_field_is_refused(self) -> None:
        """A verdict must carry every dimension."""
        verdict = _verdict("demo:000")
        del verdict["pii"]

        with pytest.raises(ValueError, match="no pii"):
            validate_batch([verdict], [{"id": "demo:000", "text": "x"}])


class TestJudgingRun:
    """Batching, bounding and failure handling across a whole run."""

    def test_documents_are_judged_in_batches(self, tmp_path: Path) -> None:
        """The sample is split into calls of the requested size."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 5)
        destination = tmp_path / "verdicts.jsonl"
        command = _stub_judge(tmp_path / "judge.py", _ANSWER_EVERY_DOCUMENT)

        counts = judge_documents(
            source, destination, _rubric(tmp_path), backend="cli", batch_size=2, concurrency=1, command=command
        )

        assert counts["batches"] == 3
        assert counts["judged"] == 5

    def test_max_batches_bounds_the_run(self, tmp_path: Path) -> None:
        """A run stops after the requested number of calls, leaving the rest for later."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 10)
        destination = tmp_path / "verdicts.jsonl"
        command = _stub_judge(tmp_path / "judge.py", _ANSWER_EVERY_DOCUMENT)

        counts = judge_documents(
            source,
            destination,
            _rubric(tmp_path),
            backend="cli",
            batch_size=2,
            concurrency=1,
            max_batches=2,
            command=command,
        )

        assert counts["batches"] == 2
        assert counts["judged"] == 4

    def test_a_failing_batch_leaves_the_others_written(self, tmp_path: Path) -> None:
        """One bad call costs its own batch, not the run."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 4)
        destination = tmp_path / "verdicts.jsonl"
        command = _stub_judge(
            tmp_path / "judge.py",
            "prompt = sys.stdin.read()\n"
            'documents = json.loads(prompt[prompt.index("["):prompt.rindex("]") + 1])\n'
            'if documents[0]["id"] == "demo:000":\n'
            '    sys.exit("the judge broke")\n'
            "print(json.dumps([{\n"
            '    "id": d["id"], "language": "sl", "text_type": "prose", "adult_or_spam": False,\n'
            '    "coherence": 4, "machine_translated": False, "pii": False, "domain": "news", "note": "",\n'
            "} for d in documents]))",
        )

        counts = judge_documents(
            source, destination, _rubric(tmp_path), backend="cli", batch_size=2, concurrency=1, command=command
        )

        assert counts["judged"] == 2
        assert counts["failed"] == 2
        assert judged_ids(destination) == {"demo:002", "demo:003"}

    def test_an_unparsable_reply_fails_its_batch(self, tmp_path: Path) -> None:
        """A reply with no verdicts in it is dropped, not written."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 2)
        destination = tmp_path / "verdicts.jsonl"
        command = _stub_judge(tmp_path / "judge.py", 'sys.stdin.read()\nprint("I am not going to do that.")')

        counts = judge_documents(
            source, destination, _rubric(tmp_path), backend="cli", batch_size=2, concurrency=1, command=command
        )

        assert counts == {"documents": 2, "skipped": 0, "batches": 1, "judged": 0, "failed": 2}
        assert not destination.exists() or destination.read_text(encoding="utf-8") == ""

    def test_the_model_reaches_the_command(self, tmp_path: Path) -> None:
        """The chosen model is passed to the executable."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 1)
        destination = tmp_path / "verdicts.jsonl"
        recorded = tmp_path / "argv.txt"
        command = _stub_judge(
            tmp_path / "judge.py",
            f'open({str(recorded)!r}, "w").write(" ".join(sys.argv[1:]))\n' + _ANSWER_EVERY_DOCUMENT,
        )

        judge_documents(
            source, destination, _rubric(tmp_path), backend="cli", model="claude-opus-5", concurrency=1, command=command
        )

        assert recorded.read_text(encoding="utf-8") == "-p --model claude-opus-5"

    def test_batches_run_concurrently(self, tmp_path: Path) -> None:
        """Several calls are in flight at once and every verdict still lands."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 12)
        destination = tmp_path / "verdicts.jsonl"
        command = _stub_judge(tmp_path / "judge.py", _ANSWER_EVERY_DOCUMENT)

        counts = judge_documents(
            source, destination, _rubric(tmp_path), backend="cli", batch_size=2, concurrency=4, command=command
        )

        assert counts["judged"] == 12
        assert len(judged_ids(destination)) == 12

    def test_a_missing_command_fails_the_batch(self, tmp_path: Path) -> None:
        """A judge that cannot be run is reported, not swallowed as success."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 1)

        with pytest.raises((FileNotFoundError, subprocess.SubprocessError, OSError)):
            judge_documents(
                source,
                tmp_path / "verdicts.jsonl",
                _rubric(tmp_path),
                backend="cli",
                concurrency=1,
                command=str(tmp_path / "does-not-exist"),
            )


class _FakeBatches:
    """Stands in for `client.messages.batches` so no key or spend is involved."""

    def __init__(self, replies: Dict[str, str], statuses: Optional[List[str]] = None) -> None:
        """Record what the fake batch should return.

        Args:
            replies: Reply text keyed by `custom_id`; a missing key errors.
            statuses: Statuses to report before `ended`, for polling tests.
        """
        self.replies = replies
        self.statuses = list(statuses or [])
        self.submitted: List[Any] = []

    def create(self, requests: List[Any]) -> Any:
        """Accept a batch and hand back its id.

        Args:
            requests: The batch's requests.

        Returns:
            An object carrying the batch id.
        """
        self.submitted = requests
        return SimpleNamespace(id="batch_test", processing_status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        """Report the batch's status, ending once `statuses` is spent.

        Args:
            batch_id: The batch to check.

        Returns:
            An object carrying the processing status.
        """
        status = self.statuses.pop(0) if self.statuses else "ended"
        return SimpleNamespace(
            status_id=batch_id, processing_status=status, request_counts=SimpleNamespace(processing=1)
        )

    def results(self, batch_id: str) -> Any:
        """Yield one result per submitted request, in reverse order.

        Args:
            batch_id: The batch to read.

        Yields:
            Result objects shaped like the SDK's.
        """
        for request in reversed(self.submitted):
            custom_id = request["custom_id"]
            reply = self.replies.get(custom_id)
            if reply is None:
                yield SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))
                continue
            message = SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)])
            yield SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


def _fake_client(replies: Dict[str, str], statuses: Optional[List[str]] = None) -> Any:
    """Build a client whose `messages.batches` is the fake above.

    Args:
        replies: Reply text keyed by `custom_id`.
        statuses: Statuses to report before `ended`.

    Returns:
        An object with a `messages.batches` attribute.
    """
    return SimpleNamespace(messages=SimpleNamespace(batches=_FakeBatches(replies, statuses)))


def _reply(*document_ids: str) -> str:
    """Render a structured-output reply for the given documents.

    Args:
        document_ids: Ids to return verdicts for.

    Returns:
        JSON matching `VERDICT_SCHEMA`.
    """
    return json.dumps({"verdicts": [_verdict(document_id) for document_id in document_ids]})


class TestBatchBackend:
    """The API backend submits one request per group and keys results by id."""

    def test_requests_carry_the_schema_and_no_thinking(self, tmp_path: Path) -> None:
        """Every request asks for schema-valid JSON and spends nothing on thinking."""
        client = _fake_client({})
        groups = [[{"id": "demo:000", "text": "besedilo"}]]

        submit_batch(client, "claude-sonnet-5", groups, _RUBRIC, 2000)

        params = client.messages.batches.submitted[0]["params"]
        assert params["output_config"]["format"]["schema"] == VERDICT_SCHEMA
        assert params["thinking"] == {"type": "disabled"}
        assert params["model"] == "claude-sonnet-5"

    def test_haiku_is_sent_no_thinking_key(self, tmp_path: Path) -> None:
        """Haiku does not think unless asked, so it must not be sent `disabled`."""
        client = _fake_client({})

        submit_batch(client, "claude-haiku-4-5", [[{"id": "demo:000", "text": "x"}]], _RUBRIC, 2000)

        assert "thinking" not in client.messages.batches.submitted[0]["params"]

    def test_results_are_matched_by_custom_id_not_order(self, tmp_path: Path) -> None:
        """The fake returns results reversed; every group must still find its own."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 4)
        destination = tmp_path / "verdicts.jsonl"
        client = _fake_client(
            {"group-00000": _reply("demo:000", "demo:001"), "group-00001": _reply("demo:002", "demo:003")}
        )

        counts = judge_documents(
            source,
            destination,
            _rubric(tmp_path),
            backend="api",
            batch_size=2,
            client=client,
        )

        assert counts["judged"] == 4
        assert judged_ids(destination) == {"demo:000", "demo:001", "demo:002", "demo:003"}

    def test_an_errored_request_is_left_for_the_next_run(self, tmp_path: Path) -> None:
        """A failed request costs its own group, and its documents stay unjudged."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 4)
        destination = tmp_path / "verdicts.jsonl"
        client = _fake_client({"group-00001": _reply("demo:002", "demo:003")})

        counts = judge_documents(source, destination, _rubric(tmp_path), backend="api", batch_size=2, client=client)

        assert counts == {"documents": 4, "skipped": 0, "batches": 2, "judged": 2, "failed": 2}
        assert judged_ids(destination) == {"demo:002", "demo:003"}

    def test_polling_waits_for_the_batch_to_end(self, tmp_path: Path) -> None:
        """A batch still processing is polled until it reports `ended`."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 1)
        client = _fake_client({"group-00000": _reply("demo:000")}, statuses=["in_progress", "in_progress"])

        counts = judge_documents(
            source,
            tmp_path / "verdicts.jsonl",
            _rubric(tmp_path),
            backend="api",
            client=client,
            poll_seconds=0,
        )

        assert counts["judged"] == 1
        assert client.messages.batches.statuses == []

    def test_an_unknown_backend_is_refused(self, tmp_path: Path) -> None:
        """A typo in the backend name stops the run rather than picking one."""
        source = tmp_path / "sample.jsonl"
        _sample(source, 1)

        with pytest.raises(ValueError, match="unknown backend"):
            judge_documents(source, tmp_path / "out.jsonl", _rubric(tmp_path), backend="batch")


def _pair_sample(path: Path, datasets: Sequence[str] = ("alpha",), per_cell: int = 3) -> List[Dict[str, Any]]:
    """Write a sample the pairwise drawing can be run over.

    Args:
        path: File to write.
        datasets: Sources to write documents for.
        per_cell: Documents per source, stage and decision.

    Returns:
        The documents written.
    """
    documents = []
    for dataset in datasets:
        for stage in PAIRWISE_STAGES:
            for decision in ("kept", "dropped"):
                for index in range(per_cell):
                    documents.append(
                        {
                            "id": f"{dataset}:{stage}:{decision}:{index}",
                            "dataset": dataset,
                            "cells": [{"stage": stage, "decision": decision}],
                            "text": f"besedilo {dataset} {stage} {decision} {index}",
                        }
                    )
    path.write_text("\n".join(json.dumps(doc) for doc in documents) + "\n", encoding="utf-8")
    return documents


#: A stub that answers every comparison by naming the first text.
_ANSWER_EVERY_PAIR = """
prompt = sys.stdin.read()
pairs = json.loads(prompt[prompt.index("["):prompt.rindex("]") + 1])
print(json.dumps([{"id": p["id"], "better": "a", "note": ""} for p in pairs]))
"""


class TestDrawingPairs:
    """Each source and stage contributes the same comparisons in both orders."""

    def test_every_pair_is_asked_in_both_orders(self, tmp_path: Path) -> None:
        """A pair is asked twice, once with the kept document in each slot."""
        documents = _pair_sample(tmp_path / "sample.jsonl", per_cell=2)

        pairs = draw_pairs(documents, pairs_per_cell=2)

        assert len(pairs) == len(PAIRWISE_STAGES) * 2 * 2
        first, second = pairs[0], pairs[1]
        assert (first["kept"], second["kept"]) == ("a", "b")
        assert (first["a_id"], first["b_id"]) == (second["b_id"], second["a_id"])

    def test_a_pair_holds_one_kept_and_one_dropped_document(self, tmp_path: Path) -> None:
        """The two documents differ only in what the stage decided about them."""
        documents = _pair_sample(tmp_path / "sample.jsonl")

        for pair in draw_pairs(documents):
            kept_slot, dropped_slot = (
                (pair["a_id"], pair["b_id"]) if pair["kept"] == "a" else (pair["b_id"], pair["a_id"])
            )
            assert f":{pair['stage']}:kept:" in kept_slot
            assert f":{pair['stage']}:dropped:" in dropped_slot

    def test_a_stage_with_no_drops_is_skipped(self, tmp_path: Path) -> None:
        """Nothing is compared where the stage dropped nothing in that source."""
        documents = [doc for doc in _pair_sample(tmp_path / "sample.jsonl") if ":spam:dropped:" not in doc["id"]]

        assert not [pair for pair in draw_pairs(documents) if pair["stage"] == "spam"]

    def test_the_cap_is_bounded_by_what_the_sample_holds(self, tmp_path: Path) -> None:
        """Asking for more pairs than there are documents draws what exists."""
        documents = _pair_sample(tmp_path / "sample.jsonl", per_cell=2)

        pairs = draw_pairs(documents, stages=["quality"], pairs_per_cell=20)

        assert len({pair["pair"] for pair in pairs}) == 2

    def test_the_same_seed_draws_the_same_pairs(self, tmp_path: Path) -> None:
        """The drawing is reproducible, so a rerun asks about the same pairs."""
        documents = _pair_sample(tmp_path / "sample.jsonl", datasets=("alpha", "beta"))

        assert draw_pairs(documents, seed=7) == draw_pairs(documents, seed=7)
        assert draw_pairs(documents, seed=7) != draw_pairs(documents, seed=8)


class TestPairwiseJudging:
    """A comparison sends two texts and stores which slot won, never which was kept."""

    def test_the_prompt_carries_both_texts_and_no_decision(self, tmp_path: Path) -> None:
        """The judge sees two texts under neutral names and nothing else."""
        unit = {"id": "alpha|quality|00|a", "a": "prvo besedilo", "b": "drugo besedilo", "kept": "a"}

        prompt = render_prompt(_RUBRIC, [unit], 2000, PAIRWISE_TASK.text_keys)

        assert '"a": "prvo besedilo"' in prompt
        assert '"b": "drugo besedilo"' in prompt
        assert "kept" not in prompt

    def test_a_verdict_is_narrowed_to_the_choice(self, tmp_path: Path) -> None:
        """Only `better` and `note` survive validation."""
        batch = [{"id": "alpha|quality|00|a", "a": "x", "b": "y"}]
        verdicts = [{"id": "alpha|quality|00|a", "better": "b", "note": "b is cleaner", "extra": 1}]

        assert validate_batch(verdicts, batch, PAIRWISE_TASK.checks) == [
            {"id": "alpha|quality|00|a", "better": "b", "note": "b is cleaner"}
        ]

    @pytest.mark.parametrize("value", ["A", "neither", "", 1])
    def test_a_choice_outside_the_prompt_is_refused(self, value: Any) -> None:
        """A judge that answers anything but a, b or tie fails its batch."""
        batch = [{"id": "alpha|quality|00|a", "a": "x", "b": "y"}]

        with pytest.raises(ValueError, match="better="):
            validate_batch([{"id": "alpha|quality|00|a", "better": value}], batch, PAIRWISE_TASK.checks)

    def test_the_drawing_is_written_beside_the_verdicts(self, tmp_path: Path) -> None:
        """Which slot held the kept document is kept on disk, not in the prompt."""
        source = tmp_path / "sample.jsonl"
        _pair_sample(source, per_cell=1)
        destination = tmp_path / "pairwise.jsonl"
        command = _stub_judge(tmp_path / "judge.py", _ANSWER_EVERY_PAIR)

        counts = judge_pairs(source, destination, _rubric(tmp_path), backend="cli", concurrency=1, command=command)

        pairs = [json.loads(line) for line in (tmp_path / "pairwise.pairs.jsonl").read_text().splitlines()]
        assert counts["judged"] == len(pairs) == len(PAIRWISE_STAGES) * 2
        assert judged_ids(destination) == {pair["id"] for pair in pairs}
        assert {json.loads(line)["better"] for line in destination.read_text().splitlines()} == {"a"}

    def test_a_drawing_on_disk_is_reused(self, tmp_path: Path) -> None:
        """A resumed run asks about the pairs already drawn, not a fresh draw."""
        source = tmp_path / "sample.jsonl"
        _pair_sample(source, per_cell=3)
        pairs_path = tmp_path / "chosen.jsonl"
        command = _stub_judge(tmp_path / "judge.py", _ANSWER_EVERY_PAIR)
        judge_pairs(
            source,
            tmp_path / "pairwise.jsonl",
            _rubric(tmp_path),
            pairs_path=pairs_path,
            pairs_per_cell=1,
            backend="cli",
            concurrency=1,
            command=command,
        )
        drawn = pairs_path.read_text()

        counts = judge_pairs(
            source,
            tmp_path / "pairwise.jsonl",
            _rubric(tmp_path),
            pairs_path=pairs_path,
            pairs_per_cell=3,
            backend="cli",
            concurrency=1,
            command=command,
        )

        assert pairs_path.read_text() == drawn
        assert counts["batches"] == 0


def test_the_experiment_pairwise_prompt_carries_the_placeholder() -> None:
    """The comparison prompt shipped with the experiment can actually be filled."""
    prompt = Path("experiments/data/curation-quality-slovenian/configs/judge-pairwise.md")

    assert DOCUMENTS_PLACEHOLDER in prompt.read_text(encoding="utf-8")


def test_the_experiment_rubric_carries_the_placeholder() -> None:
    """The rubric shipped with the experiment can actually be filled."""
    rubric = Path("experiments/data/curation-quality-slovenian/configs/judge-rubric.md")

    assert DOCUMENTS_PLACEHOLDER in rubric.read_text(encoding="utf-8")
    assert os.path.isfile(rubric)
