"""Tests for the part-of-speech backend (slm4ie/data/tasks/converters/pos.py)."""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from slm4ie.data.tasks.run import convert_tasks

# ===============================
# Fixtures
# ===============================

#: The 17 Universal Dependencies part-of-speech tags.
UPOS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]


def _write_layout(tmp_path: Path, annotations: list[dict[str, Any]]) -> Path:
    """Write an extracted text file, its annotations sidecar and a one-entry registry.

    Args:
        tmp_path: Test root; every file is written beneath it.
        annotations: One annotations row per document (`forms`, `upos`, `sentences`).

    Returns:
        Path to the written task registry.
    """
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    (tmp_path / "raw").mkdir()

    # write the text file and the annotations sidecar
    with (extracted / "toy.jsonl").open("w", encoding="utf-8") as fh:
        for i, _ in enumerate(annotations):
            fh.write(json.dumps({"text": "x", "source": "toy", "domain": "web", "doc_id": f"d{i}", "uid": f"toy:d{i}"}))
            fh.write("\n")
    with gzip.open(extracted / "toy.annotations.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i, row in enumerate(annotations):
            fh.write(json.dumps({"doc_id": f"d{i}", "uid": f"toy:d{i}", **row}))
            fh.write("\n")

    # write the registry
    registry = {
        "roots": {"extracted": str(extracted), "raw": str(tmp_path / "raw"), "tasks": str(tmp_path / "tasks")},
        "converters": {"pos": "pos"},
        "entries": {
            "pos/toy": {
                "role": "finetune_and_eval",
                "source": {"kind": "extracted", "keys": ["toy"]},
                "splits": {"train": "train.jsonl.gz", "val": "val.jsonl.gz", "test": "test.jsonl.gz"},
                "labels": UPOS_TAGS,
                "suite": None,
                "language": "sl",
                "license": "cc-by-sa-4.0",
            },
        },
    }
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(yaml.safe_dump(registry))
    return config_path


def _read_examples(tmp_path: Path) -> list[dict[str, Any]]:
    """Read every written example of `pos/toy` across its splits.

    Args:
        tmp_path: Test root holding the `tasks` tree.

    Returns:
        The examples, sorted by id.
    """
    rows: list[dict[str, Any]] = []
    for path in (tmp_path / "tasks" / "pos" / "toy").glob("*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return sorted(rows, key=lambda row: row["id"])


# ===============================
# Tests
# ===============================


def test_writes_one_example_per_sentence_with_one_tag_per_word(tmp_path: Path) -> None:
    """Each sentence becomes one example whose tags run parallel to its words."""
    annotations = [
        {
            "forms": ["Ana", "bere", ".", "Dežuje", "."],
            "upos": ["PROPN", "VERB", "PUNCT", "VERB", "PUNCT"],
            "sentences": [[0, 2], [3, 4]],
        }
    ]
    config_path = _write_layout(tmp_path, annotations)

    convert_tasks(config_path, entry_keys=["pos/toy"], max_workers=1)

    assert _read_examples(tmp_path) == [
        {"id": "toy:d0:s0", "words": ["Ana", "bere", "."], "tags": ["PROPN", "VERB", "PUNCT"]},
        {"id": "toy:d0:s1", "words": ["Dežuje", "."], "tags": ["VERB", "PUNCT"]},
    ]


def test_refuses_and_counts_a_sentence_whose_word_and_tag_counts_differ(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A sentence with a missing tag is not written, and the refusal is logged with its count."""
    annotations = [
        {
            "forms": ["Ana", "bere", ".", "Dežuje", "."],
            "upos": ["PROPN", "VERB", "PUNCT", "VERB", None],
            "sentences": [[0, 2], [3, 4]],
        }
    ]
    config_path = _write_layout(tmp_path, annotations)

    with caplog.at_level(logging.WARNING, logger="slm4ie.data.tasks.converters.pos"):
        convert_tasks(config_path, entry_keys=["pos/toy"], max_workers=1)

    assert [row["id"] for row in _read_examples(tmp_path)] == ["toy:d0:s0"]
    assert "pos/toy: refused 1 sentence(s) whose word and tag counts differ" in caplog.text
