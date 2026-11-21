"""Tests for slm4ie/data/pretokenize/check.py (verifying a finished output)."""

import json
from pathlib import Path

from slm4ie.data.pretokenize.check import check_pretokenized
from slm4ie.data.pretokenize.formats import Sample, get_format
from slm4ie.data.pretokenize.run import convert_entry
from slm4ie.data.pretokenize.splits import is_validation
from tests.data.pretokenize.conftest import CORPUS, make_entry, write_corpus


def _converted(tmp_path: Path, tokenizer_dir: Path, **overrides):
    """Write the corpus and convert it.

    Args:
        tmp_path: Test directory.
        tokenizer_dir: The exported tokenizer folder.
        **overrides: Entry fields to change.

    Returns:
        The entry.
    """
    write_corpus(tmp_path / "corpus")
    entry = make_entry(tmp_path, tokenizer_dir, **overrides)
    convert_entry(entry)
    return entry


def _edit_manifest(entry, edit) -> None:
    """Change the manifest in place.

    Args:
        entry: The conversion settings.
        edit: Mutates the parsed manifest.
    """
    path = entry.output_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def test_finished_output_passes(tmp_path: Path, tokenizer_dir: Path):
    """A freshly converted output passes with every format's totals equal to the manifest's."""
    entry = _converted(tmp_path, tokenizer_dir)
    report = check_pretokenized(entry.output_dir)
    manifest = json.loads((entry.output_dir / "manifest.json").read_text())
    assert report.ok, report.problems
    assert report.totals["parquet"]["train"]["tokens"] == manifest["totals"]["train"]["tokens"]


def test_token_total_off_the_manifest_is_reported(tmp_path: Path, tokenizer_dir: Path):
    """A manifest whose token total disagrees with the samples fails the check, naming format and split."""
    entry = _converted(tmp_path, tokenizer_dir)
    _edit_manifest(entry, lambda manifest: manifest["totals"]["train"].update(tokens=1))
    report = check_pretokenized(entry.output_dir)
    assert not report.ok and any("mds/train" in problem and "tokens" in problem for problem in report.problems)


def test_document_in_both_splits_is_reported(tmp_path: Path, tokenizer_dir: Path):
    """A validation document that also has a train sample fails the check."""
    entry = _converted(tmp_path, tokenizer_dir, formats=("parquet",))
    train_dir = entry.output_dir / "parquet" / "train"
    validation_id = next(iter(get_format("parquet").read_split(entry.output_dir / "parquet" / "validation"))).id
    writer = get_format("parquet").open_part(train_dir / "99999.parquet", 300, 1 << 20)
    writer.write(Sample((2, 5, 3), validation_id.rsplit("#", 1)[0] + "#9"))
    writer.close()
    report = check_pretokenized(entry.output_dir)
    assert any("both splits" in problem for problem in report.problems)


def test_formats_that_differ_are_reported(tmp_path: Path, tokenizer_dir: Path):
    """An extra sample in one format fails the check as a difference between formats."""
    entry = _converted(tmp_path, tokenizer_dir, formats=("mds", "parquet"))
    writer = get_format("parquet").open_part(entry.output_dir / "parquet" / "train" / "99999.parquet", 300, 1 << 20)
    writer.write(Sample((2, 5, 3), "extra#0"))
    writer.close()
    report = check_pretokenized(entry.output_dir)
    assert any("formats differ" in problem for problem in report.problems)


def test_stored_len_off_the_ids_is_reported(tmp_path: Path, tokenizer_dir: Path):
    """A stored len that differs from the sample's id count fails the check, naming the format."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    entry = _converted(tmp_path, tokenizer_dir, formats=("parquet",))
    path = entry.output_dir / "parquet" / "train" / "00000.parquet"
    table = pq.read_table(str(path))
    lengths = pa.array([value + 1 for value in table.column("len").to_pylist()], type=pa.int32())
    pq.write_table(table.set_column(1, "len", lengths), str(path))
    report = check_pretokenized(entry.output_dir)
    assert any("parquet/train" in problem and "stores len" in problem for problem in report.problems)


def test_small_validation_split_warns_without_failing(tmp_path: Path, tokenizer_dir: Path):
    """A validation split below the minimum is a warning, since the hash split meets it only on average."""
    entry = _converted(tmp_path, tokenizer_dir)
    _edit_manifest(entry, lambda manifest: manifest["settings"].update(min_validation_docs=1000))
    report = check_pretokenized(entry.output_dir)
    assert report.ok and any("min_validation_docs" in warning for warning in report.warnings)


def test_output_without_manifest_is_reported(tmp_path: Path):
    """An output with no manifest is reported as not complete."""
    report = check_pretokenized(tmp_path / "missing")
    assert not report.ok and "not complete" in report.problems[0]


def test_empty_validation_split_passes(tmp_path: Path, tokenizer_dir: Path):
    """An output whose validation split is empty is read back without error."""
    # the minimum of one document raises the share to 1/5; pick a seed that still sends none
    doc_ids = [doc_id for docs in CORPUS.values() for doc_id, _ in docs]
    seed = next(s for s in range(1000) if not any(is_validation(doc_id, s, 0.2) for doc_id in doc_ids))
    entry = _converted(tmp_path, tokenizer_dir, seed=seed, validation_share=0.01, min_validation_docs=1)
    report = check_pretokenized(entry.output_dir)
    assert report.ok, report.problems
    assert report.totals["mds"]["validation"] == {"samples": 0, "tokens": 0}
