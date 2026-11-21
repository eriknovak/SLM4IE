"""Tests for slm4ie/data/pretokenize/run.py (shard conversion and the full run)."""

import dataclasses
import json
from pathlib import Path

import pytest

from slm4ie.data.pretokenize import run
from slm4ie.data.pretokenize.config import PretokenizeConfig
from slm4ie.data.pretokenize.formats import get_format
from slm4ie.data.pretokenize.run import convert_entry, convert_shard, load_tokenizer, part_marker, run_pretokenize
from slm4ie.data.pretokenize.splits import SPLITS, split_of
from tests.data.pretokenize.conftest import CORPUS, make_entry, write_corpus


def _read_all(entry, name: str) -> dict[str, list]:
    """Merge one part per split of a format and read its samples back.

    Args:
        entry: The conversion settings.
        name: Registry name of the format.

    Returns:
        Split to its samples.
    """
    output_format = get_format(name)
    samples = {}
    for split in SPLITS:
        split_dir = entry.output_dir / name / split
        output_format.merge_parts(split_dir, ["00000"])
        samples[split] = list(output_format.read_split(split_dir))
    return samples


class TestBatches:
    """Documents are tokenized in batches bounded by count and by total characters."""

    def test_a_batch_ends_before_it_passes_the_character_budget(self):
        """Batches stay within the budget; a document longer than it forms a batch alone."""
        documents = [(f"doc-{index}", "c4", "x" * length) for index, length in enumerate([10, 10, 10, 30, 5])]
        batches = list(run._batches(iter(documents), max_docs=100, max_chars=25))
        assert [[doc_id for doc_id, _, _ in batch] for batch in batches] == [
            ["doc-0", "doc-1"],
            ["doc-2"],
            ["doc-3"],
            ["doc-4"],
        ]

    def test_a_batch_ends_at_the_document_count(self):
        """Short documents are cut into batches of at most max_docs."""
        documents = [(f"doc-{index}", "c4", "x") for index in range(5)]
        batches = list(run._batches(iter(documents), max_docs=2, max_chars=1000))
        assert [len(batch) for batch in batches] == [2, 2, 1]


class TestConvertShard:
    """convert_shard turns one input shard into one checked part per format and split."""

    def test_samples_fit_the_length_and_carry_the_template(self, tmp_path: Path, tokenizer_dir: Path):
        """Every sample is at most max_seq_len long and starts with <s> and ends with </s>."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir)
        convert_shard(shard, "00000", entry, share=0.4)
        tokenizer = load_tokenizer(str(tokenizer_dir))
        samples = [sample for split in _read_all(entry, "mds").values() for sample in split]
        assert samples and all(sample.len <= entry.max_seq_len for sample in samples)
        assert all(sample.input_ids[0] == tokenizer.cls_token_id for sample in samples)
        assert all(sample.input_ids[-1] == tokenizer.sep_token_id for sample in samples)

    def test_no_token_is_lost_apart_from_dropped_chunks(self, tmp_path: Path, tokenizer_dir: Path):
        """Each document's chunk bodies, with dropped chunks counted, add up to its tokenization."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir, min_chunk_tokens=1)
        result = convert_shard(shard, "00000", entry, share=0.4)
        tokenizer = load_tokenizer(str(tokenizer_dir))
        expected = sum(len(tokenizer(text, add_special_tokens=False)["input_ids"]) for _, text in CORPUS["c4"])
        bodies = sum(sample.len - 2 for split in _read_all(entry, "mds").values() for sample in split)
        assert bodies == expected
        assert sum(counts["dropped_tokens"] for counts in result["counts"]["c4"].values()) == 0

    def test_chunks_of_a_document_share_its_split(self, tmp_path: Path, tokenizer_dir: Path):
        """All samples of a document land in the split its id hashes to."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir)
        convert_shard(shard, "00000", entry, share=0.4)
        for split, samples in _read_all(entry, "parquet").items():
            assert all(split_of(sample.id.split("#")[0], entry.seed, 0.4) == split for sample in samples)

    def test_a_small_character_budget_leaves_the_samples_unchanged(
        self, tmp_path: Path, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Tokenizing one document per batch writes the same samples as the default batches."""
        shard = write_corpus(tmp_path / "corpus")[0]
        default_entry = make_entry(tmp_path / "default", tokenizer_dir)
        convert_shard(shard, "00000", default_entry, share=0.4)
        monkeypatch.setattr(run, "BATCH_CHARS", 1)
        small_entry = make_entry(tmp_path / "small", tokenizer_dir)
        convert_shard(shard, "00000", small_entry, share=0.4)
        assert _read_all(small_entry, "mds") == _read_all(default_entry, "mds")

    def test_every_format_holds_the_same_samples(self, tmp_path: Path, tokenizer_dir: Path):
        """The mds, mds-streaming and parquet parts hold identical samples in the same order."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir)
        convert_shard(shard, "00000", entry, share=0.4)
        reads = [_read_all(entry, name) for name in entry.formats]
        assert reads[0] == reads[1] == reads[2]

    def test_short_chunks_are_dropped_and_counted(self, tmp_path: Path, tokenizer_dir: Path):
        """A chunk shorter than min_chunk_tokens is not written and is counted as dropped."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir, min_chunk_tokens=6)
        result = convert_shard(shard, "00000", entry, share=0.4)
        samples = [sample for split in _read_all(entry, "mds").values() for sample in split]
        assert all(sample.len >= 6 for sample in samples)
        assert sum(counts["dropped_chunks"] for counts in result["counts"]["c4"].values()) >= 1

    def test_counts_match_the_written_samples(self, tmp_path: Path, tokenizer_dir: Path):
        """The part's counts give the documents, samples and tokens written per split."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir)
        result = convert_shard(shard, "00000", entry, share=0.4)
        written = _read_all(entry, "mds")
        for split in SPLITS:
            counts = result["counts"]["c4"][split]
            assert counts["samples"] == len(written[split])
            assert counts["tokens"] == sum(sample.len for sample in written[split])
        assert sum(counts["documents"] for counts in result["counts"]["c4"].values()) == len(CORPUS["c4"])

    def test_complete_part_is_skipped(self, tmp_path: Path, tokenizer_dir: Path):
        """A part with a completion marker is not converted again."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir)
        first = convert_shard(shard, "00000", entry, share=0.4)
        part_path = get_format("parquet").part_path(entry.output_dir / "parquet" / "train", "00000")
        stamp = part_path.stat().st_mtime_ns
        assert convert_shard(shard, "00000", entry, share=0.4) == first
        assert part_path.stat().st_mtime_ns == stamp

    def test_interrupted_part_leaves_no_marker_and_is_redone(self, tmp_path: Path, tokenizer_dir: Path):
        """A part interrupted before its marker is written is converted again on the next run."""
        shard = write_corpus(tmp_path / "corpus")[0]
        entry = make_entry(tmp_path, tokenizer_dir)
        convert_shard(shard, "00000", entry, share=0.4)
        part_marker(entry.output_dir, "00000").unlink()
        (entry.output_dir / "_partial" / "00000" / "mds").mkdir(parents=True)
        result = convert_shard(shard, "00000", entry, share=0.4)
        assert part_marker(entry.output_dir, "00000").exists()
        assert not (entry.output_dir / "_partial" / "00000").exists()
        assert result["counts"]["c4"]


class TestConvertEntry:
    """convert_entry converts every shard, merges the parts and writes the manifest last."""

    @staticmethod
    def _converted(tmp_path: Path, tokenizer_dir: Path, **overrides):
        """Write the corpus and convert it once.

        Args:
            tmp_path: Test directory.
            tokenizer_dir: The exported tokenizer folder.
            **overrides: Entry fields to change.

        Returns:
            The entry and the parsed manifest.
        """
        write_corpus(tmp_path / "corpus")
        entry = make_entry(tmp_path, tokenizer_dir, **overrides)
        assert convert_entry(entry) == ("written", [])
        return entry, json.loads((entry.output_dir / "manifest.json").read_text())

    @staticmethod
    def _samples(entry, name: str, split: str) -> list:
        """Read a merged split of a format.

        Args:
            entry: The conversion settings.
            name: Registry name of the format.
            split: The split.

        Returns:
            The samples, in order.
        """
        return list(get_format(name).read_split(entry.output_dir / name / split))

    def test_manifest_tokens_equal_the_samples_of_every_format(self, tmp_path: Path, tokenizer_dir: Path):
        """The manifest's token total per split equals the summed lengths in every format."""
        entry, manifest = self._converted(tmp_path, tokenizer_dir)
        for name in entry.formats:
            for split in SPLITS:
                assert manifest["totals"][split]["tokens"] == sum(s.len for s in self._samples(entry, name, split))

    def test_formats_hold_identical_samples_after_merge(self, tmp_path: Path, tokenizer_dir: Path):
        """The merged mds, mds-streaming and parquet splits hold the same samples in the same order."""
        entry, _ = self._converted(tmp_path, tokenizer_dir)
        for split in SPLITS:
            reads = [self._samples(entry, name, split) for name in entry.formats]
            assert reads[0] == reads[1] == reads[2]

    def test_no_document_spans_both_splits(self, tmp_path: Path, tokenizer_dir: Path):
        """The documents behind the train and validation samples are disjoint."""
        entry, _ = self._converted(tmp_path, tokenizer_dir)
        documents = {split: {s.id.split("#")[0] for s in self._samples(entry, "mds", split)} for split in SPLITS}
        assert documents["train"] and documents["validation"]
        assert not documents["train"] & documents["validation"]

    def test_manifest_counts_documents_per_source(self, tmp_path: Path, tokenizer_dir: Path):
        """The manifest counts every input document under its source."""
        _, manifest = self._converted(tmp_path, tokenizer_dir)
        for source, docs in CORPUS.items():
            assert sum(manifest["counts"][source][split]["documents"] for split in SPLITS) == len(docs)

    def test_manifest_records_tokenizer_inputs_and_settings(self, tmp_path: Path, tokenizer_dir: Path):
        """The manifest names the tokenizer and its digest, the input shards and their digest, and the seed."""
        _, manifest = self._converted(tmp_path, tokenizer_dir)
        assert manifest["settings"]["tokenizer"] == str(tokenizer_dir) and manifest["settings"]["tokenizer_digest"]
        assert manifest["inputs"]["digest"].startswith("sha256:")
        assert [shard["path"] for shard in manifest["inputs"]["shards"]] == ["c4/00000.jsonl.gz", "kas/00000.jsonl.gz"]
        assert manifest["settings"]["seed"] == 13

    def test_complete_output_is_skipped_unless_forced(self, tmp_path: Path, tokenizer_dir: Path):
        """A second run skips a complete output; `force` converts it again."""
        entry, _ = self._converted(tmp_path, tokenizer_dir)
        assert convert_entry(entry) == ("skipped", [])
        assert convert_entry(entry, force=True) == ("written", [])

    def test_interrupted_run_resumes_without_rewriting_finished_parts(self, tmp_path: Path, tokenizer_dir: Path):
        """With the manifest and one part's marker gone, a rerun converts only that part."""
        entry, _ = self._converted(tmp_path, tokenizer_dir)
        finished = get_format("parquet").part_path(entry.output_dir / "parquet" / "train", "00000")
        stamp = finished.stat().st_mtime_ns
        (entry.output_dir / "manifest.json").unlink()
        part_marker(entry.output_dir, "00001").unlink()
        assert convert_entry(entry) == ("written", [])
        assert finished.stat().st_mtime_ns == stamp
        assert part_marker(entry.output_dir, "00001").exists()

    def test_parts_from_other_settings_are_refused(self, tmp_path: Path, tokenizer_dir: Path):
        """Unfinished parts built with another length are not mixed in without `force`."""
        entry, _ = self._converted(tmp_path, tokenizer_dir)
        (entry.output_dir / "manifest.json").unlink()
        with pytest.raises(ValueError, match="force"):
            convert_entry(dataclasses.replace(entry, max_seq_len=32))

    def test_parts_from_another_corpus_are_refused(self, tmp_path: Path, tokenizer_dir: Path):
        """Unfinished parts built before the corpus changed are not mixed in without `force`."""
        entry, _ = self._converted(tmp_path, tokenizer_dir)
        (entry.output_dir / "manifest.json").unlink()
        write_corpus(tmp_path / "corpus", {"c4": [*CORPUS["c4"], ("c4:doc-3", "nova hiša ob cesti")]})
        with pytest.raises(ValueError, match="force"):
            convert_entry(entry)

    def test_resumed_run_reports_throughput_of_its_own_parts(self, tmp_path: Path, tokenizer_dir: Path):
        """After a resume, the run's token count covers only the parts it converted."""
        entry, first = self._converted(tmp_path, tokenizer_dir)
        (entry.output_dir / "manifest.json").unlink()
        part_marker(entry.output_dir, "00001").unlink()
        convert_entry(entry)
        last_run = json.loads((entry.output_dir / "manifest.json").read_text())["last_run"]
        kas_tokens = sum(first["counts"]["kas"][split]["tokens"] for split in SPLITS)
        assert (last_run["parts_converted"], last_run["parts_resumed"]) == (1, 1)
        assert last_run["tokens_converted"] == kas_tokens

    def test_run_log_records_throughput_and_dropped_chunks(self, tmp_path: Path, tokenizer_dir: Path):
        """The run writes its tokens per second, time and dropped-chunk count to its log folder."""
        self._converted(tmp_path, tokenizer_dir)
        line = (tmp_path / "logs" / "pretokenize" / "summary.log").read_text()
        assert "tokens/s" in line and " s (" in line and "chunks dropped" in line

    def test_parallel_run_gives_the_serial_output(self, tmp_path: Path, tokenizer_dir: Path):
        """Two workers write the same samples as one."""
        serial, _ = self._converted(tmp_path / "serial", tokenizer_dir)
        write_corpus(tmp_path / "parallel" / "corpus")
        parallel = make_entry(tmp_path / "parallel", tokenizer_dir)
        assert convert_entry(parallel, max_workers=2) == ("written", [])
        for split in SPLITS:
            assert self._samples(parallel, "mds", split) == self._samples(serial, "mds", split)

    def test_manifest_pairs_each_part_with_its_shard_whatever_order_parts_finish(
        self, tmp_path: Path, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Parts finishing in reverse order still name their own shard and its document count."""
        original = run.run_parallel

        def reversed_run_parallel(*args, **kwargs):
            results, failures = original(*args, **kwargs)
            return dict(sorted(results.items(), reverse=True)), failures

        monkeypatch.setattr(run, "run_parallel", reversed_run_parallel)
        write_corpus(tmp_path / "corpus")
        entry = make_entry(tmp_path, tokenizer_dir)
        assert convert_entry(entry) == ("written", [])
        manifest = json.loads((entry.output_dir / "manifest.json").read_text(encoding="utf-8"))
        shards = manifest["inputs"]["shards"]
        expected_paths = sorted(f"{source}/00000.jsonl.gz" for source in CORPUS)
        assert [shard["part"] for shard in shards] == [f"{index:05d}" for index in range(len(CORPUS))]
        assert [shard["path"] for shard in shards] == expected_paths
        assert [shard["documents"] for shard in shards] == [len(CORPUS[path.split("/")[0]]) for path in expected_paths]


def test_run_pretokenize_rejects_an_unknown_entry(tmp_path: Path, tokenizer_dir: Path):
    """Asking for an entry the registry does not declare fails, listing the declared ones."""
    entry = make_entry(tmp_path, tokenizer_dir)
    with pytest.raises(KeyError, match="corpus/tiny-16"):
        run_pretokenize(PretokenizeConfig(entries={entry.key: entry}), ["corpus/other-8"])


def test_run_pretokenize_reports_written_entries(tmp_path: Path, tokenizer_dir: Path):
    """A converted entry is listed as written."""
    write_corpus(tmp_path / "corpus")
    entry = make_entry(tmp_path, tokenizer_dir)
    summary = run_pretokenize(PretokenizeConfig(entries={entry.key: entry}), [entry.key])
    assert (summary.written, summary.skipped, summary.failed) == ([entry.key], [], {})
