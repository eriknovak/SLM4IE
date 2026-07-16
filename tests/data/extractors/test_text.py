"""Tests for TextExtractor, focused on deterministic doc_id assignment."""

from pathlib import Path
from typing import List

from slm4ie.data.extractors.text import TextExtractor
from slm4ie.data.processing import _chunk_files

_TWO_BLOCKS = "Prvi dokument.\nDruga vrstica.\n\nDrugi dokument.\n"


def _ids(docs: List) -> List[str]:
    """Return the doc_ids of the given Documents.

    Args:
        docs (List): Documents yielded by an extractor.

    Returns:
        List[str]: The `doc_id` of each document, in order.
    """
    return [d.doc_id for d in docs]


class TestTextExtractorDocIds:
    """Tests for the file-relative, per-file-indexed doc_id scheme."""

    def test_doc_id_is_file_relative_with_block_index(
        self, tmp_path: Path
    ) -> None:
        """doc_id is `<rel-path-without-suffix>:<block index>`."""
        (tmp_path / "sl.txt").write_text(_TWO_BLOCKS, encoding="utf-8")

        docs = list(TextExtractor().extract(tmp_path, "cc100", "web"))

        assert _ids(docs) == ["sl:000000", "sl:000001"]
        assert docs[0].uid == "cc100:sl:000000"

    def test_nested_paths_keep_posix_separators(self, tmp_path: Path) -> None:
        """A nested input file yields a `/`-joined relative key."""
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "sl.txt").write_text(_TWO_BLOCKS, encoding="utf-8")

        docs = list(TextExtractor().extract(tmp_path, "cc100", "web"))

        assert _ids(docs) == ["a/b/sl:000000", "a/b/sl:000001"]

    def test_block_index_resets_per_file(self, tmp_path: Path) -> None:
        """Each file restarts its block index at zero; ids stay unique."""
        (tmp_path / "a.txt").write_text(_TWO_BLOCKS, encoding="utf-8")
        (tmp_path / "b.txt").write_text(_TWO_BLOCKS, encoding="utf-8")

        docs = list(TextExtractor().extract(tmp_path, "cc100", "web"))

        assert _ids(docs) == [
            "a:000000",
            "a:000001",
            "b:000000",
            "b:000001",
        ]
        assert len(set(_ids(docs))) == len(docs)

    def test_ids_are_stable_across_runs(self, tmp_path: Path) -> None:
        """Re-extracting the same input reproduces identical ids."""
        (tmp_path / "sl.txt").write_text(_TWO_BLOCKS, encoding="utf-8")

        first = list(TextExtractor().extract(tmp_path, "cc100", "web"))
        second = list(TextExtractor().extract(tmp_path, "cc100", "web"))

        assert _ids(first) == _ids(second)

    def test_whitespace_only_blocks_do_not_consume_an_index(
        self, tmp_path: Path
    ) -> None:
        """Blocks that strip to nothing are skipped without leaving a gap."""
        (tmp_path / "sl.txt").write_text(
            "Prvi.\n\n   \n\nDrugi.\n", encoding="utf-8"
        )

        docs = list(TextExtractor().extract(tmp_path, "cc100", "web"))

        assert _ids(docs) == ["sl:000000", "sl:000001"]
        assert [d.text for d in docs] == ["Prvi.", "Drugi."]

    def test_serial_and_sharded_ids_are_identical(self, tmp_path: Path) -> None:
        """Splitting the file list into shards does not change any id.

        The sharded writer splits whole files across workers and
        concatenates the shard outputs in order, so mirroring that split
        here must reproduce the serial writer's ids exactly.
        """
        for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
            (tmp_path / name).write_text(_TWO_BLOCKS, encoding="utf-8")

        extractor = TextExtractor()
        files = extractor.iter_input_files(tmp_path)
        serial = _ids(list(extractor.extract(tmp_path, "cc100", "web")))

        sharded: List[str] = []
        for chunk in _chunk_files(files, 3):
            sharded.extend(
                _ids(
                    list(
                        extractor.extract_files(
                            chunk, "cc100", "web", tmp_path
                        )
                    )
                )
            )

        assert sharded == serial

    def test_doc_id_falls_back_to_filename_outside_input_dir(
        self, tmp_path: Path
    ) -> None:
        """A file outside input_dir keys off its filename, not a traversal."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "sl.txt").write_text(_TWO_BLOCKS, encoding="utf-8")
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        docs = list(
            TextExtractor().extract_files(
                [outside / "sl.txt"], "cc100", "web", input_dir
            )
        )

        assert _ids(docs) == ["sl:000000", "sl:000001"]
