"""Plain-text extractor for line-oriented corpora (e.g. CC100).

Treats each blank-line-separated block in a .txt file as a single
document. Streams files line-by-line so multi-GB inputs do not need
to fit in memory.

Example:
    A .txt file with documents separated by blank lines:

        Prvi dokument, prva vrstica.
        Prvi dokument, druga vrstica.

        Drugi dokument, ena sama vrstica.

        Tretji dokument.

    Schema mapping:
        text:        joined non-empty lines of one block.
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      `<rel>:<block_idx>`, e.g. `sl:000123` (see below).
        metadata:    not produced.
        annotations: not produced.

Document ids:
    Plain text carries no native id, so `doc_id` is the positional id
    `<rel>:<block_idx>` of `assembly.positional_doc_id`, where `rel` is
    the input file's unit name (path relative to `input_dir`, suffix
    removed, separators `/`) and `block_idx` the 0-based index of the
    emitted document within that file, padded to six digits. So
    `<input_dir>/sl.txt` yields `sl:000000`, `sl:000001`, ... and `uid`
    becomes `cc100:sl:000000`.

    The scheme is deterministic and independent of worker count: the
    orchestrator shards by whole files and preserves per-file block
    order, so a document's id is the same under the serial and sharded
    writers. It replaces the orchestrator's positional `idx-` fallback,
    which varied with both worker count and file-discovery order.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.extractors.assembly import positional_doc_id, relative_unit
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)

# Six digits: the shared corpus keys on CC100 ids minted at this width.
_BLOCK_WIDTH = 6


class TextExtractor(FileBasedExtractor):
    """Extracts Documents from plain .txt files.

    Documents are delimited by blank lines (the CC100 convention).
    Recursively discovers all .txt files under input_dir (sorted) and
    yields one Document per non-empty block, each carrying a
    deterministic file-relative `doc_id` (see the module docstring).
    No annotations are produced.
    """

    def iter_input_files(self, input_dir: Path) -> List[Path]:
        """Return sorted .txt files under input_dir.

        Args:
            input_dir (Path): Directory searched recursively.

        Returns:
            List[Path]: Sorted .txt file paths.
        """
        return sorted(input_dir.rglob("*.txt"))

    def extract_files(
        self,
        files: List[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given .txt files.

        Args:
            files (List[Path]): .txt files to parse, in order.
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.
            input_dir (Path): Dataset root; each document's `doc_id` is
                keyed off its file's path relative to this directory.
            metadata (Optional[Dict[str, Any]]): Ignored.

        Yields:
            Document: One document per blank-line-separated block.
        """
        del metadata
        for filepath in files:
            yield from self._parse_file(filepath, source, domain, input_dir)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        input_dir: Path,
    ) -> Iterator[Document]:
        """Stream a file and yield one Document per blank-line block.

        Args:
            filepath (Path): Path to the text file.
            source (str): Dataset key.
            domain (str): Domain label.
            input_dir (Path): Dataset root, used to derive the
                file-relative part of each `doc_id`.

        Yields:
            Document: One document per blank-line-separated block, with
                `doc_id` = `<rel>:<block index within this file>`.
        """
        rel = relative_unit(filepath, input_dir)
        buffer: List[str] = []
        block_idx = 0

        def flush() -> Iterator[Document]:
            nonlocal block_idx
            if not buffer:
                return
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=positional_doc_id(rel, block_idx, width=_BLOCK_WIDTH),
                )
                block_idx += 1

        with filepath.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if line == "":
                    yield from flush()
                else:
                    buffer.append(line)

        yield from flush()


register_extractor("text", TextExtractor)
