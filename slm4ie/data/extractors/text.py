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
        doc_id:      not produced.
        metadata:    not produced.
        annotations: not produced.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from slm4ie.data.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)


class TextExtractor(FileBasedExtractor):
    """Extracts Documents from plain .txt files.

    Documents are delimited by blank lines (the CC100 convention).
    Recursively discovers all .txt files under input_dir (sorted) and
    yields one Document per non-empty block. No annotations are
    produced.
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
            input_dir (Path): Unused; this extractor has no sidecar.
            metadata (Optional[Dict[str, Any]]): Ignored.

        Yields:
            Document: One document per blank-line-separated block.
        """
        del input_dir, metadata
        for filepath in files:
            yield from self._parse_file(filepath, source, domain)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Stream a file and yield one Document per blank-line block.

        Args:
            filepath (Path): Path to the text file.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per blank-line-separated block.
        """
        buffer: List[str] = []

        def flush() -> Iterator[Document]:
            if not buffer:
                return
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                yield Document(text=text, source=source, domain=domain)

        with filepath.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if line == "":
                    yield from flush()
                else:
                    buffer.append(line)

        yield from flush()


register_extractor("text", TextExtractor)
