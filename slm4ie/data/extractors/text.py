"""Plain-text extractor for line-oriented corpora (e.g. CC100).

Treats each blank-line-separated block in a ``.txt`` file as a single
document.  Streams files line-by-line so multi-GB inputs do not need
to fit in memory.
"""

import logging
from pathlib import Path
from typing import Iterator, List

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    """Extracts Documents from plain ``.txt`` files.

    Documents are delimited by blank lines (the CC100 convention).
    Recursively discovers all ``*.txt`` files under *input_dir* (sorted)
    and yields one :class:`~slm4ie.data.schema.Document` per non-empty
    block. No annotations are produced.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all ``.txt`` files under *input_dir*.

        Args:
            input_dir (Path): Directory containing ``.txt`` files
                (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per blank-line-separated block.
        """
        files: List[Path] = sorted(input_dir.rglob("*.txt"))

        for filepath in files:
            yield from self._parse_file(filepath, source, domain)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Stream *filepath* and yield one Document per blank-line block.

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
