"""CoNLL-U format extractor for the SLM4IE pipeline."""

from pathlib import Path
from typing import Iterator, List, Optional

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.schema import Annotations, Document, Token


def _blank_to_none(value: str) -> Optional[str]:
    """Convert underscore placeholder to None.

    Args:
        value (str): CoNLL-U field value.

    Returns:
        Optional[str]: None if value is ``_``, otherwise the value itself.
    """
    return None if value == "_" else value


def _parse_block(
    lines: List[str],
    source: str,
    domain: str,
) -> Optional[Document]:
    """Parse a single CoNLL-U sentence block into a Document.

    Skips multiword and empty-node lines (ID contains ``-`` or ``.``).

    Args:
        lines (List[str]): Non-empty lines of the sentence block.
        source (str): Dataset key to assign to the Document.
        domain (str): Domain label to assign to the Document.

    Returns:
        Optional[Document]: Parsed Document, or None if the block had
            no token lines.
    """
    text = ""
    sent_id: Optional[str] = None
    tokens: List[Token] = []

    for line in lines:
        if line.startswith("#"):
            if line.startswith("# text = "):
                text = line[len("# text = "):]
            elif line.startswith("# sent_id = "):
                sent_id = line[len("# sent_id = "):]
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            continue

        token_id = parts[0]
        # Skip multiword tokens (e.g. "1-2") and empty nodes (e.g. "1.1")
        if "-" in token_id or "." in token_id:
            continue

        form = parts[1]
        lemma = _blank_to_none(parts[2])
        upos = _blank_to_none(parts[3])
        feats = _blank_to_none(parts[5])

        tokens.append(
            Token(
                form=form,
                lemma=lemma,
                upos=upos,
                feats=feats,
            )
        )

    if not tokens:
        return None

    annotations = Annotations(
        tokens=tokens,
        sentences=[[0, len(tokens) - 1]],
    )
    return Document(
        text=text,
        source=source,
        domain=domain,
        doc_id=sent_id,
        annotations=annotations,
    )


class ConlluExtractor(BaseExtractor):
    """Extracts Documents from CoNLL-U / CoNLL files.

    One :class:`~slm4ie.data.schema.Document` is produced per sentence
    block. Recursively discovers all ``*.conllu`` and ``*.conll`` files
    under the given directory (sorted).
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Yield Documents from all CoNLL-U files under *input_dir*.

        Args:
            input_dir (Path): Directory containing ``.conllu``/``.conll``
                files (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.

        Yields:
            Document: One document per sentence block.
        """
        patterns = ["*.conllu", "*.conll"]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(input_dir.rglob(pattern))
        files.sort()

        for filepath in files:
            yield from self._parse_file(filepath, source, domain)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse a single CoNLL-U file and yield Documents.

        Args:
            filepath (Path): Path to the CoNLL-U file.
            source (str): Dataset key.
            domain (str): Domain label.

        Yields:
            Document: One document per sentence block.
        """
        current_block: List[str] = []

        with filepath.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if line == "":
                    if current_block:
                        doc = _parse_block(current_block, source, domain)
                        if doc is not None:
                            yield doc
                        current_block = []
                else:
                    current_block.append(line)

        # Handle files that don't end with a blank line
        if current_block:
            doc = _parse_block(current_block, source, domain)
            if doc is not None:
                yield doc


register_extractor("conllu", ConlluExtractor)
