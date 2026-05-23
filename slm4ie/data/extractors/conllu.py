"""CoNLL-U format extractor for the SLM4IE pipeline.

Reads .conllu and .conll files and yields one `Document` per real
document. Documents are bounded by `# newdoc id = ...` markers when
present; files lacking that marker collapse to a single Document per
file (`doc_id` derived from the filename stem). Sentence-level
structure is preserved inside `Annotations.sentences` as inclusive
`[start, end]` token-index spans.

Multiword tokens (ID containing `-`) and empty nodes (ID containing
`.`) are skipped. Columns are tab-separated: ID, FORM, LEMMA, UPOS,
XPOS, FEATS, HEAD, DEPREL, DEPS, MISC. An underscore (`_`) denotes
a missing value.

Example:
    Input (two sentences grouped under one `# newdoc id`):

        # newdoc id = doc1
        # sent_id = doc1.s1
        # text = Predsednik je odprl sejo.
        1	Predsednik	predsednik	NOUN	Ncmsn	Case=Nom	3	nsubj	_	NER=O
        2	je	biti	AUX	Va-r3s-n	Tense=Pres	3	aux	_	NER=O
        3	odprl	odpreti	VERB	Vmep-sm	VerbForm=Part	0	root	_	NER=O
        4	sejo	seja	NOUN	Ncfsa	Case=Acc	3	obj	_	NER=O
        5	.	.	PUNCT	Z	_	3	punct	_	NER=O

        # sent_id = doc1.s2
        # text = Hvala.
        1	Hvala	hvala	NOUN	Ncfsn	Case=Nom	0	root	_	NER=O
        2	.	.	PUNCT	Z	_	1	punct	_	NER=O

    Yields one `Document` with `doc_id == "doc1"`, `text` formed
    by joining the two sentence strings with a newline, 7 flat
    tokens in `annotations.tokens`, and `annotations.sentences ==
    [[0, 4], [5, 6]]`.

    Schema mapping:
        text:        per-sentence `# text = ...` comments joined
                     with newlines; falls back to reconstructed text
                     from FORM columns (honouring SpaceAfter=No)
                     when the comment is absent.
        source:      provided by caller.
        domain:      provided by caller.
        doc_id:      value of `# newdoc id = ...` when present,
                     otherwise the file's stem.
        metadata:    empty by default; populated per-file when the
                     `metadata:` config block is supplied (see
                     `MetadataLookup`).
        annotations:
            tokens:    flat concatenation of every sentence's tokens.
            sentences: one inclusive `[start, end]` index pair per
                       sentence, in reading order.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.extractors import BaseExtractor, register_extractor
from slm4ie.data.metadata_lookup import MetadataLookup
from slm4ie.data.schema import Annotations, Document, Token


def _blank_to_none(value: str) -> Optional[str]:
    """Convert underscore placeholder to None.

    Args:
        value (str): CoNLL-U field value.

    Returns:
        Optional[str]: None if value is "_", otherwise the value
            itself.
    """
    return None if value == "_" else value


def _parse_block(lines: List[str]) -> Optional[Tuple[str, List[Token]]]:
    """Parse a single CoNLL-U sentence block into (text, tokens).

    Skips multiword and empty-node lines (ID contains `-` or `.`).
    Text is taken from a `# text = ...` comment when present, else
    reconstructed from token forms honouring `SpaceAfter=No`.

    Args:
        lines (List[str]): Non-empty lines of the sentence block.

    Returns:
        Optional[Tuple[str, List[Token]]]: `(text, tokens)` for the
            sentence, or None when the block had no token lines.
    """
    text = ""
    tokens: List[Token] = []

    for line in lines:
        if line.startswith("#"):
            if line.startswith("# text = "):
                text = line[len("# text = "):]
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            continue

        token_id = parts[0]
        # Skip multiword tokens (e.g. "1-2") and empty nodes (e.g. "1.1").
        if "-" in token_id or "." in token_id:
            continue

        form = parts[1]
        tokens.append(
            Token(
                form=form,
                lemma=_blank_to_none(parts[2]),
                upos=_blank_to_none(parts[3]),
                feats=_blank_to_none(parts[5]),
            )
        )

    if not tokens:
        return None

    if not text:
        text = _reconstruct_text(lines)

    return text, tokens


def _reconstruct_text(lines: List[str]) -> str:
    """Reconstruct sentence text from CoNLL-U token rows.

    Joins token forms with spaces unless the token's MISC field
    (column 10) contains SpaceAfter=No.

    Args:
        lines (List[str]): Non-empty lines of the sentence block.

    Returns:
        str: The reconstructed sentence text.
    """
    parts: List[str] = []
    for line in lines:
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 10:
            continue
        token_id = cols[0]
        if "-" in token_id or "." in token_id:
            continue
        form = cols[1]
        misc = cols[9]
        parts.append(form)
        if "SpaceAfter=No" not in misc:
            parts.append(" ")
    return "".join(parts).rstrip()


def _newdoc_id(lines: List[str]) -> Optional[str]:
    """Return the value of `# newdoc id = ...` in *lines*, if any.

    Args:
        lines (List[str]): Non-empty lines of a sentence block.

    Returns:
        Optional[str]: The newdoc id when the marker is present on
            this block, else `None`.
    """
    for line in lines:
        if line.startswith("# newdoc id = "):
            return line[len("# newdoc id = "):]
        # Comments precede tokens; bail as soon as we hit a token row.
        if not line.startswith("#"):
            return None
    return None


def _build_document(
    sentence_texts: List[str],
    sentence_tokens: List[List[Token]],
    doc_id: str,
    source: str,
    domain: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Document:
    """Combine per-sentence pieces into one document-level Document.

    Args:
        sentence_texts (List[str]): Text for each sentence in order.
        sentence_tokens (List[List[Token]]): Tokens for each sentence
            in the same order as *sentence_texts*.
        doc_id (str): Identifier for the resulting Document.
        source (str): Dataset key.
        domain (str): Domain label.
        extra_metadata (Optional[Dict[str, Any]]): Per-document
            fields copied verbatim into `Document.metadata` (e.g.
            from `MetadataLookup`). Empty when no sidecar TSV is
            configured.

    Returns:
        Document: One Document whose `text` is the sentence texts
            joined with newlines, whose `annotations.tokens` is the
            flat concatenation of every sentence's tokens, and whose
            `annotations.sentences` carries one inclusive
            `[start, end]` token-index span per sentence.
    """
    flat_tokens: List[Token] = []
    spans: List[List[int]] = []
    cursor = 0
    for tokens in sentence_tokens:
        start = cursor
        flat_tokens.extend(tokens)
        cursor += len(tokens)
        spans.append([start, cursor - 1])

    annotations = Annotations(tokens=flat_tokens, sentences=spans)
    return Document(
        text="\n".join(sentence_texts),
        source=source,
        domain=domain,
        doc_id=doc_id,
        metadata=dict(extra_metadata) if extra_metadata else {},
        annotations=annotations,
    )


class ConlluExtractor(BaseExtractor):
    """Extracts Documents from CoNLL-U / CoNLL files.

    Groups sentence blocks into Documents using `# newdoc id`
    markers when present; falls back to one Document per file
    (`doc_id` = filename stem) for sources that don't emit the
    marker. Recursively discovers all .conllu and .conll files under
    the given directory (sorted).
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from all CoNLL-U files under input_dir.

        Args:
            input_dir (Path): Directory containing .conllu and
                .conll files (searched recursively).
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.
            metadata (Optional[Dict[str, Any]]): Optional `metadata:`
                config block describing an external per-document TSV.
                When given, every Document is enriched with the row
                matched on the source filename. See `MetadataLookup`
                for the expected schema.

        Yields:
            Document: One Document per `# newdoc id` block, or one
                Document per file when the marker is absent.
        """
        lookup: Optional[MetadataLookup] = (
            MetadataLookup.from_config(input_dir, metadata)
            if metadata
            else None
        )

        patterns = ["*.conllu", "*.conll"]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(p for p in input_dir.rglob(pattern) if p.is_file())
        files.sort()

        for filepath in files:
            extra = lookup.get_for_path(filepath) if lookup else {}
            yield from self._parse_file(filepath, source, domain, extra)

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Parse a single CoNLL-U file and yield Documents.

        Walks blank-line-separated sentence blocks, accumulating them
        into the current document. A new document starts on each
        `# newdoc id = ...` marker; the final document is flushed
        at EOF. Files with no `# newdoc id` markers produce one
        Document whose `doc_id` is the file's stem.

        Args:
            filepath (Path): Path to the CoNLL-U file.
            source (str): Dataset key.
            domain (str): Domain label.
            extra_metadata (Optional[Dict[str, Any]]): Per-document
                fields copied into every yielded `Document.metadata`.
                The same dict applies to every Document produced from
                this file (sidecar TSV rows are keyed by filename).

        Yields:
            Document: One Document per newdoc block, or one per file
                when no markers are present.
        """
        current_block: List[str] = []
        sentence_texts: List[str] = []
        sentence_tokens: List[List[Token]] = []
        current_doc_id: Optional[str] = None

        def _flush() -> Optional[Document]:
            if not sentence_tokens:
                return None
            doc_id = current_doc_id if current_doc_id is not None else filepath.stem
            return _build_document(
                sentence_texts=sentence_texts,
                sentence_tokens=sentence_tokens,
                doc_id=doc_id,
                source=source,
                domain=domain,
                extra_metadata=extra_metadata,
            )

        def _consume(block: List[str]) -> Iterator[Document]:
            nonlocal sentence_texts, sentence_tokens, current_doc_id
            newdoc = _newdoc_id(block)
            if newdoc is not None and sentence_tokens:
                doc = _flush()
                if doc is not None:
                    yield doc
                sentence_texts = []
                sentence_tokens = []
            if newdoc is not None:
                current_doc_id = newdoc

            parsed = _parse_block(block)
            if parsed is not None:
                text, tokens = parsed
                sentence_texts.append(text)
                sentence_tokens.append(tokens)

        with filepath.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if line == "":
                    if current_block:
                        yield from _consume(current_block)
                        current_block = []
                else:
                    current_block.append(line)

        # Handle files that don't end with a blank line.
        if current_block:
            yield from _consume(current_block)

        final = _flush()
        if final is not None:
            yield final


register_extractor("conllu", ConlluExtractor)
