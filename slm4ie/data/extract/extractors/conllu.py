"""CoNLL-U format extractor for the SLM4IE pipeline.

Reads .conllu and .conll files and yields one `Document` per real
document. Document boundaries are detected in this order:

1. `# newdoc id = ...` markers (the standard CoNLL-U signal).
2. Changes in the leading component of a hierarchical `# sent_id`
   (e.g. `solar1.1.1` → `solar2.1.1` opens a new document with
   `doc_id = "solar2"`). Only used when no `# newdoc id` marker
   has appeared in the file — once one is seen the file is treated
   as marker-driven and the prefix heuristic is disabled.
3. Per file (`doc_id` derived from the filename stem) when neither
   signal is present.

Sentence-level structure is preserved inside `Annotations.sentences`
as inclusive `[start, end]` token-index spans.

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
        doc_id:      value of `# newdoc id = ...` when present;
                     otherwise the leading `# sent_id` prefix
                     when hierarchical; otherwise the file's stem.
        metadata:    empty by default; populated per-file when the
                     `metadata:` config block is supplied (see
                     `MetadataTable`).
        annotations:
            tokens:    flat concatenation of every sentence's tokens.
            sentences: one inclusive `[start, end]` index pair per
                       sentence, in reading order.
            spans:     entity spans decoded from `NER=` IOB tags in
                       MISC, as `[start, end, label]` character
                       offsets into `text` with upper-case labels;
                       absent when no token carries the attribute.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.metadata_table import MetadataTable
from slm4ie.data.schema import Annotations, Document, Token, render_sentence

logger = logging.getLogger(__name__)

#: IOB tag of a token outside any entity, also assumed for a token
#: whose MISC lacks `NER=` inside an otherwise tagged sentence.
_OUTSIDE = "O"


def _blank_to_none(value: str) -> Optional[str]:
    """Convert underscore placeholder to None.

    Args:
        value (str): CoNLL-U field value.

    Returns:
        Optional[str]: None if value is "_", otherwise the value
            itself.
    """
    return None if value == "_" else value


def _parse_misc(misc: str) -> Dict[str, str]:
    """Parse a MISC column into its `|`-separated `key=value` pairs.

    Args:
        misc (str): The raw MISC field; `_` denotes no attributes.

    Returns:
        Dict[str, str]: Attribute values keyed by name. A bare flag
            without `=` maps to the empty string.
    """
    if misc == "_":
        return {}
    pairs: Dict[str, str] = {}
    for item in misc.split("|"):
        key, sep, value = item.partition("=")
        pairs[key] = value if sep else ""
    return pairs


def _parse_block(lines: List[str]) -> Optional[Tuple[str, List[Token], Optional[List[str]]]]:
    """Parse a single CoNLL-U sentence block into (text, tokens, NER tags).

    Skips multiword and empty-node lines (ID contains `-` or `.`).
    Each token's `space_after` is derived from the MISC field
    (column 10): False when it carries `SpaceAfter=No`, else True.
    Text is taken from a `# text = ...` comment when present, else
    reconstructed from the tokens honouring `space_after` — both the
    text and the persisted `space_after` array come from the same
    MISC signal, so they can never diverge. The `NER=` attribute of
    MISC is kept as one IOB tag per token; a token without it inside
    a tagged sentence counts as outside any entity.

    Args:
        lines (List[str]): Non-empty lines of the sentence block.

    Returns:
        Optional[Tuple[str, List[Token], Optional[List[str]]]]:
            `(text, tokens, tags)` for the sentence, where `tags` is
            None when no token carried `NER=`; None when the block had
            no token lines.
    """
    text = ""
    tokens: List[Token] = []
    tags: List[str] = []
    tagged = False

    for line in lines:
        if line.startswith("#"):
            if line.startswith("# text = "):
                text = line[len("# text = ") :]
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            continue

        token_id = parts[0]
        # Skip multiword tokens (e.g. "1-2") and empty nodes (e.g. "1.1").
        if "-" in token_id or "." in token_id:
            continue

        misc = _parse_misc(parts[9])
        tokens.append(
            Token(
                form=parts[1],
                lemma=_blank_to_none(parts[2]),
                upos=_blank_to_none(parts[3]),
                feats=_blank_to_none(parts[5]),
                space_after=misc.get("SpaceAfter") != "No",
            )
        )
        tag = misc.get("NER")
        tagged = tagged or tag is not None
        tags.append(tag if tag else _OUTSIDE)

    if not tokens:
        return None

    if not text:
        text = render_sentence(tokens)

    return text, tokens, (tags if tagged else None)


def _decode_iob(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Decode a sentence's IOB tag sequence into token-index entity spans.

    `B-<label>` opens a span and `I-<label>` with the same label extends
    it. An `I-<label>` with no open span or a different label opens a
    new span, so a stray continuation tag keeps its entity instead of
    dropping it. `O` and the end of the sequence close the open span.

    Args:
        tags (List[str]): One IOB tag per token.

    Returns:
        List[Tuple[int, int, str]]: `(first, last, label)` token-index
            triples, `last` inclusive, with upper-case labels.
    """
    spans: List[Tuple[int, int, str]] = []
    open_start: Optional[int] = None
    open_label: Optional[str] = None

    def _close(last: int) -> None:
        nonlocal open_start, open_label
        if open_start is not None and open_label is not None:
            spans.append((open_start, last, open_label))
        open_start = open_label = None

    for index, tag in enumerate(tags):
        prefix, sep, label = tag.partition("-")
        label = label.upper()
        if not sep or prefix not in ("B", "I"):
            _close(index - 1)
        elif prefix == "B" or label != open_label:
            _close(index - 1)
            open_start, open_label = index, label
    _close(len(tags) - 1)
    return spans


def _align_tokens(text: str, tokens: List[Token]) -> Optional[List[Tuple[int, int]]]:
    """Locate each token's character range in its sentence text.

    Walks the text with a cursor that skips whitespace before every
    token and requires the token form to start exactly there, so a
    `# text` line that disagrees with the tokens is detected rather
    than matched to a later occurrence.

    Args:
        text (str): The sentence text the tokens were drawn from.
        tokens (List[Token]): The sentence's tokens in reading order.

    Returns:
        Optional[List[Tuple[int, int]]]: One `(start, end)` pair per
            token, `end` exclusive, or None when any token cannot be
            placed.
    """
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(token.form, cursor):
            return None
        offsets.append((cursor, cursor + len(token.form)))
        cursor += len(token.form)
    return offsets


def _entity_spans(
    sentence_texts: List[str],
    sentence_tokens: List[List[Token]],
    sentence_tags: List[Optional[List[str]]],
) -> Tuple[Optional[List[List[Any]]], int]:
    """Turn per-sentence IOB tags into document-level character spans.

    Each sentence's token-index spans are mapped onto the sentence
    text by `_align_tokens` and shifted by the sentence's offset in
    the newline-joined document text. A sentence that fails alignment
    contributes no spans and is counted instead of raising.

    Args:
        sentence_texts (List[str]): Text for each sentence in order.
        sentence_tokens (List[List[Token]]): Tokens per sentence.
        sentence_tags (List[Optional[List[str]]]): IOB tags per
            sentence, None for a sentence without `NER=` tags.

    Returns:
        Tuple[Optional[List[List[Any]]], int]: The document's
            `[start, end, label]` spans, or None when no sentence was
            tagged, and the number of sentences that failed alignment.
    """
    if all(tags is None for tags in sentence_tags):
        return None, 0

    spans: List[List[Any]] = []
    misaligned = 0
    offset = 0
    for text, tokens, tags in zip(sentence_texts, sentence_tokens, sentence_tags):
        if tags:
            offsets = _align_tokens(text, tokens)
            if offsets is None:
                misaligned += 1
            else:
                for first, last, label in _decode_iob(tags):
                    spans.append([offset + offsets[first][0], offset + offsets[last][1], label])
        offset += len(text) + 1
    return spans, misaligned


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
            return line[len("# newdoc id = ") :]
        # Comments precede tokens; bail as soon as we hit a token row.
        if not line.startswith("#"):
            return None
    return None


def _sent_id_prefix(lines: List[str]) -> Optional[str]:
    """Return the leading component of `# sent_id = <prefix>.X.Y...`.

    Sources such as Solar pack thousands of essays into a single
    `.conllu` file without ever emitting `# newdoc id` markers, but
    encode the essay identifier as the first dot-separated component
    of every `# sent_id` (e.g. `solar1.1.1`, `solar2.4.7`). When
    that prefix changes between sentence blocks the corpus intends a
    document boundary.

    Args:
        lines (List[str]): Non-empty lines of a sentence block.

    Returns:
        Optional[str]: The portion of `# sent_id` before the first
            `.`, or `None` when no `# sent_id` is present or the
            value does not contain a `.` (in which case the prefix
            cannot be distinguished from the full sentence id).
    """
    for line in lines:
        if line.startswith("# sent_id = "):
            value = line[len("# sent_id = ") :]
            if "." in value:
                return value.split(".", 1)[0]
            return None
        if not line.startswith("#"):
            return None
    return None


def _build_document(
    sentence_texts: List[str],
    sentence_tokens: List[List[Token]],
    sentence_tags: List[Optional[List[str]]],
    doc_id: str,
    source: str,
    domain: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    native_id: Optional[str] = None,
) -> Tuple[Document, int]:
    """Combine per-sentence pieces into one document-level Document.

    Args:
        sentence_texts (List[str]): Text for each sentence in order.
        sentence_tokens (List[List[Token]]): Tokens for each sentence
            in the same order as *sentence_texts*.
        sentence_tags (List[Optional[List[str]]]): IOB tags for each
            sentence, None where the sentence carried no `NER=`.
        doc_id (str): Identifier for the resulting Document.
        source (str): Dataset key.
        domain (str): Domain label.
        extra_metadata (Optional[Dict[str, Any]]): Per-document
            fields copied verbatim into `Document.metadata` (e.g.
            from `MetadataTable`). Empty when no sidecar TSV is
            configured.
        native_id (Optional[str]): The file's own document id
            (`# newdoc id` or the `sent_id` prefix); None when the
            document is the whole file and `doc_id` is the filename.

    Returns:
        Tuple[Document, int]: One Document whose `text` is the sentence
            texts joined with newlines, whose `annotations.tokens` is
            the flat concatenation of every sentence's tokens, whose
            `annotations.sentences` carries one inclusive `[start, end]`
            token-index span per sentence and whose `annotations.spans`
            holds the entity spans when any sentence was tagged; and
            the number of sentences whose entity spans were dropped
            because their tokens did not align with their text.
    """
    flat_tokens: List[Token] = []
    sentences: List[List[int]] = []
    cursor = 0
    for tokens in sentence_tokens:
        start = cursor
        flat_tokens.extend(tokens)
        cursor += len(tokens)
        sentences.append([start, cursor - 1])

    spans, misaligned = _entity_spans(sentence_texts, sentence_tokens, sentence_tags)
    annotations = Annotations(tokens=flat_tokens, sentences=sentences, spans=spans)
    document = Document(
        text="\n".join(sentence_texts),
        source=source,
        domain=domain,
        doc_id=doc_id,
        native_id=native_id,
        metadata=dict(extra_metadata) if extra_metadata else {},
        annotations=annotations,
    )
    return document, misaligned


class ConlluExtractor(FileBasedExtractor):
    """Extracts Documents from CoNLL-U / CoNLL files.

    Groups sentence blocks into Documents using `# newdoc id`
    markers when present; otherwise uses changes in the leading
    component of hierarchical `# sent_id` values as boundaries;
    otherwise falls back to one Document per file (`doc_id` =
    filename stem). Recursively discovers all .conllu and .conll
    files under the given directory (sorted).
    """

    def iter_input_files(self, input_dir: Path) -> List[Path]:
        """Return sorted .conllu and .conll files under input_dir.

        Args:
            input_dir (Path): Directory searched recursively.

        Returns:
            List[Path]: Sorted CoNLL-U/CoNLL file paths.
        """
        files: List[Path] = []
        for pattern in ("*.conllu", "*.conll"):
            files.extend(p for p in input_dir.rglob(pattern) if p.is_file())
        files.sort()
        return files

    def extract_files(
        self,
        files: List[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given CoNLL-U/CoNLL files.

        Args:
            files (List[Path]): Files to parse, in order.
            source (str): Dataset key assigned to every Document.
            domain (str): Domain label assigned to every Document.
            input_dir (Path): Dataset root, used to locate an optional
                `MetadataTable` TSV.
            metadata (Optional[Dict[str, Any]]): Optional `metadata:`
                config block describing an external per-document TSV.

        Yields:
            Document: One Document per `# newdoc id` block when those
                markers are present; otherwise one per leading
                `# sent_id` prefix; otherwise one per file.
        """
        table: Optional[MetadataTable] = MetadataTable.from_config(input_dir, metadata) if metadata else None
        misaligned = [0]
        for filepath in files:
            extra = table.get_for_path(filepath) if table else {}
            yield from self._parse_file(filepath, source, domain, extra, misaligned)
        if misaligned[0]:
            logger.warning(
                "Dropped entity spans of %d sentence(s) in '%s' whose tokens did not align with their `# text`",
                misaligned[0],
                source,
            )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        misaligned: Optional[List[int]] = None,
    ) -> Iterator[Document]:
        """Parse a single CoNLL-U file and yield Documents.

        Walks blank-line-separated sentence blocks, accumulating them
        into the current document. Document boundaries come from (in
        priority order) `# newdoc id` markers, changes in the
        leading component of hierarchical `# sent_id` values, or the
        file itself. The final document is flushed at EOF.

        Args:
            filepath (Path): Path to the CoNLL-U file.
            source (str): Dataset key.
            domain (str): Domain label.
            extra_metadata (Optional[Dict[str, Any]]): Per-document
                fields copied into every yielded `Document.metadata`.
                The same dict applies to every Document produced from
                this file (sidecar TSV rows are keyed by filename).
            misaligned (Optional[List[int]]): One-cell counter the
                caller shares across files; incremented by the number
                of sentences whose entity spans were dropped.

        Yields:
            Document: One Document per detected boundary, or one per
                file when no boundary signal is present.
        """
        current_block: List[str] = []
        sentence_texts: List[str] = []
        sentence_tokens: List[List[Token]] = []
        sentence_tags: List[Optional[List[str]]] = []
        current_doc_id: Optional[str] = None
        # Once any `# newdoc id` appears the file is treated as
        # marker-driven and the sent_id-prefix heuristic is off.
        seen_newdoc_marker = False
        current_prefix: Optional[str] = None
        counter = misaligned if misaligned is not None else [0]

        def _flush() -> Optional[Document]:
            if not sentence_tokens:
                return None
            doc_id = current_doc_id if current_doc_id is not None else filepath.stem
            document, dropped = _build_document(
                sentence_texts=sentence_texts,
                sentence_tokens=sentence_tokens,
                sentence_tags=sentence_tags,
                doc_id=doc_id,
                source=source,
                domain=domain,
                extra_metadata=extra_metadata,
                native_id=current_doc_id,
            )
            counter[0] += dropped
            return document

        def _reset() -> None:
            nonlocal sentence_texts, sentence_tokens, sentence_tags
            sentence_texts = []
            sentence_tokens = []
            sentence_tags = []

        def _consume(block: List[str]) -> Iterator[Document]:
            nonlocal current_doc_id, seen_newdoc_marker, current_prefix

            newdoc = _newdoc_id(block)
            if newdoc is not None:
                if sentence_tokens:
                    doc = _flush()
                    if doc is not None:
                        yield doc
                    _reset()
                seen_newdoc_marker = True
                current_doc_id = newdoc
                current_prefix = None
            elif not seen_newdoc_marker:
                prefix = _sent_id_prefix(block)
                if prefix is not None:
                    if current_prefix is None:
                        current_prefix = prefix
                        current_doc_id = prefix
                    elif prefix != current_prefix:
                        if sentence_tokens:
                            doc = _flush()
                            if doc is not None:
                                yield doc
                            _reset()
                        current_prefix = prefix
                        current_doc_id = prefix

            parsed = _parse_block(block)
            if parsed is not None:
                text, tokens, tags = parsed
                sentence_texts.append(text)
                sentence_tokens.append(tokens)
                sentence_tags.append(tags)

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
