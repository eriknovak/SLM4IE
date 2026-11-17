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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm4ie.data.extract.extractors import FileBasedExtractor, register_extractor
from slm4ie.data.extract.metadata_table import MetadataTable
from slm4ie.data.schema import Annotations, Document, Token, render_sentence

# ===============================
# Logging
# ===============================

logger = logging.getLogger(__name__)

# ===============================
# Defaults
# ===============================

#: IOB tag outside any entity; assumed when a tagged sentence's token lacks `NER=`.
_OUTSIDE = "O"

#: Joins sentence texts into the document text; entity offsets shift by it.
_SENTENCE_SEP = "\n"

# ===============================
# Helper functions
# ===============================


def _blank_to_none(value: str) -> str | None:
    """Convert underscore placeholder to None.

    Args:
        value: CoNLL-U field value.

    Returns:
        None if value is "_", otherwise the value
        itself.
    """
    return None if value == "_" else value


def _parse_misc(misc: str) -> dict[str, str]:
    """Parse a MISC column into its `|`-separated `key=value` pairs.

    Args:
        misc: The raw MISC field; `_` denotes no attributes.

    Returns:
        Attribute values keyed by name. A bare flag
        without `=` maps to the empty string.
    """
    if misc == "_":
        return {}
    pairs: dict[str, str] = {}
    for item in misc.split("|"):
        key, sep, value = item.partition("=")
        pairs[key] = value if sep else ""
    return pairs


def _parse_block(lines: list[str]) -> tuple[str, list[Token], list[str] | None, list[str | None]] | None:
    """Parse a single CoNLL-U sentence block into (text, tokens, IOB tags, surfaces).

    Skips empty-node lines (ID contains `.`). A multiword token line
    (ID `3-4`) yields no token, but its form is the surface its words
    are written as in the text, so it is kept for aligning them.
    Each token's `space_after` is derived from the MISC field
    (column 10): False when it carries `SpaceAfter=No`, else True.
    Text is taken from a `# text = ...` comment when present, else
    reconstructed from the tokens honouring `space_after` — both the
    text and the persisted `space_after` array come from the same
    MISC signal, so they can never diverge. The `NER=` attribute of
    MISC is kept as one IOB tag per token; a token without it inside
    a tagged sentence counts as outside any entity.

    Args:
        lines: Non-empty lines of the sentence block.

    Returns:
        `(text, tokens, tags, surfaces)` for the sentence, where `tags`
        is None when no token carried `NER=` and `surfaces` holds, per
        token, the form written in the text: its own form, its multiword
        token's form for the first word of one, and None for the later
        words, which share that surface. None when the block had no
        token lines.
    """
    # parse the token lines
    text = ""
    tokens: list[Token] = []
    tags: list[str] = []
    surfaces: list[str | None] = []
    tagged = False
    multiword_form, multiword_first, multiword_last = "", 0, -1

    for line in lines:
        if line.startswith("#"):
            if line.startswith("# text = "):
                text = line[len("# text = ") :]
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            continue

        token_id = parts[0]
        # skip empty nodes ("1.1"); keep a multiword token's ("1-2") surface
        if "." in token_id:
            continue
        if "-" in token_id:
            first, _, last = token_id.partition("-")
            multiword_form, multiword_first, multiword_last = parts[1], int(first), int(last)
            continue

        # a word inside a multiword token is written as that token's form
        word_id = int(token_id)
        if multiword_first <= word_id <= multiword_last:
            surfaces.append(multiword_form if word_id == multiword_first else None)
        else:
            surfaces.append(parts[1])

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

    # build the sentence
    if not tokens:
        return None

    # rendered text is written from the words, so they align on their own forms
    if not text:
        text = render_sentence(tokens)
        surfaces = [token.form for token in tokens]

    return text, tokens, (tags if tagged else None), surfaces


def _decode_iob(tags: list[str]) -> list[tuple[int, int, str]]:
    """Decode a sentence's IOB tag sequence into token-index entity spans.

    `B-<label>` opens a span and `I-<label>` with the same label extends
    it. An `I-<label>` with no open span or a different label opens a
    new span, so a stray continuation tag keeps its entity instead of
    dropping it. `O` and the end of the sequence close the open span.

    Args:
        tags: One IOB tag per token.

    Returns:
        `(first, last, label)` token-index
        triples, `last` inclusive, with upper-case labels.
    """
    # track the open span
    spans: list[tuple[int, int, str]] = []
    open_start: int | None = None
    open_label: str | None = None

    def _close(last: int) -> None:
        nonlocal open_start, open_label
        if open_start is not None and open_label is not None:
            spans.append((open_start, last, open_label))
        open_start = open_label = None

    # walk the tags and close or open spans
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


def _align_tokens(text: str, surfaces: list[str | None]) -> list[tuple[int, int]] | None:
    """Locate each token's character range in its sentence text.

    Walks the text with a cursor that skips whitespace before every
    token and requires the token's surface to start exactly there, so a
    `# text` line that disagrees with the tokens is detected rather
    than matched to a later occurrence.

    Args:
        text: The sentence text the tokens were drawn from.
        surfaces: Per token in reading order, the form written in the
            text, or None for a later word of a multiword token, which
            takes the range of the word before it.

    Returns:
        One `(start, end)` pair per
        token, `end` exclusive, or None when any token cannot be
        placed.
    """
    # place each token after the previous one
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for surface in surfaces:
        if surface is None:
            offsets.append(offsets[-1])
            continue
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(surface, cursor):
            return None
        offsets.append((cursor, cursor + len(surface)))
        cursor += len(surface)
    return offsets


def _entity_spans(
    sentence_texts: list[str],
    sentence_surfaces: list[list[str | None]],
    sentence_tags: list[list[str] | None],
) -> tuple[list[list[Any]] | None, int]:
    """Turn per-sentence IOB tags into document-level character spans.

    Each sentence's token-index spans are mapped onto the sentence
    text by `_align_tokens` and shifted by the sentence's offset in
    the newline-joined document text. A sentence that fails alignment
    contributes no spans and is counted instead of raising.

    Args:
        sentence_texts: Text for each sentence in order.
        sentence_surfaces: Per sentence, each token's surface form as
            `_parse_block` returns it.
        sentence_tags: IOB tags per
            sentence, None for a sentence without `NER=` tags.

    Returns:
        The document's
        `[start, end, label]` spans, or None when no sentence was
        tagged, and the number of sentences that failed alignment.
    """
    if all(tags is None for tags in sentence_tags):
        return None, 0

    # shift each sentence's spans by its offset in the document
    spans: list[list[Any]] = []
    misaligned = 0
    offset = 0
    for text, surfaces, tags in zip(sentence_texts, sentence_surfaces, sentence_tags):
        if tags:
            offsets = _align_tokens(text, surfaces)
            if offsets is None:
                misaligned += 1
            else:
                for first, last, label in _decode_iob(tags):
                    spans.append([offset + offsets[first][0], offset + offsets[last][1], label])
        offset += len(text) + len(_SENTENCE_SEP)
    return spans, misaligned


def _newdoc_id(lines: list[str]) -> str | None:
    """Return the value of `# newdoc id = ...` in *lines*, if any.

    Args:
        lines: Non-empty lines of a sentence block.

    Returns:
        The newdoc id when the marker is present on
        this block, else `None`.
    """
    for line in lines:
        if line.startswith("# newdoc id = "):
            return line[len("# newdoc id = ") :]
        # comments precede tokens: stop at the first token row
        if not line.startswith("#"):
            return None
    return None


def _sent_id(lines: list[str]) -> str | None:
    """Return a sentence block's `# sent_id` value.

    Args:
        lines: Lines of one sentence block.

    Returns:
        The id, or None when the block has no `# sent_id` comment.
    """
    for line in lines:
        if line.startswith("# sent_id = "):
            return line[len("# sent_id = ") :]
    return None


def _sent_id_prefix(lines: list[str]) -> str | None:
    """Return the leading component of `# sent_id = <prefix>.X.Y...`.

    Sources such as Solar pack thousands of essays into a single
    `.conllu` file without ever emitting `# newdoc id` markers, but
    encode the essay identifier as the first dot-separated component
    of every `# sent_id` (e.g. `solar1.1.1`, `solar2.4.7`). When
    that prefix changes between sentence blocks the corpus intends a
    document boundary.

    Args:
        lines: Non-empty lines of a sentence block.

    Returns:
        The portion of `# sent_id` before the first
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
    sentence_texts: list[str],
    sentence_tokens: list[list[Token]],
    sentence_tags: list[list[str] | None],
    sentence_surfaces: list[list[str | None]],
    doc_id: str,
    source: str,
    domain: str,
    extra_metadata: dict[str, Any] | None = None,
    native_id: str | None = None,
) -> tuple[Document, int]:
    """Combine per-sentence pieces into one document-level Document.

    Args:
        sentence_texts: Text for each sentence in order.
        sentence_tokens: Tokens for each sentence
            in the same order as *sentence_texts*.
        sentence_tags: IOB tags for each
            sentence, None where the sentence carried no `NER=`.
        sentence_surfaces: Each sentence's token surfaces, used to place
            the entity spans in the text.
        doc_id: Identifier for the resulting Document.
        source: Dataset key.
        domain: Domain label.
        extra_metadata: Per-document
            fields copied verbatim into `Document.metadata` (e.g.
            from `MetadataTable`). Empty when no sidecar TSV is
            configured.
        native_id: The file's own document id
            (`# newdoc id` or the `sent_id` prefix); None when the
            document is the whole file and `doc_id` is the filename.

    Returns:
        One Document whose `text` is the sentence
        texts joined with newlines, whose `annotations.tokens` is
        the flat concatenation of every sentence's tokens, whose
        `annotations.sentences` carries one inclusive `[start, end]`
        token-index span per sentence and whose `annotations.spans`
        holds the entity spans when any sentence was tagged; and
        the number of sentences whose entity spans were dropped
        because their tokens did not align with their text.
    """
    # flatten the tokens and record the sentence spans
    flat_tokens: list[Token] = []
    sentences: list[list[int]] = []
    cursor = 0
    for tokens in sentence_tokens:
        start = cursor
        flat_tokens.extend(tokens)
        cursor += len(tokens)
        sentences.append([start, cursor - 1])

    # build the document
    spans, misaligned = _entity_spans(sentence_texts, sentence_surfaces, sentence_tags)
    annotations = Annotations(tokens=flat_tokens, sentences=sentences, spans=spans)
    document = Document(
        text=_SENTENCE_SEP.join(sentence_texts),
        source=source,
        domain=domain,
        doc_id=doc_id,
        native_id=native_id,
        metadata=dict(extra_metadata) if extra_metadata else {},
        annotations=annotations,
    )
    return document, misaligned


# ===============================
# Extractor class
# ===============================


class ConlluExtractor(FileBasedExtractor):
    """Extracts Documents from CoNLL-U / CoNLL files.

    Groups sentence blocks into Documents using `# newdoc id`
    markers when present; otherwise uses changes in the leading
    component of hierarchical `# sent_id` values as boundaries;
    otherwise falls back to one Document per file (`doc_id` =
    filename stem). Recursively discovers all .conllu and .conll
    files under the given directory (sorted).
    """

    def iter_input_files(self, input_dir: Path) -> list[Path]:
        """Return sorted .conllu and .conll files under input_dir.

        Args:
            input_dir: Directory searched recursively.

        Returns:
            Sorted CoNLL-U/CoNLL file paths.
        """
        # collect the files
        files: list[Path] = []
        for pattern in ("*.conllu", "*.conll"):
            files.extend(p for p in input_dir.rglob(pattern) if p.is_file())
        files.sort()
        return files

    def extract_files(
        self,
        files: list[Path],
        source: str,
        domain: str,
        input_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Document]:
        """Yield Documents from the given CoNLL-U/CoNLL files.

        Args:
            files: Files to parse, in order.
            source: Dataset key assigned to every Document.
            domain: Domain label assigned to every Document.
            input_dir: Dataset root, used to locate an optional
                `MetadataTable` TSV.
            metadata: Optional `metadata:`
                config block: an external per-document TSV (when it has a
                `path`) and `sentence_documents`, the document ids whose
                sentences are unrelated and become one Document each.

        Yields:
            Document: One Document per `# newdoc id` block when those
                markers are present; otherwise one per leading
                `# sent_id` prefix; otherwise one per file. A document
                listed in `sentence_documents` yields one per sentence.
        """
        # load the optional metadata table and sentence-document ids
        metadata = metadata or {}
        table: MetadataTable | None = MetadataTable.from_config(input_dir, metadata) if "path" in metadata else None
        sentence_documents = frozenset(metadata.get("sentence_documents", []))
        misaligned = [0]

        # parse each file
        for filepath in files:
            extra = table.get_for_path(filepath) if table else {}
            yield from self._parse_file(filepath, source, domain, extra, misaligned, sentence_documents)

        # report the dropped entity spans
        if misaligned[0]:
            logger.warning(
                "Dropped entity spans of %d sentence(s) in '%s' whose tokens did not align with their `# text`",
                misaligned[0],
                source,
            )

    # ===============================
    # Private methods
    # ===============================

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
        extra_metadata: dict[str, Any] | None,
        misaligned: list[int],
        sentence_documents: frozenset[str] = frozenset(),
    ) -> Iterator[Document]:
        """Parse a single CoNLL-U file and yield Documents.

        Walks blank-line-separated sentence blocks, accumulating them
        into the current document. Document boundaries come from (in
        priority order) `# newdoc id` markers, changes in the
        leading component of hierarchical `# sent_id` values, or the
        file itself. The final document is flushed at EOF.

        Args:
            filepath: Path to the CoNLL-U file.
            source: Dataset key.
            domain: Domain label.
            extra_metadata: Per-document
                fields copied into every yielded `Document.metadata`.
                The same dict applies to every Document produced from
                this file (sidecar TSV rows are keyed by filename).
            misaligned: One-cell counter the caller shares
                across files; incremented by the number of sentences
                whose entity spans were dropped.
            sentence_documents: Document ids whose sentences are
                yielded as one Document each, keyed by `# sent_id`.

        Yields:
            Document: One Document per detected boundary, or one per
                file when no boundary signal is present.
        """
        # track the current block and document
        current_block: list[str] = []
        sentence_texts: list[str] = []
        sentence_tokens: list[list[Token]] = []
        sentence_tags: list[list[str] | None] = []
        sentence_surfaces: list[list[str | None]] = []
        sentence_ids: list[str | None] = []
        current_doc_id: str | None = None
        # after a `# newdoc id` the sent_id-prefix heuristic is off
        seen_newdoc_marker = False
        current_prefix: str | None = None

        def _flush() -> list[Document]:
            if not sentence_tokens:
                return []
            doc_id = current_doc_id if current_doc_id is not None else filepath.stem

            # one group of sentences per document: all of them, or one each
            if doc_id in sentence_documents:
                groups = [([i], sentence_ids[i] or f"{doc_id}.{i}") for i in range(len(sentence_tokens))]
            else:
                groups = [(list(range(len(sentence_tokens))), doc_id)]

            documents: list[Document] = []
            for indices, group_id in groups:
                document, dropped = _build_document(
                    sentence_texts=[sentence_texts[i] for i in indices],
                    sentence_tokens=[sentence_tokens[i] for i in indices],
                    sentence_tags=[sentence_tags[i] for i in indices],
                    sentence_surfaces=[sentence_surfaces[i] for i in indices],
                    doc_id=group_id,
                    source=source,
                    domain=domain,
                    extra_metadata=extra_metadata,
                    native_id=current_doc_id,
                )
                misaligned[0] += dropped
                documents.append(document)
            return documents

        def _reset() -> None:
            nonlocal sentence_texts, sentence_tokens, sentence_tags, sentence_surfaces, sentence_ids
            sentence_texts = []
            sentence_tokens = []
            sentence_tags = []
            sentence_surfaces = []
            sentence_ids = []

        def _consume(block: list[str]) -> Iterator[Document]:
            nonlocal current_doc_id, seen_newdoc_marker, current_prefix

            newdoc = _newdoc_id(block)
            if newdoc is not None:
                if sentence_tokens:
                    yield from _flush()
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
                            yield from _flush()
                            _reset()
                        current_prefix = prefix
                        current_doc_id = prefix

            parsed = _parse_block(block)
            if parsed is not None:
                text, tokens, tags, surfaces = parsed
                sentence_texts.append(text)
                sentence_tokens.append(tokens)
                sentence_tags.append(tags)
                sentence_surfaces.append(surfaces)
                sentence_ids.append(_sent_id(block))

        # read the sentence blocks
        with filepath.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\r\n")
                if line == "":
                    if current_block:
                        yield from _consume(current_block)
                        current_block = []
                else:
                    current_block.append(line)

        # handle files that do not end with a blank line
        if current_block:
            yield from _consume(current_block)

        # flush the last document
        yield from _flush()


register_extractor("conllu", ConlluExtractor)
