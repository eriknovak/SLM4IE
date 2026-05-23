"""Tests for ConlluExtractor."""

import textwrap
from pathlib import Path
from typing import List


from slm4ie.data.extractors.conllu import ConlluExtractor
from slm4ie.data.schema import Document


# ---------------------------------------------------------------------------
# Shared test fixtures / constants
# ---------------------------------------------------------------------------

SENTENCE_1 = textwrap.dedent("""\
    # newdoc id = doc1
    # sent_id = doc1.s1
    # text = Predsednik je odprl sejo.
    1\tPredsednik\tpredsednik\tNOUN\tNcmsn\tCase=Nom|Gender=Masc|Number=Sing\t3\tnsubj\t_\tNER=O
    2\tje\tbiti\tAUX\tVa-r3s-n\tMood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin\t3\taux\t_\tNER=O
    3\todprl\todpreti\tVERB\tVmep-sm\tGender=Masc|Number=Sing|VerbForm=Part\t0\troot\t_\tNER=O
    4\tsejo\tseja\tNOUN\tNcfsa\tCase=Acc|Gender=Fem|Number=Sing\t3\tobj\t_\tNER=O
    5\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t_\tNER=O
""")

SENTENCE_2 = textwrap.dedent("""\
    # sent_id = doc1.s2
    # text = Seja se je začela ob devetih.
    1\tSeja\tseja\tNOUN\tNcfsn\tCase=Nom|Gender=Fem|Number=Sing\t4\tnsubj\t_\tNER=O
    2\tse\tsebe\tPRON\tPx------y\t_\t4\texpl\t_\tNER=O
    3\tje\tbiti\tAUX\tVa-r3s-n\tMood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin\t4\taux\t_\tNER=O
    4\tzačela\tzačeti\tVERB\tVmep-sf\tGender=Fem|Number=Sing|VerbForm=Part\t0\troot\t_\tNER=O
    5\tob\tob\tADP\tSl\tCase=Loc\t4\tobl\t_\tNER=O
    6\tdevetih\tdevet\tNUM\tMl-pl\tCase=Loc|Number=Plur\t5\tnmod\t_\tNER=O
    7\t.\t.\tPUNCT\tZ\t_\t4\tpunct\t_\tNER=O
""")

# `SENTENCE_2` opens a fresh CoNLL-U newdoc so the two sentences land in
# separate documents when concatenated. Used by the multi-newdoc tests.
SENTENCE_2_NEWDOC = "# newdoc id = doc2\n" + SENTENCE_2

# Two sentences sharing a single newdoc — one Document with 12 tokens.
ONE_NEWDOC_TWO_SENTENCES = SENTENCE_1 + "\n" + SENTENCE_2

# Two newdoc blocks — two Documents, one per block.
TWO_NEWDOCS = SENTENCE_1 + "\n" + SENTENCE_2_NEWDOC

# Sentences with no newdoc marker at all — per-file fallback kicks in.
NO_NEWDOC_MARKER = textwrap.dedent("""\
    # sent_id = a.s1
    # text = Prva poved.
    1\tPrva\tprv\tADJ\tAgpfsn\tCase=Nom|Gender=Fem|Number=Sing\t2\tamod\t_\t_
    2\tpoved\tpoved\tNOUN\tNcfsn\tCase=Nom|Gender=Fem|Number=Sing\t0\troot\t_\t_
    3\t.\t.\tPUNCT\tZ\t_\t2\tpunct\t_\t_

    # sent_id = a.s2
    # text = Druga poved.
    1\tDruga\tdrug\tADJ\tAgpfsn\tCase=Nom|Gender=Fem|Number=Sing\t2\tamod\t_\t_
    2\tpoved\tpoved\tNOUN\tNcfsn\tCase=Nom|Gender=Fem|Number=Sing\t0\troot\t_\t_
    3\t.\t.\tPUNCT\tZ\t_\t2\tpunct\t_\t_
""")

NER_SENTENCE = textwrap.dedent("""\
    # sent_id = ner.s1
    # text = Janez Novak je iz Ljubljane.
    1\tJanez\tJanez\tPROPN\tNpmsn\tCase=Nom|Gender=Masc|Number=Sing\t3\tnsubj\t_\tNER=B-PER
    2\tNovak\tNovak\tPROPN\tNpmsn\tCase=Nom|Gender=Masc|Number=Sing\t1\tflat\t_\tNER=I-PER
    3\tje\tbiti\tAUX\tVa-r3s-n\tMood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin\t0\troot\t_\tNER=O
    4\tiz\tiz\tADP\tSg\tCase=Gen\t3\tobl\t_\tNER=O
    5\tLjubljane\tLjubljana\tPROPN\tNpfsg\tCase=Gen|Gender=Fem|Number=Sing\t4\tnmod\t_\tNER=B-LOC
    6\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t_\tNER=O
""")

UNDERSCORE_SENTENCE = textwrap.dedent("""\
    # sent_id = under.s1
    # text = Primer.
    1\tPrimer\tprimer\tNOUN\t_\t_\t0\troot\t_\t_
    2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_
""")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_conllu(tmp_path: Path, filename: str, content: str) -> Path:
    """Write *content* to *tmp_path / filename* and return the path.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        filename (str): File name to create.
        content (str): CoNLL-U content to write.

    Returns:
        Path: Path to the written file.
    """
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _extract(tmp_path: Path, content: str, filename: str = "test.conllu") -> List[Document]:
    """Write content to a .conllu file and extract documents.

    Args:
        tmp_path (Path): Temporary directory.
        content (str): CoNLL-U content.
        filename (str): Name of the .conllu file to write (drives the
            per-file-fallback doc_id when no `# newdoc id` markers
            appear in the content).

    Returns:
        List[Document]: Extracted documents.
    """
    _write_conllu(tmp_path, filename, content)
    extractor = ConlluExtractor()
    return list(extractor.extract(tmp_path, source="test", domain="news"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConlluExtractor:
    """Tests for ConlluExtractor."""

    def test_one_newdoc_two_sentences_yields_single_document(
        self, tmp_path: Path
    ) -> None:
        """Sentences sharing one `# newdoc id` collapse to one Document."""
        docs = _extract(tmp_path, ONE_NEWDOC_TWO_SENTENCES)
        assert len(docs) == 1
        # Text is the two sentence strings joined by a sentence-boundary
        # newline so datatrove's sentence splitter retains the cue.
        assert docs[0].text == (
            "Predsednik je odprl sejo.\nSeja se je začela ob devetih."
        )

    def test_one_newdoc_flat_tokens_across_sentences(
        self, tmp_path: Path
    ) -> None:
        """Annotations.tokens concatenates tokens from every sentence."""
        docs = _extract(tmp_path, ONE_NEWDOC_TWO_SENTENCES)
        tokens = docs[0].annotations.tokens  # type: ignore[union-attr]
        # 5 tokens from sentence 1 + 7 from sentence 2.
        assert len(tokens) == 12
        assert tokens[0].form == "Predsednik"
        assert tokens[0].upos == "NOUN"
        # First token of sentence 2 sits at flat index 5.
        assert tokens[5].form == "Seja"
        assert tokens[5].upos == "NOUN"

    def test_one_newdoc_sentence_spans(self, tmp_path: Path) -> None:
        """Annotations.sentences carries one [start, end] per sentence."""
        docs = _extract(tmp_path, ONE_NEWDOC_TWO_SENTENCES)
        ann = docs[0].annotations
        assert ann is not None
        assert ann.sentences == [[0, 4], [5, 11]]

    def test_doc_id_from_newdoc_marker(self, tmp_path: Path) -> None:
        """doc_id comes from `# newdoc id`, not per-sentence `# sent_id`."""
        docs = _extract(tmp_path, ONE_NEWDOC_TWO_SENTENCES)
        assert docs[0].doc_id == "doc1"

    def test_two_newdocs_yield_two_documents(self, tmp_path: Path) -> None:
        """Each `# newdoc id` marker starts a fresh Document."""
        docs = _extract(tmp_path, TWO_NEWDOCS)
        assert len(docs) == 2
        assert docs[0].doc_id == "doc1"
        assert docs[1].doc_id == "doc2"
        # Each document carries only its own sentences.
        assert docs[0].annotations.sentences == [[0, 4]]  # type: ignore[union-attr]
        assert docs[1].annotations.sentences == [[0, 6]]  # type: ignore[union-attr]
        assert docs[0].text == "Predsednik je odprl sejo."
        assert docs[1].text == "Seja se je začela ob devetih."

    def test_per_file_fallback_when_no_newdoc(self, tmp_path: Path) -> None:
        """A file with no `# newdoc id` collapses to one Document per file."""
        docs = _extract(tmp_path, NO_NEWDOC_MARKER, filename="essay.conllu")
        assert len(docs) == 1
        assert docs[0].doc_id == "essay"
        ann = docs[0].annotations
        assert ann is not None
        # Both sentences end up in the one Document.
        assert ann.sentences == [[0, 2], [3, 5]]
        assert docs[0].text == "Prva poved.\nDruga poved."

    def test_source_and_domain(self, tmp_path: Path) -> None:
        """Source and domain are passed through to each Document."""
        _write_conllu(tmp_path, "a.conllu", SENTENCE_1)
        extractor = ConlluExtractor()
        docs = list(
            extractor.extract(tmp_path, source="ssj500k", domain="news")
        )
        assert docs[0].source == "ssj500k"
        assert docs[0].domain == "news"

    def test_processes_multiple_files(self, tmp_path: Path) -> None:
        """Both .conllu and .conll files are discovered and processed."""
        _write_conllu(tmp_path, "file1.conllu", SENTENCE_1)
        # SENTENCE_2 alone has no newdoc id; per-file fallback gives it
        # its own Document keyed off the file stem.
        _write_conllu(tmp_path, "file2.conll", SENTENCE_2)
        extractor = ConlluExtractor()
        docs = list(
            extractor.extract(tmp_path, source="test", domain="test")
        )
        assert len(docs) == 2
        assert docs[0].doc_id == "doc1"          # from `# newdoc id`
        assert docs[1].doc_id == "file2"         # per-file fallback

    def test_handles_underscore_feats(self, tmp_path: Path) -> None:
        """_ in feats column is converted to None."""
        docs = _extract(tmp_path, UNDERSCORE_SENTENCE)
        tokens = docs[0].annotations.tokens  # type: ignore[union-attr]
        assert tokens[0].feats is None

    def test_text_reconstructed_when_comment_absent(
        self, tmp_path: Path
    ) -> None:
        """Text is rebuilt from token forms when # text is absent."""
        content = textwrap.dedent("""\
            # sent_id = x.s1
            1\tLepa\tlep\tADJ\t_\t_\t2\tamod\t_\t_
            2\tbeseda\tbeseda\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No
            3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
        """)
        docs = _extract(tmp_path, content)
        assert len(docs) == 1
        # Per-file fallback yields one Document whose text is the single
        # reconstructed sentence — no trailing newline because there is
        # only one sentence to join.
        assert docs[0].text == "Lepa beseda."

    def test_directory_named_like_conll_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """Glob must not return directories whose name ends in .conll."""
        (tmp_path / "KZB.conll").mkdir()
        _write_conllu(tmp_path / "KZB.conll", "inner.conllu", SENTENCE_1)
        extractor = ConlluExtractor()
        docs = list(
            extractor.extract(tmp_path, source="kzb", domain="x")
        )
        # Should find the .conllu inside the directory and not crash.
        assert len(docs) == 1

    def test_registered_as_conllu(self) -> None:
        """ConlluExtractor is registered under the 'conllu' key."""
        from slm4ie.data.extractors import get_extractor

        extractor = get_extractor("conllu")
        assert isinstance(extractor, ConlluExtractor)


class TestConlluMetadata:
    """End-to-end metadata injection from an external TSV."""

    def test_metadata_attached_to_every_doc_in_file(self, tmp_path: Path) -> None:
        """Every Document from a file shares the file's per-doc metadata.

        Uses the two-newdoc fixture so the file yields two Documents
        — both should carry the same metadata since sidecar TSV rows
        are keyed by filename, not by `# newdoc id`.
        """
        _write_conllu(tmp_path, "kzb-001.conllu", TWO_NEWDOCS)
        tsv = tmp_path / "meta.tsv"
        tsv.write_text(
            "ID\tFields\tType\n"
            "kzb-001\tgeografija,zgodovina\tmonografija\n",
            encoding="utf-8",
        )

        extractor = ConlluExtractor()
        docs = list(
            extractor.extract(
                tmp_path,
                source="kzb",
                domain="scientific",
                metadata={
                    "path": "meta.tsv",
                    "key_column": "ID",
                    "fields": {"Fields": "field", "Type": "doctype"},
                    "splits": {"Fields": ","},
                },
            )
        )

        assert len(docs) == 2
        expected = {"field": ["geografija", "zgodovina"], "doctype": "monografija"}
        assert docs[0].metadata == expected
        assert docs[1].metadata == expected

    def test_regex_key_pattern_extracts_id(self, tmp_path: Path) -> None:
        """key_pattern lets the OSS numeric ID drive the lookup."""
        _write_conllu(tmp_path, "oss-42.conllu", SENTENCE_1)
        tsv = tmp_path / "meta.tsv"
        tsv.write_text("id\tudc\n42\t502\n", encoding="utf-8")

        extractor = ConlluExtractor()
        docs = list(
            extractor.extract(
                tmp_path,
                source="oss",
                domain="scientific",
                metadata={
                    "path": "meta.tsv",
                    "key_column": "id",
                    "key_pattern": r"^oss-(\d+)$",
                    "fields": {"udc": "udc"},
                },
            )
        )

        assert docs[0].metadata == {"udc": "502"}

    def test_no_metadata_kwarg_keeps_empty_dict(self, tmp_path: Path) -> None:
        """Without the kwarg, metadata stays empty (backward compatible)."""
        docs = _extract(tmp_path, SENTENCE_1)
        assert docs[0].metadata == {}
