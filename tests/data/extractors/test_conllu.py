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

TWO_SENTENCE_CONLLU = SENTENCE_1 + "\n" + SENTENCE_2

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


def _extract(tmp_path: Path, content: str) -> List[Document]:
    """Write content to a .conllu file and extract documents.

    Args:
        tmp_path (Path): Temporary directory.
        content (str): CoNLL-U content.

    Returns:
        List[Document]: Extracted documents.
    """
    _write_conllu(tmp_path, "test.conllu", content)
    extractor = ConlluExtractor()
    return list(extractor.extract(tmp_path, source="test", domain="news"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConlluExtractor:
    """Tests for ConlluExtractor."""

    def test_extracts_text(self, tmp_path: Path) -> None:
        """Text field is populated from # text = comment."""
        docs = _extract(tmp_path, TWO_SENTENCE_CONLLU)
        assert len(docs) == 2
        assert docs[0].text == "Predsednik je odprl sejo."
        assert docs[1].text == "Seja se je začela ob devetih."

    def test_extracts_tokens(self, tmp_path: Path) -> None:
        """Token fields (form, lemma, upos, feats) are parsed."""
        docs = _extract(tmp_path, TWO_SENTENCE_CONLLU)
        tokens = docs[0].annotations.tokens  # type: ignore[union-attr]
        assert len(tokens) == 5

        t0 = tokens[0]
        assert t0.form == "Predsednik"
        assert t0.lemma == "predsednik"
        assert t0.upos == "NOUN"
        assert t0.feats == "Case=Nom|Gender=Masc|Number=Sing"

        t4 = tokens[4]
        assert t4.form == "."
        assert t4.upos == "PUNCT"

    def test_doc_id_from_sent_id(self, tmp_path: Path) -> None:
        """doc_id is set from the # sent_id = comment."""
        docs = _extract(tmp_path, TWO_SENTENCE_CONLLU)
        assert docs[0].doc_id == "doc1.s1"
        assert docs[1].doc_id == "doc1.s2"

    def test_sentence_boundaries(self, tmp_path: Path) -> None:
        """sentences list is [[0, N-1]] for each document."""
        docs = _extract(tmp_path, TWO_SENTENCE_CONLLU)
        ann0 = docs[0].annotations  # type: ignore[union-attr]
        assert ann0.sentences == [[0, 4]]  # 5 tokens, indices 0-4

        ann1 = docs[1].annotations  # type: ignore[union-attr]
        assert ann1.sentences == [[0, 6]]  # 7 tokens, indices 0-6

    def test_source_and_domain(self, tmp_path: Path) -> None:
        """source and domain are passed through to each Document."""
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
        _write_conllu(tmp_path, "file2.conll", SENTENCE_2)
        extractor = ConlluExtractor()
        docs = list(
            extractor.extract(tmp_path, source="test", domain="test")
        )
        assert len(docs) == 2

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
