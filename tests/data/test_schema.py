"""Tests for the unified Document/Token/Annotations output schema."""

import json


from slm4ie.data.schema import Annotations, Document, Token, render_sentence


class TestRenderSentence:
    """Tests for the render_sentence text-reconstruction helper."""

    def test_default_spacing(self):
        """All-default tokens are joined with single spaces."""
        tokens = [Token(form="Dober"), Token(form="dan")]
        assert render_sentence(tokens) == "Dober dan"

    def test_space_after_false_drops_space(self):
        """space_after=False glues the next token on."""
        tokens = [
            Token(form="uprava", space_after=False),
            Token(form="."),
        ]
        assert render_sentence(tokens) == "uprava."

    def test_no_trailing_space(self):
        """The final token never emits a trailing space."""
        tokens = [Token(form="Hvala"), Token(form=".")]
        assert render_sentence(tokens) == "Hvala ."

    def test_empty(self):
        """No tokens yields an empty string."""
        assert render_sentence([]) == ""


class TestToken:
    """Tests for the Token dataclass."""

    def test_minimal_token(self):
        """Token with only form required."""
        token = Token(form="hello")
        assert token.form == "hello"
        assert token.lemma is None
        assert token.upos is None
        assert token.feats is None
        assert token.space_after is True

    def test_full_token(self):
        """Token with all fields populated."""
        token = Token(
            form="running",
            lemma="run",
            upos="VERB",
            feats="Aspect=Prog",
        )
        assert token.form == "running"
        assert token.lemma == "run"
        assert token.upos == "VERB"
        assert token.feats == "Aspect=Prog"

    def test_to_dict_excludes_none(self):
        """to_dict should not include keys with None values."""
        token = Token(form="hello", upos="NOUN")
        result = token.to_dict()
        assert result == {"form": "hello", "upos": "NOUN", "space_after": True}
        assert "lemma" not in result

    def test_to_dict_full_token(self):
        """to_dict includes all fields when all are set."""
        token = Token(
            form="running",
            lemma="run",
            upos="VERB",
            feats="Aspect=Prog",
            space_after=False,
        )
        result = token.to_dict()
        assert result == {
            "form": "running",
            "lemma": "run",
            "upos": "VERB",
            "feats": "Aspect=Prog",
            "space_after": False,
        }

    def test_to_dict_minimal_token(self):
        """to_dict with only form still carries the space_after bool."""
        token = Token(form="test")
        result = token.to_dict()
        assert result == {"form": "test", "space_after": True}


class TestAnnotations:
    """Tests for the Annotations dataclass."""

    def test_annotations_creation(self):
        """Create Annotations with tokens and sentence boundaries."""
        tokens = [Token(form="Hello"), Token(form="world")]
        sentences = [[0, 1]]
        ann = Annotations(tokens=tokens, sentences=sentences)
        assert len(ann.tokens) == 2
        assert ann.sentences == [[0, 1]]

    def test_to_dict(self):
        """to_dict returns tokens as dicts and sentences."""
        tokens = [
            Token(form="Hello", upos="INTJ"),
            Token(form="world", upos="NOUN"),
        ]
        sentences = [[0, 1]]
        ann = Annotations(tokens=tokens, sentences=sentences)
        result = ann.to_dict()
        assert result == {
            "tokens": [
                {"form": "Hello", "upos": "INTJ", "space_after": True},
                {"form": "world", "upos": "NOUN", "space_after": True},
            ],
            "sentences": [[0, 1]],
        }

    def test_to_dict_multiple_sentences(self):
        """to_dict with multiple sentence boundaries."""
        tokens = [
            Token(form="Hello"),
            Token(form="world"),
            Token(form="Goodbye"),
        ]
        sentences = [[0, 1], [2, 2]]
        ann = Annotations(tokens=tokens, sentences=sentences)
        result = ann.to_dict()
        assert result["sentences"] == [[0, 1], [2, 2]]
        assert len(result["tokens"]) == 3


class TestDocument:
    """Tests for the Document dataclass."""

    def test_minimal_document(self):
        """Document with only required fields."""
        doc = Document(
            text="Hello world",
            source="ssj500k",
            domain="web",
        )
        assert doc.text == "Hello world"
        assert doc.source == "ssj500k"
        assert doc.domain == "web"
        assert doc.doc_id is None
        assert doc.metadata == {}
        assert doc.annotations is None

    def test_to_jsonl_line_no_newline(self):
        """to_jsonl_line output must not contain newlines."""
        doc = Document(
            text="Hello world",
            source="ssj500k",
            domain="web",
        )
        line = doc.to_jsonl_line()
        assert "\n" not in line

    def test_to_jsonl_line_none_fields_excluded(self):
        """to_jsonl_line excludes None and empty fields."""
        doc = Document(
            text="Hello world",
            source="ssj500k",
            domain="web",
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert "doc_id" not in data
        assert "annotations" not in data
        assert "metadata" not in data

    def test_to_jsonl_line_required_fields(self):
        """to_jsonl_line includes required fields."""
        doc = Document(
            text="Hello world",
            source="ssj500k",
            domain="web",
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert data["text"] == "Hello world"
        assert data["source"] == "ssj500k"
        assert data["domain"] == "web"

    def test_to_jsonl_line_with_doc_id(self):
        """doc_id preserved in JSONL output."""
        doc = Document(
            text="Hello world",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert data["doc_id"] == "doc-001"

    def test_uid_combines_source_and_doc_id(self):
        """The uid is `{source}:{doc_id}` when doc_id is set."""
        doc = Document(
            text="Hello",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
        )
        assert doc.uid == "ssj500k:doc-001"

    def test_uid_none_without_doc_id(self):
        """The uid is None when doc_id is absent."""
        doc = Document(text="Hello", source="ssj500k", domain="web")
        assert doc.uid is None

    def test_to_jsonl_line_includes_uid(self):
        """The uid is emitted alongside doc_id in JSONL output."""
        doc = Document(
            text="Hello",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
        )
        data = json.loads(doc.to_jsonl_line())
        assert data["uid"] == "ssj500k:doc-001"

    def test_to_jsonl_line_uid_excluded_without_doc_id(self):
        """The uid is omitted when there is no doc_id."""
        doc = Document(text="Hello", source="ssj500k", domain="web")
        data = json.loads(doc.to_jsonl_line())
        assert "uid" not in data

    def test_to_annotation_line_includes_uid(self):
        """The uid is emitted in annotation output when doc_id is set."""
        ann = Annotations(tokens=[Token(form="x")], sentences=[[0, 0]])
        doc = Document(
            text="x",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
            annotations=ann,
        )
        line = doc.to_annotation_line()
        assert line is not None
        data = json.loads(line)
        assert data["uid"] == "ssj500k:doc-001"

    def test_uid_disambiguates_across_sources(self):
        """Same doc_id under different sources yields distinct uids."""
        a = Document(text="x", source="ds_a", domain="web", doc_id="1")
        b = Document(text="y", source="ds_b", domain="web", doc_id="1")
        assert a.uid != b.uid

    def test_to_jsonl_line_with_metadata(self):
        """Metadata is preserved in JSONL output."""
        doc = Document(
            text="Hello world",
            source="ssj500k",
            domain="web",
            metadata={"year": 2020, "url": "http://example.com"},
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert data["metadata"] == {"year": 2020, "url": "http://example.com"}

    def test_to_jsonl_line_excludes_annotations(self):
        """to_jsonl_line must NOT include annotations."""
        tokens = [
            Token(form="Zdravo", upos="INTJ"),
            Token(form="svet", upos="NOUN"),
        ]
        ann = Annotations(tokens=tokens, sentences=[[0, 1]])
        doc = Document(
            text="Zdravo svet",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
            annotations=ann,
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert "annotations" not in data
        assert data["text"] == "Zdravo svet"
        assert data["doc_id"] == "doc-001"

    def test_to_jsonl_line_with_annotations_still_excludes_them(self):
        """Even with annotations set, to_jsonl_line excludes them."""
        tokens = [
            Token(form="Zdravo", upos="INTJ"),
            Token(form="svet", upos="NOUN"),
        ]
        ann = Annotations(tokens=tokens, sentences=[[0, 1]])
        doc = Document(
            text="Zdravo svet",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
            annotations=ann,
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert "annotations" not in data
        assert data["text"] == "Zdravo svet"

    def test_to_annotation_line_compact_format(self):
        """to_annotation_line returns compact parallel-array format."""
        tokens = [
            Token(form="Zdravo", lemma="zdrav", upos="INTJ", feats=None),
            Token(form="svet", lemma="svet", upos="NOUN", feats="Case=Nom"),
        ]
        ann = Annotations(tokens=tokens, sentences=[[0, 1]])
        doc = Document(
            text="Zdravo svet",
            source="ssj500k",
            domain="web",
            doc_id="doc-001",
            annotations=ann,
        )
        line = doc.to_annotation_line()
        data = json.loads(line)
        assert data["doc_id"] == "doc-001"
        assert data["forms"] == ["Zdravo", "svet"]
        assert data["lemmas"] == ["zdrav", "svet"]
        assert data["upos"] == ["INTJ", "NOUN"]
        assert data["feats"] == [None, "Case=Nom"]
        assert data["space_after"] == [True, True]
        assert data["sentences"] == [[0, 1]]

    def test_to_annotation_line_space_after_array(self):
        """space_after is a bool array parallel to forms."""
        tokens = [
            Token(form="uprava", space_after=False),
            Token(form="."),
        ]
        ann = Annotations(tokens=tokens, sentences=[[0, 1]])
        doc = Document(
            text="uprava.",
            source="kas",
            domain="academic",
            doc_id="d1",
            annotations=ann,
        )
        data = json.loads(doc.to_annotation_line())
        assert data["space_after"] == [False, True]
        assert len(data["space_after"]) == len(data["forms"])
        assert all(isinstance(v, bool) for v in data["space_after"])

    def test_to_annotation_line_none_when_no_annotations(self):
        """to_annotation_line returns None when no annotations."""
        doc = Document(
            text="Zdravo svet",
            source="ssj500k",
            domain="web",
        )
        assert doc.to_annotation_line() is None

    def test_to_annotation_line_no_newline(self):
        """to_annotation_line output must not contain newlines."""
        tokens = [Token(form="test")]
        ann = Annotations(tokens=tokens, sentences=[[0, 0]])
        doc = Document(
            text="test",
            source="t",
            domain="t",
            doc_id="d1",
            annotations=ann,
        )
        line = doc.to_annotation_line()
        assert "\n" not in line

    def test_to_jsonl_line_ensure_ascii_false(self):
        """Non-ASCII characters are not escaped."""
        doc = Document(
            text="Šola je dobra",
            source="ssj500k",
            domain="web",
        )
        line = doc.to_jsonl_line()
        assert "Šola" in line

    def test_empty_metadata_excluded(self):
        """Empty metadata dict is excluded from JSONL output."""
        doc = Document(
            text="test",
            source="ssj500k",
            domain="web",
            metadata={},
        )
        line = doc.to_jsonl_line()
        data = json.loads(line)
        assert "metadata" not in data
