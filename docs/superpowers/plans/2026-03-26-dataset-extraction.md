# Dataset Extraction & Text Processing Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build extraction infrastructure that decompresses raw dataset archives and converts them into a unified JSONL format preserving text and linguistic annotations (POS, NER, morphology, dependency parse) for Slovenian LM training.

**Architecture:** A base `Extractor` ABC with format-specific subclasses (CoNLL-U, JSONL, TEI XML, HuggingFace Arrow). Each extractor reads from `data/raw/<dataset>/` and writes JSONL to `data/processed/<dataset>/`. A registry maps dataset keys to extractors. An `extract_datasets` orchestrator (mirroring `download_datasets`) drives the pipeline via config.

**Tech Stack:** Python 3.13, stdlib `gzip`/`zipfile`/`tarfile` for decompression, `xml.etree.ElementTree` for TEI XML, `datasets` library for Arrow, pytest for testing.

---

## Output Schema

Every extractor produces JSONL files where each line is a JSON object:

```json
{
  "text": "Predsednik je odprl sejo.",
  "source": "parlamint_si",
  "domain": "parliamentary",
  "doc_id": "ParlaMint-SI_2022-01-15.seg42",
  "metadata": {},
  "annotations": {
    "tokens": [
      {
        "form": "Predsednik",
        "lemma": "predsednik",
        "upos": "NOUN",
        "xpos": "Ncmsn",
        "feats": "Case=Nom|Gender=Masc|Number=Sing",
        "head": 3,
        "deprel": "nsubj",
        "ner": "O"
      }
    ],
    "sentences": [[0, 4]]
  }
}
```

- `text`: Required. Raw sentence/document text.
- `source`: Required. Dataset key from config.
- `domain`: Required. One of: `web`, `parliamentary`, `academic`, `legal`, `medical`, `scientific`, `forum`, `blog`, `news`, `student`, `wiki`, `literary`.
- `doc_id`: Optional. Document/sentence identifier from source.
- `metadata`: Optional dict. Source-specific metadata (speaker, date, url, quality_score, etc.).
- `annotations`: Optional. Present when source has linguistic markup.
  - `tokens`: List of token dicts. Fields: `form` (required), `lemma`, `upos`, `xpos`, `feats`, `head`, `deprel`, `ner`.
  - `sentences`: List of `[start_idx, end_idx]` pairs indexing into `tokens`.

---

## File Structure

```
slm4ie/data/
├── extract.py          # NEW — archive decompression (gz, zip, tgz)
├── extractors/         # NEW — format-specific text extractors
│   ├── __init__.py     #       registry + base class
│   ├── conllu.py       #       CoNLL-U extractor
│   ├── jsonl.py        #       JSONL extractor (CLASSLA-web)
│   ├── tei.py          #       TEI XML extractor
│   ├── macocu.py       #       MaCoCu XML extractor
│   └── huggingface.py  #       HuggingFace Arrow extractor
├── processing.py       # MODIFY — add extract_datasets orchestrator
├── download.py         # existing, no changes
└── ...

scripts/data/
├── extract.py          # NEW — CLI entrypoint for extraction

configs/data/
├── extract.yaml        # NEW — extraction config (maps dataset → extractor + domain)

tests/data/
├── test_extract.py     # NEW — archive decompression tests
├── extractors/         # NEW
│   ├── __init__.py
│   ├── test_conllu.py
│   ├── test_jsonl.py
│   ├── test_tei.py
│   ├── test_macocu.py
│   └── test_huggingface.py
└── test_processing.py  # NEW — orchestrator tests
```

---

## Task 1: Output Schema Dataclass

**Files:**
- Create: `slm4ie/data/schema.py`
- Test: `tests/data/test_schema.py`

- [ ] **Step 1: Write failing tests for schema dataclasses**

```python
# tests/data/test_schema.py
"""Tests for slm4ie.data.schema module."""

import json

from slm4ie.data.schema import (
    Document,
    Token,
    Annotations,
)


class TestToken:
    """Tests for Token dataclass."""

    def test_minimal_token(self):
        token = Token(form="Hello")
        assert token.form == "Hello"
        assert token.lemma is None
        assert token.upos is None

    def test_full_token(self):
        token = Token(
            form="Predsednik",
            lemma="predsednik",
            upos="NOUN",
            xpos="Ncmsn",
            feats="Case=Nom|Gender=Masc|Number=Sing",
            head=3,
            deprel="nsubj",
            ner="O",
        )
        assert token.form == "Predsednik"
        assert token.upos == "NOUN"
        assert token.head == 3

    def test_token_to_dict(self):
        token = Token(form="je", lemma="biti", upos="AUX")
        d = token.to_dict()
        assert d == {
            "form": "je",
            "lemma": "biti",
            "upos": "AUX",
        }
        assert "xpos" not in d  # None fields excluded


class TestAnnotations:
    """Tests for Annotations dataclass."""

    def test_annotations_with_tokens(self):
        tokens = [
            Token(form="A", upos="NOUN"),
            Token(form="B", upos="VERB"),
        ]
        ann = Annotations(
            tokens=tokens, sentences=[[0, 1]]
        )
        assert len(ann.tokens) == 2
        assert ann.sentences == [[0, 1]]

    def test_annotations_to_dict(self):
        tokens = [Token(form="A")]
        ann = Annotations(tokens=tokens, sentences=[[0, 0]])
        d = ann.to_dict()
        assert d == {
            "tokens": [{"form": "A"}],
            "sentences": [[0, 0]],
        }


class TestDocument:
    """Tests for Document dataclass."""

    def test_minimal_document(self):
        doc = Document(
            text="Hello world.",
            source="test",
            domain="web",
        )
        assert doc.text == "Hello world."
        assert doc.source == "test"
        assert doc.domain == "web"
        assert doc.doc_id is None
        assert doc.metadata == {}
        assert doc.annotations is None

    def test_document_to_jsonl_line(self):
        doc = Document(
            text="Test.",
            source="test_ds",
            domain="web",
        )
        line = doc.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed == {
            "text": "Test.",
            "source": "test_ds",
            "domain": "web",
        }
        assert "\n" not in line

    def test_document_with_annotations_to_jsonl(self):
        doc = Document(
            text="A B",
            source="test_ds",
            domain="web",
            doc_id="doc1",
            metadata={"url": "http://example.com"},
            annotations=Annotations(
                tokens=[
                    Token(form="A", upos="NOUN"),
                    Token(form="B", upos="VERB"),
                ],
                sentences=[[0, 1]],
            ),
        )
        line = doc.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["doc_id"] == "doc1"
        assert parsed["metadata"]["url"] == "http://example.com"
        assert len(parsed["annotations"]["tokens"]) == 2
        assert parsed["annotations"]["sentences"] == [[0, 1]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'slm4ie.data.schema'`

- [ ] **Step 3: Implement schema dataclasses**

```python
# slm4ie/data/schema.py
"""Unified output schema for extracted dataset documents."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Token:
    """A single token with optional linguistic annotations.

    Attributes:
        form: Surface form of the token.
        lemma: Lemmatized form.
        upos: Universal POS tag.
        xpos: Language-specific POS tag.
        feats: Morphological features string.
        head: Index of syntactic head token.
        deprel: Dependency relation label.
        ner: Named entity tag (IOB2 format).
    """

    form: str
    lemma: Optional[str] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    feats: Optional[str] = None
    head: Optional[int] = None
    deprel: Optional[str] = None
    ner: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dict, excluding None fields."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None
        }


@dataclass
class Annotations:
    """Linguistic annotations for a document.

    Attributes:
        tokens: List of annotated tokens.
        sentences: Sentence boundaries as [start, end] index
            pairs into the tokens list.
    """

    tokens: List[Token]
    sentences: List[List[int]]

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "tokens": [t.to_dict() for t in self.tokens],
            "sentences": self.sentences,
        }


@dataclass
class Document:
    """A single document/sentence in the unified output format.

    Attributes:
        text: Raw text content.
        source: Dataset key identifier.
        domain: Domain category.
        doc_id: Optional document/sentence identifier.
        metadata: Source-specific metadata.
        annotations: Optional linguistic annotations.
    """

    text: str
    source: str
    domain: str
    doc_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    annotations: Optional[Annotations] = None

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSON line.

        Returns:
            str: JSON string with no trailing newline.
                None/empty fields are excluded.
        """
        d: Dict = {
            "text": self.text,
            "source": self.source,
            "domain": self.domain,
        }
        if self.doc_id is not None:
            d["doc_id"] = self.doc_id
        if self.metadata:
            d["metadata"] = self.metadata
        if self.annotations is not None:
            d["annotations"] = self.annotations.to_dict()
        return json.dumps(d, ensure_ascii=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_schema.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/schema.py tests/data/test_schema.py
git commit -m "feat: add unified Document/Token/Annotations output schema"
```

---

## Task 2: Archive Decompression

**Files:**
- Create: `slm4ie/data/extract.py`
- Test: `tests/data/test_extract.py`

- [ ] **Step 1: Write failing tests for archive extraction**

```python
# tests/data/test_extract.py
"""Tests for slm4ie.data.extract module."""

import gzip
import tarfile
import zipfile
from pathlib import Path

import pytest

from slm4ie.data.extract import extract_archive


class TestExtractArchive:
    """Tests for extract_archive function."""

    def test_extract_gzip(self, tmp_path: Path):
        gz_file = tmp_path / "test.jsonl.gz"
        content = b'{"text": "hello"}\n'
        with gzip.open(gz_file, "wb") as f:
            f.write(content)

        result = extract_archive(gz_file, tmp_path)
        expected = tmp_path / "test.jsonl"
        assert result == expected
        assert expected.read_bytes() == content

    def test_extract_zip(self, tmp_path: Path):
        zip_file = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("inner/file.txt", "hello zip")

        result = extract_archive(zip_file, tmp_path)
        assert result == tmp_path
        assert (tmp_path / "inner" / "file.txt").read_text() == (
            "hello zip"
        )

    def test_extract_tar_gz(self, tmp_path: Path):
        tgz_file = tmp_path / "test.tgz"
        inner_file = tmp_path / "inner.txt"
        inner_file.write_text("hello tar")
        with tarfile.open(tgz_file, "w:gz") as tf:
            tf.add(inner_file, arcname="inner.txt")
        inner_file.unlink()

        result = extract_archive(tgz_file, tmp_path)
        assert result == tmp_path
        assert (tmp_path / "inner.txt").read_text() == "hello tar"

    def test_extract_tar_gz_long_ext(self, tmp_path: Path):
        tgz_file = tmp_path / "test.tar.gz"
        inner_file = tmp_path / "data.txt"
        inner_file.write_text("hello")
        with tarfile.open(tgz_file, "w:gz") as tf:
            tf.add(inner_file, arcname="data.txt")
        inner_file.unlink()

        result = extract_archive(tgz_file, tmp_path)
        assert (tmp_path / "data.txt").read_text() == "hello"

    def test_extract_skips_if_already_extracted(
        self, tmp_path: Path
    ):
        gz_file = tmp_path / "test.jsonl.gz"
        content = b"original"
        with gzip.open(gz_file, "wb") as f:
            f.write(content)

        expected = tmp_path / "test.jsonl"
        expected.write_bytes(b"already here")

        result = extract_archive(gz_file, tmp_path)
        assert expected.read_bytes() == b"already here"

    def test_extract_unknown_format_raises(
        self, tmp_path: Path
    ):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("???")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_archive(bad_file, tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement archive extraction**

```python
# slm4ie/data/extract.py
"""Archive decompression utilities for raw dataset files."""

import gzip
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract a compressed archive to the output directory.

    Supports .gz, .zip, .tgz, and .tar.gz formats.
    Skips extraction if output already exists.

    Args:
        archive_path: Path to the archive file.
        output_dir: Directory to extract into.

    Returns:
        Path: Path to extracted file (gz) or output_dir
            (zip/tar).

    Raises:
        ValueError: If archive format is not supported.
    """
    name = archive_path.name
    suffixes = archive_path.suffixes

    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return _extract_tar(archive_path, output_dir)
    elif name.endswith(".gz"):
        return _extract_gzip(archive_path, output_dir)
    elif name.endswith(".zip"):
        return _extract_zip(archive_path, output_dir)
    else:
        raise ValueError(
            f"Unsupported archive format: {name}"
        )


def _extract_gzip(gz_path: Path, output_dir: Path) -> Path:
    """Extract a .gz file, stripping the .gz extension.

    Args:
        gz_path: Path to gzip file.
        output_dir: Directory to write extracted file.

    Returns:
        Path: Path to extracted file.
    """
    stem = gz_path.name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    dest = output_dir / stem

    if dest.exists():
        logger.info("Already extracted: %s", dest.name)
        return dest

    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as f_in:
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info("Extracted: %s -> %s", gz_path.name, dest.name)
    return dest


def _extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract a .zip archive.

    Args:
        zip_path: Path to zip file.
        output_dir: Directory to extract into.

    Returns:
        Path: The output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    logger.info("Extracted: %s -> %s", zip_path.name, output_dir)
    return output_dir


def _extract_tar(tar_path: Path, output_dir: Path) -> Path:
    """Extract a .tgz or .tar.gz archive.

    Args:
        tar_path: Path to tar archive.
        output_dir: Directory to extract into.

    Returns:
        Path: The output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(output_dir, filter="data")

    logger.info("Extracted: %s -> %s", tar_path.name, output_dir)
    return output_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_extract.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extract.py tests/data/test_extract.py
git commit -m "feat: add archive decompression for gz, zip, and tar.gz"
```

---

## Task 3: Base Extractor ABC + Registry

**Files:**
- Create: `slm4ie/data/extractors/__init__.py`
- Test: `tests/data/extractors/__init__.py`, `tests/data/extractors/test_registry.py`

- [ ] **Step 1: Write failing tests for base extractor and registry**

```python
# tests/data/extractors/__init__.py
```

```python
# tests/data/extractors/test_registry.py
"""Tests for extractor base class and registry."""

from pathlib import Path
from typing import Iterator

import pytest

from slm4ie.data.extractors import (
    BaseExtractor,
    get_extractor,
    register_extractor,
)
from slm4ie.data.schema import Document


class _DummyExtractor(BaseExtractor):
    """Concrete extractor for testing."""

    def extract(
        self, input_dir: Path, source: str, domain: str
    ) -> Iterator[Document]:
        yield Document(
            text="test", source=source, domain=domain
        )


class TestBaseExtractor:
    """Tests for BaseExtractor ABC."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_concrete_subclass_works(self):
        ext = _DummyExtractor()
        docs = list(
            ext.extract(Path("."), "test", "web")
        )
        assert len(docs) == 1
        assert docs[0].text == "test"


class TestRegistry:
    """Tests for extractor registry."""

    def test_register_and_get(self):
        register_extractor("dummy", _DummyExtractor)
        ext = get_extractor("dummy")
        assert isinstance(ext, _DummyExtractor)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="no_such"):
            get_extractor("no_such")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/extractors/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement base extractor and registry**

```python
# slm4ie/data/extractors/__init__.py
"""Dataset text extractors — base class and registry."""

import abc
from pathlib import Path
from typing import Dict, Iterator, Type

from slm4ie.data.schema import Document

_REGISTRY: Dict[str, Type["BaseExtractor"]] = {}


class BaseExtractor(abc.ABC):
    """Abstract base class for dataset text extractors."""

    @abc.abstractmethod
    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extract documents from a raw dataset directory.

        Args:
            input_dir: Path to the raw dataset files.
            source: Dataset key identifier.
            domain: Domain category label.

        Yields:
            Document: Extracted documents.
        """


def register_extractor(
    name: str, cls: Type[BaseExtractor]
) -> None:
    """Register an extractor class by name.

    Args:
        name: Extractor identifier.
        cls: Extractor class to register.
    """
    _REGISTRY[name] = cls


def get_extractor(name: str) -> BaseExtractor:
    """Get an extractor instance by name.

    Args:
        name: Registered extractor identifier.

    Returns:
        BaseExtractor: An instance of the extractor.

    Raises:
        KeyError: If no extractor registered for name.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown extractor: '{name}'"
        )
    return _REGISTRY[name]()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/extractors/test_registry.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extractors/__init__.py \
        tests/data/extractors/__init__.py \
        tests/data/extractors/test_registry.py
git commit -m "feat: add BaseExtractor ABC and extractor registry"
```

---

## Task 4: CoNLL-U Extractor

Handles: CLASSLAWiki-sl, OSS, KZB, Solar (`.conllu` / `.conll` files).

**Files:**
- Create: `slm4ie/data/extractors/conllu.py`
- Test: `tests/data/extractors/test_conllu.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/extractors/test_conllu.py
"""Tests for CoNLL-U extractor."""

from pathlib import Path

from slm4ie.data.extractors.conllu import ConlluExtractor


SAMPLE_CONLLU = """\
# newdoc id = doc1
# sent_id = doc1.s1
# text = Predsednik je odprl sejo.
1\tPredsednik\tpredsednik\tNOUN\tNcmsn\tCase=Nom|Gender=Masc|Number=Sing\t3\tnsubj\t_\tNER=O
2\tje\tbiti\tAUX\tVa-r3s-n\tMood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin\t3\taux\t_\tNER=O
3\todprl\todpreti\tVERB\tVmep-sm\tGender=Masc|Number=Sing|VerbForm=Part\t0\troot\t_\tNER=O
4\tsejo\tseja\tNOUN\tNcfsa\tCase=Acc|Gender=Fem|Number=Sing\t3\tobj\t_\tNER=O
5\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t_\tNER=O

# sent_id = doc1.s2
# text = Seja je trajala dve uri.
1\tSeja\tseja\tNOUN\tNcfsn\tCase=Nom|Gender=Fem|Number=Sing\t3\tnsubj\t_\tNER=O
2\tje\tbiti\tAUX\tVa-r3s-n\tMood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin\t3\taux\t_\tNER=O
3\ttrajala\ttrajati\tVERB\tVmep-sf\tGender=Fem|Number=Sing|VerbForm=Part\t0\troot\t_\tNER=O
4\tdve\tdva\tNUM\tMlcfda\tCase=Acc|Gender=Fem|Number=Dual|NumType=Card\t5\tnummod\t_\tNER=O
5\turi\tura\tNOUN\tNcfda\tCase=Acc|Gender=Fem|Number=Dual\t3\tobl\t_\tNER=O
6\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t_\tSpaceAfter=No|NER=O

"""

SAMPLE_NER = """\
# sent_id = s1
# text = Janez Novak živi v Ljubljani.
1\tJanez\tJanez\tPROPN\tNpmsn\tCase=Nom|Gender=Masc|Number=Sing\t3\tnsubj\t_\tNER=B-PER
2\tNovak\tNovak\tPROPN\tNpmsn\tCase=Nom|Gender=Masc|Number=Sing\t1\tflat:name\t_\tNER=I-PER
3\tživi\tživeti\tVERB\tVmpr3s\tMood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin\t0\troot\t_\tNER=O
4\tv\tv\tADP\tSl\tCase=Loc\t5\tcase\t_\tNER=O
5\tLjubljani\tLjubljana\tPROPN\tNpfsl\tCase=Loc|Gender=Fem|Number=Sing\t3\tobl\t_\tNER=B-LOC
6\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t_\tSpaceAfter=No|NER=O

"""


class TestConlluExtractor:
    """Tests for ConlluExtractor."""

    def test_extracts_text(self, tmp_path: Path):
        (tmp_path / "test.conllu").write_text(SAMPLE_CONLLU)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "wiki"))
        assert len(docs) == 2
        assert docs[0].text == "Predsednik je odprl sejo."
        assert docs[1].text == "Seja je trajala dve uri."

    def test_extracts_tokens(self, tmp_path: Path):
        (tmp_path / "test.conllu").write_text(SAMPLE_CONLLU)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "wiki"))
        tokens = docs[0].annotations.tokens
        assert len(tokens) == 5
        assert tokens[0].form == "Predsednik"
        assert tokens[0].lemma == "predsednik"
        assert tokens[0].upos == "NOUN"
        assert tokens[0].xpos == "Ncmsn"
        assert tokens[0].feats == (
            "Case=Nom|Gender=Masc|Number=Sing"
        )
        assert tokens[0].head == 3
        assert tokens[0].deprel == "nsubj"
        assert tokens[0].ner == "O"

    def test_extracts_ner_tags(self, tmp_path: Path):
        (tmp_path / "test.conllu").write_text(SAMPLE_NER)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "web"))
        tokens = docs[0].annotations.tokens
        assert tokens[0].ner == "B-PER"
        assert tokens[1].ner == "I-PER"
        assert tokens[4].ner == "B-LOC"

    def test_doc_id_from_sent_id(self, tmp_path: Path):
        (tmp_path / "test.conllu").write_text(SAMPLE_CONLLU)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "wiki"))
        assert docs[0].doc_id == "doc1.s1"
        assert docs[1].doc_id == "doc1.s2"

    def test_sentence_boundaries(self, tmp_path: Path):
        (tmp_path / "test.conllu").write_text(SAMPLE_CONLLU)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "wiki"))
        assert docs[0].annotations.sentences == [[0, 4]]
        assert docs[1].annotations.sentences == [[0, 5]]

    def test_source_and_domain(self, tmp_path: Path):
        (tmp_path / "test.conllu").write_text(SAMPLE_CONLLU)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "wiki"))
        assert docs[0].source == "test_ds"
        assert docs[0].domain == "wiki"

    def test_processes_multiple_files(self, tmp_path: Path):
        (tmp_path / "a.conllu").write_text(SAMPLE_NER)
        (tmp_path / "b.conll").write_text(SAMPLE_NER)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "web"))
        assert len(docs) == 2

    def test_handles_underscore_feats(self, tmp_path: Path):
        content = """\
# sent_id = s1
# text = .
1\t.\t.\tPUNCT\tZ\t_\t0\troot\t_\t_

"""
        (tmp_path / "test.conllu").write_text(content)
        ext = ConlluExtractor()
        docs = list(ext.extract(tmp_path, "test_ds", "web"))
        assert docs[0].annotations.tokens[0].feats is None
        assert docs[0].annotations.tokens[0].ner is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/extractors/test_conllu.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CoNLL-U extractor**

```python
# slm4ie/data/extractors/conllu.py
"""CoNLL-U format text extractor.

Handles .conllu and .conll files. Extracts text from
# text = comments and full token annotations from tab-
separated columns.
"""

import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

from slm4ie.data.extractors import (
    BaseExtractor,
    register_extractor,
)
from slm4ie.data.schema import Annotations, Document, Token

logger = logging.getLogger(__name__)

_NER_RE = re.compile(r"NER=([^\s|]+)")


def _parse_ner(misc: str) -> Optional[str]:
    """Extract NER tag from the misc column.

    Args:
        misc: The CoNLL-U misc column value.

    Returns:
        Optional[str]: NER tag if present, else None.
    """
    if misc == "_":
        return None
    m = _NER_RE.search(misc)
    return m.group(1) if m else None


def _blank_to_none(value: str) -> Optional[str]:
    """Convert CoNLL-U underscore placeholder to None.

    Args:
        value: Column value.

    Returns:
        Optional[str]: None if underscore, else value.
    """
    return None if value == "_" else value


class ConlluExtractor(BaseExtractor):
    """Extracts documents from CoNLL-U formatted files.

    Reads .conllu and .conll files, parsing # text = lines
    for raw text and token rows for linguistic annotations.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extract annotated documents from CoNLL-U files.

        Args:
            input_dir: Directory containing .conllu/.conll files.
            source: Dataset key identifier.
            domain: Domain category label.

        Yields:
            Document: One document per sentence.
        """
        files = sorted(
            f
            for f in input_dir.iterdir()
            if f.suffix in (".conllu", ".conll")
            and f.is_file()
        )
        for filepath in files:
            yield from self._parse_file(
                filepath, source, domain
            )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse a single CoNLL-U file.

        Args:
            filepath: Path to the CoNLL-U file.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per sentence block.
        """
        text: Optional[str] = None
        sent_id: Optional[str] = None
        tokens: List[Token] = []

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                if line.startswith("# text = "):
                    text = line[9:]
                elif line.startswith("# sent_id = "):
                    sent_id = line[12:]
                elif line == "" and tokens:
                    yield Document(
                        text=text or "",
                        source=source,
                        domain=domain,
                        doc_id=sent_id,
                        annotations=Annotations(
                            tokens=tokens,
                            sentences=[
                                [0, len(tokens) - 1]
                            ],
                        ),
                    )
                    text = None
                    sent_id = None
                    tokens = []
                elif line and not line.startswith("#"):
                    cols = line.split("\t")
                    if len(cols) < 10:
                        continue
                    # Skip multiword token lines (e.g. "1-2")
                    if "-" in cols[0] or "." in cols[0]:
                        continue
                    head_val = _blank_to_none(cols[6])
                    tokens.append(
                        Token(
                            form=cols[1],
                            lemma=_blank_to_none(cols[2]),
                            upos=_blank_to_none(cols[3]),
                            xpos=_blank_to_none(cols[4]),
                            feats=_blank_to_none(cols[5]),
                            head=(
                                int(head_val)
                                if head_val
                                else None
                            ),
                            deprel=_blank_to_none(cols[7]),
                            ner=_parse_ner(cols[9]),
                        )
                    )

        # Handle file not ending with blank line
        if tokens:
            yield Document(
                text=text or "",
                source=source,
                domain=domain,
                doc_id=sent_id,
                annotations=Annotations(
                    tokens=tokens,
                    sentences=[[0, len(tokens) - 1]],
                ),
            )


register_extractor("conllu", ConlluExtractor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/extractors/test_conllu.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extractors/conllu.py \
        tests/data/extractors/test_conllu.py
git commit -m "feat: add CoNLL-U extractor with full annotation support"
```

---

## Task 5: JSONL Extractor (CLASSLA-web)

Handles: CLASSLA-web.sl 2.0 (`.jsonl` files with annotated content).

**Files:**
- Create: `slm4ie/data/extractors/jsonl.py`
- Test: `tests/data/extractors/test_jsonl.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/extractors/test_jsonl.py
"""Tests for JSONL extractor."""

import json
from pathlib import Path

from slm4ie.data.extractors.jsonl import JsonlExtractor


def _write_jsonl(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


class TestJsonlExtractor:
    """Tests for JsonlExtractor."""

    def test_extracts_text_field(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "data.jsonl",
            [{"text": "Dober dan."}, {"text": "Nasvidenje."}],
        )
        ext = JsonlExtractor()
        docs = list(ext.extract(tmp_path, "classla", "web"))
        assert len(docs) == 2
        assert docs[0].text == "Dober dan."
        assert docs[1].text == "Nasvidenje."

    def test_preserves_annotations(self, tmp_path: Path):
        record = {
            "text": "Janez spi.",
            "paragraphs": [
                {
                    "sentences": [
                        {
                            "tokens": [
                                {
                                    "form": "Janez",
                                    "lemma": "Janez",
                                    "upos": "PROPN",
                                    "xpos": "Npmsn",
                                    "feats": "Case=Nom",
                                    "head": 2,
                                    "deprel": "nsubj",
                                    "ner": "B-PER",
                                },
                                {
                                    "form": "spi",
                                    "lemma": "spati",
                                    "upos": "VERB",
                                    "xpos": "Vmpr3s",
                                    "feats": "Mood=Ind",
                                    "head": 0,
                                    "deprel": "root",
                                    "ner": "O",
                                },
                                {
                                    "form": ".",
                                    "lemma": ".",
                                    "upos": "PUNCT",
                                    "xpos": "Z",
                                    "head": 2,
                                    "deprel": "punct",
                                    "ner": "O",
                                },
                            ]
                        }
                    ]
                }
            ],
        }
        _write_jsonl(tmp_path / "data.jsonl", [record])
        ext = JsonlExtractor()
        docs = list(ext.extract(tmp_path, "classla", "web"))
        assert docs[0].annotations is not None
        tokens = docs[0].annotations.tokens
        assert tokens[0].form == "Janez"
        assert tokens[0].ner == "B-PER"
        assert tokens[0].upos == "PROPN"

    def test_text_only_no_annotations(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "data.jsonl",
            [{"text": "Hello."}],
        )
        ext = JsonlExtractor()
        docs = list(ext.extract(tmp_path, "test", "web"))
        assert docs[0].annotations is None

    def test_skips_empty_text(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "data.jsonl",
            [{"text": ""}, {"text": "Valid."}],
        )
        ext = JsonlExtractor()
        docs = list(ext.extract(tmp_path, "test", "web"))
        assert len(docs) == 1
        assert docs[0].text == "Valid."

    def test_preserves_metadata_fields(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "data.jsonl",
            [
                {
                    "text": "Hello.",
                    "url": "http://example.com",
                    "doc_id": "d1",
                }
            ],
        )
        ext = JsonlExtractor()
        docs = list(ext.extract(tmp_path, "test", "web"))
        assert docs[0].metadata.get("url") == (
            "http://example.com"
        )
        assert docs[0].doc_id == "d1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/extractors/test_jsonl.py -v`
Expected: FAIL

- [ ] **Step 3: Implement JSONL extractor**

```python
# slm4ie/data/extractors/jsonl.py
"""JSONL format extractor for CLASSLA-web annotated data."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from slm4ie.data.extractors import (
    BaseExtractor,
    register_extractor,
)
from slm4ie.data.schema import Annotations, Document, Token

logger = logging.getLogger(__name__)

# Fields that map to Document attributes, not metadata
_RESERVED_FIELDS = {"text", "paragraphs", "doc_id"}


def _parse_tokens_from_paragraphs(
    paragraphs: List[Dict],
) -> Optional[Annotations]:
    """Parse CLASSLA-style paragraph/sentence/token structure.

    Args:
        paragraphs: List of paragraph dicts with nested
            sentences and tokens.

    Returns:
        Optional[Annotations]: Annotations if tokens found.
    """
    all_tokens: List[Token] = []
    sentences: List[List[int]] = []

    for para in paragraphs:
        for sent in para.get("sentences", []):
            start = len(all_tokens)
            for tok in sent.get("tokens", []):
                all_tokens.append(
                    Token(
                        form=tok.get("form", ""),
                        lemma=tok.get("lemma"),
                        upos=tok.get("upos"),
                        xpos=tok.get("xpos"),
                        feats=tok.get("feats"),
                        head=tok.get("head"),
                        deprel=tok.get("deprel"),
                        ner=tok.get("ner"),
                    )
                )
            if all_tokens:
                sentences.append(
                    [start, len(all_tokens) - 1]
                )

    if not all_tokens:
        return None

    return Annotations(
        tokens=all_tokens, sentences=sentences
    )


class JsonlExtractor(BaseExtractor):
    """Extracts documents from JSONL files.

    Handles CLASSLA-web annotated JSONL with nested
    paragraphs/sentences/tokens structure. Also works
    with plain text JSONL (text field only).
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extract documents from .jsonl files.

        Args:
            input_dir: Directory containing .jsonl files.
            source: Dataset key identifier.
            domain: Domain category label.

        Yields:
            Document: One per JSONL record with non-empty text.
        """
        files = sorted(
            f
            for f in input_dir.iterdir()
            if f.suffix == ".jsonl" and f.is_file()
        )
        for filepath in files:
            yield from self._parse_file(
                filepath, source, domain
            )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse a single JSONL file.

        Args:
            filepath: Path to the JSONL file.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per valid JSONL line.
        """
        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON at "
                        "%s:%d",
                        filepath.name,
                        line_num,
                    )
                    continue

                text = record.get("text", "")
                if not text:
                    continue

                annotations = None
                paragraphs = record.get("paragraphs")
                if paragraphs:
                    annotations = (
                        _parse_tokens_from_paragraphs(
                            paragraphs
                        )
                    )

                metadata = {
                    k: v
                    for k, v in record.items()
                    if k not in _RESERVED_FIELDS
                }

                yield Document(
                    text=text,
                    source=source,
                    domain=domain,
                    doc_id=record.get("doc_id"),
                    metadata=metadata if metadata else {},
                    annotations=annotations,
                )


register_extractor("jsonl", JsonlExtractor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/extractors/test_jsonl.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extractors/jsonl.py \
        tests/data/extractors/test_jsonl.py
git commit -m "feat: add JSONL extractor for CLASSLA-web annotated data"
```

---

## Task 6: TEI XML Extractor

Handles: ParlaMint-SI, siParl, KAS, Janes-Forum, Janes-Blog, Janes-News.

**Files:**
- Create: `slm4ie/data/extractors/tei.py`
- Test: `tests/data/extractors/test_tei.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/extractors/test_tei.py
"""Tests for TEI XML extractor."""

from pathlib import Path

from slm4ie.data.extractors.tei import TeiExtractor


SAMPLE_TEI_ANNOTATED = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <u xml:id="u1" who="#speaker1">
        <seg xml:id="seg1">
          <s xml:id="s1">
            <w lemma="predsednik" msd="UPosTag=NOUN|Case=Nom"
               >Predsednik</w>
            <w lemma="biti" msd="UPosTag=AUX">je</w>
            <w lemma="odpreti" msd="UPosTag=VERB">odprl</w>
            <w lemma="seja" msd="UPosTag=NOUN|Case=Acc"
               >sejo</w>
            <pc msd="UPosTag=PUNCT">.</pc>
          </s>
        </seg>
      </u>
    </body>
  </text>
</TEI>
"""

SAMPLE_TEI_NER = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <p xml:id="p1">
        <s xml:id="s1">
          <name type="PER">
            <w lemma="Janez" msd="UPosTag=PROPN">Janez</w>
            <w lemma="Novak" msd="UPosTag=PROPN">Novak</w>
          </name>
          <w lemma="živeti" msd="UPosTag=VERB">živi</w>
          <w lemma="v" msd="UPosTag=ADP">v</w>
          <name type="LOC">
            <w lemma="Ljubljana" msd="UPosTag=PROPN"
               >Ljubljani</w>
          </name>
          <pc msd="UPosTag=PUNCT">.</pc>
        </s>
      </p>
    </body>
  </text>
</TEI>
"""

SAMPLE_TEI_PLAIN = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <p>Dober dan. Kako ste?</p>
      <p>Hvala, dobro.</p>
    </body>
  </text>
</TEI>
"""


class TestTeiExtractor:
    """Tests for TeiExtractor."""

    def test_extracts_text_from_w_elements(
        self, tmp_path: Path
    ):
        (tmp_path / "test.xml").write_text(
            SAMPLE_TEI_ANNOTATED
        )
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "parlamint", "parliamentary")
        )
        assert len(docs) == 1
        assert docs[0].text == "Predsednik je odprl sejo ."

    def test_extracts_lemma_and_msd(self, tmp_path: Path):
        (tmp_path / "test.xml").write_text(
            SAMPLE_TEI_ANNOTATED
        )
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "parlamint", "parliamentary")
        )
        tokens = docs[0].annotations.tokens
        assert tokens[0].form == "Predsednik"
        assert tokens[0].lemma == "predsednik"
        assert "NOUN" in tokens[0].upos

    def test_extracts_ner_from_name_elements(
        self, tmp_path: Path
    ):
        (tmp_path / "test.xml").write_text(SAMPLE_TEI_NER)
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "janes", "forum")
        )
        tokens = docs[0].annotations.tokens
        assert tokens[0].ner == "B-PER"
        assert tokens[1].ner == "I-PER"
        assert tokens[2].ner == "O"
        assert tokens[4].ner == "B-LOC"

    def test_extracts_plain_text_paragraphs(
        self, tmp_path: Path
    ):
        (tmp_path / "test.xml").write_text(SAMPLE_TEI_PLAIN)
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "test", "web")
        )
        assert len(docs) == 2
        assert docs[0].text == "Dober dan. Kako ste?"
        assert docs[1].text == "Hvala, dobro."
        assert docs[0].annotations is None

    def test_doc_id_from_xml_id(self, tmp_path: Path):
        (tmp_path / "test.xml").write_text(
            SAMPLE_TEI_ANNOTATED
        )
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "test", "parliamentary")
        )
        assert docs[0].doc_id is not None

    def test_processes_multiple_xml_files(
        self, tmp_path: Path
    ):
        (tmp_path / "a.xml").write_text(SAMPLE_TEI_PLAIN)
        (tmp_path / "b.xml").write_text(SAMPLE_TEI_PLAIN)
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "test", "web")
        )
        assert len(docs) == 4

    def test_handles_nested_dirs(self, tmp_path: Path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "test.xml").write_text(SAMPLE_TEI_PLAIN)
        ext = TeiExtractor()
        docs = list(
            ext.extract(tmp_path, "test", "web")
        )
        assert len(docs) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/extractors/test_tei.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TEI XML extractor**

```python
# slm4ie/data/extractors/tei.py
"""TEI XML text extractor.

Handles annotated TEI (ParlaMint, siParl, KAS, Janes-*)
and plain TEI files. Extracts tokens from <w>/<pc> elements
with lemma/MSD annotations, and NER from <name> elements.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from slm4ie.data.extractors import (
    BaseExtractor,
    register_extractor,
)
from slm4ie.data.schema import Annotations, Document, Token

logger = logging.getLogger(__name__)

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_XML_NS = "http://www.w3.org/XML/1998/namespace"


def _parse_msd(msd: Optional[str]) -> Tuple[
    Optional[str], Optional[str]
]:
    """Extract UPOS from MSD attribute.

    MSD format: "UPosTag=NOUN|Case=Nom|..." where first
    feature is UPosTag.

    Args:
        msd: MSD attribute string.

    Returns:
        Tuple of (upos, remaining_feats).
    """
    if not msd:
        return None, None
    parts = msd.split("|")
    upos = None
    feats = []
    for part in parts:
        if part.startswith("UPosTag="):
            upos = part.split("=", 1)[1]
        else:
            feats.append(part)
    feat_str = "|".join(feats) if feats else None
    return upos, feat_str


class TeiExtractor(BaseExtractor):
    """Extracts documents from TEI XML files.

    Supports two modes:
    1. Annotated TEI: extracts <w>/<pc> tokens with
       lemma, MSD, and NER from <name> elements.
    2. Plain TEI: extracts text from <p> elements.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extract documents from .xml files recursively.

        Args:
            input_dir: Directory containing TEI XML files.
            source: Dataset key identifier.
            domain: Domain category label.

        Yields:
            Document: One per sentence/segment (annotated)
                or paragraph (plain).
        """
        files = sorted(input_dir.rglob("*.xml"))
        for filepath in files:
            try:
                yield from self._parse_file(
                    filepath, source, domain
                )
            except ET.ParseError:
                logger.warning(
                    "Failed to parse XML: %s", filepath
                )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse a single TEI XML file.

        Detects annotated vs plain by checking for <w>
        elements.

        Args:
            filepath: Path to the XML file.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: Extracted documents.
        """
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Check if annotated (has <w> elements)
        w_elements = root.findall(
            f".//{{{_TEI_NS}}}w"
        )
        if w_elements:
            yield from self._parse_annotated(
                root, source, domain
            )
        else:
            yield from self._parse_plain(
                root, source, domain
            )

    def _parse_annotated(
        self,
        root: ET.Element,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse annotated TEI with <w>/<pc> tokens.

        Looks for <s> (sentence) elements and extracts
        tokens from <w> and <pc> children, handling <name>
        wrappers for NER.

        Args:
            root: XML root element.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per <s> element.
        """
        for s_elem in root.iter(f"{{{_TEI_NS}}}s"):
            tokens, text_parts = (
                self._extract_tokens_from_sentence(s_elem)
            )
            if not tokens:
                continue

            text = " ".join(text_parts)
            doc_id = s_elem.get(f"{{{_XML_NS}}}id")

            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=doc_id,
                annotations=Annotations(
                    tokens=tokens,
                    sentences=[[0, len(tokens) - 1]],
                ),
            )

    def _extract_tokens_from_sentence(
        self, s_elem: ET.Element
    ) -> Tuple[List[Token], List[str]]:
        """Extract tokens from a <s> element.

        Handles <name> wrappers for NER tagging. Tokens
        inside <name type="X"> get B-X/I-X tags.

        Args:
            s_elem: The <s> XML element.

        Returns:
            Tuple of (tokens list, text parts list).
        """
        tokens: List[Token] = []
        text_parts: List[str] = []

        for child in s_elem:
            tag = child.tag.split("}")[-1]

            if tag in ("w", "pc"):
                token = self._make_token(child, ner="O")
                tokens.append(token)
                text_parts.append(token.form)

            elif tag == "name":
                ner_type = child.get("type", "MISC")
                first = True
                for w_elem in child:
                    w_tag = w_elem.tag.split("}")[-1]
                    if w_tag in ("w", "pc"):
                        prefix = "B" if first else "I"
                        ner = f"{prefix}-{ner_type}"
                        token = self._make_token(
                            w_elem, ner=ner
                        )
                        tokens.append(token)
                        text_parts.append(token.form)
                        first = False

        return tokens, text_parts

    def _make_token(
        self,
        elem: ET.Element,
        ner: Optional[str] = None,
    ) -> Token:
        """Create a Token from a <w> or <pc> element.

        Args:
            elem: The XML element.
            ner: NER tag to assign.

        Returns:
            Token: Populated token.
        """
        form = (elem.text or "").strip()
        lemma = elem.get("lemma")
        msd = elem.get("msd")
        upos, feats = _parse_msd(msd)

        return Token(
            form=form,
            lemma=lemma,
            upos=upos,
            feats=feats,
            ner=ner,
        )

    def _parse_plain(
        self,
        root: ET.Element,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse plain TEI, extracting text from <p> elements.

        Args:
            root: XML root element.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per <p> element with text.
        """
        for p_elem in root.iter(f"{{{_TEI_NS}}}p"):
            text = "".join(p_elem.itertext()).strip()
            if not text:
                continue

            doc_id = p_elem.get(f"{{{_XML_NS}}}id")

            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=doc_id,
            )


register_extractor("tei", TeiExtractor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/extractors/test_tei.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extractors/tei.py \
        tests/data/extractors/test_tei.py
git commit -m "feat: add TEI XML extractor with NER and MSD annotation support"
```

---

## Task 7: MaCoCu XML Extractor

Handles: MaCoCu-sl 2.0 (`.xml` bilingual corpus format).

**Files:**
- Create: `slm4ie/data/extractors/macocu.py`
- Test: `tests/data/extractors/test_macocu.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/extractors/test_macocu.py
"""Tests for MaCoCu XML extractor."""

from pathlib import Path

from slm4ie.data.extractors.macocu import MacocuExtractor


SAMPLE_MACOCU = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE cesDoc SYSTEM "MaCoCu-monolingual.dtd">
<cesDoc version="4">
  <cesHeader/>
  <text>
    <group>
      <tu id="1" score="0.95">
        <tuv lang="sl">
          <p>Dober dan. Kako ste?</p>
        </tuv>
      </tu>
      <tu id="2" score="0.42">
        <tuv lang="sl">
          <p>Slab dokument.</p>
        </tuv>
      </tu>
      <tu id="3" score="0.80">
        <tuv lang="sl">
          <p>Slovenija je lepa dežela.</p>
        </tuv>
      </tu>
    </group>
  </text>
</cesDoc>
"""


class TestMacocuExtractor:
    """Tests for MacocuExtractor."""

    def test_extracts_all_tu_elements(self, tmp_path: Path):
        (tmp_path / "corpus.xml").write_text(SAMPLE_MACOCU)
        ext = MacocuExtractor()
        docs = list(ext.extract(tmp_path, "macocu_sl", "web"))
        assert len(docs) == 3

    def test_extracts_text_content(self, tmp_path: Path):
        (tmp_path / "corpus.xml").write_text(SAMPLE_MACOCU)
        ext = MacocuExtractor()
        docs = list(ext.extract(tmp_path, "macocu_sl", "web"))
        assert docs[0].text == "Dober dan. Kako ste?"
        assert docs[2].text == "Slovenija je lepa dežela."

    def test_preserves_score_in_metadata(
        self, tmp_path: Path
    ):
        (tmp_path / "corpus.xml").write_text(SAMPLE_MACOCU)
        ext = MacocuExtractor()
        docs = list(ext.extract(tmp_path, "macocu_sl", "web"))
        assert docs[0].metadata["score"] == "0.95"

    def test_doc_id_from_tu_id(self, tmp_path: Path):
        (tmp_path / "corpus.xml").write_text(SAMPLE_MACOCU)
        ext = MacocuExtractor()
        docs = list(ext.extract(tmp_path, "macocu_sl", "web"))
        assert docs[0].doc_id == "1"

    def test_source_and_domain(self, tmp_path: Path):
        (tmp_path / "corpus.xml").write_text(SAMPLE_MACOCU)
        ext = MacocuExtractor()
        docs = list(ext.extract(tmp_path, "macocu_sl", "web"))
        assert docs[0].source == "macocu_sl"
        assert docs[0].domain == "web"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/extractors/test_macocu.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MaCoCu extractor**

```python
# slm4ie/data/extractors/macocu.py
"""MaCoCu monolingual corpus XML extractor.

Handles MaCoCu-sl 2.0 XML format with <tu> translation
units containing <p> text elements and quality scores.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from slm4ie.data.extractors import (
    BaseExtractor,
    register_extractor,
)
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)


class MacocuExtractor(BaseExtractor):
    """Extracts documents from MaCoCu XML corpus files.

    Reads <tu> elements, extracting text from nested <p>
    and preserving quality scores as metadata.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extract documents from .xml files.

        Args:
            input_dir: Directory with MaCoCu XML files.
            source: Dataset key identifier.
            domain: Domain category label.

        Yields:
            Document: One per <tu> element.
        """
        files = sorted(
            f
            for f in input_dir.iterdir()
            if f.suffix == ".xml" and f.is_file()
        )
        for filepath in files:
            try:
                yield from self._parse_file(
                    filepath, source, domain
                )
            except ET.ParseError:
                logger.warning(
                    "Failed to parse XML: %s", filepath
                )

    def _parse_file(
        self,
        filepath: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Parse a single MaCoCu XML file.

        Args:
            filepath: Path to the XML file.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per <tu> element with text.
        """
        tree = ET.parse(filepath)
        root = tree.getroot()

        for tu in root.iter("tu"):
            tu_id = tu.get("id")
            score = tu.get("score")

            text_parts = []
            for p in tu.iter("p"):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    text_parts.append(p_text)

            text = " ".join(text_parts)
            if not text:
                continue

            metadata = {}
            if score is not None:
                metadata["score"] = score

            yield Document(
                text=text,
                source=source,
                domain=domain,
                doc_id=tu_id,
                metadata=metadata,
            )


register_extractor("macocu", MacocuExtractor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/extractors/test_macocu.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extractors/macocu.py \
        tests/data/extractors/test_macocu.py
git commit -m "feat: add MaCoCu XML extractor with quality scores"
```

---

## Task 8: HuggingFace Arrow Extractor

Handles: FinePDF, FineWeb-2, CulturaX, Legal-mC4, mC4, HPLT.

**Files:**
- Create: `slm4ie/data/extractors/huggingface.py`
- Test: `tests/data/extractors/test_huggingface.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/extractors/test_huggingface.py
"""Tests for HuggingFace Arrow extractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from slm4ie.data.extractors.huggingface import (
    HuggingFaceExtractor,
)


class TestHuggingFaceExtractor:
    """Tests for HuggingFaceExtractor."""

    @patch("slm4ie.data.extractors.huggingface.load_from_disk")
    def test_extracts_text_column(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        config_dir = tmp_path / "sl"
        config_dir.mkdir()

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"text": "Hello.", "url": "http://a.com"},
                    {"text": "World.", "url": "http://b.com"},
                ]
            )
        )
        mock_ds.column_names = ["text", "url"]
        mock_load.return_value = mock_ds

        ext = HuggingFaceExtractor()
        docs = list(ext.extract(tmp_path, "fineweb2", "web"))
        assert len(docs) == 2
        assert docs[0].text == "Hello."
        assert docs[1].text == "World."

    @patch("slm4ie.data.extractors.huggingface.load_from_disk")
    def test_preserves_metadata_columns(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        config_dir = tmp_path / "sl"
        config_dir.mkdir()

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "text": "Hello.",
                        "url": "http://a.com",
                        "language_score": 0.95,
                    },
                ]
            )
        )
        mock_ds.column_names = [
            "text",
            "url",
            "language_score",
        ]
        mock_load.return_value = mock_ds

        ext = HuggingFaceExtractor()
        docs = list(ext.extract(tmp_path, "fineweb2", "web"))
        assert docs[0].metadata["url"] == "http://a.com"
        assert docs[0].metadata["language_score"] == 0.95

    @patch("slm4ie.data.extractors.huggingface.load_from_disk")
    def test_skips_empty_text(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        config_dir = tmp_path / "sl"
        config_dir.mkdir()

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter(
                [{"text": ""}, {"text": "Valid."}]
            )
        )
        mock_ds.column_names = ["text"]
        mock_load.return_value = mock_ds

        ext = HuggingFaceExtractor()
        docs = list(ext.extract(tmp_path, "test", "web"))
        assert len(docs) == 1

    @patch("slm4ie.data.extractors.huggingface.load_from_disk")
    def test_handles_dataset_dict(
        self, mock_load: MagicMock, tmp_path: Path
    ):
        config_dir = tmp_path / "sl"
        config_dir.mkdir()

        mock_split = MagicMock()
        mock_split.__iter__ = MagicMock(
            return_value=iter([{"text": "Train row."}])
        )
        mock_split.column_names = ["text"]

        mock_dd = MagicMock()
        mock_dd.__contains__ = MagicMock(return_value=True)
        mock_dd.__getitem__ = MagicMock(
            return_value=mock_split
        )
        mock_dd.keys = MagicMock(
            return_value=["train"]
        )
        # Distinguish DatasetDict from Dataset
        mock_dd.column_names = {"train": ["text"]}
        mock_load.return_value = mock_dd

        ext = HuggingFaceExtractor()
        docs = list(ext.extract(tmp_path, "test", "web"))
        assert len(docs) == 1
        assert docs[0].text == "Train row."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/extractors/test_huggingface.py -v`
Expected: FAIL

- [ ] **Step 3: Implement HuggingFace extractor**

```python
# slm4ie/data/extractors/huggingface.py
"""HuggingFace Arrow dataset extractor.

Reads datasets saved via save_to_disk() (Arrow format)
and extracts text + metadata columns.
"""

import logging
from pathlib import Path
from typing import Iterator

from datasets import load_from_disk

from slm4ie.data.extractors import (
    BaseExtractor,
    register_extractor,
)
from slm4ie.data.schema import Document

logger = logging.getLogger(__name__)


class HuggingFaceExtractor(BaseExtractor):
    """Extracts documents from HuggingFace Arrow datasets.

    Reads datasets saved to disk by the download step.
    Extracts the 'text' column and preserves all other
    columns as metadata.
    """

    def extract(
        self,
        input_dir: Path,
        source: str,
        domain: str,
    ) -> Iterator[Document]:
        """Extract documents from Arrow dataset directories.

        Iterates over subdirectories (one per HF config)
        and loads each as a dataset.

        Args:
            input_dir: Directory containing config subdirs.
            source: Dataset key identifier.
            domain: Domain category label.

        Yields:
            Document: One per dataset row with non-empty text.
        """
        config_dirs = sorted(
            d for d in input_dir.iterdir() if d.is_dir()
        )
        for config_dir in config_dirs:
            try:
                ds = load_from_disk(str(config_dir))
            except Exception:
                logger.warning(
                    "Failed to load Arrow dataset: %s",
                    config_dir,
                )
                continue

            yield from self._iterate_dataset(
                ds, source, domain
            )

    def _iterate_dataset(
        self, ds, source: str, domain: str
    ) -> Iterator[Document]:
        """Iterate over a dataset or DatasetDict.

        Args:
            ds: A Dataset or DatasetDict instance.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per row.
        """
        # DatasetDict has dict-typed column_names
        if isinstance(ds.column_names, dict):
            for split_name in ds.keys():
                yield from self._iterate_split(
                    ds[split_name], source, domain
                )
        else:
            yield from self._iterate_split(
                ds, source, domain
            )

    def _iterate_split(
        self, split, source: str, domain: str
    ) -> Iterator[Document]:
        """Iterate over a single dataset split.

        Args:
            split: A Dataset split.
            source: Dataset key.
            domain: Domain label.

        Yields:
            Document: One per row with non-empty text.
        """
        meta_cols = [
            c
            for c in split.column_names
            if c != "text"
        ]

        for row in split:
            text = row.get("text", "")
            if not text:
                continue

            metadata = {
                k: row[k]
                for k in meta_cols
                if row.get(k) is not None
            }

            yield Document(
                text=text,
                source=source,
                domain=domain,
                metadata=metadata if metadata else {},
            )


register_extractor("huggingface", HuggingFaceExtractor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/extractors/test_huggingface.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add slm4ie/data/extractors/huggingface.py \
        tests/data/extractors/test_huggingface.py
git commit -m "feat: add HuggingFace Arrow extractor with metadata preservation"
```

---

## Task 9: Extraction Config + Orchestrator

**Files:**
- Create: `configs/data/extract.yaml`
- Modify: `slm4ie/data/processing.py`
- Test: `tests/data/test_processing.py`

- [ ] **Step 1: Create extraction config**

```yaml
# configs/data/extract.yaml
input_dir: data/raw
output_dir: data/processed

datasets:
  classla_web_sl:
    extractor: jsonl
    domain: web
    decompress: [gz]

  classlawiki_sl:
    extractor: conllu
    domain: wiki
    decompress: [gz]

  macocu_sl:
    extractor: macocu
    domain: web
    decompress: [zip]

  parlamint_si:
    extractor: tei
    domain: parliamentary
    decompress: [tgz]

  kas:
    extractor: tei
    domain: academic
    decompress: [tar.gz]

  coleslaw:
    extractor: tei
    domain: legal
    decompress: [zip]

  povejmo_vemo_med:
    extractor: tei
    domain: medical
    decompress: [zip]

  oss:
    extractor: conllu
    domain: scientific
    decompress: [zip]

  siparl:
    extractor: tei
    domain: parliamentary
    decompress: [zip]

  janes_forum:
    extractor: tei
    domain: forum
    decompress: [zip]

  janes_blog:
    extractor: tei
    domain: blog
    decompress: [zip]

  janes_news:
    extractor: tei
    domain: news
    decompress: [zip]

  kzb:
    extractor: conllu
    domain: scientific
    decompress: [zip]

  solar:
    extractor: conllu
    domain: student
    decompress: [zip]

  finepdf:
    extractor: huggingface
    domain: web

  fineweb2:
    extractor: huggingface
    domain: web

  culturax:
    extractor: huggingface
    domain: web

  legal_mc4:
    extractor: huggingface
    domain: legal

  c4:
    extractor: huggingface
    domain: web

  hplt:
    extractor: huggingface
    domain: web
```

- [ ] **Step 2: Write failing tests for orchestrator**

```python
# tests/data/test_processing.py
"""Tests for slm4ie.data.processing module."""

import json
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest
import yaml

from slm4ie.data.extractors import (
    BaseExtractor,
    register_extractor,
)
from slm4ie.data.processing import (
    ExtractionConfig,
    extract_datasets,
    load_extraction_config,
)
from slm4ie.data.schema import Document


class _StubExtractor(BaseExtractor):
    def extract(
        self, input_dir: Path, source: str, domain: str
    ) -> Iterator[Document]:
        yield Document(
            text="stub", source=source, domain=domain
        )


register_extractor("stub", _StubExtractor)


class TestLoadExtractionConfig:
    """Tests for load_extraction_config."""

    def test_loads_config(self, tmp_path: Path):
        config_data = {
            "input_dir": "data/raw",
            "output_dir": "data/processed",
            "datasets": {
                "ds1": {
                    "extractor": "stub",
                    "domain": "web",
                },
            },
        }
        cfg_file = tmp_path / "extract.yaml"
        cfg_file.write_text(yaml.dump(config_data))
        cfg = load_extraction_config(cfg_file)
        assert cfg.input_dir == "data/raw"
        assert cfg.output_dir == "data/processed"
        assert "ds1" in cfg.datasets
        assert cfg.datasets["ds1"]["extractor"] == "stub"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_extraction_config(
                Path("/nonexistent.yaml")
            )


class TestExtractDatasets:
    """Tests for extract_datasets orchestrator."""

    def test_extracts_single_dataset(self, tmp_path: Path):
        raw = tmp_path / "raw" / "ds1"
        raw.mkdir(parents=True)
        (raw / "dummy.txt").write_text("placeholder")

        config_data = {
            "input_dir": str(tmp_path / "raw"),
            "output_dir": str(tmp_path / "processed"),
            "datasets": {
                "ds1": {
                    "extractor": "stub",
                    "domain": "web",
                },
            },
        }
        cfg_file = tmp_path / "extract.yaml"
        cfg_file.write_text(yaml.dump(config_data))

        extract_datasets(cfg_file)

        output = tmp_path / "processed" / "ds1.jsonl"
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["text"] == "stub"
        assert parsed["source"] == "ds1"
        assert parsed["domain"] == "web"

    def test_extracts_selected_datasets(
        self, tmp_path: Path
    ):
        for name in ("ds1", "ds2"):
            d = tmp_path / "raw" / name
            d.mkdir(parents=True)
            (d / "f.txt").write_text("x")

        config_data = {
            "input_dir": str(tmp_path / "raw"),
            "output_dir": str(tmp_path / "processed"),
            "datasets": {
                "ds1": {
                    "extractor": "stub",
                    "domain": "web",
                },
                "ds2": {
                    "extractor": "stub",
                    "domain": "legal",
                },
            },
        }
        cfg_file = tmp_path / "extract.yaml"
        cfg_file.write_text(yaml.dump(config_data))

        extract_datasets(cfg_file, dataset_keys=["ds2"])

        assert not (
            tmp_path / "processed" / "ds1.jsonl"
        ).exists()
        assert (
            tmp_path / "processed" / "ds2.jsonl"
        ).exists()

    def test_unknown_key_raises(self, tmp_path: Path):
        config_data = {
            "input_dir": str(tmp_path / "raw"),
            "output_dir": str(tmp_path / "processed"),
            "datasets": {},
        }
        cfg_file = tmp_path / "extract.yaml"
        cfg_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="Unknown"):
            extract_datasets(
                cfg_file, dataset_keys=["missing"]
            )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_processing.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Implement orchestrator in processing.py**

```python
# slm4ie/data/processing.py
"""Data cleaning, formatting, and splitting utilities."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Import extractors to trigger registration
import slm4ie.data.extractors.conllu  # noqa: F401
import slm4ie.data.extractors.huggingface  # noqa: F401
import slm4ie.data.extractors.jsonl  # noqa: F401
import slm4ie.data.extractors.macocu  # noqa: F401
import slm4ie.data.extractors.tei  # noqa: F401
from slm4ie.data.extractors import get_extractor

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for dataset extraction pipeline.

    Attributes:
        input_dir: Base directory for raw datasets.
        output_dir: Base directory for processed output.
        datasets: Dict mapping dataset key to config dict
            with 'extractor' and 'domain' keys.
    """

    input_dir: str
    output_dir: str
    datasets: Dict[str, Dict] = field(
        default_factory=dict
    )


def load_extraction_config(
    config_path: Path,
) -> ExtractionConfig:
    """Load extraction config from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        ExtractionConfig: Parsed config.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return ExtractionConfig(
        input_dir=raw.get("input_dir", "data/raw"),
        output_dir=raw.get(
            "output_dir", "data/processed"
        ),
        datasets=raw.get("datasets", {}),
    )


def extract_datasets(
    config_path: Path,
    dataset_keys: Optional[List[str]] = None,
) -> None:
    """Extract and convert datasets to unified JSONL.

    Args:
        config_path: Path to extraction YAML config.
        dataset_keys: Specific dataset keys to extract.
            If None, extracts all configured datasets.

    Raises:
        ValueError: If any requested key is unknown.
    """
    cfg = load_extraction_config(config_path)

    if dataset_keys:
        unknown = set(dataset_keys) - set(
            cfg.datasets.keys()
        )
        if unknown:
            raise ValueError(
                f"Unknown dataset keys: "
                f"{', '.join(sorted(unknown))}"
            )
        selected = {
            k: v
            for k, v in cfg.datasets.items()
            if k in dataset_keys
        }
    else:
        selected = cfg.datasets

    output_base = Path(cfg.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for key, ds_cfg in selected.items():
        extractor_name = ds_cfg["extractor"]
        domain = ds_cfg["domain"]
        input_dir = Path(cfg.input_dir) / key

        if not input_dir.exists():
            logger.warning(
                "Input dir not found for '%s': %s",
                key,
                input_dir,
            )
            continue

        logger.info(
            "Extracting '%s' with %s extractor",
            key,
            extractor_name,
        )

        extractor = get_extractor(extractor_name)
        output_file = output_base / f"{key}.jsonl"

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in extractor.extract(
                input_dir, key, domain
            ):
                f.write(doc.to_jsonl_line())
                f.write("\n")
                count += 1

        logger.info(
            "Extracted %d documents from '%s' -> %s",
            count,
            key,
            output_file,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_processing.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add slm4ie/data/processing.py \
        configs/data/extract.yaml \
        tests/data/test_processing.py
git commit -m "feat: add extraction orchestrator with YAML config and JSONL output"
```

---

## Task 10: CLI Entrypoint

**Files:**
- Create: `scripts/data/extract.py`

- [ ] **Step 1: Implement CLI**

```python
# scripts/data/extract.py
"""Extract and convert raw datasets to unified JSONL format."""

import argparse
import logging
import sys
from pathlib import Path

from slm4ie.data.processing import extract_datasets


def _find_project_root() -> Path:
    """Find the project root by locating pyproject.toml.

    Returns:
        Path: The project root directory.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw datasets to unified JSONL format."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset keys to extract "
            "(default: all configured)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to extraction YAML config "
            "(default: configs/data/extract.yaml)."
        ),
    )
    return parser.parse_args(argv)


def main():
    """Run dataset extraction pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(name)s: "
            "%(message)s"
        ),
    )

    args = parse_args()
    project_root = _find_project_root()

    config_path = (
        Path(args.config)
        if args.config
        else project_root
        / "configs"
        / "data"
        / "extract.yaml"
    )

    try:
        extract_datasets(
            config_path=config_path,
            dataset_keys=args.datasets,
        )
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(
            "Extraction failed: %s", e
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI manually**

Run: `uv run python -m scripts.data.extract --help`
Expected: Shows `--datasets`, `--config` flags.

- [ ] **Step 3: Commit**

```bash
git add scripts/data/extract.py
git commit -m "feat: add CLI entrypoint for dataset extraction"
```

---

## Dataset → Extractor Mapping Reference

| Dataset | Extractor | Domain | Archive Format |
|---------|-----------|--------|----------------|
| classla_web_sl | `jsonl` | web | .jsonl.gz |
| classlawiki_sl | `conllu` | wiki | .conllu.gz |
| macocu_sl | `macocu` | web | .xml.zip |
| parlamint_si | `tei` | parliamentary | .tgz |
| kas | `tei` | academic | .tar.gz |
| coleslaw | `tei` | legal | .zip |
| povejmo_vemo_med | `tei` | medical | .zip |
| oss | `conllu` | scientific | .zip (×3) |
| siparl | `tei` | parliamentary | .zip |
| janes_forum | `tei` | forum | .zip |
| janes_blog | `tei` | blog | .zip |
| janes_news | `tei` | news | .zip |
| kzb | `conllu` | scientific | .zip |
| solar | `conllu` | student | .zip |
| finepdf | `huggingface` | web | Arrow |
| fineweb2 | `huggingface` | web | Arrow |
| culturax | `huggingface` | web | Arrow |
| legal_mc4 | `huggingface` | legal | Arrow |
| c4 | `huggingface` | web | Arrow |
| hplt | `huggingface` | web | Arrow |
