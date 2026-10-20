"""Fit a supervised domain classifier on judged documents and label new ones.

The lexicon classifier of `domains.py` reads marker words; this one learns
from documents a judge already labelled. It is one logistic regression over
the judge's labels, fed by one of two feature backends: `tfidf`, word and
character n-grams weighted by TF-IDF (how often a term occurs in a document,
discounted by how many documents carry it), which runs on a CPU; or
`embeddings`, one sentence-embedding vector per document from a pretrained
encoder, which wants a GPU. The regularisation strength is picked by
cross-validation on the fitting documents, with the features refitted inside
every fold and copies of one text kept in one fold, so the held-out side of
the gold is read once, with the saved model.

Every text is cut to a fixed length before it is read, fitting and labelling
alike, so a classifier measured on cut gold text labels corpus text the same
way.

The candidates and their settings live in the experiment's YAML config; see
`load_candidate` for its shape. `label_corpus` runs a saved classifier over
every document of a built corpus, counting documents and words per source and
label, and `track_corpus_labels` logs that count as an MLflow run.
"""

import gzip
import hashlib
import json
import multiprocessing
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from tqdm import tqdm

from slm4ie.data.curate.lineage import read_sentinel
from slm4ie.data.curate.stages.statistics import CorpusStats
from slm4ie.utils import mlflow as ml
from slm4ie.utils.config import load_yaml
from slm4ie.utils.versioning import corpus_digest, file_sha256

#: Turns a batch of texts into one vector per text.
Encoder = Callable[[Sequence[str]], np.ndarray]


def _require(config: Mapping[str, Any], key: str, where: str) -> Any:
    """Return `config[key]`, naming the key and where it was expected when missing."""
    if key not in config:
        raise ValueError(f"classifier config key '{key}' missing in {where}")
    return config[key]


def tfidf_features(
    word_ngrams: Tuple[int, int], char_ngrams: Tuple[int, int], min_df: int, max_features: int
) -> FeatureUnion:
    """Build the TF-IDF backend: word and character n-grams side by side.

    Args:
        word_ngrams: Shortest and longest word n-gram.
        char_ngrams: Shortest and longest character n-gram, kept inside word boundaries.
        min_df: Documents an n-gram must occur in to be kept.
        max_features: Most frequent n-grams kept, per n-gram kind.

    Returns:
        An unfitted transformer from texts to sparse TF-IDF rows.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=word_ngrams, min_df=min_df, max_features=max_features, sublinear_tf=True),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngrams,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """The embeddings backend: one vector per text from a pretrained encoder, loaded on first use.

    Args:
        model: The encoder's name on the Hugging Face hub.
        revision: The commit of the encoder to load, so the vectors stay the same across hub updates.
        prefix: Text put before every document, as the encoder was trained with.
        batch_size: Documents encoded at once.
        max_length: Tokens read per document; the rest is cut.
        encoder: A ready encoder in place of the named model, for tests and reuse.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        prefix: str,
        batch_size: int,
        max_length: int,
        encoder: Optional[Encoder] = None,
    ) -> None:
        """Keep the settings; the encoder is loaded on the first transform unless one is passed."""
        self.model = model
        self.revision = revision
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self.encoder = encoder

    def fit(self, texts: Sequence[str], labels: Any = None) -> "EmbeddingFeatures":
        """Nothing to learn: the encoder is pretrained."""
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unit-length embedding per text."""
        if self.encoder is None:
            self.encoder = self._load_encoder()
        return self.encoder([self.prefix + text for text in texts])

    def _load_encoder(self) -> Encoder:
        """Load the named sentence-embedding model; the import is deferred to keep the TF-IDF path light."""
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(self.model, revision=self.revision)
        encoder.max_seq_length = self.max_length
        return lambda texts: encoder.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle the settings only; the encoder is reloaded from its name."""
        return {key: value for key, value in self.__dict__.items() if key != "encoder"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the settings with no encoder loaded."""
        self.__dict__.update(state)
        self.encoder = None


@dataclass(frozen=True)
class Candidate:
    """One classifier to fit: its feature backend and the fitting procedure's settings.

    Attributes:
        name: The candidate's key in the config.
        features: Its feature backend, an unfitted transformer from texts to rows.
        max_chars: Characters read per text, fitting and labelling alike.
        regularisation: The inverse regularisation strengths tried, in order.
        folds: Cross-validation folds that pick among them.
        workers: Folds fitted at once.
        seed: The fold assignment's seed.
        max_iterations: Solver iterations before a fit is declared not converged.
    """

    name: str
    features: Any
    max_chars: int
    regularisation: Tuple[float, ...]
    folds: int
    workers: int
    seed: int
    max_iterations: int


def _build_features(name: str, settings: Mapping[str, Any], encoder: Optional[Encoder]) -> Any:
    """Build the feature backend a candidate names."""
    where = f"candidate '{name}'"
    kind = _require(settings, "features", where)
    if kind == "tfidf":
        return tfidf_features(
            word_ngrams=tuple(_require(settings, "word_ngrams", where)),
            char_ngrams=tuple(_require(settings, "char_ngrams", where)),
            min_df=int(_require(settings, "min_df", where)),
            max_features=int(_require(settings, "max_features", where)),
        )
    if kind == "embeddings":
        return EmbeddingFeatures(
            model=str(_require(settings, "model", where)),
            revision=str(_require(settings, "revision", where)),
            prefix=str(_require(settings, "prefix", where)),
            batch_size=int(_require(settings, "batch_size", where)),
            max_length=int(_require(settings, "max_length", where)),
            encoder=encoder,
        )
    raise ValueError(f"{where} names features '{kind}', not one of tfidf, embeddings")


def load_candidate(
    config: Union[str, Path, Mapping[str, Any]], name: str, encoder: Optional[Encoder] = None
) -> Candidate:
    """Read one candidate from the classifier config.

    The config has `max_chars`, `folds`, `workers`, `seed`, `regularisation`
    (a list of C values), `max_iterations` and `candidates`, a mapping from
    name to settings with a `features` key of `tfidf` (`word_ngrams`,
    `char_ngrams`, `min_df`, `max_features`) or `embeddings` (`model`,
    `revision`, `prefix`, `batch_size`, `max_length`).

    Args:
        config: The config mapping, or the path of its YAML file.
        name: The candidate's key under `candidates`.
        encoder: A ready encoder for an embeddings candidate, in place of the named model.

    Returns:
        The candidate, unfitted.

    Raises:
        ValueError: If a key is missing, the candidate is unknown, or its feature kind is.
    """
    if isinstance(config, (str, Path)):
        config = load_yaml(Path(config))
    candidates = _require(config, "candidates", "the config")
    if name not in candidates:
        raise ValueError(f"candidate '{name}' is not one of {sorted(candidates)}")
    return Candidate(
        name=name,
        features=_build_features(name, candidates[name], encoder),
        max_chars=int(_require(config, "max_chars", "the config")),
        regularisation=tuple(float(value) for value in _require(config, "regularisation", "the config")),
        folds=int(_require(config, "folds", "the config")),
        workers=int(_require(config, "workers", "the config")),
        seed=int(_require(config, "seed", "the config")),
        max_iterations=int(_require(config, "max_iterations", "the config")),
    )


@dataclass
class DomainClassifier:
    """A fitted classifier: the features and the logistic regression over them, as one pipeline.

    Attributes:
        pipeline: The fitted features followed by the fitted regression.
        regularisation: The inverse regularisation strength the fit used.
        max_chars: Characters read per text before labelling.
    """

    pipeline: Pipeline
    regularisation: float
    max_chars: int

    def predict(self, texts: Sequence[str]) -> List[str]:
        """Label texts, each cut to `max_chars` first.

        Args:
            texts: The documents.

        Returns:
            One judge label per text.
        """
        return [str(label) for label in self.pipeline.predict(cut_texts(texts, self.max_chars))]

    def save(self, path: Path) -> None:
        """Pickle the classifier to `path`, creating its folder."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "DomainClassifier":
        """Unpickle a classifier `save` wrote; an embeddings backend reloads its encoder on first use."""
        return pickle.loads(path.read_bytes())


def cut_texts(texts: Sequence[str], max_chars: int) -> List[str]:
    """Return the first `max_chars` characters of every text."""
    return [text[:max_chars] for text in texts]


def _pipeline(features: Any, regularisation: float, max_iterations: int) -> Pipeline:
    """Fresh features followed by a logistic regression weighting every class the same whatever its size."""
    regression = LogisticRegression(C=regularisation, class_weight="balanced", max_iter=max_iterations)
    return Pipeline([("features", clone(features)), ("regression", regression)])


def fit_classifier(
    texts: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    label_set: Sequence[str],
    candidate: Candidate,
) -> Tuple[DomainClassifier, List[Dict[str, Any]]]:
    """Fit a candidate, picking its regularisation by cross-validation on the fitting documents.

    Each regularisation strength is scored by the macro-F1 over the label
    set of its out-of-fold predictions, the features refitted inside each
    fold and every group kept in one fold; the highest wins, ties to the
    stronger regularisation, and the winner is refitted on every document.

    Args:
        texts: The fitting documents.
        labels: Their judge labels, in the same order.
        groups: A group key per document; copies of one text share one and stay in one fold.
        label_set: The labels macro-F1 averages over, each counting once whether or not it occurs.
        candidate: What to fit.

    Returns:
        The fitted classifier, and one row per regularisation strength with
        its cross-validated macro-F1 and whether it was picked.

    Raises:
        ValueError: If a training label is not in `label_set`, where macro-F1 would silently drop it.
    """
    outside = sorted(set(labels) - set(label_set))
    if outside:
        raise ValueError(f"training labels {outside} are not in the scored label set {list(label_set)}")
    cut = cut_texts(texts, candidate.max_chars)
    truth = np.asarray(labels)
    folds = StratifiedGroupKFold(n_splits=candidate.folds, shuffle=True, random_state=candidate.seed)
    rows = []
    for regularisation in tqdm(candidate.regularisation, desc=f"{candidate.name} regularisation", unit="C"):
        pipeline = _pipeline(candidate.features, regularisation, candidate.max_iterations)
        predicted = cross_val_predict(pipeline, cut, truth, groups=groups, cv=folds, n_jobs=candidate.workers)
        macro = f1_score(truth, predicted, labels=list(label_set), average="macro", zero_division=0)
        rows.append({"regularisation": regularisation, "macro_f1": float(macro)})
    picked = max(rows, key=lambda row: (row["macro_f1"], -row["regularisation"]))
    pipeline = _pipeline(candidate.features, picked["regularisation"], candidate.max_iterations).fit(cut, truth)
    classifier = DomainClassifier(
        pipeline=pipeline, regularisation=picked["regularisation"], max_chars=candidate.max_chars
    )
    return classifier, [{**row, "picked": row is picked} for row in rows]


#: Documents labelled at once within a shard; bounds the feature matrix held in memory.
LABEL_BATCH = 2_000

#: Per-process state of a corpus pass: the classifier, the word counter and the gold ids.
_labeller: Dict[str, Any] = {}

#: The name the curation build logs its final corpus under, so MLflow links the pass to the build.
FINAL_CORPUS_DATASET = "pretrain/06_sentence_dedup"


@dataclass(frozen=True)
class CorpusLabels:
    """What a classifier pass over a corpus found.

    Attributes:
        documents: Documents per source and assigned label.
        words: Words per source and assigned label, counted as the statistics stage counts them.
        gold_ids: The gold document ids the corpus still holds.
    """

    documents: Dict[str, Dict[str, int]]
    words: Dict[str, Dict[str, int]]
    gold_ids: List[str]


def _start_labeller(classifier_path: Path, stopwords: Set[str], gold_ids: FrozenSet[str]) -> None:
    """Load the classifier and word counter once per worker process."""
    _labeller["classifier"] = DomainClassifier.load(classifier_path)
    _labeller["counter"] = CorpusStats(stopwords=stopwords)
    _labeller["gold_ids"] = gold_ids


def _label_batch(batch: List[Dict[str, Any]], documents: Counter, words: Counter) -> None:
    """Label one batch of documents and add it to the per-label counts."""
    labels = _labeller["classifier"].predict([document["text"] for document in batch])
    for document, label in zip(batch, labels):
        documents[label] += 1
        # The statistics stage's own word rule, so tokens match the source-label counts.
        words[label] += len(_labeller["counter"]._tokenize_words(document["text"]))


def _label_shard(shard: Path) -> Dict[str, Any]:
    """Label every document of one shard.

    Args:
        shard: A gzipped JSONL shard of `Document` records.

    Returns:
        Documents and words per assigned label, and the gold ids the shard holds.
    """
    documents: Counter = Counter()
    words: Counter = Counter()
    found: List[str] = []
    batch: List[Dict[str, Any]] = []
    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        for line in handle:
            document = json.loads(line)
            if document["id"] in _labeller["gold_ids"]:
                found.append(document["id"])
            batch.append(document)
            if len(batch) == LABEL_BATCH:
                _label_batch(batch, documents, words)
                batch = []
    if batch:
        _label_batch(batch, documents, words)
    return {"documents": dict(documents), "words": dict(words), "gold_ids": found}


def _pass_key(document_digest: str, classifier_path: Path, stopwords: Set[str], gold_ids: FrozenSet[str]) -> str:
    """Hash everything a shard's counts depend on, so a checkpoint is reused only for the same pass."""
    payload = json.dumps([document_digest, file_sha256(classifier_path), sorted(stopwords), sorted(gold_ids)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resume_checkpoint(checkpoint: Path, key: str) -> Dict[str, Dict[str, Any]]:
    """Read the shards a checkpoint already holds and leave it ready to append to.

    A line cut short by a crash is dropped, and the file rewritten without it,
    so its shard is labelled again.

    Args:
        checkpoint: JSONL file: a header line with the pass key, then one line per finished shard.
        key: The key of the pass about to run.

    Returns:
        The finished shards' counts, keyed by shard path relative to the stage.

    Raises:
        ValueError: If the checkpoint was written by a pass with another key.
    """
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        checkpoint.write_text(json.dumps({"key": key}) + "\n", encoding="utf-8")
        return {}
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    written = json.loads(lines[0])["key"]
    if written != key:
        raise ValueError(f"checkpoint {checkpoint} belongs to pass {written}, not {key}; delete it to start over")
    done: Dict[str, Dict[str, Any]] = {}
    for number, line in enumerate(lines[1:], start=2):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines):
                raise
            break
        done[row["shard"]] = row
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in [{"key": key}, *done.values()]), encoding="utf-8")
    return done


def label_corpus(
    stage_dir: Path,
    classifier_path: Path,
    stopwords: Set[str],
    gold_ids: FrozenSet[str],
    checkpoint: Path,
    workers: int,
    shards_per_source: Optional[int] = None,
) -> CorpusLabels:
    """Label every document of a corpus stage with a saved classifier.

    Each finished shard's counts are appended to `checkpoint`, so a pass that
    stops part way resumes from the shards it has not finished.

    Args:
        stage_dir: A completed stage folder holding one directory of gzipped JSONL shards per source.
        classifier_path: A classifier `DomainClassifier.save` wrote.
        stopwords: Words left out of the word counts, as the statistics stage leaves them out.
        gold_ids: Gold document ids to look for, so a reader learns which survived.
        checkpoint: JSONL file of finished shards, reused while the stage, classifier, stopwords and gold ids match.
        workers: Shards labelled at once, one process each.
        shards_per_source: Read only the first this many shards of each source, for a smoke run.

    Returns:
        Documents and words per source and label, and the gold ids found.

    Raises:
        FileNotFoundError: If `stage_dir` has no completed sentinel with a document digest, or holds no shard.
        ValueError: If a full pass counts other than the sentinel's documents, or the checkpoint
            belongs to another pass.
    """
    sentinel = read_sentinel(stage_dir)
    if sentinel is None or sentinel.document_digest is None:
        raise FileNotFoundError(f"no completed sentinel with a document digest at {stage_dir}")
    sources = [folder for folder in sorted(stage_dir.iterdir()) if folder.is_dir()]
    shards = [shard for folder in sources for shard in sorted(folder.glob("*.jsonl.gz"))[:shards_per_source]]
    if not shards:
        raise FileNotFoundError(f"no *.jsonl.gz shard under {stage_dir}")
    done = _resume_checkpoint(checkpoint, _pass_key(sentinel.document_digest, classifier_path, stopwords, gold_ids))
    pending = [shard for shard in shards if str(shard.relative_to(stage_dir)) not in done]
    # spawn: forking a process that already runs BLAS and tokenizer threads can deadlock the children.
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_labeller,
        initargs=(classifier_path, stopwords, gold_ids),
    )
    try:
        futures = {pool.submit(_label_shard, shard): shard for shard in pending}
        with checkpoint.open("a", encoding="utf-8") as handle:
            for future in tqdm(
                as_completed(futures), total=len(shards), initial=len(shards) - len(pending), desc="label shards"
            ):
                shard = futures[future]
                row = {"shard": str(shard.relative_to(stage_dir)), "source": shard.parent.name, **future.result()}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done[row["shard"]] = row
    finally:
        # Without cancelling, a failed shard would wait for every queued shard before raising.
        pool.shutdown(cancel_futures=True)
    labels = _sum_shards([done[str(shard.relative_to(stage_dir))] for shard in shards], [f.name for f in sources])
    counted = sum(sum(counts.values()) for counts in labels.documents.values())
    if shards_per_source is None and counted != sentinel.records_out:
        raise ValueError(f"labelled {counted} documents in {stage_dir}, its sentinel records {sentinel.records_out}")
    return labels


def _sum_shards(rows: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> CorpusLabels:
    """Sum per-shard counts into per-source counts, every source present even when empty."""
    documents: Dict[str, Counter] = {source: Counter() for source in sources}
    words: Dict[str, Counter] = {source: Counter() for source in sources}
    for row in rows:
        documents[row["source"]].update(row["documents"])
        words[row["source"]].update(row["words"])
    return CorpusLabels(
        documents={source: dict(sorted(counts.items())) for source, counts in documents.items()},
        words={source: dict(sorted(counts.items())) for source, counts in words.items()},
        gold_ids=sorted(gold_id for row in rows for gold_id in row["gold_ids"]),
    )


def track_corpus_labels(
    labels: CorpusLabels,
    destination: Path,
    stage_dir: Path,
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    tags: Mapping[str, str],
) -> Optional[str]:
    """Write a corpus pass's counts to JSON and log them as one MLflow run.

    The run carries the documents and words per label summed over sources,
    the JSON as an artifact, and the corpus stage as a dataset input under
    the name and digest the curation build logged it with.

    Args:
        labels: What `label_corpus` found.
        destination: JSON file for the counts, its folder created.
        stage_dir: The corpus stage the pass read.
        experiment: MLflow experiment name, created when absent.
        run_name: Display name of the run.
        params: Settings the pass ran with.
        tags: Run tags; the current commit is added as `base_commit`.

    Returns:
        The MLflow run id, or None when tracking is unavailable.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(labels), indent=1), encoding="utf-8")
    documents: Counter = sum((Counter(counts) for counts in labels.documents.values()), Counter())
    words: Counter = sum((Counter(counts) for counts in labels.words.values()), Counter())
    metrics = {
        "documents": sum(documents.values()),
        "words": sum(words.values()),
        "gold_documents_kept": len(labels.gold_ids),
        **{f"documents__{label}": count for label, count in documents.items()},
        **{f"words__{label}": count for label, count in words.items()},
    }
    if not ml.ensure_experiment(experiment):
        return None
    with ml.mlflow_run(run_name, tags={**tags, "base_commit": ml.git_commit()}) as run:
        ml.log_params(dict(params))
        ml.log_metrics(metrics)
        ml.log_artifact(destination)
        ml.log_dataset_input(FINAL_CORPUS_DATASET, corpus_digest(stage_dir), str(stage_dir), context="labelled")
        return run.info.run_id
