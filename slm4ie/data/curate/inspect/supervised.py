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
`load_candidate` for its shape.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from tqdm import tqdm

from slm4ie.utils.config import load_yaml

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
