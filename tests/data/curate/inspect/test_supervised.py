"""Tests for the supervised domain classifier's config, fitting and feature backends."""

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion

from slm4ie.data.curate.inspect.supervised import (
    Candidate,
    DomainClassifier,
    EmbeddingFeatures,
    fit_classifier,
    load_candidate,
)

CONFIG: Dict[str, Any] = {
    "max_chars": 30,
    "folds": 2,
    "workers": 1,
    "seed": 7,
    "regularisation": [10, 1],
    "max_iterations": 200,
    "candidates": {
        "tfidf": {"features": "tfidf", "word_ngrams": [1, 1], "char_ngrams": [2, 3], "min_df": 1, "max_features": 500},
        "embeddings": {
            "features": "embeddings",
            "model": "stub",
            "revision": "none",
            "prefix": "query: ",
            "batch_size": 4,
            "max_length": 8,
        },
    },
}
#: Six documents per label, each label with its own vocabulary, each document its own group.
TEXTS: List[str] = [f"zdravilo bolnik odmerek {i}" for i in range(6)] + [f"sodišče zakon člen {i}" for i in range(6)]
LABELS: List[str] = ["medical"] * 6 + ["legal"] * 6
LABEL_SET = ("medical", "legal", "news")
GROUPS: List[str] = [str(i) for i in range(12)]


def stub_encoder(texts: Sequence[str]) -> np.ndarray:
    """Embed each text as the counts of two marker words, so the labels are separable."""
    return np.array([[text.count("zdravilo"), text.count("sodišče")] for text in texts], dtype=float)


def _candidate(name: str) -> Candidate:
    """Load a fixture candidate, the embeddings one with the stub encoder."""
    return load_candidate(CONFIG, name, encoder=stub_encoder)


def _fit(name: str) -> DomainClassifier:
    """Fit a fixture candidate on the fixture documents."""
    return fit_classifier(TEXTS, LABELS, GROUPS, LABEL_SET, _candidate(name))[0]


class TestLoadCandidate:
    """Candidates come out of the config with their backend, and bad configs are rejected."""

    def test_tfidf_candidate_builds_its_backend(self) -> None:
        """The tfidf candidate carries the word-and-character TF-IDF union."""
        assert isinstance(_candidate("tfidf").features, FeatureUnion)

    def test_shared_settings_are_read(self) -> None:
        """The cut length, regularisation grid, folds and seed come from the top level of the config."""
        candidate = _candidate("tfidf")
        assert (candidate.max_chars, candidate.regularisation, candidate.folds, candidate.seed) == (
            30,
            (10.0, 1.0),
            2,
            7,
        )

    def test_embeddings_candidate_uses_the_encoder_passed_in(self) -> None:
        """An encoder passed in replaces the named model and does the encoding."""
        features = _candidate("embeddings").features
        assert features.transform(["zdravilo"]).tolist() == [[1.0, 0.0]]

    def test_unknown_candidate_is_rejected(self) -> None:
        """A candidate the config does not list names itself and the known ones."""
        with pytest.raises(ValueError, match=r"candidate 'bert' is not one of \['embeddings', 'tfidf'\]"):
            load_candidate(CONFIG, "bert")

    def test_unknown_feature_kind_is_rejected(self) -> None:
        """A feature kind that is neither backend names itself."""
        config = {**CONFIG, "candidates": {"odd": {"features": "hashing"}}}
        with pytest.raises(ValueError, match="features 'hashing'"):
            load_candidate(config, "odd")

    def test_missing_key_names_itself_and_its_place(self) -> None:
        """A candidate without `min_df` says which candidate lacks it."""
        config = {
            **CONFIG,
            "candidates": {"tfidf": {"features": "tfidf", "word_ngrams": [1, 1], "char_ngrams": [2, 3]}},
        }
        with pytest.raises(ValueError, match="key 'min_df' missing in candidate 'tfidf'"):
            load_candidate(config, "tfidf")

    def test_loads_from_a_yaml_path(self, tmp_path: Path) -> None:
        """A path is read as the YAML config."""
        path = tmp_path / "classifier-supervised.yaml"
        path.write_text(
            "max_chars: 30\nfolds: 2\nworkers: 1\nseed: 1\nregularisation: [1]\nmax_iterations: 100\n"
            "candidates:\n  tfidf:\n    features: tfidf\n    word_ngrams: [1, 1]\n    char_ngrams: [2, 3]\n"
            "    min_df: 1\n    max_features: 10\n",
            encoding="utf-8",
        )
        assert load_candidate(str(path), "tfidf").regularisation == (1.0,)


class TestFitClassifier:
    """Fitting picks a regularisation on the fitting documents and labels new ones."""

    def test_tfidf_classifier_labels_held_out_texts(self) -> None:
        """Two separable vocabularies are learnt from word and character n-grams."""
        assert _fit("tfidf").predict(["bolnik dobi zdravilo", "tožba na sodišče"]) == ["medical", "legal"]

    def test_embeddings_classifier_labels_held_out_texts(self) -> None:
        """The stub encoder's two counts are enough for the regression."""
        assert _fit("embeddings").predict(["zdravilo", "sodišče"]) == ["medical", "legal"]

    def test_grid_marks_exactly_one_pick(self) -> None:
        """One row per regularisation strength in config order, one of them picked."""
        _, grid = fit_classifier(TEXTS, LABELS, GROUPS, LABEL_SET, _candidate("tfidf"))
        assert ([row["regularisation"] for row in grid], sum(row["picked"] for row in grid)) == ([10.0, 1.0], 1)

    def test_ties_go_to_the_stronger_regularisation(self) -> None:
        """Both strengths fit the separable fixture perfectly, so the smaller C wins although it is listed second."""
        classifier, grid = fit_classifier(TEXTS, LABELS, GROUPS, LABEL_SET, _candidate("tfidf"))
        assert ([row["macro_f1"] for row in grid], classifier.regularisation) == ([grid[0]["macro_f1"]] * 2, 1.0)

    def test_texts_are_cut_before_labelling(self) -> None:
        """Only the first `max_chars` characters are read, so a medical opening wins over a long legal tail."""
        classifier = _fit("tfidf")
        assert classifier.predict(["zdravilo bolnik odmerek tableta " + "sodišče zakon člen " * 20]) == ["medical"]

    def test_copies_stay_in_one_fold(self) -> None:
        """Documents sharing a group never split across folds, so a group as large as a fold still fits."""
        groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
        classifier, _ = fit_classifier(TEXTS, LABELS, groups, LABEL_SET, _candidate("tfidf"))
        assert classifier.predict(["zdravilo"]) == ["medical"]

    def test_training_label_outside_the_scored_set_is_rejected(self) -> None:
        """A label the macro-F1 would silently drop stops the fit, naming it."""
        with pytest.raises(ValueError, match=r"training labels \['legal'\] are not in the scored label set"):
            fit_classifier(TEXTS, LABELS, GROUPS, ("medical", "news"), _candidate("tfidf"))


class TestEmbeddingFeatures:
    """The embeddings backend prefixes texts and keeps no encoder when saved."""

    def test_prefix_reaches_the_encoder(self) -> None:
        """Every text reaches the encoder with the configured prefix in front."""
        seen: List[str] = []

        def spy(texts: Sequence[str]) -> np.ndarray:
            seen.extend(texts)
            return stub_encoder(texts)

        EmbeddingFeatures("stub", "none", "query: ", 4, 8, encoder=spy).transform(["zdravilo"])
        assert seen == ["query: zdravilo"]

    def test_saving_drops_the_encoder(self, tmp_path: Path) -> None:
        """A classifier with an unpicklable encoder still saves, since the encoder is reloaded from its name."""
        candidate = load_candidate(CONFIG, "embeddings", encoder=lambda texts: stub_encoder(texts))
        classifier, _ = fit_classifier(TEXTS, LABELS, GROUPS, LABEL_SET, candidate)
        classifier.save(tmp_path / "classifier.pkl")
        assert DomainClassifier.load(tmp_path / "classifier.pkl").regularisation == classifier.regularisation


class TestSaveLoad:
    """A saved classifier predicts the same after loading."""

    def test_tfidf_round_trip(self, tmp_path: Path) -> None:
        """The TF-IDF vocabularies and the regression survive pickling."""
        classifier = _fit("tfidf")
        classifier.save(tmp_path / "model" / "classifier.pkl")
        loaded = DomainClassifier.load(tmp_path / "model" / "classifier.pkl")
        assert loaded.predict(["bolnik dobi zdravilo"]) == classifier.predict(["bolnik dobi zdravilo"])
