"""Tests for the lexicon domain classifier's matching and decision rule."""

from pathlib import Path
from typing import Any, Dict

import pytest

from slm4ie.data.curate.inspect.domains import (
    DecisionRule,
    DomainLexicons,
    assign_domain,
    label_document,
    load_lexicons,
    score_text,
)

CONFIG: Dict[str, Any] = {
    "weights": {"strong": 3, "weak": 1},
    "lexicons": {
        "medical": {"strong": ["zdravil*", "bolnik*"], "weak": ["telo"]},
        "legal": {"strong": ["sodišč*", "člen*"], "weak": ["dne"]},
    },
    "fallback": {"mixed": "other"},
    "rule": {"threshold": 50.0, "margin": 20.0, "min_words": 3},
}
RULE = DecisionRule(threshold=50.0, margin=20.0, min_words=3)
#: Ten words with two strong medical hits (6) and one weak (1): medical 700, legal 0.
MEDICAL_TEXT = "Zdravila za bolnike in telo so tu da pomagajo vsem"
#: Ten words with one strong medical hit and three weak legal ones: medical 300, legal 300.
TIED_TEXT = "Zdravila dne dne dne so bila tu za vse nas"
#: Thirty words with one weak medical hit: medical 33, legal 0.
WEAK_TEXT = "telo " + "in " * 29


def _config(**lexicons: Dict[str, Any]) -> Dict[str, Any]:
    """Return the fixture config with its lexicons replaced."""
    return {**CONFIG, "lexicons": lexicons}


@pytest.fixture
def lexicons() -> DomainLexicons:
    """Lexicons built from the fixture config."""
    return load_lexicons(CONFIG)


class TestLoadLexicons:
    """The config compiles into exact and prefix tables, and bad configs are rejected."""

    def test_domains_keep_config_order(self, lexicons: DomainLexicons) -> None:
        """The domain order of the config is the tie-breaking order."""
        assert lexicons.domains == ("medical", "legal")

    def test_exact_and_prefix_entries_are_kept_apart(self, lexicons: DomainLexicons) -> None:
        """A whole-word entry goes to the exact table, a starred one to the prefix table without its star."""
        assert lexicons.exact["telo"] == (("medical", 1.0),)
        assert lexicons.prefixes["zdravil"] == (("medical", 3.0),)
        assert "zdravil*" not in lexicons.exact

    def test_unknown_tier_is_rejected(self) -> None:
        """A tier with no weight names itself in the error."""
        with pytest.raises(ValueError, match="tier 'loud'"):
            load_lexicons(_config(medical={"loud": ["x"]}))

    def test_duplicate_entry_in_one_domain_is_rejected(self) -> None:
        """One word listed in two tiers of one domain is an error."""
        with pytest.raises(ValueError, match="'telo'"):
            load_lexicons(_config(medical={"strong": ["telo"], "weak": ["telo"]}))

    def test_entry_covered_by_a_shorter_prefix_is_rejected(self) -> None:
        """An entry a shorter prefix of equal or higher weight already matches can never change a score."""
        with pytest.raises(ValueError, match="'zdravila' is covered by its prefix 'zdrav\\*'"):
            load_lexicons(_config(medical={"strong": ["zdrav*"], "weak": ["zdravila"]}))

    def test_stronger_entry_under_a_weaker_prefix_is_kept(self) -> None:
        """A strong whole word under a weak prefix still raises that word's weight."""
        lexicons = load_lexicons(_config(medical={"weak": ["zdrav*"], "strong": ["zdravila"]}))
        assert lexicons.word_weights("zdravila") == (("medical", 3.0),)

    def test_domain_outside_the_judge_labels_is_rejected(self) -> None:
        """A misspelt domain cannot load."""
        with pytest.raises(ValueError, match="'medicall'"):
            load_lexicons(_config(medicall={"strong": ["x"]}))

    def test_fallback_target_outside_the_judge_labels_is_rejected(self) -> None:
        """A fallback must land on one of the judge's labels."""
        with pytest.raises(ValueError, match="'mixed' -> 'misc'"):
            load_lexicons({**CONFIG, "fallback": {"mixed": "misc"}})

    def test_missing_key_names_itself(self) -> None:
        """A config without a rule says so."""
        with pytest.raises(ValueError, match="key 'rule'"):
            load_lexicons({key: value for key, value in CONFIG.items() if key != "rule"})

    def test_loads_from_a_yaml_path(self, tmp_path: Path) -> None:
        """A path, as a string or a Path, is read as the YAML config."""
        path = tmp_path / "classifier-lexicon.yaml"
        path.write_text(
            "weights: {strong: 3}\nlexicons:\n  medical:\n    strong: [zdravil*]\n"
            "fallback: {}\nrule: {threshold: 1, margin: 0, min_words: 1}\n",
            encoding="utf-8",
        )
        assert load_lexicons(path).domains == ("medical",)
        assert load_lexicons(str(path)).rule == DecisionRule(threshold=1.0, margin=0.0, min_words=1)


class TestScoreText:
    """Scores are weighted hits per thousand words."""

    def test_scores_are_weighted_hits_per_thousand_words(self, lexicons: DomainLexicons) -> None:
        """Two strong hits and one weak one in ten words score 700."""
        assert score_text(MEDICAL_TEXT, lexicons) == ({"medical": 700.0, "legal": 0.0}, 10)

    def test_prefix_matches_inflected_forms_and_exact_does_not(self, lexicons: DomainLexicons) -> None:
        """`sodišč*` matches two case forms; the whole word `telo` does not match `telesa`."""
        scores, _ = score_text("sodišču sodišče telesa", lexicons)
        assert scores == {"medical": 0.0, "legal": pytest.approx(2000.0)}

    def test_a_word_counts_once_per_domain_at_its_strongest_entry(self) -> None:
        """A word under a weak prefix and a strong whole-word entry scores the strong weight once."""
        lexicons = load_lexicons(_config(medical={"weak": ["zdrav*"], "strong": ["zdravila"]}))
        assert score_text("zdravila", lexicons) == ({"medical": 3000.0}, 1)

    def test_matching_ignores_case_and_punctuation(self, lexicons: DomainLexicons) -> None:
        """Capitals and punctuation around a word do not stop a match."""
        assert score_text("ZDRAVILA, Bolnik.", lexicons)[0]["medical"] == 3000.0

    def test_underscores_split_words(self, lexicons: DomainLexicons) -> None:
        """A word is letters and digits, so `zdravila_x` is two words with one hit."""
        assert score_text("zdravila_x", lexicons) == ({"medical": 1500.0, "legal": 0.0}, 2)

    def test_empty_text_scores_zero(self, lexicons: DomainLexicons) -> None:
        """No words means zero everywhere and a word count of zero."""
        assert score_text("", lexicons) == ({"medical": 0.0, "legal": 0.0}, 0)


class TestAssignDomain:
    """The leader is assigned at or above the threshold and margin, else the fallback."""

    def test_leader_wins_when_above_threshold_and_ahead_by_margin(self) -> None:
        """A clear leader is assigned."""
        assert assign_domain({"medical": 80.0, "legal": 10.0}, 10, RULE, "web") == "medical"

    def test_leader_exactly_at_threshold_and_margin_is_assigned(self) -> None:
        """Both comparisons are at-or-above, so a score of 50 leading by 20 passes."""
        assert assign_domain({"medical": 50.0, "legal": 30.0}, 10, RULE, "web") == "medical"

    def test_falls_back_just_below_threshold(self) -> None:
        """A leader at 49.9 keeps the source label."""
        assert assign_domain({"medical": 49.9, "legal": 0.0}, 10, RULE, "web") == "web"

    def test_falls_back_just_inside_margin(self) -> None:
        """A lead of 19.9 keeps the source label."""
        assert assign_domain({"medical": 80.0, "legal": 60.1}, 10, RULE, "news") == "news"

    def test_tie_falls_back_under_a_positive_margin(self) -> None:
        """Two equal leaders have no lead, so the source label stays."""
        assert assign_domain({"medical": 80.0, "legal": 80.0}, 10, RULE, "web") == "web"

    def test_tie_breaks_towards_the_earlier_domain_at_zero_margin(self) -> None:
        """With no margin the earlier domain wins a tie."""
        rule = DecisionRule(threshold=50.0, margin=0.0, min_words=3)
        assert assign_domain({"medical": 80.0, "legal": 80.0}, 10, rule, "web") == "medical"

    def test_short_text_falls_back_whatever_its_scores(self) -> None:
        """Fewer words than `min_words` keeps the source label even with a huge lead."""
        assert assign_domain({"medical": 3000.0, "legal": 0.0}, 2, RULE, "web") == "web"


class TestLabelDocument:
    """Documents are labelled end to end with the configured rule."""

    def test_assigns_the_leading_domain(self, lexicons: DomainLexicons) -> None:
        """A text full of medical markers from a web source is medical."""
        assert label_document(MEDICAL_TEXT, "web", lexicons) == "medical"

    def test_keeps_the_source_label_below_the_threshold(self, lexicons: DomainLexicons) -> None:
        """One weak hit in thirty words scores 33, under the threshold of 50."""
        assert label_document(WEAK_TEXT, "news", lexicons) == "news"

    def test_keeps_the_source_label_inside_the_margin(self, lexicons: DomainLexicons) -> None:
        """Medical and legal tied at 300 clear the threshold but not the margin."""
        assert label_document(TIED_TEXT, "news", lexicons) == "news"

    def test_fallback_maps_a_source_label_outside_the_domains(self, lexicons: DomainLexicons) -> None:
        """A mixed source with no hits counts as other."""
        assert label_document("nič posebnega tukaj ni", "mixed", lexicons) == "other"

    def test_fallback_keeps_a_source_label_among_the_domains(self, lexicons: DomainLexicons) -> None:
        """A news source with no hits stays news."""
        assert label_document("nič posebnega tukaj ni", "news", lexicons) == "news"

    def test_short_document_keeps_its_source_label(self, lexicons: DomainLexicons) -> None:
        """Two words, one of them a strong hit, are too few to read."""
        assert label_document("zdravila tu", "web", lexicons) == "web"
