"""Tests for the adult/SEO-spam filter stage (`slm4ie.data.curate.stages.spam`)."""

from pathlib import Path
from typing import Optional

import pytest

pytest.importorskip("datatrove")

from datatrove.data import Document  # noqa: E402

from slm4ie.data.curate.stages import spam as spam_module
from slm4ie.data.curate.stages.spam import (
    SpamConfig,
    SpamFilter,
    collapse_keys,
    load_spam_assets,
    load_spam_domains,
    load_spam_lexicon,
    stem_key,
)


def _doc(text: str, *, language: Optional[str] = "sl", url: Optional[str] = None) -> Document:
    """Build a `Document` with the given text and optional language/url metadata.

    Args:
        text: Document body.
        language: Value for `metadata.language`; omitted when `None`.
        url: Value for `metadata.url`; omitted when `None`.

    Returns:
        A datatrove `Document` ready to feed to `SpamFilter.filter`.
    """
    metadata = {}
    if language is not None:
        metadata["language"] = language
    if url is not None:
        metadata["url"] = url
    return Document(text=text, id="t", metadata=metadata)


def _kept(result) -> bool:
    """Return whether a `SpamFilter.filter` result means the doc is kept.

    Args:
        result: Either a bool or a `(bool, reason)` tuple, matching
            datatrove's `BaseFilter.filter` contract.

    Returns:
        True when the document should be kept.
    """
    return result[0] if isinstance(result, tuple) else result


# --- asset loaders --------------------------------------------------------


def test_load_spam_lexicon_sl_has_known_offenders() -> None:
    """The Slovenian lexicon flags the offenders found in the corpus stats."""
    adult, spam, raw = load_spam_lexicon("sl")
    assert {"prostitutka", "porno", "seks", "kurba"} <= adult
    assert "viagra" in spam
    assert raw  # non-empty bytes for sentinel hashing


def test_load_spam_lexicon_excludes_ambiguous_common_words() -> None:
    """Ambiguous common words are deliberately kept out of the lexicon."""
    adult, spam, _ = load_spam_lexicon("sl")
    assert "ženske" not in adult and "ženske" not in spam
    assert "masaža" not in adult and "masaža" not in spam
    ambiguous = {
        "replika",
        "replike",
        "ponaredek",
        "ponaredki",
        "igralnica",
        "igralnice",
        "oralno",
        "oralni",
        "analno",
        "analni",
        "spolnost",
    }
    assert not ambiguous & (adult | spam)


def test_lexicon_headers_name_every_excluded_word() -> None:
    """Each word kept out of the lexicon is named in a header exclusion list."""
    folder = Path(spam_module.__file__).resolve().parents[1] / "resources" / "spam" / "sl"
    header = "".join(
        line for path in folder.glob("*.txt") for line in path.read_text().splitlines(True) if line.startswith("#")
    )
    for word in ("replika", "igralnice", "oralno", "analni", "spolnost", "erotika", "golota", "bordeli"):
        assert word in header


def test_stem_key_drops_one_vowel_from_long_last_token() -> None:
    """Only the last token of 5+ letters loses one trailing vowel."""
    assert stem_key("joške") == "jošk"
    assert stem_key("seks") == "seks"
    assert stem_key("anal") == "anal"
    assert stem_key("erotična  masaža") == "erotična masaž"


def test_collapse_keys_folds_one_stem_family() -> None:
    """Entries sharing a stem, or extending it by up to 3 letters, collapse to one key."""
    assert collapse_keys({"seks", "seksi"}) == {"seks"}
    assert collapse_keys({"kurba", "kurbe", "kurbir"}) == {"kurb"}
    assert collapse_keys({"seks", "seks oglasi"}) == {"seks", "seks oglas"}


def test_load_spam_lexicon_unknown_language_raises() -> None:
    """An unknown language code raises ValueError listing what is available."""
    with pytest.raises(ValueError):
        load_spam_lexicon("zz")


def test_load_spam_domains_returns_set_and_bytes() -> None:
    """The domain blocklist loads into a non-empty set plus raw bytes."""
    domains, raw = load_spam_domains()
    assert isinstance(domains, set) and domains
    assert raw


def test_load_spam_assets_combines_languages_and_domains() -> None:
    """The asset bundle merges per-language lexicons and the domain list."""
    assets = load_spam_assets(["sl", "en"])
    assert "porno" in assets.adult_words["sl"]
    assert "sex" in assets.adult_words["en"]
    assert "viagra" in assets.spam_words["sl"]
    assert assets.domains
    assert assets.raw_bytes


def test_load_spam_assets_url_blocklist_off_omits_domains() -> None:
    """Disabling the URL blocklist yields an empty domain set."""
    assets = load_spam_assets(["sl"], url_blocklist=False)
    assert assets.domains == set()


# --- filter behavior ------------------------------------------------------


def _filter(**overrides) -> SpamFilter:
    """Build a `SpamFilter` from the real sl/en assets with config overrides.

    Args:
        **overrides: Keyword overrides applied to `SpamConfig`.

    Returns:
        A `SpamFilter` with LDNOOBW auto-loading disabled (offline tests).
    """
    sl_adult, sl_spam, _ = load_spam_lexicon("sl")
    en_adult, en_spam, _ = load_spam_lexicon("en")
    domains, _ = load_spam_domains()
    config = SpamConfig(use_ldnoobw=False, **overrides)
    return SpamFilter(
        adult_words={"sl": sl_adult, "en": en_adult},
        spam_words={"sl": sl_spam, "en": en_spam},
        domains=domains,
        config=config,
        seed=0,
    )


def test_clean_document_is_kept() -> None:
    """A clean Slovenian sentence passes the filter."""
    f = _filter()
    assert _kept(f.filter(_doc("Danes je lep sončen dan v Ljubljani.")))


def test_two_adult_hits_drop_the_document() -> None:
    """Reaching the adult-hit threshold drops the document."""
    f = _filter(min_adult_hits=2)
    result = f.filter(_doc("Oglas: porno in seks vsebine na voljo."))
    assert _kept(result) is False


def test_single_adult_hit_below_threshold_is_kept() -> None:
    """A lone adult-term occurrence stays below the default threshold."""
    f = _filter(min_adult_hits=2)
    assert _kept(f.filter(_doc("Predavanje o tem, kaj je seks v biologiji.")))


def test_spam_hits_drop_the_document() -> None:
    """Reaching the SEO/scam-hit threshold drops the document."""
    f = _filter(min_spam_hits=2)
    result = f.filter(_doc("Kupi viagra poceni, cialis na spletu!"))
    assert _kept(result) is False


def test_one_term_repeated_counts_once() -> None:
    """One term used twice, even in two inflections, stays below threshold 2."""
    f = _filter(min_adult_hits=2, min_spam_hits=2)
    assert _kept(f.filter(_doc("Kupi viagra poceni, viagra na spletu!")))
    assert _kept(f.filter(_doc("O seksu in seksa v biologiji, spet seks.")))


@pytest.mark.parametrize("text", ["o joškah", "z joškami", "pri kurbah", "brez seksa"])
def test_inflected_forms_fire(text: str) -> None:
    """Inflected forms of a lexicon entry match its stem."""
    f = _filter(min_adult_hits=1)
    assert _kept(f.filter(_doc(text))) is False


@pytest.mark.parametrize(
    "text",
    [
        "replika poslanca in replike drugih",
        "o replikah",
        "v igralnicah in igralnici",
        "oralno pršilo za grlo",
        "analiza podatkov, analni del",
        "spolnost v šoli",
    ],
)
def test_ambiguous_words_never_fire(text: str) -> None:
    """Removed ambiguous words, and their stems, never fire."""
    f = _filter(min_adult_hits=1, min_spam_hits=1)
    assert _kept(f.filter(_doc(text)))


@pytest.mark.parametrize("text", ["spletna igralnica", "spletna igralnicah", "oralni seks", "Oralni\nseks"])
def test_phrase_forms_fire(text: str) -> None:
    """Phrase-qualified forms still fire, inflecting only their last token."""
    f = _filter(min_adult_hits=1, min_spam_hits=1)
    assert _kept(f.filter(_doc(text))) is False


def test_dropped_doc_records_reason_and_terms() -> None:
    """A dropped document carries its reason and its sorted distinct stem keys."""
    f = _filter(min_adult_hits=2)
    doc = _doc("Porno, seks in spet seksi.")
    assert _kept(f.filter(doc)) is False
    assert doc.metadata["spam_reason"] == "adult_lexicon"
    assert doc.metadata["spam_terms"] == ["porn", "seks"]


def test_blocklisted_url_drops_the_document() -> None:
    """A document whose URL host is on the blocklist is dropped."""
    domains, _ = load_spam_domains()
    host = sorted(domains)[0]
    f = _filter()
    result = f.filter(_doc("Povsem nedolžno besedilo.", url=f"https://www.{host}/page"))
    assert _kept(result) is False


def test_missing_url_skips_url_check() -> None:
    """A clean document without a URL is kept (no URL signal to trip)."""
    f = _filter()
    assert _kept(f.filter(_doc("Čisto navadno besedilo brez povezave.", url=None)))


def test_keep_fraction_retains_flagged_and_tags_metadata() -> None:
    """keep_fraction=1.0 retains a flagged doc and records the reason."""
    f = _filter(min_adult_hits=2, keep_fraction=1.0)
    doc = _doc("Oglas: porno in seks vsebine na voljo.")
    assert _kept(f.filter(doc))
    assert doc.metadata.get("spam_reason")


def test_word_boundary_avoids_substring_false_positive() -> None:
    """An adult term as a substring of a benign word does not match."""
    f = _filter(min_adult_hits=1)
    # 'sex' is an English adult term, but 'Sussex' must not match it.
    assert _kept(f.filter(_doc("Brighton and Sussex on the coast.", language="en")))


def test_language_falls_back_to_default_when_metadata_missing() -> None:
    """A doc lacking metadata.language uses default_language for the lexicon."""
    f = _filter(min_adult_hits=2, default_language="sl")
    result = f.filter(_doc("porno in seks oglas", language=None))
    assert _kept(result) is False


def test_model_hook_flags_when_score_exceeds_threshold() -> None:
    """An injected model scorer flags docs even with no lexicon/URL hit."""
    sl_adult, sl_spam, _ = load_spam_lexicon("sl")
    domains, _ = load_spam_domains()
    config = SpamConfig(use_ldnoobw=False, model_threshold=0.5)
    f = SpamFilter(
        adult_words={"sl": sl_adult},
        spam_words={"sl": sl_spam},
        domains=domains,
        config=config,
        seed=0,
        model_fn=lambda text: 0.9,
    )
    assert _kept(f.filter(_doc("Popolnoma nedolžno besedilo."))) is False
