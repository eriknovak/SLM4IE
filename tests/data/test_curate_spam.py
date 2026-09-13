"""Tests for the adult/SEO-spam filter stage (`slm4ie.data.curate.spam`)."""

from typing import Optional

import pytest

pytest.importorskip("datatrove")

from datatrove.data import Document  # noqa: E402

from slm4ie.data.curate.spam import (  # noqa: E402
    SpamConfig,
    SpamFilter,
    load_spam_assets,
    load_spam_domains,
    load_spam_lexicon,
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
    assert {"prostitutke", "porno", "seks", "kurba"} <= adult
    assert "viagra" in spam
    assert raw  # non-empty bytes for sentinel hashing


def test_load_spam_lexicon_excludes_ambiguous_common_words() -> None:
    """Ambiguous common words are deliberately kept out of the lexicon."""
    adult, spam, _ = load_spam_lexicon("sl")
    assert "ženske" not in adult and "ženske" not in spam
    assert "masaža" not in adult and "masaža" not in spam


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
    result = f.filter(_doc("Kupi viagra poceni, viagra na spletu!"))
    assert _kept(result) is False


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
