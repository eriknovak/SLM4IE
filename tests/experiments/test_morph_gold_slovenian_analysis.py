"""Tests for the morph-gold experiment's scored-form cut and form counts in its analysis.py."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from slm4ie.tokenizers.morphology import LexemeSplit, MorphemeSegmentation, MorphLexicon

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "validation" / "morph-gold-slovenian" / "analysis.py"


def _load() -> ModuleType:
    """Import the experiment's analysis.py, whose folder name is no package name.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("morph_gold_slovenian_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = _load()


def _lexicon(*forms: str) -> MorphLexicon:
    """Return a lexicon holding each form as one stem.

    Args:
        *forms: The forms to hold.

    Returns:
        A lexicon keyed by those forms.
    """
    lexicon = MorphLexicon()
    for form in forms:
        lexicon.by_form[form] = MorphemeSegmentation(form, [form], ["stem"], form, msd="Sozei", verified=True)
    return lexicon


def test_counts_fold_case():
    """A sentence-initial capital counts toward the lowercase form."""
    counts = analysis.count_forms_in_sample(["Mačka je tu. Mačka in mačka."], ["mačka"])
    assert counts == {"mačka": 3}


def test_counts_skip_words_inside_longer_runs():
    """A form is counted only as a whole run of letters."""
    counts = analysis.count_forms_in_sample(["mačkah mačka"], ["mačka"])
    assert counts == {"mačka": 1}


def test_form_with_non_letter_never_counted():
    """A form holding a digit or hyphen cannot match a run of letters."""
    counts = analysis.count_forms_in_sample(["APZ-ja a380 ja"], ["APZ-ja", "a380"])
    assert counts == {"APZ-ja": 0, "a380": 0}


def test_cut_carries_lexeme_and_keeps_only_heldout():
    """The cut keeps held-out forms all golds cover and tags each with its lexeme."""
    golds = {"sloleks": _lexicon("mačka", "pes", "hiša"), "heuristic": _lexicon("mačka", "hiša")}
    split = LexemeSplit(
        heldout_forms=frozenset({"mačka", "pes"}),
        table_forms=frozenset({"hiša"}),
        lexeme_of={"mačka": "n1", "pes": "n2"},
        counts=[],
    )
    cut = analysis.restrict_to_scored_forms(golds, split)
    assert list(cut["heuristic"].by_form) == ["mačka"]
    assert cut["heuristic"].by_form["mačka"].lexeme == "n1"
