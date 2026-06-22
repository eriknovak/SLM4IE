"""Tests for slm4ie/data/sloleks_relations.py."""

from pathlib import Path

from slm4ie.data.sloleks_relations import (
    find_word_relations_tsv,
    iter_word_relation_segmentations,
    iter_word_relations,
    split_morphemes,
)

#: A header row (rejected by the decomposed-equals-lemma check) followed by data
#: rows: a score-2 verified pair, a score-0 pair (dropped), an unscored pair, a
#: second unscored occurrence of an already-verified lemma, and a malformed row
#: whose decomposition does not reproduce its lemma.
_ROWS = [
    "original\trelated\torig_dec\trel_dec\tmte_o\tmte_r\toid\trid\toverlap\trule_id\tpattern",
    "pisati\tpisatelj\tpis_ati\tpis_at_elj\tG\tSom\t1\t2\tpis\tG.Som.5\t[G]_ati->[G]_at_elj\t2",
    "pisati\tpisanje\tpis_ati\tpis_a_nje\tG\tSon\t1\t3\tpis\tG.Son.1\tx\t0",
    "hiša\thišica\thiš_a\thiš_ic_a\tSozei\tSozei\t4\t5\thiš\tS.S.1\ty",
    "delati\tpisatelj\tdel_ati\tpis_at_elj\tG\tSom\t6\t7\tx\tG.Som.9\tz",
    "x\tnoise\tx\tno_pe\tA\tB\t8\t9\tn\tR.1\tp",
]


def _write_tsv(path: Path) -> Path:
    """Write the fixture rows to a TSV file.

    Args:
        path (Path): Destination `.tsv` path.

    Returns:
        Path: `path`, for chaining.
    """
    path.write_text("\n".join(_ROWS) + "\n", encoding="utf-8")
    return path


class TestSplitMorphemes:
    """Tests for the underscore splitter."""

    def test_splits_on_underscore(self):
        """Underscore-separated pieces are returned in order."""
        assert split_morphemes("pis_at_elj") == ["pis", "at", "elj"]

    def test_drops_empty_pieces(self):
        """Leading, trailing, and doubled underscores yield no empty pieces."""
        assert split_morphemes("_pis__at_") == ["pis", "at"]


class TestIterWordRelations:
    """Tests for the per-row parser."""

    def test_header_and_malformed_rows_rejected(self, tmp_path: Path):
        """Rows whose decomposition does not reproduce the lemma are skipped."""
        records = list(iter_word_relations(_write_tsv(tmp_path / "wr.tsv")))
        lemmas = [r["lemma"] for r in records]
        assert "noise" not in lemmas  # malformed: no_pe != noise
        assert "related" not in lemmas  # header line

    def test_score_zero_dropped_and_verified_flag(self, tmp_path: Path):
        """Score-0 rows are dropped; a present score marks the row verified."""
        records = list(iter_word_relations(_write_tsv(tmp_path / "wr.tsv")))
        by_lemma = {r["lemma"]: r for r in records if r["lemma"] != "pisatelj"}
        assert "pisanje" not in by_lemma  # score 0 -> dropped
        assert by_lemma["hišica"]["verified"] is False  # no score column
        verified_pisatelj = [r for r in records if r["lemma"] == "pisatelj" and r["verified"]]
        assert verified_pisatelj and verified_pisatelj[0]["morphemes"] == ["pis", "at", "elj"]

    def test_msd_and_rule_id_carried(self, tmp_path: Path):
        """The related-lemma MSD and rule id are carried through."""
        records = list(iter_word_relations(_write_tsv(tmp_path / "wr.tsv")))
        hisica = next(r for r in records if r["lemma"] == "hišica")
        assert hisica["msd"] == "Sozei"
        assert hisica["rule_id"] == "S.S.1"


class TestIterWordRelationSegmentations:
    """Tests for the deduplicated, lemma-keyed view."""

    def test_dedup_prefers_verified(self, tmp_path: Path):
        """A verified segmentation wins over a later unverified one."""
        records = list(iter_word_relation_segmentations(_write_tsv(tmp_path / "wr.tsv")))
        lemmas = [r["lemma"] for r in records]
        assert lemmas.count("pisatelj") == 1
        pisatelj = next(r for r in records if r["lemma"] == "pisatelj")
        assert pisatelj["verified"] is True
        assert pisatelj["morphemes"] == ["pis", "at", "elj"]


class TestFindWordRelationsTsv:
    """Tests for locating the data TSV in a download directory."""

    def test_prefers_data_over_rules_file(self, tmp_path: Path):
        """The data file is chosen over the sibling rules file."""
        (tmp_path / "nssss_sloleks_word_relations_1.1.tsv").write_text("", encoding="utf-8")
        (tmp_path / "nssss_sloleks_word_relation_rules_1.1.tsv").write_text("", encoding="utf-8")
        found = find_word_relations_tsv(tmp_path)
        assert found is not None and "rules" not in found.name

    def test_returns_none_when_absent(self, tmp_path: Path):
        """No matching TSV yields None."""
        assert find_word_relations_tsv(tmp_path) is None
