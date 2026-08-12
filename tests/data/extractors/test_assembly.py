"""Tests for the shared record-assembly seam."""

from slm4ie.data.extractors.assembly import probe_doc_id, project_metadata


class TestProbeDocId:
    """Tests for probe_doc_id."""

    def test_returns_first_present_candidate(self) -> None:
        """The first candidate key with a value wins."""
        record = {"doc_id": "a", "id": "b"}
        assert probe_doc_id(record, ["doc_id", "id"]) == "a"

    def test_falls_through_to_later_candidate(self) -> None:
        """A missing earlier key falls through to a later one."""
        record = {"id": "b"}
        assert probe_doc_id(record, ["doc_id", "id"]) == "b"

    def test_skips_none_valued_candidate(self) -> None:
        """A None value is skipped in favor of the next candidate."""
        record = {"doc_id": None, "id": "b"}
        assert probe_doc_id(record, ["doc_id", "id"]) == "b"

    def test_skips_empty_string_candidate(self) -> None:
        """An empty-string value is skipped, not returned."""
        record = {"doc_id": "", "id": "b"}
        assert probe_doc_id(record, ["doc_id", "id"]) == "b"

    def test_coerces_non_string_to_str(self) -> None:
        """A non-string value is coerced via str()."""
        record = {"id": 42}
        assert probe_doc_id(record, ["id"]) == "42"

    def test_returns_none_when_no_candidate_present(self) -> None:
        """None is returned when no candidate key is present."""
        record = {"other": "x"}
        assert probe_doc_id(record, ["doc_id", "id"]) is None

    def test_returns_none_for_empty_key_list(self) -> None:
        """An empty candidate list yields None."""
        assert probe_doc_id({"id": "b"}, []) is None


class TestProjectMetadata:
    """Tests for project_metadata."""

    def test_exclude_branch_drops_excluded_keys(self) -> None:
        """Keys in the exclude set are omitted."""
        record = {"text": "hi", "keep": 1, "drop": 2}
        result = project_metadata(record, exclude={"text", "drop"})
        assert result == {"keep": 1}

    def test_exclude_branch_drops_none_values(self) -> None:
        """None-valued fields are dropped in the exclude branch."""
        record = {"keep": 1, "gone": None}
        result = project_metadata(record, exclude=set())
        assert result == {"keep": 1}

    def test_exclude_branch_preserves_order(self) -> None:
        """The exclude branch preserves record iteration order."""
        record = {"b": 1, "a": 2, "c": 3}
        result = project_metadata(record, exclude={"a"})
        assert list(result) == ["b", "c"]

    def test_whitelist_keeps_only_present_listed_keys(self) -> None:
        """Only whitelisted keys present on the record are kept."""
        record = {"title": "T", "url": "u", "other": "x"}
        result = project_metadata(record, whitelist=["title", "url", "absent"])
        assert result == {"title": "T", "url": "u"}

    def test_whitelist_drops_none_values(self) -> None:
        """A whitelisted key with a None value is dropped."""
        record = {"title": "T", "missing": None}
        result = project_metadata(record, whitelist=["title", "missing"])
        assert result == {"title": "T"}

    def test_whitelist_controls_order(self) -> None:
        """The whitelist order controls the metadata order."""
        record = {"a": 1, "b": 2, "c": 3}
        result = project_metadata(record, whitelist=["c", "a"])
        assert list(result) == ["c", "a"]

    def test_value_transform_applied_to_kept_values(self) -> None:
        """value_transform is applied to each kept value."""
        record = {"n": 2, "m": 3}
        result = project_metadata(record, value_transform=lambda v: v * 10)
        assert result == {"n": 20, "m": 30}

    def test_value_transform_never_called_on_none(self) -> None:
        """value_transform is never invoked on a None value."""

        def boom(value: object) -> object:
            assert value is not None, "transform called on None"
            return value

        record = {"keep": 1, "gone": None}
        result = project_metadata(record, value_transform=boom)
        assert result == {"keep": 1}

    def test_default_is_identity_no_exclude(self) -> None:
        """With no exclude/whitelist, every non-None field is kept as-is."""
        record = {"a": 1, "b": "x"}
        assert project_metadata(record) == {"a": 1, "b": "x"}
