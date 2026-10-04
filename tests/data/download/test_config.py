"""Tests for slm4ie.data.download.config module."""

from pathlib import Path

import pytest
import yaml

from slm4ie.data.download.config import (
    ConfigError,
    DatasetConfig,
    load_download_config,
    resolve_selection,
    validate_relations,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_clarin_dataset_from_dict(self):
        """A clarin-source payload populates urls and output_dir."""
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "Test Dataset",
            "urls": ["https://example.com/file.gz"],
            "output_dir": "test_dataset",
        }
        config = DatasetConfig.from_dict("test", data)
        assert config.key == "test"
        assert config.name == "Test Dataset"
        assert config.enabled is True
        assert config.source == "clarin"
        assert config.urls == ["https://example.com/file.gz"]
        assert config.output_dir == "test_dataset"
        assert config.manual is False
        assert config.repo_id is None
        assert config.configs is None
        assert config.note is None

    def test_huggingface_dataset_from_dict(self):
        """A huggingface-source payload populates repo_id and configs."""
        data = {
            "enabled": True,
            "source": "huggingface",
            "name": "FinePDF",
            "repo_id": "HuggingFaceFW/finepdfs",
            "configs": ["slv_Latn"],
            "output_dir": "finepdf",
        }
        config = DatasetConfig.from_dict("finepdf", data)
        assert config.source == "huggingface"
        assert config.repo_id == "HuggingFaceFW/finepdfs"
        assert config.configs == ["slv_Latn"]
        assert config.urls == []

    def test_manual_dataset_from_dict(self):
        """`manual: true` and a note round-trip into the config."""
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "KAS 2.0",
            "manual": True,
            "urls": ["https://example.com/handle"],
            "output_dir": "kas",
            "note": "Download manually.",
        }
        config = DatasetConfig.from_dict("kas", data)
        assert config.manual is True
        assert config.note == "Download manually."

    def test_disabled_dataset_from_dict(self):
        """A disabled entry defaults source and output_dir to empty strings."""
        data = {
            "enabled": False,
            "name": "Gigafida 2.0",
            "note": "Not available.",
        }
        config = DatasetConfig.from_dict("gigafida", data)
        assert config.enabled is False
        assert config.source == ""
        assert config.output_dir == ""

    def test_pretraining_dataset_defaults_role_pretrain(self):
        """Datasets default `role` to `pretrain` with empty tasks."""
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "DS",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "ds",
        }
        config = DatasetConfig.from_dict("ds", data)
        assert config.role == "pretrain"
        assert config.tasks == []

    def test_benchmark_dataset_from_dict(self):
        """`role: benchmark` and tasks list survive into the config."""
        data = {
            "enabled": True,
            "role": "benchmark",
            "source": "clarin",
            "name": "SUK",
            "urls": ["https://example.com/suk.zip"],
            "output_dir": "suk",
            "tasks": ["POS", "NER", "DEP"],
        }
        config = DatasetConfig.from_dict("suk", data)
        assert config.role == "benchmark"
        assert config.tasks == ["POS", "NER", "DEP"]

    def test_lexicon_dataset_from_dict(self):
        """`role: lexicon` round-trips into the config."""
        data = {
            "enabled": True,
            "role": "lexicon",
            "source": "http",
            "name": "Sloleks",
            "urls": ["https://example.com/sloleks.zip"],
            "output_dir": "sloleks",
            "tasks": ["TOKENIZER"],
        }
        config = DatasetConfig.from_dict("sloleks", data)
        assert config.role == "lexicon"

    def test_unknown_role_raises_config_error(self):
        """An unknown `role` value is rejected with a ConfigError."""
        data = {
            "enabled": True,
            "role": "bogus",
            "source": "http",
            "name": "DS",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "ds",
        }
        with pytest.raises(ConfigError) as excinfo:
            DatasetConfig.from_dict("ds", data)
        msg = str(excinfo.value)
        assert "bogus" in msg
        assert "role" in msg

    def test_provider_field_round_trips(self):
        """The optional `publisher` field round-trips through from_dict."""
        data = {
            "enabled": True,
            "source": "http",
            "name": "Test",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "test",
            "publisher": "clarin.si",
        }
        config = DatasetConfig.from_dict("test", data)
        assert config.publisher == "clarin.si"

    def test_provider_field_default_none(self):
        """The `publisher` field defaults to None when omitted."""
        data = {
            "enabled": True,
            "source": "http",
            "name": "Test",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "test",
        }
        config = DatasetConfig.from_dict("test", data)
        assert config.publisher is None

    def test_enabled_non_manual_requires_output_dir(self):
        """An enabled non-manual entry without output_dir is rejected."""
        data = {
            "enabled": True,
            "source": "http",
            "name": "Test",
            "urls": ["https://example.com/x.gz"],
        }
        with pytest.raises(ConfigError) as excinfo:
            DatasetConfig.from_dict("test", data)
        assert "output_dir" in str(excinfo.value)
        assert excinfo.value.problems == ["test: missing or empty 'output_dir'"]

    def test_disabled_entry_skips_output_dir_check(self):
        """Disabled entries may omit output_dir without raising."""
        data = {
            "enabled": False,
            "name": "Gigafida",
            "note": "Not available.",
        }
        config = DatasetConfig.from_dict("gigafida", data)
        assert config.output_dir == ""

    def test_manual_entry_skips_output_dir_check(self):
        """Manual entries may omit output_dir without raising."""
        data = {
            "enabled": True,
            "manual": True,
            "source": "http",
            "name": "KAS",
            "note": "Download manually.",
        }
        config = DatasetConfig.from_dict("kas", data)
        assert config.manual is True
        assert config.output_dir == ""


class TestConfigError:
    """Tests for ConfigError construction and message format."""

    def test_problems_attribute_round_trips(self):
        """The list of problems is preserved on the exception instance."""
        problems = ["ds1: missing field", "ds2: bad source 'bogus'"]
        err = ConfigError(problems)
        assert err.problems == problems

    def test_message_includes_count_and_each_problem(self):
        """The rendered message lists count and each problem."""
        problems = ["ds1: missing field", "ds2: bad source 'bogus'"]
        err = ConfigError(problems)
        text = str(err)
        assert "2 problem(s)" in text
        assert "ds1: missing field" in text
        assert "ds2: bad source 'bogus'" in text

    def test_independent_problems_list_is_copied(self):
        """Mutating the input list after construction does not leak in."""
        problems = ["ds1: missing field"]
        err = ConfigError(problems)
        problems.append("ds2: extra")
        assert err.problems == ["ds1: missing field"]


class TestLoadConfig:
    """Tests for load_download_config."""

    def test_load_valid_config(self, tmp_path: Path):
        """`load_download_config` parses output_dir and the datasets mapping."""
        config_data = {
            "output_dir": "data/raw",
            "datasets": {
                "test_ds": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "Test",
                    "urls": ["https://example.com/f.gz"],
                    "output_dir": "test_ds",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        catalog = load_download_config(config_file)
        datasets = catalog.datasets
        assert catalog.output_dir == "data/raw"
        assert len(datasets) == 1
        assert "test_ds" in datasets
        assert datasets["test_ds"].name == "Test"

    def test_load_config_file_not_found(self):
        """A missing config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_download_config(Path("/nonexistent/config.yaml"))

    def test_load_config_multiple_datasets(self, tmp_path: Path):
        """Multiple datasets keep their individual enabled flags."""
        config_data = {
            "output_dir": "data/raw",
            "datasets": {
                "ds1": {
                    "enabled": True,
                    "source": "clarin",
                    "name": "DS1",
                    "urls": ["https://example.com/1.gz"],
                    "output_dir": "ds1",
                },
                "ds2": {
                    "enabled": False,
                    "name": "DS2",
                    "note": "Disabled.",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(config_data))
        datasets = load_download_config(config_file).datasets
        assert len(datasets) == 2
        assert datasets["ds1"].enabled is True
        assert datasets["ds2"].enabled is False


class TestLoadConfigOverlay:
    """Tests for the sibling `*.local.yaml` overlay merge."""

    def test_overlay_patches_individual_fields(self, tmp_path: Path):
        """A local overlay patches fields without restating the entry."""
        base = {
            "output_dir": "data/raw",
            "datasets": {
                "gigafida": {
                    "enabled": False,
                    "name": "Gigafida 2.2",
                    "source": "http",
                    "output_dir": "gigafida",
                    "note": "Stub.",
                },
            },
        }
        overlay = {
            "datasets": {
                "gigafida": {
                    "enabled": True,
                    "urls": ["https://example.com/a.gz"],
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(base))
        (tmp_path / "download.local.yaml").write_text(yaml.dump(overlay))

        datasets = load_download_config(config_file).datasets
        gigafida = datasets["gigafida"]
        # Overlay wins on patched fields.
        assert gigafida.enabled is True
        assert gigafida.urls == ["https://example.com/a.gz"]
        # Base fields survive the merge.
        assert gigafida.name == "Gigafida 2.2"
        assert gigafida.source == "http"
        assert gigafida.output_dir == "gigafida"
        assert gigafida.note == "Stub."

    def test_overlay_can_add_new_dataset(self, tmp_path: Path):
        """An overlay may introduce a dataset absent from the base."""
        base = {"output_dir": "data/raw", "datasets": {}}
        overlay = {
            "datasets": {
                "secret_ds": {
                    "enabled": True,
                    "source": "http",
                    "name": "Secret",
                    "urls": ["https://example.com/s.gz"],
                    "output_dir": "secret_ds",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(base))
        (tmp_path / "download.local.yaml").write_text(yaml.dump(overlay))

        datasets = load_download_config(config_file).datasets
        assert "secret_ds" in datasets
        assert datasets["secret_ds"].urls == ["https://example.com/s.gz"]

    def test_no_overlay_leaves_base_unchanged(self, tmp_path: Path):
        """Without a sibling overlay the base config loads verbatim."""
        base = {
            "output_dir": "data/raw",
            "datasets": {
                "gigafida": {
                    "enabled": False,
                    "name": "Gigafida 2.2",
                    "note": "Stub.",
                },
            },
        }
        config_file = tmp_path / "download.yaml"
        config_file.write_text(yaml.dump(base))

        datasets = load_download_config(config_file).datasets
        assert datasets["gigafida"].enabled is False
        assert datasets["gigafida"].urls == []


def _write_catalog(tmp_path: Path, datasets: dict) -> Path:
    """Write a minimal download catalog holding *datasets* and return its path."""
    path = tmp_path / "download.yaml"
    path.write_text(yaml.dump({"output_dir": "data/raw", "datasets": datasets}))
    return path


def _entry(**fields) -> dict:
    """Return an enabled pretraining entry with *fields* layered on top."""
    return {"source": "http", "output_dir": "x", **fields}


class TestContainmentFields:
    """The `contains` and `overlaps` relations on a catalog entry."""

    def test_fields_default_to_empty(self):
        """An entry without relations has empty `contains` and `overlaps`."""
        config = DatasetConfig.from_dict("a", _entry())
        assert config.contains == []
        assert config.overlaps == []

    def test_fields_round_trip(self, tmp_path: Path):
        """Declared relations are parsed into the entry."""
        path = _write_catalog(tmp_path, {"a": _entry(contains=["b"], overlaps=["c"]), "b": _entry(), "c": _entry()})
        catalog = load_download_config(path)
        assert catalog.datasets["a"].contains == ["b"]
        assert catalog.datasets["a"].overlaps == ["c"]

    def test_unknown_key_raises(self, tmp_path: Path):
        """A relation naming a key outside the registry is a config error."""
        path = _write_catalog(tmp_path, {"a": _entry(contains=["ghost"], overlaps=["phantom"])})
        with pytest.raises(ConfigError) as excinfo:
            load_download_config(path)
        assert any("ghost" in p for p in excinfo.value.problems)
        assert any("phantom" in p for p in excinfo.value.problems)

    def test_self_reference_raises(self, tmp_path: Path):
        """An entry may not contain or overlap itself."""
        path = _write_catalog(tmp_path, {"a": _entry(contains=["a"]), "b": _entry(overlaps=["b"])})
        with pytest.raises(ConfigError) as excinfo:
            load_download_config(path)
        assert len(excinfo.value.problems) == 2

    def test_cycle_raises(self, tmp_path: Path):
        """A `contains` cycle, direct or through a third entry, is a config error."""
        path = _write_catalog(
            tmp_path, {"a": _entry(contains=["b"]), "b": _entry(contains=["c"]), "c": _entry(contains=["a"])}
        )
        with pytest.raises(ConfigError, match="cycle"):
            load_download_config(path)

    def test_non_list_raises(self):
        """A relation must be a list of keys."""
        with pytest.raises(ConfigError, match="contains"):
            DatasetConfig.from_dict("a", _entry(contains="b"))

    def test_shipped_registry_loads(self):
        """The committed catalog passes relation validation."""
        root = Path(__file__).resolve().parents[3]
        catalog = load_download_config(root / "configs" / "data" / "download.yaml")
        assert catalog.datasets


def _catalog(**entries: dict) -> dict:
    """Parse keyword entries into catalog `DatasetConfig`s, validating relations."""
    datasets = {key: DatasetConfig.from_dict(key, _entry(**fields)) for key, fields in entries.items()}
    validate_relations(datasets)
    return datasets


class TestResolveSelection:
    """`resolve_selection` picks one representative per containment group."""

    def test_contained_member_is_skipped(self):
        """A member of an enabled superset is skipped and names the superset."""
        selection = resolve_selection(_catalog(big={"contains": ["small"]}, small={}, other={}))
        assert selection.selected == ["big", "other"]
        assert selection.status["small"] == "skipped: contained in big"

    def test_containment_is_transitive(self):
        """The outermost enabled superset wins over every nested member."""
        selection = resolve_selection(_catalog(top={"contains": ["mid"]}, mid={"contains": ["leaf"]}, leaf={}))
        assert selection.selected == ["top"]
        assert selection.status["mid"] == "skipped: contained in top"
        assert selection.status["leaf"] == "skipped: contained in top"

    def test_disabled_superset_leaves_members_selected(self):
        """A disabled superset is not a representative; its members stay."""
        selection = resolve_selection(
            _catalog(top={"enabled": False, "contains": ["mid"]}, mid={"contains": ["leaf"]}, leaf={})
        )
        assert selection.selected == ["mid"]
        assert selection.status["top"] == "skipped: disabled"
        assert selection.status["leaf"] == "skipped: contained in mid"

    def test_non_pretrain_superset_is_not_a_representative(self):
        """A benchmark that contains a corpus does not displace it."""
        selection = resolve_selection(_catalog(gold={"role": "benchmark", "contains": ["corpus"]}, corpus={}))
        assert selection.selected == ["corpus"]
        assert selection.status["gold"] == "skipped: role benchmark"

    def test_overlaps_only_warn(self):
        """An overlapping pair is selected in full and reported once."""
        selection = resolve_selection(_catalog(a={"overlaps": ["b"]}, b={"overlaps": ["a"]}, c={"overlaps": ["a"]}))
        assert selection.selected == ["a", "b", "c"]
        assert selection.overlaps == [("a", "b"), ("a", "c")]

    def test_overlaps_are_not_transitive(self):
        """a~b and b~c does not report a~c."""
        selection = resolve_selection(_catalog(a={"overlaps": ["b"]}, b={"overlaps": ["c"]}, c={}))
        assert selection.overlaps == [("a", "b"), ("b", "c")]

    def test_overlap_with_skipped_entry_is_not_reported(self):
        """Only pairs that both enter the corpus warn."""
        selection = resolve_selection(_catalog(a={"overlaps": ["b"]}, b={"enabled": False}))
        assert selection.overlaps == []

    def test_every_entry_has_a_status(self):
        """Every catalog entry is listed, in declaration order."""
        datasets = _catalog(a={}, b={"enabled": False}, c={"role": "lexicon"})
        assert list(resolve_selection(datasets).status) == ["a", "b", "c"]
