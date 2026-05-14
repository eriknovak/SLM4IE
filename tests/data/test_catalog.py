"""Tests for slm4ie.data.catalog module."""

from pathlib import Path

import pytest
import yaml

from slm4ie.data.catalog import ConfigError, DatasetConfig, load_config


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

    def test_pretraining_dataset_defaults_benchmark_false(self):
        """Pretraining datasets default `benchmark` to False with empty tasks."""
        data = {
            "enabled": True,
            "source": "clarin",
            "name": "DS",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "ds",
        }
        config = DatasetConfig.from_dict("ds", data)
        assert config.benchmark is False
        assert config.tasks == []

    def test_benchmark_dataset_from_dict(self):
        """`benchmark: true` and tasks list survive into the config."""
        data = {
            "enabled": True,
            "benchmark": True,
            "source": "clarin",
            "name": "SUK",
            "urls": ["https://example.com/suk.zip"],
            "output_dir": "suk",
            "tasks": ["POS", "NER", "DEP"],
        }
        config = DatasetConfig.from_dict("suk", data)
        assert config.benchmark is True
        assert config.tasks == ["POS", "NER", "DEP"]

    def test_provider_field_round_trips(self):
        """The optional `provider` field round-trips through from_dict."""
        data = {
            "enabled": True,
            "source": "http",
            "name": "Test",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "test",
            "provider": "clarin.si",
        }
        config = DatasetConfig.from_dict("test", data)
        assert config.provider == "clarin.si"

    def test_provider_field_default_none(self):
        """The `provider` field defaults to None when omitted."""
        data = {
            "enabled": True,
            "source": "http",
            "name": "Test",
            "urls": ["https://example.com/x.gz"],
            "output_dir": "test",
        }
        config = DatasetConfig.from_dict("test", data)
        assert config.provider is None

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
        assert excinfo.value.problems == [
            "test: missing or empty 'output_dir'"
        ]

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
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path: Path):
        """`load_config` parses output_dir and the datasets mapping."""
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
        output_dir, datasets = load_config(config_file)
        assert output_dir == "data/raw"
        assert len(datasets) == 1
        assert "test_ds" in datasets
        assert datasets["test_ds"].name == "Test"

    def test_load_config_file_not_found(self):
        """A missing config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

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
        _output_dir, datasets = load_config(config_file)
        assert len(datasets) == 2
        assert datasets["ds1"].enabled is True
        assert datasets["ds2"].enabled is False
