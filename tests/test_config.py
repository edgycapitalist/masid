"""Tests for masid.config."""

from pathlib import Path

import pytest
import yaml

from masid.config import MASIDConfig, _deep_merge, load_config


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"model": {"name": "llama", "temp": 0.7}}
        override = {"model": {"name": "qwen"}}
        result = _deep_merge(base, override)
        assert result["model"]["name"] == "qwen"
        assert result["model"]["temp"] == 0.7

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}


class TestMASIDConfig:
    def test_defaults(self):
        config = MASIDConfig()
        assert config.model.provider == "ollama"
        assert config.model.name == "llama3.1:8b"
        assert config.experiment.trials_per_cell == 12
        assert "irm" in config.architectures
        assert "software_dev" in config.domains

    def test_custom_values(self):
        config = MASIDConfig(
            model={"provider": "openai", "name": "gpt-4o"},
            experiment={"trials_per_cell": 5},
        )
        assert config.model.provider == "openai"
        assert config.experiment.trials_per_cell == 5


class TestLoadConfig:
    def test_load_defaults(self):
        config = load_config()
        assert isinstance(config, MASIDConfig)
        assert config.model.provider == "ollama"

    def test_load_with_overrides(self):
        config = load_config(overrides={"model": {"name": "mistral:7b"}})
        assert config.model.name == "mistral:7b"

    def test_load_from_file(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "model": {"name": "test-model"},
            "experiment": {"trials_per_cell": 3},
        }))
        config = load_config(config_path=cfg_file)
        assert config.model.name == "test-model"
        assert config.experiment.trials_per_cell == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/path.yaml")
