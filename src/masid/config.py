"""Configuration management for MASID experiments.

Loads YAML config files with defaults, validates settings via Pydantic,
and merges overrides from CLI flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Settings for the LLM endpoint."""

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    name: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120


class ExperimentConfig(BaseModel):
    """Core experiment parameters."""

    trials_per_cell: int = 12
    max_rounds: int = 3
    seed: Optional[int] = 42


class EvaluationConfig(BaseModel):
    """Evaluation / LLM-as-judge settings."""

    judge_model: str = "llama3.1:8b"
    judge_provider: str = "ollama"
    spot_check_pct: float = 0.10


class RobustnessConfig(BaseModel):
    """Robustness / fault-injection settings."""

    enabled: bool = False
    trials_per_cell: int = 20
    fault_types: list[str] = Field(default_factory=lambda: ["degraded_prompt"])


class StorageConfig(BaseModel):
    """Database storage settings."""

    db_path: str = "data/experiments.db"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    file: Optional[str] = None


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class MASIDConfig(BaseModel):
    """Root configuration object for an experiment run."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    architectures: list[str] = Field(default_factory=lambda: ["irm", "jro", "iamd"])
    domains: list[str] = Field(
        default_factory=lambda: ["software_dev", "research_synthesis", "project_planning"]
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

_DEFAULTS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[dict] = None,
) -> MASIDConfig:
    """Load and validate a MASID configuration.

    1. Start from built-in defaults (``configs/default.yaml``).
    2. If *config_path* is given, merge that file on top.
    3. If *overrides* dict is given, merge those on top.
    4. Validate via Pydantic and return a ``MASIDConfig``.
    """
    # Load defaults
    base: dict = {}
    if _DEFAULTS_PATH.exists():
        with open(_DEFAULTS_PATH) as fh:
            base = yaml.safe_load(fh) or {}

    # Merge config file
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        _deep_merge(base, file_cfg)

    # Merge CLI overrides
    if overrides:
        _deep_merge(base, overrides)

    return MASIDConfig(**base)
