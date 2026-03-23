"""Configuration for CrewAI adapter experiments."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CrewAIModelConfig(BaseModel):
    """Settings for the CrewAI LLM (Ollama endpoint)."""

    model: str = "ollama/llama3.3:70b"
    base_url: str = "http://localhost:11434"
    api_key: str = "dummy"
    temperature: float = 0.7
    max_tokens: int = 2048


class JudgeConfig(BaseModel):
    """Settings for the LLM-as-judge (reuses MASID evaluation)."""

    judge_provider: str = "openai"
    judge_model: str = "gpt-4.1-mini"
    judge_base_url: str | None = None


class ExperimentConfig(BaseModel):
    """Core experiment parameters."""

    trials_per_cell: int = 10
    max_rounds: int = 3


class StorageConfig(BaseModel):
    """Database storage settings."""

    db_path: str = "data/crewai_experiments.db"


class CrewAIConfig(BaseModel):
    """Root configuration for CrewAI adapter experiments."""

    agent_model: CrewAIModelConfig = Field(default_factory=CrewAIModelConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    architectures: list[str] = Field(
        default_factory=lambda: ["irm", "jro", "iamd"]
    )
    domains: list[str] = Field(
        default_factory=lambda: ["software_dev", "research_synthesis", "project_planning"]
    )
    task_ids: dict[str, str] = Field(
        default_factory=lambda: {
            "software_dev": "sw_001",
            "research_synthesis": "rs_001",
            "project_planning": "pp_001",
        }
    )


def load_crewai_config(config_path: str | Path | None = None) -> CrewAIConfig:
    """Load a CrewAI experiment config from YAML."""
    if config_path is None:
        return CrewAIConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}

    return CrewAIConfig(**data)
