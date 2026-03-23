# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MASID (Multi-Agent System Incentive Design) is a research benchmark comparing three game-theoretically grounded incentive architectures (IRM, JRO, IAMD) for LLM multi-agent systems across three task domains (software dev, research synthesis, project planning). The experiment runs 510 controlled trials (3 architectures × 3 domains × ~10 tasks × 17 trials).

## Setup and Commands

```bash
# Install
pip install -e ".[dev]"          # core + dev deps
pip install -e ".[all]"          # core + dev + analysis

# CLI (entry point: masid = masid.cli:main)
python -m masid.cli smoke-test                                    # verify LLM connectivity
python -m masid.cli run --architecture irm --domain software_dev  # single trial
python -m masid.cli batch --config configs/groq_full.yaml         # full batch (resumable)
python -m masid.cli results                                       # view results

# Testing
pytest tests/ -v                          # all 76 tests
pytest tests/ -m "not integration" -v     # skip tests requiring live LLM
pytest tests/test_architectures.py -v     # single test file

# Linting & types
ruff check src/                           # lint (rules: E, F, I, N, W, UP; line-length 100)
mypy src/                                 # type checking (disallow_untyped_defs)

# Export data
python export_full_results.py
```

API keys go in `.env` (GROQ_API_KEY, OPENAI_API_KEY for judge).

## Architecture

Source layout: `src/masid/` package with Click CLI entry point.

**Core flow:** CLI (`cli.py`) → `TrialRunner` (orchestrator) → Architecture + Agents → Evaluation → SQLite storage

Key modules:

- **`cli.py`** — Click CLI with commands: `smoke-test`, `run`, `batch`, `results`, `list-domains`. Batch is resumable via DB seed tracking.
- **`config.py`** — YAML config loading with Pydantic validation. Merges defaults → YAML → CLI overrides. Configs live in `configs/` (primary: `groq_full.yaml`).
- **`agents/`** — `Agent` class (maintains conversation history, generates via LLMClient) and `roles.py` (16 `RoleSpec` definitions across 3 domains, 4 agents per domain).
- **`architectures/`** — `BaseArchitecture` ABC with three implementations:
  - **IRM** — agents maximize own quality, upstream-only info flow, no feedback
  - **JRO** — shared objective, full transparency (all agents see all outputs), ~2× token cost
  - **IAMD** — self-interested agents + explicit performance scorecards between rounds
  - `registry.py` — `get_architecture()` factory
- **`domains/`** — `TaskSpec` definitions: `software_dev.py` (4 tasks), `research_synthesis.py` (3 tasks, info-separated), `project_planning.py` (3 tasks). `registry.py` provides `get_tasks(domain)`.
- **`models/`** — `LLMClient` wrapping LiteLLM. Supports Ollama/Groq/OpenAI/Anthropic. Retries with backoff, strips `<think>` tags.
- **`orchestrator/`** — `TrialRunner.run()` orchestrates: create agents → run N rounds → evaluate → compute scores → persist.
- **`evaluation/`** — `judge.py` (LLM-as-judge, 4 dimensions: correctness/completeness/coherence/integration, 0-1 normalized), `sandbox.py` (subprocess code execution with 30s timeout + pytest for software_dev).
- **`storage/`** — `ExperimentDB` using SQLite (`data/experiments.db`). Tables: `trials`, `agent_outputs`. Supports resume via `count_trials_for_cell()` and `max_seed()`.

## Key Design Decisions

- **Controlled variable**: only the architecture (incentive framing + info/feedback flow) changes between trials; same model, tasks, roles, and round count.
- **Research synthesis** tasks deliberately partition source material between researchers (info separation).
- **Software dev** evaluation uses real code execution in sandboxed subprocesses, not just LLM judging.
- Registry/factory pattern for architectures and domains enables adding new ones without touching orchestration code.
