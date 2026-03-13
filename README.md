# MASID — Multi-Agent System Incentive Design Benchmark

**Incentive Architecture and Collective Outcomes in Task-Oriented LLM Multi-Agent Systems**

An empirical comparison of Independent Reward Maximization (IRM), Joint Reward Optimization (JRO), and Incentive-Aligned Mechanism Design (IAMD) in practical, task-oriented LLM multi-agent systems.

## Overview

When multiple LLM agents collaborate on real tasks, does it matter how you structure their incentives? This benchmark framework tests three game-theoretically grounded incentive architectures across three task domains to find out.

| Architecture | Agent Objective | Game-Theoretic Model |
|---|---|---|
| **IRM** (Independent Reward Maximization) | Maximize own sub-task quality | Non-cooperative / Nash Equilibrium |
| **JRO** (Joint Reward Optimization) | Maximize collective project quality | Cooperative / Pareto Optimality |
| **IAMD** (Incentive-Aligned Mechanism Design) | Maximize own objective (designed to align with collective) | Mechanism Design / Incentive Compatibility |

### Task Domains

1. **Collaborative Software Development** — 4 agents (Architect, Coder, Tester, Reviewer)
2. **Multi-Source Research Synthesis** — 4 agents (Researcher A, Researcher B, Synthesizer, Fact-Checker)
3. **Resource-Constrained Project Planning** — 4 agents (Scheduler, Resource Manager, Risk Analyst, QA Lead)

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) for local model serving
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/masid.git
cd masid

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Pull a model via Ollama (run Ollama first with: ollama serve)
ollama pull llama3.1:8b
```

### Verify Installation

```bash
# Run unit tests
pytest tests/ -v

# Run a single smoke-test trial (requires Ollama running with a model)
python -m masid.cli smoke-test
```

### Run Experiments

```bash
# Run a single trial
python -m masid.cli run --architecture irm --domain software_dev --model llama3.1:8b

# Run a batch of trials
python -m masid.cli batch --config configs/pilot_batch.yaml

# View results
python -m masid.cli results --format table
```

## Project Structure

```
masid/
├── configs/                 # Experiment configuration files
├── src/masid/               # Main package
│   ├── agents/              # Agent definitions and roles
│   ├── architectures/       # IRM, JRO, IAMD implementations
│   ├── domains/             # Task domain specifications
│   ├── evaluation/          # Metrics and LLM-as-judge
│   ├── models/              # Unified LLM client
│   ├── orchestrator/        # Core experiment engine
│   └── storage/             # SQLite experiment logging
├── tests/                   # Unit tests
├── prompts/                 # System prompt templates (per architecture × domain)
├── data/                    # Experiment results (gitignored)
└── notebooks/               # Analysis notebooks
```

## Hardware

| Machine | Role |
|---|---|
| MacBook Pro (M-series, 24GB) | Development, testing, analysis |
| RTX 4090 Desktop (24GB VRAM, Ubuntu) | Full experiment runs |

## Models

| Tier | Models | Purpose |
|---|---|---|
| Primary | Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B v0.3 | All main trials |
| Secondary | Llama 3.1 70B (4-bit), Qwen 2.5 14B | Scale comparisons |
| Tertiary | Claude Sonnet 4.6, GPT-4o | Commercial model comparison |

## License

MIT — see [LICENSE](LICENSE) for details.
