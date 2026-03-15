# MASID — Multi-Agent System Incentive Design Benchmark

**Incentive Architecture and Collective Outcomes in Task-Oriented LLM Multi-Agent Systems**

*An Empirical Comparison of Independent Reward Maximization, Joint Reward Optimization, and Incentive-Aligned Mechanism Design*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Key Finding

> **No single incentive architecture dominates.** The optimal choice depends on the task:
> - **Information synthesis** → Cooperative optimization (JRO) wins
> - **Independent planning** → Self-interested agents (IRM) win
> - **Software development** → IRM and mechanism design (IAMD) are statistically tied; both beat cooperation
>
> Cooperative architectures (JRO) consistently cost **2–2.5× more tokens** than alternatives without proportional quality gains.

## Overview

When multiple LLM agents collaborate on practical tasks, does it matter how you structure their incentives? Should agents optimize for themselves, for the team, or should you design the system so that self-interest naturally produces good collective outcomes?

This benchmark tests three game-theoretically grounded incentive architectures across three task domains with **510 controlled trials**, providing the first empirical answer.

### The Three Architectures

| Architecture | What agents are told | Game Theory | Real-world analog |
|---|---|---|---|
| **IRM** (Independent Reward Maximization) | "Maximize YOUR output quality" | Nash Equilibrium | Independent contractors |
| **JRO** (Joint Reward Optimization) | "Maximize TEAM success" + full transparency | Pareto Optimality | Tightly-knit team |
| **IAMD** (Incentive-Aligned Mechanism Design) | "Maximize YOUR output" + performance scorecards | Mechanism Design | Well-designed marketplace |

### Task Domains

| Domain | Agents | Tasks | Evaluation |
|---|---|---|---|
| **Software Development** | Architect → Coder → Tester → Reviewer | 4 tasks | Code execution + LLM judge |
| **Research Synthesis** | Researcher A, Researcher B → Synthesizer → Fact-Checker | 3 tasks | LLM judge (information separation) |
| **Project Planning** | Scheduler → Resource Manager → Risk Analyst → QA Lead | 3 tasks | LLM judge |

## Results

### 510 Trials — Primary Experiment (Llama 3.3 70B, GPT-4.1-mini judge)

| Domain | Best | 2nd | 3rd | Significant? |
|---|---|---|---|---|
| **Software Dev** | IRM (0.536) | IAMD (0.530) | JRO (0.500) | No (p=0.51) |
| **Research Synthesis** | JRO (0.712) | IRM (0.637) | IAMD (0.635) | Yes (p<0.001) |
| **Project Planning** | IRM (0.731) | JRO (0.702) | IAMD (0.671) | Yes (p<0.001) |

### Token Efficiency

JRO uses 2–2.5× more tokens than IRM/IAMD across all domains:

| Domain | IRM | JRO | IAMD |
|---|---|---|---|
| Software Dev | 82K | **178K** | 75K |
| Research Synthesis | 34K | **90K** | 40K |
| Project Planning | 67K | **121K** | 65K |

### Key Statistical Results

- **ANOVA (Research Synthesis):** F=19.4, p<0.001 — JRO significantly outperforms both IRM and IAMD
- **ANOVA (Project Planning):** F=13.0, p<0.001 — IRM significantly outperforms IAMD (d=1.16, large effect)
- **ANOVA (Software Dev):** F=0.67, p=0.51 — No significant difference between architectures
- **Token cost:** F>900, p<0.001 in all domains — JRO's cost premium is highly significant
- **Friedman test:** Architecture rankings differ across domains (interaction effect), confirming the task-dependent finding

## Quick Start

### Prerequisites

- Python 3.11+
- API key for [Groq](https://console.groq.com/) (free tier works) or another LLM provider
- API key for [OpenAI](https://platform.openai.com/) (for the judge model)

### Installation

```bash
git clone https://github.com/edgycapitalist/masid.git
cd masid

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your-groq-key
OPENAI_API_KEY=your-openai-key
```

### Run Experiments

```bash
# Run the full experiment (510 trials, ~4-5 hours)
python -m masid.cli batch --config configs/groq_full.yaml

# The batch command is resumable — if interrupted, re-run the same command
# and it picks up where it left off (skips completed cells)

# View results
python -m masid.cli results
```

### Export Data

```bash
# Export all data (CSV, Excel, per-trial details, summary statistics)
python export_full_results.py

# Output in data/export/:
#   trials_summary.csv          — all trial scores
#   agent_outputs_full.csv      — every agent response
#   masid_full_export.xlsx      — combined Excel workbook
#   summary_stats.txt           — aggregate statistics
#   trial_details/              — one text file per trial
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
masid/
├── configs/                     # Experiment configurations
│   ├── groq_full.yaml           #   Primary experiment (Llama 70B + GPT-4.1-mini judge)
│   └── openai_gpt41mini.yaml   #   Cross-model validation
├── src/masid/
│   ├── agents/                  # Agent base class and role definitions
│   ├── architectures/           # IRM, JRO, IAMD implementations
│   │   ├── irm.py               #   Upstream-only info flow, no feedback
│   │   ├── jro.py               #   Full transparency, team objective
│   │   └── iamd.py              #   Upstream-only + performance scorecards
│   ├── domains/                 # Task specifications (with information separation)
│   ├── evaluation/              # LLM-as-judge + code sandbox
│   ├── models/                  # Unified LLM client (Groq, OpenAI, Ollama)
│   ├── orchestrator/            # Trial runner with IAMD feedback injection
│   ├── storage/                 # SQLite with resumable batch support
│   └── cli.py                   # CLI with resumable batch command
├── tests/                       # 76 unit tests
├── data/
│   ├── experiments.db           # Primary results (510 trials)
│   └── export/                  # Exported CSV, Excel, text files
└── export_full_results.py       # Data export script
```

## Experiment Design

Each trial consists of:
1. **4 agents** assigned roles appropriate to the domain
2. **3 rounds** of iterative output (agents see previous rounds and improve)
3. **Architecture-specific incentive structure** (the only variable that changes)
4. **Automated evaluation** — code sandbox for software dev, GPT-4.1-mini judge for all domains

The key experimental control: **the only difference between IRM, JRO, and IAMD is the incentive framing in the system prompt and the information/feedback flow.** The same model, same tasks, same roles, same number of rounds.

### What Differs

| | IRM | JRO | IAMD |
|---|---|---|---|
| **Objective framing** | "Maximize YOUR quality" | "Maximize TEAM success" | "Maximize YOUR quality" + scorecards |
| **Information flow** | Upstream only | Full transparency | Upstream only |
| **Between-round feedback** | None | Implicit (sees everyone) | Explicit scorecards |
| **Token cost** | Baseline | ~2x baseline | ~Baseline |

## Models Tested

| Role | Model | Provider |
|---|---|---|
| Agent (primary) | Llama 3.3 70B Versatile | Groq API |
| Judge | GPT-4.1-mini | OpenAI API |
| Agent (cross-validation) | GPT-4.1-mini | OpenAI API |

## Theoretical Foundations

The three architectures map to established paradigms:

- **IRM** → Independent Learners in MARL (Claus & Boutilier, 1998; de Witt et al., 2020), Non-cooperative game theory (Nash, 1950)
- **JRO** → Team Reward / Dec-POMDPs (Bernstein et al., 2002), MAPPO (Yu et al., 2022), QMIX (Rashid et al., 2018)
- **IAMD** → Algorithmic Mechanism Design (Nisan & Ronen, 2001), VCG mechanisms, Incentive Compatibility (Hurwicz, Maskin, Myerson — 2007 Nobel Prize)

## Related Work

This study bridges three previously non-overlapping research areas:
1. **Game-theoretic MARL** — studies RL agents, not LLMs (MADAC, MERL)
2. **LLM strategic behavior** — tests if LLMs can play games, not which incentive structure makes teams effective (GTBench, FAIRGAME, ALYMPICS)
3. **LLM multi-agent frameworks** — assume one collaboration paradigm, don't compare architectures (ChatDev, MetaGPT, CrewAI, MARBLE/MultiAgentBench)

Key related surveys: Hao et al. (2026), "Game-Theoretic Lens on LLM-based Multi-Agent Systems" — provides the theoretical framework our empirical work validates.

## Cost

The full 510-trial experiment costs approximately:
- **Groq (Llama 70B agent):** ~$35-40
- **OpenAI (GPT-4.1-mini judge):** ~$2
- **Total: ~$40**

## Citation

Paper forthcoming on arXiv (cs.MA).

```bibtex
@article{masid2026,
  title={Incentive Architecture and Collective Outcomes in Task-Oriented LLM Multi-Agent Systems},
  author={},
  year={2026},
  note={Preprint}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.

Data and benchmark released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
