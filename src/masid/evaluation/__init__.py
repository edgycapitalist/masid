"""Evaluation pipeline for MASID.

Provides both automated metrics (token counts, duplication detection,
structural checks) and LLM-as-judge scoring for subjective quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from masid.agents import AgentOutput


@dataclass
class TrialMetrics:
    """Aggregated metrics for a single trial.

    All scores are normalized to [0, 1] where 1 is best.
    """

    trial_id: str
    architecture: str
    domain: str
    model: str

    # Task quality (primary outcome)
    quality_score: float = 0.0

    # Efficiency
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    num_rounds: int = 0

    # Coordination quality
    duplication_rate: float = 0.0
    conflict_rate: float = 0.0
    consistency_score: float = 0.0

    # Per-agent scores
    agent_scores: dict[str, float] = field(default_factory=dict)

    # Raw metadata
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_efficiency_metrics(
    all_outputs: list[list[AgentOutput]],
) -> dict[str, Any]:
    """Compute efficiency metrics from all rounds of agent outputs.

    Parameters
    ----------
    all_outputs : list of list of AgentOutput
        Outer list = rounds, inner list = agents within that round.

    Returns
    -------
    dict with keys: total_tokens, total_latency, num_rounds
    """
    total_tokens = 0
    total_latency = 0.0
    for round_outputs in all_outputs:
        for output in round_outputs:
            total_tokens += output.response.total_tokens
            total_latency += output.response.latency_seconds

    return {
        "total_tokens": total_tokens,
        "total_latency": round(total_latency, 3),
        "num_rounds": len(all_outputs),
    }


def compute_duplication_rate(outputs: list[AgentOutput]) -> float:
    """Estimate content duplication across agents in a single round.

    Uses simple n-gram overlap as a proxy. Returns a value in [0, 1].
    """
    if len(outputs) < 2:
        return 0.0

    def _get_ngrams(text: str, n: int = 4) -> set[str]:
        words = text.lower().split()
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    all_ngrams = [_get_ngrams(o.content) for o in outputs]
    total_overlap = 0
    total_possible = 0
    for i in range(len(all_ngrams)):
        for j in range(i + 1, len(all_ngrams)):
            if all_ngrams[i] and all_ngrams[j]:
                overlap = len(all_ngrams[i] & all_ngrams[j])
                smaller = min(len(all_ngrams[i]), len(all_ngrams[j]))
                total_overlap += overlap
                total_possible += smaller

    if total_possible == 0:
        return 0.0
    return round(total_overlap / total_possible, 4)
