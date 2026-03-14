"""Base architecture interface for MASID.

Each incentive architecture (IRM, JRO, IAMD) is a subclass of
``BaseArchitecture``. The architecture controls:
1. How system prompts are constructed for each agent.
2. What information agents share with each other between turns.
3. How individual agents are evaluated / scored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from masid.agents import Agent, AgentOutput
from masid.agents.roles import RoleSpec
from masid.domains import TaskSpec
from masid.models import LLMClient


class BaseArchitecture(ABC):
    """Abstract base class for incentive architectures."""

    # Human-readable name (e.g. "Independent Reward Maximization")
    name: str = ""
    # Short key used in configs and databases (e.g. "irm")
    key: str = ""

    @abstractmethod
    def build_system_prompt(self, role_spec: RoleSpec, task_description: str) -> str:
        """Construct the system prompt for an agent given its role and the task.

        The prompt encodes the architecture's incentive structure: what the
        agent is told to optimize for and what context it receives.
        """

    @abstractmethod
    def build_agents(
        self,
        role_specs: list[RoleSpec],
        task: TaskSpec,
        client: LLMClient,
    ) -> list[Agent]:
        """Instantiate the full set of agents for a trial.

        Uses ``task.get_description_for_role(role)`` to give each agent
        only the information appropriate for their role.
        """

    @abstractmethod
    def run_round(
        self,
        agents: list[Agent],
        task_description: str,
        round_number: int,
        previous_outputs: list[AgentOutput],
    ) -> list[AgentOutput]:
        """Execute one round of the multi-agent workflow.

        A *round* is a full pass where every agent acts once. The
        architecture determines the order of execution and what context
        each agent receives from others.

        Parameters
        ----------
        agents : list of Agent
        task_description : str
            The task specification for this trial.
        round_number : int
            0-indexed round number.
        previous_outputs : list of AgentOutput
            Outputs from the previous round (empty for round 0).

        Returns
        -------
        list of AgentOutput
            One output per agent for this round.
        """

    @abstractmethod
    def compute_agent_scores(
        self,
        agents: list[Agent],
        outputs: list[AgentOutput],
        collective_score: float,
    ) -> dict[str, float]:
        """Compute per-agent scores given the architecture's incentive rules.

        Parameters
        ----------
        agents : list of Agent
        outputs : list of AgentOutput
            Final-round outputs.
        collective_score : float
            The overall task quality score (from the evaluation pipeline).

        Returns
        -------
        dict mapping agent_id → individual score
        """

    def get_metadata(self) -> dict[str, Any]:
        """Return architecture metadata for logging."""
        return {"architecture": self.key, "name": self.name}
