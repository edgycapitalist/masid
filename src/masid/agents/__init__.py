"""Base agent abstraction for MASID.

Every agent in the framework (Architect, Coder, Tester, etc.) is an instance
of ``Agent`` configured with a role, system prompt, and evaluation criteria.
The agent itself is model-agnostic — it delegates inference to an ``LLMClient``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from masid.models import LLMClient, LLMResponse


@dataclass
class AgentMessage:
    """A single message in the agent's conversation history."""

    role: str  # "system", "user", "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Structured output from a single agent turn."""

    agent_id: str
    role: str
    content: str
    response: LLMResponse
    round_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """An LLM-powered agent with a specific role and system prompt.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent instance (e.g. ``"coder_0"``).
    role : str
        Human-readable role name (e.g. ``"Coder"``).
    system_prompt : str
        The system-level instruction that defines this agent's behavior.
    client : LLMClient
        The LLM client to use for inference.
    evaluation_criteria : dict or None
        Criteria weights used by the evaluation pipeline to score this
        agent's output (architecture-dependent).
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        system_prompt: str,
        client: LLMClient,
        evaluation_criteria: Optional[dict[str, float]] = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.evaluation_criteria = evaluation_criteria or {}
        self._history: list[AgentMessage] = [
            AgentMessage(role="system", content=system_prompt)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, prompt: str, round_number: int = 0) -> AgentOutput:
        """Generate a response to *prompt* and append to history.

        Parameters
        ----------
        prompt : str
            The user/orchestrator message for this turn.
        round_number : int
            Current revision round (0-indexed).

        Returns
        -------
        AgentOutput
        """
        self._history.append(AgentMessage(role="user", content=prompt))

        messages = [{"role": m.role, "content": m.content} for m in self._history]
        response = self.client.chat(messages)

        self._history.append(AgentMessage(role="assistant", content=response.content))

        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            content=response.content,
            response=response,
            round_number=round_number,
        )

    def inject_context(self, content: str, role: str = "user") -> None:
        """Add context to the agent's history without triggering generation.

        Used by architectures (e.g. JRO) to share other agents' outputs
        with this agent.
        """
        self._history.append(AgentMessage(role=role, content=content))

    def reset(self) -> None:
        """Clear conversation history (keeps only system prompt)."""
        self._history = [
            AgentMessage(role="system", content=self.system_prompt)
        ]

    @property
    def history(self) -> list[AgentMessage]:
        """Return a copy of the conversation history."""
        return list(self._history)

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id!r}, role={self.role!r})"
