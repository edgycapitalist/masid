"""Task domain definitions for MASID.

Each domain provides a set of task specifications (the concrete problems
agents will solve) and domain-specific evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """A single task specification within a domain.

    Attributes
    ----------
    task_id : str
        Unique identifier (e.g. ``"sw_001"``).
    title : str
        Short title.
    description : str
        Full task description given to agents.
    difficulty : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    expected_output_hint : str
        Brief description of what a good output looks like (used by evaluators).
    role_sources : dict or None
        Optional mapping from role name to role-specific source material.
        When present, each agent receives ONLY the sources assigned to
        their role instead of the full description. Agents not listed
        receive the full description. Used for information separation
        in the research synthesis domain.
    """

    task_id: str
    title: str
    description: str
    difficulty: str
    expected_output_hint: str
    role_sources: dict[str, str] | None = None

    def get_description_for_role(self, role: str) -> str:
        """Return the task description appropriate for a specific role.

        If role_sources is set and the role is listed, returns the base
        description with role-specific sources appended. Otherwise
        returns the full description.
        """
        if self.role_sources and role in self.role_sources:
            return self.role_sources[role]
        return self.description
