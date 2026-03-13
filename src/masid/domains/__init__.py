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
    """

    task_id: str
    title: str
    description: str
    difficulty: str
    expected_output_hint: str
