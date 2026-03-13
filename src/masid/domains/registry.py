"""Domain registry — maps domain keys to task lists."""

from __future__ import annotations

from masid.domains import TaskSpec
from masid.domains.project_planning import get_project_planning_tasks
from masid.domains.research_synthesis import get_research_synthesis_tasks
from masid.domains.software_dev import get_software_dev_tasks

_REGISTRY: dict[str, callable] = {
    "software_dev": get_software_dev_tasks,
    "research_synthesis": get_research_synthesis_tasks,
    "project_planning": get_project_planning_tasks,
}


def get_tasks(domain: str) -> list[TaskSpec]:
    """Return task specifications for a domain.

    Raises
    ------
    ValueError
        If *domain* is not registered.
    """
    if domain not in _REGISTRY:
        raise ValueError(f"Unknown domain {domain!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[domain]()


def list_domains() -> list[str]:
    """Return all registered domain keys."""
    return list(_REGISTRY.keys())
