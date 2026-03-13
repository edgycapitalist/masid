"""Role definitions for each task domain.

Each domain has 4 agent roles. This module provides the role metadata
(name, description, upstream/downstream dependencies) used by the
architecture implementations to construct system prompts and wire
inter-agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    """Specification for a single agent role within a domain."""

    role: str
    description: str
    upstream: list[str]  # roles this agent receives input from
    downstream: list[str]  # roles that receive this agent's output


# ---------------------------------------------------------------------------
# Software Development domain
# ---------------------------------------------------------------------------

SOFTWARE_DEV_ROLES: list[RoleSpec] = [
    RoleSpec(
        role="Architect",
        description=(
            "Designs the high-level structure: module decomposition, "
            "interfaces, data flow, and technology choices. "
            "Output a clear design document describing classes, methods, "
            "and their interactions."
        ),
        upstream=[],
        downstream=["Coder"],
    ),
    RoleSpec(
        role="Coder",
        description=(
            "Implements the solution in Python based on the architecture "
            "specification. You MUST output all implementation code inside "
            "```python``` markdown fences. The code must be complete, "
            "runnable, and self-contained in a single module. Do not use "
            "external packages beyond the Python standard library."
        ),
        upstream=["Architect"],
        downstream=["Tester"],
    ),
    RoleSpec(
        role="Tester",
        description=(
            "Writes pytest-style unit tests for the code. You MUST output "
            "all test code inside ```python``` markdown fences. Each test "
            "function must start with 'test_'. Import the code under test "
            "from 'solution' (e.g., 'from solution import MyClass'). "
            "Tests must be runnable with pytest."
        ),
        upstream=["Coder"],
        downstream=["Reviewer"],
    ),
    RoleSpec(
        role="Reviewer",
        description=(
            "Reviews the code and test results for correctness, readability, "
            "and adherence to the architecture. Provides a final quality assessment."
        ),
        upstream=["Architect", "Coder", "Tester"],
        downstream=[],
    ),
]

# ---------------------------------------------------------------------------
# Research Synthesis domain
# ---------------------------------------------------------------------------

RESEARCH_SYNTHESIS_ROLES: list[RoleSpec] = [
    RoleSpec(
        role="Researcher_A",
        description=(
            "Investigates the first set of source documents and produces "
            "structured findings with citations."
        ),
        upstream=[],
        downstream=["Synthesizer"],
    ),
    RoleSpec(
        role="Researcher_B",
        description=(
            "Investigates the second set of source documents and produces "
            "structured findings with citations."
        ),
        upstream=[],
        downstream=["Synthesizer"],
    ),
    RoleSpec(
        role="Synthesizer",
        description=(
            "Combines findings from both researchers into a coherent, "
            "well-structured synthesis report."
        ),
        upstream=["Researcher_A", "Researcher_B"],
        downstream=["Fact_Checker"],
    ),
    RoleSpec(
        role="Fact_Checker",
        description=(
            "Verifies claims in the synthesis against the original sources. "
            "Flags unsupported or contradictory statements."
        ),
        upstream=["Researcher_A", "Researcher_B", "Synthesizer"],
        downstream=[],
    ),
]

# ---------------------------------------------------------------------------
# Project Planning domain
# ---------------------------------------------------------------------------

PROJECT_PLANNING_ROLES: list[RoleSpec] = [
    RoleSpec(
        role="Scheduler",
        description=(
            "Creates the project timeline, assigns tasks to sprints, "
            "and manages dependencies between work items."
        ),
        upstream=[],
        downstream=["Resource_Manager", "QA_Lead"],
    ),
    RoleSpec(
        role="Resource_Manager",
        description=(
            "Allocates team members, budget, and equipment across tasks. "
            "Ensures no resource is over-allocated."
        ),
        upstream=["Scheduler"],
        downstream=["Risk_Analyst"],
    ),
    RoleSpec(
        role="Risk_Analyst",
        description=(
            "Identifies risks in the plan, estimates probability and impact, "
            "and proposes mitigation strategies."
        ),
        upstream=["Scheduler", "Resource_Manager"],
        downstream=["QA_Lead"],
    ),
    RoleSpec(
        role="QA_Lead",
        description=(
            "Defines quality gates, acceptance criteria, and test plans. "
            "Produces a final quality assurance assessment of the project plan."
        ),
        upstream=["Scheduler", "Resource_Manager", "Risk_Analyst"],
        downstream=[],
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAIN_ROLES: dict[str, list[RoleSpec]] = {
    "software_dev": SOFTWARE_DEV_ROLES,
    "research_synthesis": RESEARCH_SYNTHESIS_ROLES,
    "project_planning": PROJECT_PLANNING_ROLES,
}


def get_roles(domain: str) -> list[RoleSpec]:
    """Return the role specifications for a given domain.

    Raises
    ------
    ValueError
        If *domain* is not one of the registered domains.
    """
    if domain not in DOMAIN_ROLES:
        raise ValueError(
            f"Unknown domain {domain!r}. Available: {list(DOMAIN_ROLES.keys())}"
        )
    return DOMAIN_ROLES[domain]
