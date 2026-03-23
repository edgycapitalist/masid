"""Task adaptation utilities — bridge MASID TaskSpec to CrewAI task descriptions."""

from __future__ import annotations

from masid.agents.roles import RoleSpec
from masid.domains import TaskSpec
from masid.domains.registry import get_tasks


def get_task_for_experiment(domain: str, task_id: str) -> TaskSpec:
    """Retrieve a specific TaskSpec by domain and task_id.

    Raises
    ------
    ValueError
        If *task_id* is not found in the domain's task list.
    """
    tasks = get_tasks(domain)
    for task in tasks:
        if task.task_id == task_id:
            return task
    available = [t.task_id for t in tasks]
    raise ValueError(f"Task {task_id!r} not found in domain {domain!r}. Available: {available}")


def format_task_description(
    task: TaskSpec,
    role: RoleSpec,
    round_num: int,
    upstream_outputs: dict[str, str],
    all_outputs: dict[str, str] | None = None,
    scorecard: str | None = None,
) -> str:
    """Build the full task description string for a CrewAI Task.

    Parameters
    ----------
    task:
        The TaskSpec (provides base description, role-specific variants).
    role:
        The RoleSpec for this agent (provides upstream dependency list).
    round_num:
        0-indexed round number.
    upstream_outputs:
        Outputs from upstream roles only (used by IRM and IAMD).
    all_outputs:
        Outputs from ALL roles (used by JRO for full transparency).
        When provided, upstream_outputs is ignored.
    scorecard:
        IAMD performance feedback from the previous round.
        Prepended to the description when present.
    """
    # Get the role-specific task description (handles research_synthesis info separation)
    base_description = task.get_description_for_role(role.role)

    parts: list[str] = []

    # Prepend IAMD scorecard if present (the mechanism signal)
    if scorecard:
        parts.append(f"[PERFORMANCE SCORECARD — ROUND {round_num}]\n{scorecard}")

    parts.append(f"TASK:\n{base_description}")

    if round_num == 0:
        # First round: inject any available upstream context
        context_source = all_outputs if all_outputs is not None else upstream_outputs
        upstream_text = _format_context(context_source, role, use_all=all_outputs is not None)
        if upstream_text:
            label = "FULL TEAM CONTEXT" if all_outputs is not None else "UPSTREAM INPUT"
            parts.append(f"{label}:\n{upstream_text}")
        if not upstream_text:
            parts.append("Produce your initial output based on the task description above.")
        else:
            if all_outputs is not None:
                parts.append(
                    "Based on the full team context above, produce your output. "
                    "Ensure it integrates well with the rest of the project."
                )
            else:
                parts.append("Based on the upstream input above, produce your output.")
    else:
        # Revision round
        context_source = all_outputs if all_outputs is not None else upstream_outputs
        upstream_text = _format_context(context_source, role, use_all=all_outputs is not None)
        label = "CURRENT TEAM STATE" if all_outputs is not None else "LATEST UPSTREAM INPUT"
        if upstream_text:
            parts.append(f"{label}:\n{upstream_text}")

        if scorecard:
            parts.append(
                f"This is revision round {round_num + 1}. "
                f"Review your performance scorecard above and address the specific "
                f"issues identified. Focus on improving the areas where you scored lowest."
            )
        elif all_outputs is not None:
            parts.append(
                f"This is revision round {round_num + 1}. "
                f"Review the full project state above and improve your output "
                f"to maximize overall project quality."
            )
        else:
            parts.append(
                f"This is revision round {round_num + 1}. "
                f"Review and improve your previous output."
            )

    return "\n\n".join(parts)


def _format_context(
    outputs: dict[str, str],
    role: RoleSpec,
    use_all: bool,
) -> str:
    """Format output context as readable text.

    If use_all is True, include all outputs.
    Otherwise, include only upstream roles per the role's dependency list.
    """
    if use_all:
        lines = []
        for role_name, content in outputs.items():
            lines.append(f"--- {role_name} output ---\n{content}")
        return "\n".join(lines)
    else:
        lines = []
        for up_role in role.upstream:
            if up_role in outputs:
                lines.append(f"--- {up_role} output ---\n{outputs[up_role]}")
        return "\n".join(lines)


def format_expected_output(role: RoleSpec) -> str:
    """Derive a CrewAI expected_output string from a RoleSpec."""
    return (
        f"A complete and high-quality output from the {role.role}. "
        f"{role.description}"
    )
