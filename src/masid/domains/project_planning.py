"""Resource-Constrained Project Planning task domain.

Tasks involve planning projects with tight resource constraints.
This domain is the most game-theoretically rich — every agent's
decision affects every other agent.
"""

from __future__ import annotations

from masid.domains import TaskSpec

PROJECT_PLANNING_TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="pp_001",
        title="Mobile App Launch",
        description=(
            "Plan the development and launch of a mobile app for a small fintech "
            "startup.\n\n"
            "Constraints:\n"
            "- Team: 5 developers, 2 QA engineers, 1 designer, 1 PM\n"
            "- Budget: $200,000 total\n"
            "- Deadline: 16 weeks from start\n"
            "- Must include: user auth, payment processing, dashboard, notifications\n"
            "- Regulatory: must pass basic security audit before launch\n\n"
            "Deliverables: project schedule, resource allocation plan, risk "
            "assessment, and QA plan."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A feasible 16-week schedule with clear milestones, no resource "
            "over-allocation, identified risks with mitigations, and quality "
            "gates at key checkpoints."
        ),
    ),
    TaskSpec(
        task_id="pp_002",
        title="Data Center Migration",
        description=(
            "Plan the migration of a medium-sized company's infrastructure from "
            "on-premise to cloud.\n\n"
            "Constraints:\n"
            "- Team: 3 infrastructure engineers, 2 DevOps, 1 DBA, 1 security eng\n"
            "- Budget: $150,000 for migration (excluding ongoing cloud costs)\n"
            "- Deadline: 12 weeks\n"
            "- Must migrate: 3 databases, 12 microservices, CI/CD pipeline\n"
            "- Zero-downtime requirement for customer-facing services\n"
            "- Must maintain SOC 2 compliance throughout\n\n"
            "Deliverables: migration schedule, resource allocation, risk "
            "assessment, and QA plan."
        ),
        difficulty="hard",
        expected_output_hint=(
            "A phased migration plan with rollback procedures, careful resource "
            "scheduling avoiding single points of failure, comprehensive risk "
            "register, and quality gates ensuring compliance at each phase."
        ),
    ),
    TaskSpec(
        task_id="pp_003",
        title="Conference Organization",
        description=(
            "Plan a 2-day AI/ML conference for 500 attendees.\n\n"
            "Constraints:\n"
            "- Team: 2 event coordinators, 1 marketing lead, 1 tech lead, "
            "  3 volunteers\n"
            "- Budget: $80,000\n"
            "- Deadline: 10 weeks to event day\n"
            "- Must include: keynote, 4 tracks, workshop day, sponsor area\n"
            "- Need: venue, catering, A/V, speaker travel, marketing\n"
            "- Online streaming required for 1 track\n\n"
            "Deliverables: event schedule, resource and budget allocation, "
            "risk assessment, and quality assurance plan."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A detailed event timeline with vendor deadlines, budget breakdown "
            "within constraints, contingency plans for speaker cancellations "
            "and technical failures, and quality checks for attendee experience."
        ),
    ),
]


def get_project_planning_tasks() -> list[TaskSpec]:
    """Return all project planning task specifications."""
    return PROJECT_PLANNING_TASKS
