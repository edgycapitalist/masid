"""CrewAI crew builders for IRM, JRO, and IAMD architectures.

Each builder function constructs a CrewAI Crew configured to replicate
the corresponding MASID incentive architecture:

- IRM: Self-oriented agents, upstream-only context, no feedback.
- JRO: Team-oriented agents, full-transparency context injection.
- IAMD: Self-oriented agents, upstream-only context + scorecard feedback.

The key design choice: per-round crew rebuild.
CrewAI has no native multi-round support, so we run crew.kickoff() once
per round, rebuilding tasks each time with outputs from the prior round.
This mirrors MASID's TrialRunner round loop faithfully.

JRO uses explicit full-context injection into task descriptions (NOT
memory=True), avoiding the embedding-model dependency that CrewAI's
memory system requires. allow_delegation=False for all architectures
to keep the controlled variable clean.
"""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai import LLM as CrewAILLM

from masid.agents.roles import RoleSpec
from masid.domains import TaskSpec
from masid_crewai.config import CrewAIModelConfig
from masid_crewai.tasks import format_expected_output, format_task_description


def build_crewai_llm(config: CrewAIModelConfig) -> CrewAILLM:
    """Construct a CrewAI LLM object from config."""
    return CrewAILLM(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


# ---------------------------------------------------------------------------
# IRM — Independent Reward Maximization
# ---------------------------------------------------------------------------

def _irm_system_prompt(role: RoleSpec, task: TaskSpec) -> str:
    """Replicate IRMArchitecture.build_system_prompt()."""
    description = task.get_description_for_role(role.role)
    return (
        f"You are a {role.role}.\n\n"
        f"Role description: {role.description}\n\n"
        f"Your goal is to produce the highest quality output for YOUR "
        f"specific role. Focus exclusively on maximizing the quality "
        f"of your own work.\n\n"
        f"Task:\n{description}"
    )


def build_irm_crew(
    roles: list[RoleSpec],
    task: TaskSpec,
    round_num: int,
    prev_outputs: dict[str, str],
    llm: CrewAILLM,
) -> Crew:
    """Build an IRM crew for one round.

    Agents: self-oriented goals, memory=False, allow_delegation=False.
    Tasks: upstream-only context injected into descriptions.
    """
    crewai_agents: list[Agent] = []
    crewai_tasks: list[Task] = []

    for role in roles:
        agent = Agent(
            role=role.role,
            goal=(
                f"Produce the highest quality output for YOUR specific role as {role.role}. "
                f"Focus exclusively on maximizing the quality of your own work."
            ),
            backstory=_irm_system_prompt(role, task),
            llm=llm,
            memory=False,
            allow_delegation=False,
            verbose=False,
        )
        crewai_agents.append(agent)

        task_description = format_task_description(
            task=task,
            role=role,
            round_num=round_num,
            upstream_outputs=prev_outputs,
        )
        crew_task = Task(
            description=task_description,
            expected_output=format_expected_output(role),
            agent=agent,
        )
        crewai_tasks.append(crew_task)

    return Crew(
        agents=crewai_agents,
        tasks=crewai_tasks,
        process=Process.sequential,
        memory=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# JRO — Joint Reward Optimization
# ---------------------------------------------------------------------------

def _jro_system_prompt(role: RoleSpec, task: TaskSpec) -> str:
    """Replicate JROArchitecture.build_system_prompt()."""
    description = task.get_description_for_role(role.role)
    return (
        f"You are a {role.role} working on a TEAM project.\n\n"
        f"Role description: {role.description}\n\n"
        f"Your PRIMARY goal is the success of the OVERALL project, "
        f"not just your individual part. Consider how your output "
        f"will affect every other team member and the final "
        f"deliverable. Optimize for collective quality.\n\n"
        f"You will see outputs from ALL team members. Use this "
        f"information to ensure your work integrates well with "
        f"theirs.\n\n"
        f"Task:\n{description}"
    )


def build_jro_crew(
    roles: list[RoleSpec],
    task: TaskSpec,
    round_num: int,
    prev_outputs: dict[str, str],
    llm: CrewAILLM,
) -> Crew:
    """Build a JRO crew for one round.

    Agents: team-oriented goals, memory=False (explicit injection instead of RAG).
    Tasks: ALL previous agent outputs injected as context (full transparency).
    Note: allow_delegation=False to keep the controlled variable clean.
    """
    crewai_agents: list[Agent] = []
    crewai_tasks: list[Task] = []

    for role in roles:
        agent = Agent(
            role=role.role,
            goal=(
                f"The success of the OVERALL project is your primary goal. "
                f"As the {role.role}, optimize your contribution for collective quality, "
                f"not just your individual part."
            ),
            backstory=_jro_system_prompt(role, task),
            llm=llm,
            memory=False,
            allow_delegation=False,
            verbose=False,
        )
        crewai_agents.append(agent)

        # JRO: full transparency — inject ALL outputs into every task description
        task_description = format_task_description(
            task=task,
            role=role,
            round_num=round_num,
            upstream_outputs={},
            all_outputs=prev_outputs if prev_outputs else None,
        )
        crew_task = Task(
            description=task_description,
            expected_output=format_expected_output(role),
            agent=agent,
        )
        crewai_tasks.append(crew_task)

    return Crew(
        agents=crewai_agents,
        tasks=crewai_tasks,
        process=Process.sequential,
        memory=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# IAMD — Incentive-Aligned Mechanism Design
# ---------------------------------------------------------------------------

def _iamd_system_prompt(role: RoleSpec, task: TaskSpec) -> str:
    """Replicate IAMDArchitecture.build_system_prompt()."""
    description = task.get_description_for_role(role.role)
    return (
        f"You are a {role.role}.\n\n"
        f"Role description: {role.description}\n\n"
        f"Your goal is to produce the highest quality output for your "
        f"role. After each round, you will receive a performance "
        f"scorecard with specific feedback. Use this feedback to "
        f"improve your output in subsequent rounds.\n\n"
        f"Task:\n{description}"
    )


def build_iamd_crew(
    roles: list[RoleSpec],
    task: TaskSpec,
    round_num: int,
    prev_outputs: dict[str, str],
    scorecards: dict[str, str],
    llm: CrewAILLM,
) -> Crew:
    """Build an IAMD crew for one round.

    Agents: self-oriented goals + scorecard awareness, memory=False.
    Tasks: upstream-only context + scorecard from previous round prepended.
    Scorecards map role name → scorecard text (empty dict on round 0).
    """
    crewai_agents: list[Agent] = []
    crewai_tasks: list[Task] = []

    for role in roles:
        agent = Agent(
            role=role.role,
            goal=(
                f"Produce the highest quality output for your role as {role.role}. "
                f"Use the performance scorecard feedback to improve with each round."
            ),
            backstory=_iamd_system_prompt(role, task),
            llm=llm,
            memory=False,
            allow_delegation=False,
            verbose=False,
        )
        crewai_agents.append(agent)

        task_description = format_task_description(
            task=task,
            role=role,
            round_num=round_num,
            upstream_outputs=prev_outputs,
            scorecard=scorecards.get(role.role),
        )
        crew_task = Task(
            description=task_description,
            expected_output=format_expected_output(role),
            agent=agent,
        )
        crewai_tasks.append(crew_task)

    return Crew(
        agents=crewai_agents,
        tasks=crewai_tasks,
        process=Process.sequential,
        memory=False,
        verbose=False,
    )
