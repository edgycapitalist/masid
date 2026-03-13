"""Incentive-Aligned Mechanism Design (IAMD) architecture.

The system architect designs each agent's individual objective to
naturally correlate with collective success. Agents are self-interested
but the "rules of the game" align self-interest with group welfare.

Each agent receives structured feedback from downstream agents and its
individual score incorporates downstream impact metrics.

Game-theoretic model: Mechanism Design / Incentive Compatibility.
MARL equivalent: CTDE with shaped rewards.
"""

from __future__ import annotations

from masid.agents import Agent, AgentOutput
from masid.agents.roles import RoleSpec
from masid.architectures import BaseArchitecture
from masid.models import LLMClient

# ---------------------------------------------------------------------------
# Per-domain evaluation weight schemes
# ---------------------------------------------------------------------------
# Each role has a weighted score that incorporates downstream metrics.
# Format: {domain: {role: {metric_name: weight}}}

IAMD_EVAL_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "software_dev": {
        "Architect": {
            "design_clarity": 0.40,
            "implementability_of_design": 0.30,
            "final_quality": 0.30,
        },
        "Coder": {
            "code_correctness": 0.50,
            "code_readability": 0.25,
            "code_structure": 0.25,
        },
        "Tester": {
            "test_coverage": 0.40,
            "bugs_found_accuracy": 0.30,
            "code_improvement_impact": 0.30,
        },
        "Reviewer": {
            "review_thoroughness": 0.40,
            "actionable_feedback": 0.30,
            "final_quality": 0.30,
        },
    },
    "research_synthesis": {
        "Researcher_A": {
            "finding_quality": 0.40,
            "synthesis_usability": 0.30,
            "fact_check_pass_rate": 0.30,
        },
        "Researcher_B": {
            "finding_quality": 0.40,
            "synthesis_usability": 0.30,
            "fact_check_pass_rate": 0.30,
        },
        "Synthesizer": {
            "coherence": 0.40,
            "completeness": 0.30,
            "fact_check_pass_rate": 0.30,
        },
        "Fact_Checker": {
            "accuracy": 0.50,
            "coverage": 0.30,
            "actionable_corrections": 0.20,
        },
    },
    "project_planning": {
        "Scheduler": {
            "schedule_feasibility": 0.30,
            "resource_compatibility": 0.30,
            "risk_coverage": 0.20,
            "qa_approval": 0.20,
        },
        "Resource_Manager": {
            "utilization_efficiency": 0.30,
            "no_overallocation": 0.30,
            "schedule_compatibility": 0.20,
            "risk_coverage": 0.20,
        },
        "Risk_Analyst": {
            "risk_identification": 0.30,
            "mitigation_quality": 0.30,
            "schedule_impact_accuracy": 0.20,
            "qa_approval": 0.20,
        },
        "QA_Lead": {
            "quality_gate_coverage": 0.30,
            "acceptance_criteria_clarity": 0.30,
            "plan_feasibility_assessment": 0.20,
            "final_plan_quality": 0.20,
        },
    },
}


def _build_criteria_text(domain: str, role: str) -> str:
    """Build a human-readable evaluation criteria string for a prompt."""
    weights = IAMD_EVAL_WEIGHTS.get(domain, {}).get(role, {})
    if not weights:
        return "Your output will be evaluated on overall quality."
    lines = []
    for metric, weight in weights.items():
        pct = int(weight * 100)
        label = metric.replace("_", " ").title()
        lines.append(f"  - {label}: {pct}%")
    return "Your evaluation criteria:\n" + "\n".join(lines)


class IAMDArchitecture(BaseArchitecture):
    """Incentive-Aligned Mechanism Design — self-interest aligned with group welfare."""

    name = "Incentive-Aligned Mechanism Design"
    key = "iamd"

    def __init__(self, domain: str = "software_dev") -> None:
        self.domain = domain

    def build_system_prompt(self, role_spec: RoleSpec, task_description: str) -> str:
        criteria_text = _build_criteria_text(self.domain, role_spec.role)

        # IAMD v3: The incentive alignment comes through the evaluation
        # criteria alone — NOT by naming other roles or talking about
        # "downstream" agents. Previous versions caused role confusion
        # because the model would see "Tester" mentioned and start
        # writing tests instead of code.
        #
        # The key insight: we want the Coder to produce code that is
        # well-structured and easy to test, but we achieve this by
        # asking for "code structure" and "readability" — not by
        # mentioning testing or testers.

        return (
            f"You are a {role_spec.role}.\n\n"
            f"Role description: {role_spec.description}\n\n"
            f"Your output will be scored on these criteria:\n{criteria_text}\n\n"
            f"Produce the best possible output for your role. "
            f"You will be scored automatically.\n\n"
            f"Task:\n{task_description}"
        )

    def build_agents(
        self,
        role_specs: list[RoleSpec],
        task_description: str,
        client: LLMClient,
    ) -> list[Agent]:
        agents = []
        for i, spec in enumerate(role_specs):
            prompt = self.build_system_prompt(spec, task_description)
            weights = IAMD_EVAL_WEIGHTS.get(self.domain, {}).get(spec.role, {})
            agent = Agent(
                agent_id=f"{spec.role.lower()}_{i}",
                role=spec.role,
                system_prompt=prompt,
                client=client,
                evaluation_criteria=weights,
            )
            agents.append(agent)
        return agents

    def run_round(
        self,
        agents: list[Agent],
        task_description: str,
        round_number: int,
        previous_outputs: list[AgentOutput],
    ) -> list[AgentOutput]:
        outputs: list[AgentOutput] = []
        prev_by_role: dict[str, str] = {o.role: o.content for o in previous_outputs}

        from masid.agents.roles import DOMAIN_ROLES

        role_specs_by_name: dict[str, RoleSpec] = {}
        for specs in DOMAIN_ROLES.values():
            for s in specs:
                role_specs_by_name[s.role] = s

        for agent in agents:
            spec = role_specs_by_name.get(agent.role)
            if spec is None:
                prompt = "Please produce your output."
            else:
                # IAMD: structured information sharing — upstream outputs
                # only. Previous versions included downstream role hints
                # (e.g., "Tester expects: ...") which caused severe role
                # confusion — the Coder would read "Tester expects tests"
                # and start writing tests instead of code.
                #
                # The IAMD incentive alignment now comes purely from:
                # 1. Specific scoring criteria in the system prompt
                # 2. Structured upstream context (not full transparency)
                # 3. Execution feedback between rounds (from sandbox)
                upstream_context = ""
                for up_role in spec.upstream:
                    if up_role in prev_by_role:
                        upstream_context += (
                            f"\n--- {up_role} output ---\n{prev_by_role[up_role]}\n"
                        )

                if round_number == 0 and not upstream_context:
                    prompt = "Produce your output based on the task description."
                elif round_number == 0:
                    prompt = (
                        f"Here is upstream input:\n{upstream_context}\n"
                        f"Based on this input, produce your output."
                    )
                else:
                    prompt = (
                        f"Revision round {round_number + 1}.\n"
                        f"Upstream input:\n{upstream_context}\n\n"
                        f"Improve your output based on the scoring criteria "
                        f"in your instructions."
                    )

            output = agent.act(prompt, round_number=round_number)
            outputs.append(output)
            prev_by_role[agent.role] = output.content

        return outputs

    def compute_agent_scores(
        self,
        agents: list[Agent],
        outputs: list[AgentOutput],
        collective_score: float,
    ) -> dict[str, float]:
        # In IAMD, individual scores are weighted combinations.
        # Full implementation will use the evaluation pipeline.
        # For now, use the collective score modulated by criteria weights.
        scores = {}
        for agent in agents:
            weights = agent.evaluation_criteria
            if weights:
                # Placeholder: collective score * mean weight
                scores[agent.agent_id] = collective_score
            else:
                scores[agent.agent_id] = collective_score
        return scores