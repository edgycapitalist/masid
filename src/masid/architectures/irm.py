"""Independent Reward Maximization (IRM) architecture.

Theoretical basis: Non-cooperative game theory (Nash, 1950).
Each agent maximizes its own utility without coordination.
The resulting behavior is a Nash Equilibrium — no agent can
improve its outcome by unilaterally changing its strategy.

Real-world analog: Independent contractors on a project.

Key properties:
- System prompt: "Do your job well" (individual objective)
- Information flow: Upstream only (Architect → Coder → Tester → Reviewer)
- Between-round feedback: NONE. Agents revise blind.
- Scoring: Each agent scored independently on own output.

MARL equivalent: Independent Learners (IPPO, IQL).
"""

from __future__ import annotations

from masid.agents import Agent, AgentOutput
from masid.agents.roles import RoleSpec
from masid.architectures import BaseArchitecture
from masid.models import LLMClient


def _build_upstream_context(
    spec: RoleSpec,
    prev_by_role: dict[str, str],
) -> str:
    """Gather outputs from upstream roles only."""
    context = ""
    for up_role in spec.upstream:
        if up_role in prev_by_role:
            context += f"\n--- {up_role} output ---\n{prev_by_role[up_role]}\n"
    return context


class IRMArchitecture(BaseArchitecture):
    """Independent Reward Maximization — each agent for itself."""

    name = "Independent Reward Maximization"
    key = "irm"

    def build_system_prompt(self, role_spec: RoleSpec, task_description: str) -> str:
        return (
            f"You are a {role_spec.role}.\n\n"
            f"Role description: {role_spec.description}\n\n"
            f"Your goal is to produce the highest quality output for YOUR "
            f"specific role. Focus exclusively on maximizing the quality "
            f"of your own work.\n\n"
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
            agent = Agent(
                agent_id=f"{spec.role.lower()}_{i}",
                role=spec.role,
                system_prompt=prompt,
                client=client,
                evaluation_criteria={"own_quality": 1.0},
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
        prev_by_role: dict[str, str] = {
            o.role: o.content for o in previous_outputs
        }

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
                upstream = _build_upstream_context(spec, prev_by_role)

                if round_number == 0 and not upstream:
                    prompt = "Produce your output based on the task description."
                elif round_number == 0:
                    prompt = (
                        f"Here is the input from upstream:\n{upstream}\n\n"
                        f"Based on this input, produce your output."
                    )
                else:
                    # IRM: NO feedback between rounds. Just "improve."
                    prompt = (
                        f"This is revision round {round_number + 1}. "
                        f"Here is the latest upstream input:\n{upstream}\n\n"
                        f"Review and improve your previous output."
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
        return {agent.agent_id: collective_score for agent in agents}
