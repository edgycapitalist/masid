"""Independent Reward Maximization (IRM) architecture.

Each agent optimizes solely for its own sub-task quality. No shared
objectives, no cross-agent feedback. Agents execute sequentially
following the dependency graph, each receiving only upstream outputs.

Game-theoretic model: Non-cooperative game / Nash Equilibrium behavior.
MARL equivalent: Independent Learners (IPPO, IQL).
"""

from __future__ import annotations

from masid.agents import Agent, AgentOutput
from masid.agents.roles import RoleSpec
from masid.architectures import BaseArchitecture
from masid.models import LLMClient


class IRMArchitecture(BaseArchitecture):
    """Independent Reward Maximization — each agent for itself."""

    name = "Independent Reward Maximization"
    key = "irm"

    def build_system_prompt(self, role_spec: RoleSpec, task_description: str) -> str:
        return (
            f"You are a {role_spec.role}.\n\n"
            f"Role description: {role_spec.description}\n\n"
            f"Your goal is to produce the highest quality output for YOUR specific "
            f"role. Focus exclusively on maximizing the quality of your own work.\n\n"
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
        # Build lookup of previous outputs by role
        prev_by_role: dict[str, str] = {
            o.role: o.content for o in previous_outputs
        }
        # Build lookup of role_spec by role name
        from masid.agents.roles import DOMAIN_ROLES

        role_specs_by_name: dict[str, RoleSpec] = {}
        for specs in DOMAIN_ROLES.values():
            for s in specs:
                role_specs_by_name[s.role] = s

        for agent in agents:
            spec = role_specs_by_name.get(agent.role)
            if spec is None:
                # Fallback: just send task description
                prompt = f"Please produce your output.\n\nTask: {task_description}"
            else:
                # Gather upstream outputs
                upstream_context = ""
                for up_role in spec.upstream:
                    if up_role in prev_by_role:
                        upstream_context += (
                            f"\n--- {up_role} output ---\n{prev_by_role[up_role]}\n"
                        )

                if round_number == 0 and not upstream_context:
                    prompt = "Please produce your output based on the task description."
                elif round_number == 0:
                    prompt = (
                        f"Here is the input from upstream roles:\n{upstream_context}\n\n"
                        f"Based on this input, produce your output."
                    )
                else:
                    prompt = (
                        f"This is revision round {round_number + 1}. "
                        f"Here is the latest input from upstream roles:\n{upstream_context}\n\n"
                        f"Review and improve your previous output."
                    )

            output = agent.act(prompt, round_number=round_number)
            outputs.append(output)
            # Make this output available for downstream agents in same round
            prev_by_role[agent.role] = output.content

        return outputs

    def compute_agent_scores(
        self,
        agents: list[Agent],
        outputs: list[AgentOutput],
        collective_score: float,
    ) -> dict[str, float]:
        # In IRM, each agent's score is based only on its own output quality.
        # For now, return the collective score as a placeholder — the evaluation
        # pipeline will compute per-agent scores separately.
        return {agent.agent_id: collective_score for agent in agents}
