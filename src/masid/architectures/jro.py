"""Joint Reward Optimization (JRO) architecture.

All agents share a collective objective and explicitly optimize for
project-wide success. Full transparency between agents — every agent
sees all other agents' outputs and the overall project state.

Game-theoretic model: Cooperative game / Pareto Optimality.
MARL equivalent: Fully Cooperative (MAPPO, QMIX, VDN).
"""

from __future__ import annotations

from masid.agents import Agent, AgentOutput
from masid.agents.roles import RoleSpec
from masid.architectures import BaseArchitecture
from masid.models import LLMClient


class JROArchitecture(BaseArchitecture):
    """Joint Reward Optimization — every agent optimizes for the team."""

    name = "Joint Reward Optimization"
    key = "jro"

    def build_system_prompt(self, role_spec: RoleSpec, task_description: str) -> str:
        return (
            f"You are a {role_spec.role} working on a TEAM project.\n\n"
            f"Role description: {role_spec.description}\n\n"
            f"Your PRIMARY goal is the success of the OVERALL project, not just "
            f"your individual part. Consider how your output will affect every "
            f"other team member and the final deliverable. Optimize for "
            f"collective quality.\n\n"
            f"You will see outputs from ALL team members. Use this information "
            f"to ensure your work integrates well with theirs.\n\n"
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
                evaluation_criteria={"collective_quality": 1.0},
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

        # Build FULL context from all previous outputs (JRO = full transparency)
        all_context = ""
        for o in previous_outputs:
            all_context += f"\n--- {o.role} output ---\n{o.content}\n"

        for agent in agents:
            if round_number == 0 and not all_context:
                prompt = (
                    "Please produce your output. Remember: optimize for the "
                    "success of the entire project, not just your part."
                )
            elif round_number == 0:
                prompt = (
                    f"Here is what your teammates have produced so far:\n"
                    f"{all_context}\n\n"
                    f"Based on the full team context, produce your output. "
                    f"Ensure it integrates well with the rest of the project."
                )
            else:
                prompt = (
                    f"This is revision round {round_number + 1}. "
                    f"Here is the current state of ALL team outputs:\n{all_context}\n\n"
                    f"Review the full project state and improve your output to "
                    f"maximize overall project quality."
                )

            output = agent.act(prompt, round_number=round_number)
            outputs.append(output)
            # Update context for next agent in this round
            all_context += f"\n--- {agent.role} output ---\n{output.content}\n"

        return outputs

    def compute_agent_scores(
        self,
        agents: list[Agent],
        outputs: list[AgentOutput],
        collective_score: float,
    ) -> dict[str, float]:
        # In JRO, ALL agents receive the same collective score.
        return {agent.agent_id: collective_score for agent in agents}
