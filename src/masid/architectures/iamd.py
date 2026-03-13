"""Incentive-Aligned Mechanism Design (IAMD) architecture.

Theoretical basis: Mechanism Design (Hurwicz, Maskin, Myerson — 2007 Nobel).
Also: Adam Smith's "invisible hand" — self-interest operating within
well-designed institutional rules produces socially desirable outcomes.
Algorithmic Mechanism Design (Nisan & Ronen, 2001).

The key insight: IAMD agents have the SAME individual objective as IRM
("do your job well"). The difference is the MECHANISM — a structured
feedback system that acts as the "institution" aligning self-interest
with group welfare. Agents don't need to be altruistic; the feedback
signals naturally guide them toward collectively better outcomes.

Real-world analog: A well-designed marketplace or organization where
price signals / performance reviews guide individual behavior toward
group-optimal outcomes.

Key properties:
- System prompt: SAME as IRM ("do your job well") — agents are self-interested
- Information flow: Upstream only (same as IRM)
- Between-round feedback: THIS IS THE MECHANISM. Each agent receives:
  * Structured scoring on their output (the "price signal")
  * Specific, actionable feedback on what to improve
  * How their output affected downstream quality
  This feedback loop is what aligns individual optimization with group welfare.
- Scoring: Individual scores based on role-specific criteria.

MARL equivalent: CTDE with shaped rewards.

Analogy to real mechanism design:
- Vickrey auction: bidders act selfishly, but the auction RULES produce
  efficient allocation. The rules are the mechanism.
- Our system: agents optimize for themselves, but the FEEDBACK RULES
  guide them toward team-optimal behavior. The feedback is the mechanism.
"""

from __future__ import annotations

from masid.agents import Agent, AgentOutput
from masid.agents.roles import RoleSpec
from masid.architectures import BaseArchitecture
from masid.models import LLMClient


class IAMDArchitecture(BaseArchitecture):
    """Incentive-Aligned Mechanism Design — feedback mechanism aligns self-interest."""

    name = "Incentive-Aligned Mechanism Design"
    key = "iamd"

    def __init__(self, domain: str = "software_dev") -> None:
        self.domain = domain

    def build_system_prompt(self, role_spec: RoleSpec, task_description: str) -> str:
        # IAMD system prompt is deliberately SIMILAR to IRM.
        # The agent is self-interested — "do your job well."
        # The mechanism (feedback between rounds) is what differentiates
        # IAMD from IRM, not the system prompt.
        #
        # The only addition: agents know they will receive scored feedback.
        # This primes them to pay attention to the feedback when it arrives.
        return (
            f"You are a {role_spec.role}.\n\n"
            f"Role description: {role_spec.description}\n\n"
            f"Your goal is to produce the highest quality output for your "
            f"role. After each round, you will receive a performance "
            f"scorecard with specific feedback. Use this feedback to "
            f"improve your output in subsequent rounds.\n\n"
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
                prompt = "Produce your output."
            else:
                # IAMD: upstream-only information flow (same as IRM).
                # The differentiator is the feedback injected BETWEEN
                # rounds by the orchestrator, not the information flow.
                upstream = ""
                for up_role in spec.upstream:
                    if up_role in prev_by_role:
                        upstream += (
                            f"\n--- {up_role} output ---\n{prev_by_role[up_role]}\n"
                        )

                if round_number == 0 and not upstream:
                    prompt = "Produce your output based on the task description."
                elif round_number == 0:
                    prompt = (
                        f"Here is the input from upstream:\n{upstream}\n\n"
                        f"Based on this input, produce your output."
                    )
                else:
                    # IAMD revision: reference the feedback scorecard.
                    # The actual scorecard content was injected into the
                    # agent's conversation history by the orchestrator
                    # between rounds via agent.inject_context().
                    prompt = (
                        f"This is revision round {round_number + 1}. "
                        f"Here is the latest upstream input:\n{upstream}\n\n"
                        f"Review your performance scorecard from the previous "
                        f"round and address the specific issues identified. "
                        f"Focus on improving the areas where you scored lowest."
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
