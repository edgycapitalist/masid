"""Tests for masid.agents."""

import pytest

from masid.agents import Agent, AgentOutput
from masid.agents.roles import DOMAIN_ROLES, get_roles


class TestAgent:
    def test_creation(self, mock_client):
        agent = Agent(
            agent_id="test_0",
            role="Tester",
            system_prompt="You are a tester.",
            client=mock_client,
        )
        assert agent.agent_id == "test_0"
        assert agent.role == "Tester"
        assert len(agent.history) == 1  # system prompt only

    def test_act(self, mock_client):
        agent = Agent(
            agent_id="coder_0",
            role="Coder",
            system_prompt="You are a coder.",
            client=mock_client,
        )
        output = agent.act("Write a function.", round_number=0)
        assert isinstance(output, AgentOutput)
        assert output.agent_id == "coder_0"
        assert output.content == "This is a mock response."
        assert output.round_number == 0
        # History: system + user + assistant = 3
        assert len(agent.history) == 3

    def test_multiple_acts(self, mock_client):
        from tests.conftest import MockLLMClient

        client = MockLLMClient(responses=["response_1", "response_2"])
        agent = Agent("a", "A", "sys", client)
        o1 = agent.act("prompt1", round_number=0)
        o2 = agent.act("prompt2", round_number=1)
        assert o1.content == "response_1"
        assert o2.content == "response_2"
        # system + (user+assistant)*2 = 5
        assert len(agent.history) == 5

    def test_inject_context(self, mock_client):
        agent = Agent("a", "A", "sys", mock_client)
        agent.inject_context("some context from another agent")
        assert len(agent.history) == 2
        assert agent.history[1].content == "some context from another agent"

    def test_reset(self, mock_client):
        agent = Agent("a", "A", "sys prompt", mock_client)
        agent.act("hello", round_number=0)
        assert len(agent.history) == 3
        agent.reset()
        assert len(agent.history) == 1
        assert agent.history[0].content == "sys prompt"


class TestRoles:
    def test_all_domains_have_roles(self):
        for domain in DOMAIN_ROLES:
            roles = get_roles(domain)
            assert len(roles) == 4

    def test_software_dev_roles(self):
        roles = get_roles("software_dev")
        role_names = [r.role for r in roles]
        assert role_names == ["Architect", "Coder", "Tester", "Reviewer"]

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            get_roles("nonexistent")

    def test_dependency_consistency(self):
        """Verify that upstream/downstream references are consistent."""
        for domain, roles in DOMAIN_ROLES.items():
            role_names = {r.role for r in roles}
            for role in roles:
                for up in role.upstream:
                    assert up in role_names, (
                        f"{domain}/{role.role}: upstream {up} not found"
                    )
                for down in role.downstream:
                    assert down in role_names, (
                        f"{domain}/{role.role}: downstream {down} not found"
                    )
