"""Tests for masid.architectures."""

import pytest

from masid.agents.roles import get_roles
from masid.architectures.registry import get_architecture
from masid.architectures.irm import IRMArchitecture
from masid.architectures.jro import JROArchitecture
from masid.architectures.iamd import IAMDArchitecture


TASK_DESC = "Build a URL shortener library."


class TestArchitectureRegistry:
    def test_get_irm(self):
        arch = get_architecture("irm")
        assert isinstance(arch, IRMArchitecture)
        assert arch.key == "irm"

    def test_get_jro(self):
        arch = get_architecture("jro")
        assert isinstance(arch, JROArchitecture)
        assert arch.key == "jro"

    def test_get_iamd(self):
        arch = get_architecture("iamd", domain="software_dev")
        assert isinstance(arch, IAMDArchitecture)
        assert arch.key == "iamd"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_architecture("xyz")


class TestIRM:
    def test_build_agents(self, mock_client):
        arch = IRMArchitecture()
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        assert len(agents) == 4
        # IRM prompts should emphasize individual quality
        for agent in agents:
            assert "highest quality output for YOUR" in agent.system_prompt

    def test_run_round_zero(self, mock_client):
        arch = IRMArchitecture()
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        outputs = arch.run_round(agents, TASK_DESC, round_number=0, previous_outputs=[])
        assert len(outputs) == 4
        # Each output should have content from the mock
        for o in outputs:
            assert o.content == "This is a mock response."

    def test_run_round_with_previous(self, mock_client):
        arch = IRMArchitecture()
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        r0 = arch.run_round(agents, TASK_DESC, 0, [])
        r1 = arch.run_round(agents, TASK_DESC, 1, r0)
        assert len(r1) == 4


class TestJRO:
    def test_build_agents(self, mock_client):
        arch = JROArchitecture()
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        assert len(agents) == 4
        for agent in agents:
            assert "TEAM project" in agent.system_prompt
            assert "OVERALL project" in agent.system_prompt

    def test_run_round_zero(self, mock_client):
        arch = JROArchitecture()
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        outputs = arch.run_round(agents, TASK_DESC, 0, [])
        assert len(outputs) == 4


class TestIAMD:
    def test_build_agents(self, mock_client):
        arch = IAMDArchitecture(domain="software_dev")
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        assert len(agents) == 4
        # IAMD v4: prompt mentions scorecard/feedback mechanism
        coder = [a for a in agents if a.role == "Coder"][0]
        assert "scorecard" in coder.system_prompt.lower() or "feedback" in coder.system_prompt.lower()

    def test_iamd_prompt_no_role_leakage(self, mock_client):
        """IAMD prompts must NOT reference other agent roles to avoid role confusion."""
        arch = IAMDArchitecture(domain="software_dev")
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        coder = [a for a in agents if a.role == "Coder"][0]
        prompt_lower = coder.system_prompt.lower()
        # Check for explicit role references (not substrings like "architecture")
        assert "the tester" not in prompt_lower
        assert "the reviewer" not in prompt_lower
        assert "downstream" not in prompt_lower
        assert "other roles" not in prompt_lower

    def test_run_round_zero(self, mock_client):
        arch = IAMDArchitecture(domain="software_dev")
        roles = get_roles("software_dev")
        agents = arch.build_agents(roles, TASK_DESC, mock_client)
        outputs = arch.run_round(agents, TASK_DESC, 0, [])
        assert len(outputs) == 4
