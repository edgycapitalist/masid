"""Unit tests for the MASID CrewAI adapter.

All tests here run without a live LLM endpoint.
Integration tests (requiring vLLM) are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest

from masid.agents import AgentOutput
from masid.agents.roles import get_roles
from masid.models import LLMResponse
from masid_crewai.architectures import (
    build_crewai_llm,
    build_iamd_crew,
    build_irm_crew,
    build_jro_crew,
)
from masid_crewai.config import CrewAIConfig, CrewAIModelConfig, load_crewai_config
from masid_crewai.evaluation import (
    _build_sandbox_scorecard,
    extract_agent_outputs,
)
from masid_crewai.tasks import (
    format_expected_output,
    format_task_description,
    get_task_for_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> CrewAIConfig:
    return CrewAIConfig()


@pytest.fixture
def mock_llm(default_config):
    """Build a CrewAI LLM object (no network call at construction time)."""
    return build_crewai_llm(default_config.agent_model)


@pytest.fixture
def sw_task():
    return get_task_for_experiment("software_dev", "sw_001")


@pytest.fixture
def rs_task():
    return get_task_for_experiment("research_synthesis", "rs_001")


@pytest.fixture
def pp_task():
    return get_task_for_experiment("project_planning", "pp_001")


@pytest.fixture
def sw_roles():
    return get_roles("software_dev")


@pytest.fixture
def rs_roles():
    return get_roles("research_synthesis")


@pytest.fixture
def pp_roles():
    return get_roles("project_planning")


def _make_agent_output(role: str, content: str, round_number: int = 0) -> AgentOutput:
    """Helper to build a minimal AgentOutput."""
    return AgentOutput(
        agent_id=f"{role.lower()}_0",
        role=role,
        content=content,
        response=LLMResponse(
            content=content,
            model="mock",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency_seconds=0.1,
        ),
        round_number=round_number,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = CrewAIConfig()
        assert cfg.agent_model.base_url == "http://localhost:11434"
        assert cfg.agent_model.model == "ollama/llama3.3:70b"
        assert cfg.experiment.trials_per_cell == 10
        assert cfg.experiment.max_rounds == 3
        assert set(cfg.architectures) == {"irm", "jro", "iamd"}
        assert set(cfg.domains) == {"software_dev", "research_synthesis", "project_planning"}
        assert cfg.task_ids["software_dev"] == "sw_001"

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """\
agent_model:
  model: "openai/custom-model"
  base_url: "http://custom:8000/v1"
  api_key: "test-key"
  temperature: 0.5
  max_tokens: 1024
experiment:
  trials_per_cell: 5
  max_rounds: 2
"""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        cfg = load_crewai_config(str(cfg_file))
        assert cfg.agent_model.model == "openai/custom-model"
        assert cfg.agent_model.temperature == 0.5
        assert cfg.experiment.trials_per_cell == 5
        assert cfg.experiment.max_rounds == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_crewai_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# Task adaptation tests
# ---------------------------------------------------------------------------

class TestTaskAdaptation:
    def test_get_task_sw001(self):
        task = get_task_for_experiment("software_dev", "sw_001")
        assert task.task_id == "sw_001"
        assert "URL" in task.title or "url" in task.title.lower()

    def test_get_task_invalid_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_task_for_experiment("software_dev", "xx_999")

    def test_get_task_invalid_domain_raises(self):
        with pytest.raises(ValueError):
            get_task_for_experiment("nonexistent_domain", "sw_001")

    def test_research_synthesis_info_separation(self, rs_task, rs_roles):
        """Researcher_A and Researcher_B must receive different task descriptions."""
        role_a = next(r for r in rs_roles if r.role == "Researcher_A")
        role_b = next(r for r in rs_roles if r.role == "Researcher_B")

        desc_a = rs_task.get_description_for_role(role_a.role)
        desc_b = rs_task.get_description_for_role(role_b.role)

        assert desc_a != desc_b, "Researcher_A and Researcher_B should see different source sets"
        assert len(desc_a) > 0
        assert len(desc_b) > 0

    def test_format_task_description_round0_no_upstream(self, sw_task, sw_roles):
        architect = sw_roles[0]  # no upstream
        desc = format_task_description(sw_task, architect, round_num=0, upstream_outputs={})
        assert "TASK:" in desc
        assert "PERFORMANCE SCORECARD" not in desc

    def test_format_task_description_round0_with_upstream(self, sw_task, sw_roles):
        coder = sw_roles[1]  # upstream = [Architect]
        desc = format_task_description(
            sw_task, coder, round_num=0,
            upstream_outputs={"Architect": "Here is the design doc."},
        )
        assert "UPSTREAM INPUT" in desc
        assert "Here is the design doc." in desc

    def test_format_task_description_revision_round(self, sw_task, sw_roles):
        coder = sw_roles[1]
        desc = format_task_description(
            sw_task, coder, round_num=1,
            upstream_outputs={"Architect": "Updated design."},
        )
        assert "revision round 2" in desc.lower()

    def test_format_task_description_iamd_scorecard(self, sw_task, sw_roles):
        coder = sw_roles[1]
        desc = format_task_description(
            sw_task, coder, round_num=1,
            upstream_outputs={"Architect": "Design."},
            scorecard="CODER SCORECARD:\n  Syntax: FAIL",
        )
        assert "PERFORMANCE SCORECARD" in desc
        assert "CODER SCORECARD" in desc

    def test_format_task_description_jro_full_context(self, sw_task, sw_roles):
        reviewer = sw_roles[3]  # sees all
        all_outputs = {
            "Architect": "Design doc",
            "Coder": "def foo(): pass",
            "Tester": "def test_foo(): assert True",
        }
        desc = format_task_description(
            sw_task, reviewer, round_num=0,
            upstream_outputs={},
            all_outputs=all_outputs,
        )
        assert "FULL TEAM CONTEXT" in desc
        assert "Architect" in desc
        assert "Coder" in desc
        assert "Tester" in desc

    def test_format_expected_output(self, sw_roles):
        coder = sw_roles[1]
        expected = format_expected_output(coder)
        assert "Coder" in expected
        assert len(expected) > 20


# ---------------------------------------------------------------------------
# Architecture builder tests
# ---------------------------------------------------------------------------

class TestIRMCrew:
    def test_crew_structure(self, sw_roles, sw_task, mock_llm):
        crew = build_irm_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        assert len(crew.agents) == 4
        assert len(crew.tasks) == 4

    def test_agents_no_delegation(self, sw_roles, sw_task, mock_llm):
        crew = build_irm_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        for agent in crew.agents:
            assert agent.allow_delegation is False

    def test_self_oriented_goal(self, sw_roles, sw_task, mock_llm):
        crew = build_irm_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        for agent in crew.agents:
            goal_lower = agent.goal.lower()
            # IRM agents focus on their own quality
            assert "your" in goal_lower or "own" in goal_lower or "specific role" in goal_lower

    def test_upstream_only_context(self, sw_roles, sw_task, mock_llm):
        """Coder task description should contain Architect output, not Tester output."""
        prev = {"Architect": "ARCH_DESIGN", "Tester": "TEST_CONTENT"}
        crew = build_irm_crew(sw_roles, sw_task, round_num=0, prev_outputs=prev, llm=mock_llm)
        coder_task = crew.tasks[1]  # Coder is index 1
        assert "ARCH_DESIGN" in coder_task.description
        assert "TEST_CONTENT" not in coder_task.description  # upstream-only

    def test_no_scorecard_in_round0(self, sw_roles, sw_task, mock_llm):
        crew = build_irm_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        for task in crew.tasks:
            assert "PERFORMANCE SCORECARD" not in task.description


class TestJROCrew:
    def test_crew_structure(self, sw_roles, sw_task, mock_llm):
        crew = build_jro_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        assert len(crew.agents) == 4
        assert len(crew.tasks) == 4

    def test_team_oriented_goal(self, sw_roles, sw_task, mock_llm):
        crew = build_jro_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        for agent in crew.agents:
            goal_lower = agent.goal.lower()
            assert "overall" in goal_lower or "team" in goal_lower or "project" in goal_lower

    def test_full_context_injection(self, sw_roles, sw_task, mock_llm):
        """All agents see ALL previous outputs (full transparency)."""
        prev = {
            "Architect": "ARCH_OUTPUT",
            "Coder": "CODER_OUTPUT",
            "Tester": "TESTER_OUTPUT",
        }
        crew = build_jro_crew(sw_roles, sw_task, round_num=1, prev_outputs=prev, llm=mock_llm)
        # Architect (round 1) should see all outputs
        arch_task_desc = crew.tasks[0].description
        assert "ARCH_OUTPUT" in arch_task_desc
        assert "CODER_OUTPUT" in arch_task_desc
        assert "TESTER_OUTPUT" in arch_task_desc

    def test_jro_uses_full_team_context_label(self, sw_roles, sw_task, mock_llm):
        prev = {"Architect": "Some design", "Coder": "Some code"}
        crew = build_jro_crew(sw_roles, sw_task, round_num=0, prev_outputs=prev, llm=mock_llm)
        # At least one task should reference full team context
        assert any("FULL TEAM CONTEXT" in t.description for t in crew.tasks)

    def test_jro_no_delegation(self, sw_roles, sw_task, mock_llm):
        """allow_delegation=False keeps the controlled variable clean."""
        crew = build_jro_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        for agent in crew.agents:
            assert agent.allow_delegation is False


class TestIAMDCrew:
    def test_crew_structure(self, sw_roles, sw_task, mock_llm):
        crew = build_iamd_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, scorecards={}, llm=mock_llm)
        assert len(crew.agents) == 4
        assert len(crew.tasks) == 4

    def test_no_scorecard_in_round0(self, sw_roles, sw_task, mock_llm):
        """Round 0 has no scorecards yet."""
        crew = build_iamd_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, scorecards={}, llm=mock_llm)
        for task in crew.tasks:
            assert "PERFORMANCE SCORECARD" not in task.description

    def test_scorecard_injected_for_role(self, sw_roles, sw_task, mock_llm):
        """Scorecard appears in the task description of the matching role."""
        scorecards = {
            "Coder": "CODER SCORECARD:\n  Syntax: FAIL — missing colon",
            "Architect": "ARCHITECT SCORECARD:\n  Design is solid.",
        }
        crew = build_iamd_crew(
            sw_roles, sw_task, round_num=1, prev_outputs={}, scorecards=scorecards, llm=mock_llm
        )
        arch_task = crew.tasks[0]
        coder_task = crew.tasks[1]
        tester_task = crew.tasks[2]

        assert "PERFORMANCE SCORECARD" in arch_task.description
        assert "ARCHITECT SCORECARD" in arch_task.description
        assert "PERFORMANCE SCORECARD" in coder_task.description
        assert "CODER SCORECARD" in coder_task.description
        assert "PERFORMANCE SCORECARD" not in tester_task.description  # no scorecard for Tester

    def test_iamd_upstream_only(self, sw_roles, sw_task, mock_llm):
        """IAMD uses upstream-only context (same as IRM), not full transparency."""
        prev = {"Architect": "ARCH_DESIGN", "Tester": "TEST_CONTENT"}
        crew = build_iamd_crew(sw_roles, sw_task, round_num=0, prev_outputs=prev, scorecards={}, llm=mock_llm)
        coder_task = crew.tasks[1]
        assert "ARCH_DESIGN" in coder_task.description
        assert "TEST_CONTENT" not in coder_task.description

    def test_iamd_no_delegation(self, sw_roles, sw_task, mock_llm):
        crew = build_iamd_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, scorecards={}, llm=mock_llm)
        for agent in crew.agents:
            assert agent.allow_delegation is False

    def test_self_oriented_plus_scorecard_awareness(self, sw_roles, sw_task, mock_llm):
        """IAMD backstory mentions feedback/scorecard, unlike IRM."""
        crew = build_iamd_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, scorecards={}, llm=mock_llm)
        for agent in crew.agents:
            backstory_lower = agent.backstory.lower()
            assert "scorecard" in backstory_lower or "feedback" in backstory_lower


class TestInfoSeparation:
    """Research synthesis requires Researcher_A/B to see different source sets."""

    def test_irm_researcher_a_vs_b_different_tasks(self, rs_roles, rs_task, mock_llm):
        crew = build_irm_crew(rs_roles, rs_task, round_num=0, prev_outputs={}, llm=mock_llm)
        task_a = next(t for t in crew.tasks if "Researcher_A" in t.agent.role)
        task_b = next(t for t in crew.tasks if "Researcher_B" in t.agent.role)
        assert task_a.description != task_b.description

    def test_jro_researcher_a_vs_b_different_tasks(self, rs_roles, rs_task, mock_llm):
        crew = build_jro_crew(rs_roles, rs_task, round_num=0, prev_outputs={}, llm=mock_llm)
        task_a = next(t for t in crew.tasks if "Researcher_A" in t.agent.role)
        task_b = next(t for t in crew.tasks if "Researcher_B" in t.agent.role)
        assert task_a.description != task_b.description

    def test_iamd_researcher_a_vs_b_different_tasks(self, rs_roles, rs_task, mock_llm):
        crew = build_iamd_crew(rs_roles, rs_task, round_num=0, prev_outputs={}, scorecards={}, llm=mock_llm)
        task_a = next(t for t in crew.tasks if "Researcher_A" in t.agent.role)
        task_b = next(t for t in crew.tasks if "Researcher_B" in t.agent.role)
        assert task_a.description != task_b.description


# ---------------------------------------------------------------------------
# Evaluation bridge tests
# ---------------------------------------------------------------------------

class TestExtractAgentOutputs:
    def _make_mock_crew_output(self, texts: list[str]):
        """Build a minimal mock CrewOutput."""
        class MockTaskOutput:
            def __init__(self, raw):
                self.raw = raw

        class MockCrewOutput:
            def __init__(self, texts):
                self.tasks_output = [MockTaskOutput(t) for t in texts]

        return MockCrewOutput(texts)

    def test_correct_count(self):
        crew_out = self._make_mock_crew_output(["A", "B", "C", "D"])
        outputs = extract_agent_outputs(crew_out, ["Architect", "Coder", "Tester", "Reviewer"], round_number=0)
        assert len(outputs) == 4

    def test_role_assignment(self):
        crew_out = self._make_mock_crew_output(["design", "code", "tests", "review"])
        roles = ["Architect", "Coder", "Tester", "Reviewer"]
        outputs = extract_agent_outputs(crew_out, roles, round_number=2)
        for i, output in enumerate(outputs):
            assert output.role == roles[i]
            assert output.round_number == 2

    def test_content_preserved(self):
        texts = ["DESIGN DOC", "def solution(): pass", "def test_it(): pass", "LGTM"]
        crew_out = self._make_mock_crew_output(texts)
        outputs = extract_agent_outputs(crew_out, ["Architect", "Coder", "Tester", "Reviewer"], round_number=0)
        for i, output in enumerate(outputs):
            assert output.content == texts[i]

    def test_agent_id_format(self):
        crew_out = self._make_mock_crew_output(["x", "y"])
        outputs = extract_agent_outputs(crew_out, ["Architect", "Coder"], round_number=0)
        assert outputs[0].agent_id == "architect_0"
        assert outputs[1].agent_id == "coder_1"

    def test_token_counts_zero(self):
        """Token counts are 0 since CrewAI doesn't expose per-task usage."""
        crew_out = self._make_mock_crew_output(["output"])
        outputs = extract_agent_outputs(crew_out, ["Architect"], round_number=0)
        assert outputs[0].response.total_tokens == 0
        assert outputs[0].metadata["token_counts_available"] is False


class TestSandboxScorecards:
    def _make_exec_result(self, **kwargs):
        from masid.evaluation.sandbox import ExecutionResult
        defaults = dict(
            syntax_valid=True, code_runs=True,
            tests_run=True, tests_passed=3, tests_failed=0,
            tests_total=3, execution_score=0.9,
        )
        defaults.update(kwargs)
        return ExecutionResult(**defaults)

    def test_coder_all_pass(self):
        result = self._make_exec_result()
        card = _build_sandbox_scorecard("Coder", result, [])
        assert "CODER SCORECARD" in card
        assert "PASS" in card

    def test_coder_syntax_fail(self):
        from masid.evaluation.sandbox import ExecutionResult
        result = ExecutionResult(syntax_valid=False, syntax_error="SyntaxError: invalid syntax")
        card = _build_sandbox_scorecard("Coder", result, [])
        assert "FAIL" in card
        assert "syntax" in card.lower()

    def test_tester_scorecard(self):
        result = self._make_exec_result(tests_passed=2, tests_failed=1, tests_total=3)
        card = _build_sandbox_scorecard("Tester", result, [])
        assert "TESTER SCORECARD" in card
        assert "3" in card  # total count

    def test_architect_scorecard_pass(self):
        result = self._make_exec_result()
        card = _build_sandbox_scorecard("Architect", result, [])
        assert "ARCHITECT SCORECARD" in card
        assert "SUCCESS" in card

    def test_reviewer_scorecard(self):
        result = self._make_exec_result(execution_score=0.8)
        card = _build_sandbox_scorecard("Reviewer", result, [])
        assert "REVIEWER SCORECARD" in card
        assert "80%" in card

    def test_unknown_role(self):
        result = self._make_exec_result()
        card = _build_sandbox_scorecard("ProductManager", result, [])
        assert "ProductManager SCORECARD" in card


# ---------------------------------------------------------------------------
# Architecture differentiation tests (behavioral contracts)
# ---------------------------------------------------------------------------

class TestArchitectureDifferentiation:
    """Verify that the three architectures produce structurally different crews."""

    def test_irm_vs_jro_goals_differ(self, sw_roles, sw_task, mock_llm):
        irm = build_irm_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        jro = build_jro_crew(sw_roles, sw_task, round_num=0, prev_outputs={}, llm=mock_llm)
        irm_goal = irm.agents[0].goal.lower()
        jro_goal = jro.agents[0].goal.lower()
        assert irm_goal != jro_goal

    def test_irm_vs_jro_context_differs_when_prev_outputs(self, sw_roles, sw_task, mock_llm):
        """With prev outputs, IRM injects only upstream; JRO injects all."""
        prev = {"Architect": "ARCH", "Coder": "CODE", "Tester": "TESTS"}
        irm = build_irm_crew(sw_roles, sw_task, round_num=1, prev_outputs=prev, llm=mock_llm)
        jro = build_jro_crew(sw_roles, sw_task, round_num=1, prev_outputs=prev, llm=mock_llm)

        # Architect has no upstream, so IRM Architect sees nothing from prev; JRO sees all
        irm_arch_desc = irm.tasks[0].description
        jro_arch_desc = jro.tasks[0].description

        # JRO Architect should see Coder and Tester outputs (it sees ALL)
        assert "CODE" in jro_arch_desc
        assert "TESTS" in jro_arch_desc

    def test_iamd_vs_irm_scorecards_only_in_iamd(self, sw_roles, sw_task, mock_llm):
        scorecards = {"Coder": "CODER SCORECARD:\n  Action: fix it"}
        irm = build_irm_crew(sw_roles, sw_task, round_num=1, prev_outputs={}, llm=mock_llm)
        iamd = build_iamd_crew(sw_roles, sw_task, round_num=1, prev_outputs={}, scorecards=scorecards, llm=mock_llm)

        irm_coder_desc = irm.tasks[1].description
        iamd_coder_desc = iamd.tasks[1].description

        assert "PERFORMANCE SCORECARD" not in irm_coder_desc
        assert "PERFORMANCE SCORECARD" in iamd_coder_desc


# ---------------------------------------------------------------------------
# Integration test (requires live vLLM at http://localhost:8000/v1)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_single_trial_end_to_end(tmp_path):
    """Run one full IAMD trial against the live vLLM endpoint.

    Requires: vLLM serving meta-llama/Llama-3.3-70B-Instruct at localhost:8000
    and OPENAI_API_KEY set for the judge (gpt-4.1-mini).
    """
    from masid.storage import ExperimentDB
    from masid_crewai.config import CrewAIConfig
    from masid_crewai.runner import CrewAITrialRunner

    config = CrewAIConfig()
    config.experiment.max_rounds = 1  # single round to keep the test fast
    db = ExperimentDB(tmp_path / "integration_test.db")

    runner = CrewAITrialRunner(config=config, db=db)
    metrics = runner.run(
        architecture_key="irm",
        domain="project_planning",
        task_id="pp_001",
        seed=1,
    )

    assert metrics.quality_score >= 0.0
    assert metrics.quality_score <= 1.0
    assert metrics.num_rounds == 1
    assert metrics.architecture == "irm"
    assert metrics.domain == "project_planning"
    assert metrics.metadata.get("framework") == "crewai"

    # Verify DB persistence
    trials = db.get_all_trials()
    assert len(trials) == 1
    assert trials[0]["architecture"] == "irm"
