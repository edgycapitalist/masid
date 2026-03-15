"""Tests for masid.orchestrator (TrialRunner)."""


from masid.config import MASIDConfig
from masid.domains.registry import get_tasks
from masid.orchestrator import TrialRunner
from masid.storage import ExperimentDB
from tests.conftest import MockLLMClient


class TestTrialRunner:
    """Test the orchestrator using mock LLM clients.

    These tests verify the full pipeline without hitting a real model.
    We monkey-patch the LLMClient constructor to return our mock.
    """

    def _make_runner(self, tmp_path):
        config = MASIDConfig(
            model={"provider": "mock", "name": "mock-model"},
            experiment={"trials_per_cell": 1, "max_rounds": 1, "seed": 42},
        )
        db = ExperimentDB(tmp_path / "test.db")
        return TrialRunner(config=config, db=db), db

    def test_run_irm_trial(self, tmp_path, monkeypatch):
        runner, db = self._make_runner(tmp_path)
        tasks = get_tasks("software_dev")

        # Patch LLMClient to return mock
        mock = MockLLMClient(responses=[
            "Architecture: modular design with 3 components.",
            "def shorten(url): return hash(url)",
            "test_shorten passed. Coverage: 80%",
            "Code review: looks good. Minor style issues.",
            # Judge response
            '{"correctness": 7, "completeness": 6, "coherence": 8, '
            '"integration": 7, "overall": 7, "rationale": "Decent."}',
        ])
        monkeypatch.setattr(
            "masid.orchestrator.LLMClient",
            lambda **kwargs: mock,
        )

        metrics = runner.run(
            architecture_key="irm",
            domain="software_dev",
            task=tasks[0],
            seed=42,
        )
        assert metrics.architecture == "irm"
        assert metrics.domain == "software_dev"
        assert metrics.quality_score > 0
        assert metrics.total_tokens > 0
        assert db.count_trials() == 1

    def test_run_jro_trial(self, tmp_path, monkeypatch):
        runner, db = self._make_runner(tmp_path)
        tasks = get_tasks("software_dev")

        mock = MockLLMClient(responses=[
            "Team architecture approach...",
            "Collaborative code...",
            "Integrated tests...",
            "Team review...",
            '{"correctness": 8, "completeness": 7, "coherence": 9, '
            '"integration": 8, "overall": 8, "rationale": "Good team work."}',
        ])
        monkeypatch.setattr("masid.orchestrator.LLMClient", lambda **kwargs: mock)

        metrics = runner.run(
            architecture_key="jro",
            domain="software_dev",
            task=tasks[0],
            seed=42,
        )
        assert metrics.architecture == "jro"
        assert db.count_trials() == 1

    def test_run_iamd_trial(self, tmp_path, monkeypatch):
        runner, db = self._make_runner(tmp_path)
        tasks = get_tasks("software_dev")

        mock = MockLLMClient(responses=[
            "Incentive-aligned architecture...",
            "Code optimized for testability...",
            "Tests with improvement suggestions...",
            "Review with actionable feedback...",
            '{"correctness": 8, "completeness": 8, "coherence": 8, '
            '"integration": 9, "overall": 8, "rationale": "Well aligned."}',
        ])
        monkeypatch.setattr("masid.orchestrator.LLMClient", lambda **kwargs: mock)

        metrics = runner.run(
            architecture_key="iamd",
            domain="software_dev",
            task=tasks[0],
            seed=42,
        )
        assert metrics.architecture == "iamd"
        assert db.count_trials() == 1

    def test_fault_injection(self, tmp_path, monkeypatch):
        runner, db = self._make_runner(tmp_path)
        tasks = get_tasks("software_dev")

        mock = MockLLMClient(responses=[
            "output 1", "output 2", "output 3", "output 4",
            '{"correctness": 3, "completeness": 3, "coherence": 3, '
            '"integration": 3, "overall": 3, "rationale": "Degraded."}',
        ])
        monkeypatch.setattr("masid.orchestrator.LLMClient", lambda **kwargs: mock)

        metrics = runner.run(
            architecture_key="irm",
            domain="software_dev",
            task=tasks[0],
            fault_type="degraded_prompt",
            fault_agent_role="Coder",
        )
        assert metrics.metadata["fault_type"] == "degraded_prompt"
