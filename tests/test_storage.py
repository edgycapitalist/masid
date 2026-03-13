"""Tests for masid.storage."""

import pytest

from masid.storage import ExperimentDB


class TestExperimentDB:
    def test_create_db(self, tmp_db):
        assert tmp_db.count_trials() == 0

    def test_save_and_retrieve_trial(self, tmp_db):
        tmp_db.save_trial(
            trial_id="test_001",
            architecture="irm",
            domain="software_dev",
            model="llama3.1:8b",
            task_id="sw_001",
            scores={"overall": 0.8, "correctness": 0.9, "completeness": 0.7,
                    "coherence": 0.8, "integration": 0.7},
            efficiency={"total_tokens": 1000, "total_latency": 5.0, "num_rounds": 3},
            coordination={"duplication_rate": 0.1, "conflict_rate": 0.0,
                         "consistency_score": 0.9},
            agent_scores={"coder_0": 0.8, "tester_1": 0.7},
            judge_rationale="Good output.",
            seed=42,
        )
        assert tmp_db.count_trials() == 1

        trials = tmp_db.get_all_trials()
        assert len(trials) == 1
        assert trials[0]["trial_id"] == "test_001"
        assert trials[0]["quality_score"] == 0.8
        assert trials[0]["architecture"] == "irm"

    def test_filter_by_architecture(self, tmp_db):
        for arch in ["irm", "jro", "iamd"]:
            tmp_db.save_trial(
                trial_id=f"test_{arch}",
                architecture=arch,
                domain="software_dev",
                model="llama3.1:8b",
                task_id="sw_001",
                scores={"overall": 0.5},
                efficiency={"total_tokens": 100, "total_latency": 1.0, "num_rounds": 1},
                coordination={},
                agent_scores={},
            )
        irm_trials = tmp_db.get_trials_by(architecture="irm")
        assert len(irm_trials) == 1
        assert irm_trials[0]["architecture"] == "irm"

    def test_save_agent_output(self, tmp_db):
        tmp_db.save_trial(
            trial_id="test_out",
            architecture="irm",
            domain="software_dev",
            model="test",
            task_id="sw_001",
            scores={"overall": 0.5},
            efficiency={"total_tokens": 100, "total_latency": 1.0, "num_rounds": 1},
            coordination={},
            agent_scores={},
        )
        tmp_db.save_agent_output(
            trial_id="test_out",
            agent_id="coder_0",
            role="Coder",
            round_number=0,
            content="def hello(): pass",
            prompt_tokens=20,
            completion_tokens=10,
            latency_seconds=0.5,
        )
        # Verify it was saved (no error)
        assert tmp_db.count_trials() == 1
