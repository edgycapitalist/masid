"""Tests for masid.evaluation."""

import pytest

from masid.agents import AgentOutput
from masid.evaluation import compute_duplication_rate, compute_efficiency_metrics
from masid.evaluation.judge import _parse_judge_response
from masid.models import LLMResponse


def _make_output(role: str, content: str, tokens: int = 80) -> AgentOutput:
    return AgentOutput(
        agent_id=f"{role.lower()}_0",
        role=role,
        content=content,
        response=LLMResponse(
            content=content,
            model="mock",
            prompt_tokens=tokens // 2,
            completion_tokens=tokens // 2,
            total_tokens=tokens,
            latency_seconds=0.1,
        ),
        round_number=0,
    )


class TestEfficiencyMetrics:
    def test_single_round(self):
        outputs = [_make_output("A", "hello", 100), _make_output("B", "world", 200)]
        result = compute_efficiency_metrics([outputs])
        assert result["total_tokens"] == 300
        assert result["num_rounds"] == 1

    def test_multiple_rounds(self):
        r0 = [_make_output("A", "a", 50)]
        r1 = [_make_output("A", "b", 60)]
        result = compute_efficiency_metrics([r0, r1])
        assert result["total_tokens"] == 110
        assert result["num_rounds"] == 2


class TestDuplicationRate:
    def test_no_duplication(self):
        outputs = [
            _make_output("A", "the quick brown fox jumps over the lazy dog"),
            _make_output("B", "a completely different sentence about cats and mice"),
        ]
        rate = compute_duplication_rate(outputs)
        assert rate < 0.5

    def test_identical_outputs(self):
        text = "the quick brown fox jumps over the lazy dog again and again"
        outputs = [_make_output("A", text), _make_output("B", text)]
        rate = compute_duplication_rate(outputs)
        assert rate > 0.9

    def test_single_output(self):
        outputs = [_make_output("A", "hello")]
        assert compute_duplication_rate(outputs) == 0.0


class TestJudgeParser:
    def test_valid_json(self):
        text = '{"correctness": 8, "completeness": 7, "coherence": 9, "integration": 7, "overall": 8, "rationale": "Good."}'
        result = _parse_judge_response(text)
        assert result["correctness"] == 0.8
        assert result["overall"] == 0.8
        assert result["rationale"] == "Good."

    def test_json_with_preamble(self):
        text = 'Here is my evaluation:\n{"correctness": 6, "completeness": 5, "coherence": 7, "integration": 6, "overall": 6, "rationale": "OK."}'
        result = _parse_judge_response(text)
        assert result["correctness"] == 0.6

    def test_malformed_json_fallback(self):
        text = "I think the output is pretty good overall."
        result = _parse_judge_response(text)
        assert result["overall"] == 0.5  # fallback
        assert "Failed to parse" in result["rationale"]
