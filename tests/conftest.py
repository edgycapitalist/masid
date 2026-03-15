"""Shared test fixtures for MASID."""

from __future__ import annotations

import pytest

from masid.models import LLMClient, LLMResponse

# ---------------------------------------------------------------------------
# Mock LLM client that returns deterministic responses without a real model
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """A mock LLM client that returns canned responses without network calls."""

    def __init__(
        self,
        responses: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        # Don't call super().__init__ to avoid requiring a real endpoint
        self.provider = "mock"
        self.model_name = "mock-model"
        self.base_url = None
        self.temperature = 0.7
        self.max_tokens = 2048
        self.timeout = 10
        self._litellm_model = "mock/mock-model"

        self._responses = responses or ["This is a mock response."]
        self._call_count = 0
        self._call_log: list[list[dict]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self._call_log.append(messages)
        response_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return LLMResponse(
            content=response_text,
            model=self.model_name,
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            latency_seconds=0.1,
            raw=None,
        )


@pytest.fixture
def mock_client() -> MockLLMClient:
    """Provide a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_client_with_judge() -> MockLLMClient:
    """Provide a mock client that returns valid judge JSON."""
    judge_response = (
        '{"correctness": 8, "completeness": 7, "coherence": 9, '
        '"integration": 7, "overall": 8, "rationale": "Good work."}'
    )
    return MockLLMClient(responses=[judge_response])


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database."""
    from masid.storage import ExperimentDB
    return ExperimentDB(tmp_path / "test.db")
