"""Unified LLM client for MASID.

Provides a single interface for calling local models (via Ollama) and
commercial APIs (via LiteLLM). Every LLM interaction in the framework
goes through this module, making it trivial to swap backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import litellm

# Suppress LiteLLM's verbose logging by default
litellm.suppress_debug_info = True


@dataclass(frozen=True)
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    raw: Optional[dict] = field(default=None, repr=False)


class LLMClient:
    """Unified client for local and API-based LLMs.

    Parameters
    ----------
    provider : str
        One of ``"ollama"``, ``"openai"``, ``"anthropic"``, or any
        provider supported by LiteLLM.
    model_name : str
        Model identifier. For Ollama use the tag (e.g. ``"llama3.1:8b"``).
        For APIs use the standard name (e.g. ``"gpt-4o"``).
    base_url : str or None
        Required for Ollama (e.g. ``"http://localhost:11434"``).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate per call.
    timeout : int
        Timeout in seconds for a single call.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "llama3.1:8b",
        base_url: Optional[str] = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Build the model string LiteLLM expects
        if provider == "ollama":
            self._litellm_model = f"ollama/{model_name}"
        else:
            self._litellm_model = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat-completion request and return a structured response.

        Parameters
        ----------
        messages : list of dict
            OpenAI-style messages, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
        temperature : float or None
            Override the default temperature for this call.
        max_tokens : int or None
            Override the default max_tokens for this call.

        Returns
        -------
        LLMResponse
        """
        t0 = time.perf_counter()

        kwargs: dict = {
            "model": self._litellm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "timeout": self.timeout,
        }

        if self.base_url and self.provider == "ollama":
            kwargs["api_base"] = self.base_url

        response = litellm.completion(**kwargs)

        latency = time.perf_counter() - t0

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_seconds=round(latency, 3),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.provider!r}, model={self.model_name!r}, "
            f"base_url={self.base_url!r})"
        )
