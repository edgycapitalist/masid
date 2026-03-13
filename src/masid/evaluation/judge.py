"""LLM-as-judge evaluator for subjective quality scoring.

Uses a separate LLM call to evaluate the collective output of a trial
on dimensions like correctness, completeness, coherence, and
integration quality.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from masid.models import LLMClient

logger = logging.getLogger(__name__)

# Maximum characters of combined output to send to the judge.
# Longer outputs get truncated to avoid overwhelming small models.
_MAX_OUTPUT_CHARS = 6000

# The judge prompt — designed for reliability with small models.
# Uses a simpler format and explicit examples to improve JSON compliance.
_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator. Score the quality of a multi-agent collaboration output.

Score each dimension from 1 (worst) to 10 (best):
- correctness: Is it technically correct?
- completeness: Does it cover all parts of the task?
- coherence: Is it well-organized and consistent?
- integration: Do the agents' outputs fit together?
- overall: Overall quality considering all dimensions.

You MUST respond with ONLY this JSON, nothing else:
{"correctness": 7, "completeness": 6, "coherence": 8, "integration": 5, "overall": 7, "rationale": "brief reason"}
"""


def judge_trial_output(
    client: LLMClient,
    task_description: str,
    combined_output: str,
    expected_hint: Optional[str] = None,
    max_retries: int = 2,
) -> dict:
    """Run LLM-as-judge evaluation on a trial's combined output.

    Parameters
    ----------
    client : LLMClient
        The LLM client to use for judging (can be a different model).
    task_description : str
        The original task specification.
    combined_output : str
        Concatenated outputs from all agents.
    expected_hint : str or None
        Optional hint about what good output looks like.
    max_retries : int
        Number of retries if JSON parsing fails.

    Returns
    -------
    dict
        Scores dict with keys: correctness, completeness, coherence,
        integration, overall, rationale. Scores are normalized to [0, 1].
    """
    # Truncate very long outputs to keep the judge focused
    if len(combined_output) > _MAX_OUTPUT_CHARS:
        truncated = combined_output[:_MAX_OUTPUT_CHARS]
        combined_output = truncated + "\n\n[... output truncated for evaluation ...]"

    user_content = f"Task: {task_description[:1000]}\n\n"
    if expected_hint:
        user_content += f"Expected: {expected_hint[:500]}\n\n"
    user_content += f"Agent Output:\n{combined_output}\n\nRespond with ONLY the JSON scores."

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries + 1):
        response = client.chat(messages, temperature=0.1, max_tokens=256)
        result = _parse_judge_response(response.content)

        if result["rationale"] and not result["rationale"].startswith("Failed to parse"):
            logger.info("Judge scored trial (attempt %d): overall=%.1f", attempt + 1, result["overall"])
            return result

        logger.warning(
            "Judge parse failed (attempt %d/%d): %s",
            attempt + 1, max_retries + 1, response.content[:150],
        )

        # On retry, add a stronger nudge
        if attempt < max_retries:
            messages = [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "{"},  # Prefill to force JSON start
            ]

    return result


def _parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON response, with multiple fallback strategies."""
    # If assistant prefill was used, the text might not start with {
    if not text.strip().startswith("{"):
        text_with_brace = "{" + text
    else:
        text_with_brace = text

    # Strategy 1: Try to parse the full text as JSON
    for candidate in [text_with_brace, text]:
        try:
            raw = json.loads(candidate.strip())
            return _normalize_scores(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Strategy 2: Extract JSON object with regex (handles preamble text)
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            return _normalize_scores(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Strategy 3: Extract individual scores with regex
    scores = {}
    for key in ["correctness", "completeness", "coherence", "integration", "overall"]:
        match = re.search(rf"{key}[\"']?\s*[:=]\s*(\d+)", text, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))

    if scores:
        # We found at least some scores via regex
        if "overall" not in scores and scores:
            scores["overall"] = round(sum(scores.values()) / len(scores))
        rationale_match = re.search(r"rationale[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']", text, re.IGNORECASE)
        scores["rationale"] = rationale_match.group(1) if rationale_match else "Parsed via regex fallback"
        return _normalize_scores(scores)

    # Strategy 4: Absolute fallback
    return {
        "correctness": 0.5,
        "completeness": 0.5,
        "coherence": 0.5,
        "integration": 0.5,
        "overall": 0.5,
        "rationale": f"Failed to parse judge response: {text[:300]}",
    }


def _normalize_scores(raw: dict) -> dict:
    """Normalize scores from 1-10 to 0-1 and ensure all keys exist."""
    def _clamp(val: object, default: int = 5) -> float:
        try:
            v = float(val)
            v = max(1.0, min(10.0, v))  # clamp to 1-10
            return v / 10.0
        except (TypeError, ValueError):
            return default / 10.0

    return {
        "correctness": _clamp(raw.get("correctness", 5)),
        "completeness": _clamp(raw.get("completeness", 5)),
        "coherence": _clamp(raw.get("coherence", 5)),
        "integration": _clamp(raw.get("integration", 5)),
        "overall": _clamp(raw.get("overall", 5)),
        "rationale": str(raw.get("rationale", "")),
    }
