"""LLM-as-judge evaluator for subjective quality scoring.

Uses a separate LLM call to evaluate the collective output of a trial
on dimensions like correctness, completeness, coherence, and
integration quality.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from masid.models import LLMClient


# The judge prompt asks for structured JSON scores
_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of a multi-agent \
collaboration output. You will be given a task description and the \
combined outputs of all agents.

Evaluate on these dimensions (each scored 1-10):
1. **Correctness**: Is the output factually and technically correct?
2. **Completeness**: Does it address all parts of the task?
3. **Coherence**: Is the output internally consistent and well-organized?
4. **Integration**: Do the different agents' outputs fit together well?

Respond with ONLY a JSON object in this exact format:
{
  "correctness": <1-10>,
  "completeness": <1-10>,
  "coherence": <1-10>,
  "integration": <1-10>,
  "overall": <1-10>,
  "rationale": "<brief explanation>"
}
"""


def judge_trial_output(
    client: LLMClient,
    task_description: str,
    combined_output: str,
    expected_hint: Optional[str] = None,
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

    Returns
    -------
    dict
        Scores dict with keys: correctness, completeness, coherence,
        integration, overall, rationale. Scores are normalized to [0, 1].
    """
    user_content = f"## Task Description\n{task_description}\n\n"
    if expected_hint:
        user_content += f"## Expected Output Characteristics\n{expected_hint}\n\n"
    user_content += f"## Combined Agent Output\n{combined_output}"

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = client.chat(messages, temperature=0.1, max_tokens=512)

    return _parse_judge_response(response.content)


def _parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON response, with fallback for malformed output."""
    # Try to extract JSON from the response
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            # Normalize scores from 1-10 to 0-1
            return {
                "correctness": raw.get("correctness", 5) / 10.0,
                "completeness": raw.get("completeness", 5) / 10.0,
                "coherence": raw.get("coherence", 5) / 10.0,
                "integration": raw.get("integration", 5) / 10.0,
                "overall": raw.get("overall", 5) / 10.0,
                "rationale": raw.get("rationale", ""),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: return neutral scores
    return {
        "correctness": 0.5,
        "completeness": 0.5,
        "coherence": 0.5,
        "integration": 0.5,
        "overall": 0.5,
        "rationale": f"Failed to parse judge response: {text[:200]}",
    }
