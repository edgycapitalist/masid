"""LLM-as-judge evaluator for subjective quality scoring.

Uses a separate LLM call to evaluate the collective output of a trial.
Domain-specific rubrics with concrete grading criteria force the judge
to discriminate between good and bad outputs instead of defaulting to
"8/10 for everything."
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from masid.models import LLMClient

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 6000

# ---------------------------------------------------------------------------
# Domain-specific rubrics with concrete anchors
# ---------------------------------------------------------------------------
# The key to forcing discrimination: give the judge SPECIFIC failure modes
# to look for, and anchor scores to concrete behaviors, not vibes.

_RUBRICS: dict[str, str] = {
    "software_dev": """\
You are a strict code review judge. Evaluate this multi-agent software development output.

Score EACH dimension from 1 to 10 using these STRICT anchors:

CORRECTNESS (does the code work?):
  1-3: Code has syntax errors or crashes immediately
  4-5: Code runs but core functionality is broken
  6-7: Core functionality works but edge cases fail
  8-9: Works correctly with minor issues
  10: Fully correct, handles all edge cases

COMPLETENESS (does it cover all requirements?):
  1-3: Missing most requirements from the task
  4-5: Covers fewer than half the requirements
  6-7: Covers most requirements, misses 1-2
  8-9: Covers all stated requirements
  10: Covers all requirements plus thoughtful extras

COHERENCE (is it well-organized?):
  1-3: Disorganized, contradictory between agents
  4-5: Some structure but agents overlap or contradict
  6-7: Mostly organized, minor inconsistencies
  8-9: Well-structured, agents complement each other
  10: Excellently organized, reads as unified work

INTEGRATION (do agents' outputs fit together?):
  1-3: Agents produced incompatible outputs
  4-5: Outputs partially compatible, major gaps
  6-7: Outputs mostly compatible, tests don't match code
  8-9: Good integration, tests cover the code
  10: Perfect integration, all pieces fit seamlessly

Be STRICT. Most outputs should score 4-7. Only exceptional work gets 8+.
An output where code crashes is NOT an 8 on correctness.""",

    "research_synthesis": """\
You are a strict academic reviewer. Evaluate this multi-agent research synthesis.

Score EACH dimension from 1 to 10 using these STRICT anchors:

CORRECTNESS (are claims accurate?):
  1-3: Contains fabricated claims or major factual errors
  4-5: Some claims unsupported or misrepresented from sources
  6-7: Most claims accurate but some lack proper attribution
  8-9: All claims accurately represent the sources
  10: Perfect accuracy with nuanced representation

COMPLETENESS (does it cover all sources?):
  1-3: Ignores most source material
  4-5: Covers fewer than half the key points from sources
  6-7: Covers most key points, misses some important nuances
  8-9: Comprehensive coverage of all sources
  10: Comprehensive plus identifies connections sources missed

COHERENCE (is the synthesis well-structured?):
  1-3: Disjointed, reads like separate summaries pasted together
  4-5: Some structure but transitions are weak
  6-7: Reasonable structure, some repetition across agents
  8-9: Well-structured narrative that flows logically
  10: Excellent narrative arc with clear thematic organization

INTEGRATION (do researchers + synthesizer + fact-checker work together?):
  1-3: Synthesizer ignores researcher findings, fact-checker misses errors
  4-5: Partial integration, fact-checker adds little value
  6-7: Decent integration, fact-checker catches some issues
  8-9: Strong integration, fact-checker meaningfully improves output
  10: Seamless collaboration, each agent adds clear value

Be STRICT. Most outputs should score 4-7. Only exceptional work gets 8+.
A synthesis that just lists findings without connecting them is NOT an 8.""",

    "project_planning": """\
You are a strict project management reviewer. Evaluate this multi-agent project plan.

Score EACH dimension from 1 to 10 using these STRICT anchors:

CORRECTNESS (is the plan feasible?):
  1-3: Plan violates stated constraints (budget, timeline, resources)
  4-5: Plan is technically possible but has unrealistic assumptions
  6-7: Plan is feasible with minor issues in resource/time estimates
  8-9: Solid, realistic plan that respects all constraints
  10: Excellent plan with built-in contingency

COMPLETENESS (does it cover all aspects?):
  1-3: Missing major components (no risk plan, no QA, etc.)
  4-5: Has most components but they're superficial
  6-7: All components present but some lack detail
  8-9: Comprehensive plan with sufficient detail
  10: Comprehensive with exceptional depth

COHERENCE (are the components consistent?):
  1-3: Schedule conflicts with resources, risk plan ignores timeline
  4-5: Some consistency but components don't reference each other
  6-7: Mostly consistent, minor alignment issues
  8-9: Well-aligned components that reference each other
  10: Perfectly consistent, integrated plan

INTEGRATION (do agents' plans fit together?):
  1-3: Agents produced conflicting plans
  4-5: Partial alignment, major gaps between components
  6-7: Mostly aligned, some gaps in handoffs
  8-9: Strong alignment, clear dependencies between components
  10: Seamless integration across all planning dimensions

Be STRICT. Most outputs should score 4-7. Only exceptional work gets 8+.
A plan that exceeds the budget is NOT an 8 on correctness.""",
}

_DEFAULT_RUBRIC = _RUBRICS["software_dev"]


def judge_trial_output(
    client: LLMClient,
    task_description: str,
    combined_output: str,
    expected_hint: Optional[str] = None,
    domain: str = "software_dev",
    max_retries: int = 2,
) -> dict:
    """Run LLM-as-judge evaluation on a trial's combined output.

    Parameters
    ----------
    client : LLMClient
        The LLM client to use for judging.
    task_description : str
        The original task specification.
    combined_output : str
        Concatenated outputs from all agents.
    expected_hint : str or None
        Optional hint about what good output looks like.
    domain : str
        Task domain key — selects the appropriate rubric.
    max_retries : int
        Number of retries if JSON parsing fails.

    Returns
    -------
    dict with keys: correctness, completeness, coherence, integration,
    overall, rationale. Scores normalized to [0, 1].
    """
    if len(combined_output) > _MAX_OUTPUT_CHARS:
        combined_output = combined_output[:_MAX_OUTPUT_CHARS] + "\n\n[... truncated ...]"

    rubric = _RUBRICS.get(domain, _DEFAULT_RUBRIC)

    user_content = (
        f"Task description:\n{task_description[:1000]}\n\n"
    )
    if expected_hint:
        user_content += f"What good output looks like:\n{expected_hint[:500]}\n\n"
    user_content += (
        f"Agent output to evaluate:\n{combined_output}\n\n"
        f"Score this output. Remember: be STRICT, most work is 4-7 range.\n"
        f"Respond with ONLY JSON:\n"
        f'{"{"}"correctness": N, "completeness": N, "coherence": N, '
        f'"integration": N, "overall": N, "rationale": "brief reason"{"}"}'
    )

    messages = [
        {"role": "system", "content": rubric},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries + 1):
        response = client.chat(messages, temperature=0.1, max_tokens=256)
        result = _parse_judge_response(response.content)

        if result["rationale"] and not result["rationale"].startswith("Failed to parse"):
            logger.info(
                "Judge scored trial (attempt %d): overall=%.1f",
                attempt + 1, result["overall"],
            )
            return result

        logger.warning(
            "Judge parse failed (attempt %d/%d): %s",
            attempt + 1, max_retries + 1, response.content[:150],
        )

        if attempt < max_retries:
            messages = [
                {"role": "system", "content": rubric},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "{"},
            ]

    return result


def _parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON response, with multiple fallback strategies."""
    if not text.strip().startswith("{"):
        text_with_brace = "{" + text
    else:
        text_with_brace = text

    for candidate in [text_with_brace, text]:
        try:
            raw = json.loads(candidate.strip())
            return _normalize_scores(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            return _normalize_scores(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    scores = {}
    for key in ["correctness", "completeness", "coherence", "integration", "overall"]:
        match = re.search(rf"{key}[\"']?\s*[:=]\s*(\d+)", text, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))

    if scores:
        if "overall" not in scores and scores:
            scores["overall"] = round(sum(scores.values()) / len(scores))
        rationale_match = re.search(
            r"rationale[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']", text, re.IGNORECASE
        )
        scores["rationale"] = (
            rationale_match.group(1) if rationale_match else "Parsed via regex fallback"
        )
        return _normalize_scores(scores)

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
            v = max(1.0, min(10.0, v))
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
