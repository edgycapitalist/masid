"""Evaluation bridge — connects CrewAI outputs to MASID's evaluation pipeline.

Reuses masid.evaluation.judge and masid.evaluation.sandbox directly.
The only new code here is the mapping layer from CrewAI's TaskOutput
to MASID's AgentOutput, and the scorecard generator for IAMD.
"""

from __future__ import annotations

import logging
import time

from crewai.crews.crew_output import CrewOutput

from masid.agents import AgentOutput
from masid.domains import TaskSpec
from masid.evaluation.judge import judge_trial_output
from masid.evaluation.sandbox import ExecutionResult, evaluate_agent_code
from masid.models import LLMClient, LLMResponse
from masid_crewai.config import CrewAIConfig

logger = logging.getLogger(__name__)

# Sentinel used when CrewAI does not report token counts
_UNKNOWN_TOKENS = 0


def extract_agent_outputs(
    crew_output: CrewOutput,
    roles_in_order: list[str],
    round_number: int,
) -> list[AgentOutput]:
    """Map CrewAI TaskOutputs to MASID AgentOutputs.

    CrewAI's TaskOutput contains .raw (text), .agent (agent role string),
    and optionally usage metrics. Since CrewAI does not reliably expose
    per-task token counts, we store 0 and flag it in metadata.

    Parameters
    ----------
    crew_output:
        The CrewOutput returned by crew.kickoff().
    roles_in_order:
        The ordered list of role names matching the tasks list.
    round_number:
        0-indexed round number.
    """
    outputs: list[AgentOutput] = []
    task_outputs = crew_output.tasks_output

    for i, task_out in enumerate(task_outputs):
        role = roles_in_order[i] if i < len(roles_in_order) else f"agent_{i}"
        content = task_out.raw if task_out.raw else ""

        # CrewAI does not expose per-task token counts reliably.
        # Use 0 as placeholder; runner records this limitation in metadata.
        response = LLMResponse(
            content=content,
            model="crewai",
            prompt_tokens=_UNKNOWN_TOKENS,
            completion_tokens=_UNKNOWN_TOKENS,
            total_tokens=_UNKNOWN_TOKENS,
            latency_seconds=0.0,
        )

        agent_out = AgentOutput(
            agent_id=f"{role.lower()}_{i}",
            role=role,
            content=content,
            response=response,
            round_number=round_number,
            metadata={"token_counts_available": False},
        )
        outputs.append(agent_out)

    return outputs


def evaluate_crew_output(
    final_outputs: list[AgentOutput],
    task: TaskSpec,
    domain: str,
    judge_client: LLMClient,
) -> tuple[dict, ExecutionResult | None]:
    """Evaluate final-round outputs using MASID's judge and sandbox.

    Parameters
    ----------
    final_outputs:
        AgentOutputs from the last round.
    task:
        The TaskSpec (provides description and expected_output_hint).
    domain:
        Domain key (determines whether sandbox evaluation runs).
    judge_client:
        Configured LLMClient for the judge (OpenAI gpt-4.1-mini).

    Returns
    -------
    (judge_scores_dict, execution_result_or_None)
    """
    combined_output = "\n\n".join(
        f"=== {o.role} ===\n{o.content}" for o in final_outputs
    )

    # Software dev: run code sandbox
    execution_result: ExecutionResult | None = None
    if domain == "software_dev":
        execution_result = _run_sandbox(final_outputs)
        if execution_result:
            logger.info(
                "  Execution score: %.2f (syntax=%s, runs=%s, tests=%d/%d passed)",
                execution_result.execution_score,
                execution_result.syntax_valid,
                execution_result.code_runs,
                execution_result.tests_passed,
                execution_result.tests_total,
            )

    # LLM-as-judge (all domains)
    judge_scores = judge_trial_output(
        client=judge_client,
        task_description=task.description,
        combined_output=combined_output,
        expected_hint=task.expected_output_hint,
        domain=domain,
    )

    return judge_scores, execution_result


def build_scorecards(
    round_outputs: list[AgentOutput],
    task: TaskSpec,
    domain: str,
    judge_client: LLMClient,
) -> dict[str, str]:
    """Generate per-role IAMD scorecards after a round.

    Replicates TrialRunner._build_scorecard() and _build_llm_scorecard()
    from masid.orchestrator.

    Parameters
    ----------
    round_outputs:
        AgentOutputs from the just-completed round.
    task:
        The TaskSpec.
    domain:
        Domain key.
    judge_client:
        LLMClient for LLM-judge scoring (non-software_dev domains).

    Returns
    -------
    dict mapping role name → scorecard text
    """
    scorecards: dict[str, str] = {}

    if domain == "software_dev":
        exec_result = _run_sandbox(round_outputs)
        if exec_result is not None:
            for output in round_outputs:
                scorecards[output.role] = _build_sandbox_scorecard(output.role, exec_result, round_outputs)
    else:
        for output in round_outputs:
            scorecards[output.role] = _build_llm_scorecard(
                judge_client, output, task.description, round_outputs
            )

    return scorecards


# ---------------------------------------------------------------------------
# Internal helpers (replicate TrialRunner statics)
# ---------------------------------------------------------------------------

def _run_sandbox(outputs: list[AgentOutput]) -> ExecutionResult | None:
    """Run code sandbox on a round's Coder and Tester outputs."""
    coder_output = ""
    tester_output = ""
    for o in outputs:
        if o.role == "Coder":
            coder_output = o.content
        elif o.role == "Tester":
            tester_output = o.content

    if not coder_output:
        return None

    return evaluate_agent_code(coder_output, tester_output)


def _build_sandbox_scorecard(
    role: str,
    exec_result: ExecutionResult,
    outputs: list[AgentOutput],
) -> str:
    """Build a role-specific scorecard from code sandbox results.

    Mirrors TrialRunner._build_scorecard() exactly.
    """
    if role == "Architect":
        lines = ["ARCHITECT SCORECARD:"]
        if exec_result.syntax_valid and exec_result.code_runs:
            lines.append("  Design → Implementation: SUCCESS. Code runs.")
            lines.append(f"  Test pass rate: {exec_result.tests_passed}/{exec_result.tests_total}")
            if exec_result.tests_failed > 0:
                lines.append("  Action: Your design may need clearer interfaces or edge case handling.")
            else:
                lines.append("  Action: Design is solid. Consider edge cases for robustness.")
        elif exec_result.syntax_valid:
            lines.append("  Design → Implementation: PARTIAL. Code has syntax but crashes at runtime.")
            lines.append(f"  Error: {exec_result.code_error[:200]}")
            lines.append("  Action: Simplify your design or clarify module interfaces.")
        else:
            lines.append("  Design → Implementation: FAILED. Code has syntax errors.")
            lines.append("  Action: Your design may be too complex. Simplify.")

    elif role == "Coder":
        lines = ["CODER SCORECARD:"]
        if not exec_result.syntax_valid:
            lines.append(f"  Syntax: FAIL — {exec_result.syntax_error}")
            lines.append("  Action: Fix syntax errors. Ensure complete, valid Python.")
        elif not exec_result.code_runs:
            lines.append("  Syntax: PASS")
            lines.append(f"  Execution: FAIL — {exec_result.code_error[:200]}")
            lines.append("  Action: Fix the runtime error. Check imports and class definitions.")
        else:
            lines.append("  Syntax: PASS")
            lines.append("  Execution: PASS")
            lines.append(f"  Tests: {exec_result.tests_passed}/{exec_result.tests_total} passed")
            if exec_result.tests_total > 0 and exec_result.tests_failed > 0:
                lines.append("  Action: Fix failing tests. Check edge cases and error handling.")
            elif exec_result.tests_total == 0:
                lines.append("  Action: Code works. Ensure it is well-structured.")
            else:
                lines.append("  Action: All tests pass. Consider robustness improvements.")

    elif role == "Tester":
        lines = ["TESTER SCORECARD:"]
        if exec_result.tests_run:
            lines.append(f"  Tests executed: {exec_result.tests_total}")
            lines.append(f"  Passed: {exec_result.tests_passed}, Failed: {exec_result.tests_failed}")
            if exec_result.tests_total < 3:
                lines.append("  Action: Write more tests. Aim for at least 5 test functions.")
            elif exec_result.tests_passed == exec_result.tests_total:
                lines.append("  Action: All pass. Add edge case tests to improve coverage.")
            else:
                lines.append("  Action: Some tests fail. Verify tests match the code interface.")
        elif exec_result.code_runs:
            lines.append("  Tests: NONE EXECUTED")
            lines.append("  Action: Ensure you import from 'solution' and use 'def test_' naming.")
        else:
            lines.append("  Tests: COULD NOT RUN (code has errors)")
            lines.append("  Action: Write tests that will be ready when the code is fixed.")

    elif role == "Reviewer":
        lines = ["REVIEWER SCORECARD:"]
        lines.append(f"  Code runs: {exec_result.code_runs}")
        lines.append(f"  Tests: {exec_result.tests_passed}/{exec_result.tests_total} passed")
        lines.append(f"  Execution score: {exec_result.execution_score:.0%}")
        if exec_result.execution_score < 0.5:
            lines.append("  Action: Focus review on fundamental correctness issues.")
        else:
            lines.append("  Action: Focus review on quality, edge cases, maintainability.")

    else:
        lines = [f"{role} SCORECARD:", "  No specific metrics for this role."]

    return "\n".join(lines)


def _build_llm_scorecard(
    judge_client: LLMClient,
    agent_output: AgentOutput,
    task_description: str,
    all_outputs: list[AgentOutput],
) -> str:
    """Build a per-agent scorecard using LLM-as-judge.

    Mirrors TrialRunner._build_llm_scorecard() exactly.
    """
    others_context = ""
    for o in all_outputs:
        if o.agent_id != agent_output.agent_id:
            snippet = o.content[:500]
            others_context += f"  {o.role}: {snippet}...\n"

    prompt = (
        f"You are evaluating a {agent_output.role}'s output in a "
        f"multi-agent collaboration.\n\n"
        f"Task: {task_description[:500]}\n\n"
        f"This agent's output:\n{agent_output.content[:1000]}\n\n"
        f"Other agents' outputs (summaries):\n{others_context}\n\n"
        f"Provide a brief scorecard for the {agent_output.role}. "
        f"Score 1-10 on: quality, completeness, usefulness to other agents. "
        f"Then give ONE specific, actionable improvement suggestion.\n\n"
        f"Format:\n"
        f"SCORECARD:\n"
        f"  Quality: X/10\n"
        f"  Completeness: X/10\n"
        f"  Usefulness to team: X/10\n"
        f"  Action: [one specific suggestion]"
    )

    try:
        response = judge_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        return response.content
    except Exception as e:
        logger.warning("Failed to generate LLM scorecard for %s: %s", agent_output.role, e)
        return (
            f"{agent_output.role} SCORECARD:\n"
            f"  Quality: 5/10\n"
            f"  Completeness: 5/10\n"
            f"  Usefulness to team: 5/10\n"
            f"  Action: Review your output for completeness and clarity."
        )
