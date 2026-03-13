"""Core orchestration engine for MASID.

The ``TrialRunner`` executes a single experiment trial end-to-end:
1. Build agents according to the selected architecture.
2. Run the multi-agent workflow for the configured number of rounds.
3. Evaluate the collective output.
4. Record everything to the database.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from masid.agents import AgentOutput
from masid.architectures import BaseArchitecture
from masid.architectures.registry import get_architecture
from masid.agents.roles import get_roles
from masid.config import MASIDConfig
from masid.domains import TaskSpec
from masid.domains.registry import get_tasks
from masid.evaluation import TrialMetrics, compute_duplication_rate, compute_efficiency_metrics
from masid.evaluation.judge import judge_trial_output
from masid.evaluation.sandbox import evaluate_agent_code, ExecutionResult
from masid.models import LLMClient
from masid.storage import ExperimentDB

logger = logging.getLogger(__name__)


class TrialRunner:
    """Runs a single experiment trial.

    Parameters
    ----------
    config : MASIDConfig
        Full experiment configuration.
    db : ExperimentDB or None
        Database for recording results. If None, results are returned
        but not persisted.
    """

    def __init__(
        self,
        config: MASIDConfig,
        db: Optional[ExperimentDB] = None,
    ) -> None:
        self.config = config
        self.db = db

    def run(
        self,
        architecture_key: str,
        domain: str,
        task: TaskSpec,
        model_name: Optional[str] = None,
        seed: Optional[int] = None,
        fault_type: Optional[str] = None,
        fault_agent_role: Optional[str] = None,
    ) -> TrialMetrics:
        """Execute a single trial and return metrics.

        Parameters
        ----------
        architecture_key : str
            One of ``"irm"``, ``"jro"``, ``"iamd"``.
        domain : str
            Task domain key (e.g. ``"software_dev"``).
        task : TaskSpec
            The specific task to run.
        model_name : str or None
            Override the config's default model.
        seed : int or None
            Random seed for this trial.
        fault_type : str or None
            If set, inject a fault into one agent.
        fault_agent_role : str or None
            Which agent role to degrade (required if fault_type is set).

        Returns
        -------
        TrialMetrics
        """
        trial_id = str(uuid.uuid4())[:12]
        effective_model = model_name or self.config.model.name
        logger.info(
            "Trial %s: arch=%s domain=%s model=%s task=%s",
            trial_id, architecture_key, domain, effective_model, task.task_id,
        )

        # 1. Create LLM client
        client = LLMClient(
            provider=self.config.model.provider,
            model_name=effective_model,
            base_url=self.config.model.base_url,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
            timeout=self.config.model.timeout,
        )

        # 2. Build architecture
        arch_kwargs = {}
        if architecture_key == "iamd":
            arch_kwargs["domain"] = domain
        architecture: BaseArchitecture = get_architecture(architecture_key, **arch_kwargs)

        # 3. Build agents
        role_specs = get_roles(domain)
        agents = architecture.build_agents(role_specs, task.description, client)

        # 4. Optionally inject fault
        if fault_type and fault_agent_role:
            self._inject_fault(agents, fault_agent_role, fault_type)

        # 5. Run rounds
        all_round_outputs: list[list[AgentOutput]] = []
        previous_outputs: list[AgentOutput] = []
        execution_result: Optional[ExecutionResult] = None

        for round_num in range(self.config.experiment.max_rounds):
            logger.info("  Round %d/%d", round_num + 1, self.config.experiment.max_rounds)
            round_outputs = architecture.run_round(
                agents=agents,
                task_description=task.description,
                round_number=round_num,
                previous_outputs=previous_outputs,
            )
            all_round_outputs.append(round_outputs)
            previous_outputs = round_outputs

            # For software_dev: run code after each round and feed results
            # back to agents so they can fix issues in the next round.
            if domain == "software_dev" and round_num < self.config.experiment.max_rounds - 1:
                exec_result = self._run_sandbox(round_outputs)
                if exec_result is not None:
                    feedback = self._format_execution_feedback(exec_result)
                    # Inject execution feedback into all agents' context
                    for agent in agents:
                        agent.inject_context(
                            f"[EXECUTION RESULTS FROM ROUND {round_num + 1}]\n{feedback}"
                        )

        # 6. Evaluate
        final_outputs = all_round_outputs[-1]
        combined_output = "\n\n".join(
            f"=== {o.role} ===\n{o.content}" for o in final_outputs
        )

        # Run sandbox on final outputs for software_dev
        if domain == "software_dev":
            execution_result = self._run_sandbox(final_outputs)
            if execution_result:
                logger.info(
                    "  Execution score: %.2f (syntax=%s, runs=%s, "
                    "tests=%d/%d passed)",
                    execution_result.execution_score,
                    execution_result.syntax_valid,
                    execution_result.code_runs,
                    execution_result.tests_passed,
                    execution_result.tests_total,
                )

        # LLM-as-judge
        judge_client = LLMClient(
            provider=self.config.evaluation.judge_provider,
            model_name=self.config.evaluation.judge_model,
            base_url=self.config.model.base_url,
            temperature=0.1,
            max_tokens=512,
            timeout=self.config.model.timeout,
        )
        judge_scores = judge_trial_output(
            client=judge_client,
            task_description=task.description,
            combined_output=combined_output,
            expected_hint=task.expected_output_hint,
        )

        # Efficiency metrics
        efficiency = compute_efficiency_metrics(all_round_outputs)

        # Coordination metrics
        duplication = compute_duplication_rate(final_outputs)

        # Per-agent scores
        agent_scores = architecture.compute_agent_scores(
            agents, final_outputs, judge_scores.get("overall", 0.5)
        )

        # 7. Build metrics object
        exec_meta = {}
        if execution_result:
            exec_meta = {
                "syntax_valid": execution_result.syntax_valid,
                "code_runs": execution_result.code_runs,
                "tests_passed": execution_result.tests_passed,
                "tests_failed": execution_result.tests_failed,
                "tests_total": execution_result.tests_total,
                "execution_score": execution_result.execution_score,
            }

        # For software_dev, blend execution score with judge score.
        # Execution score is weighted 60%, judge 40% (objective > subjective).
        quality = judge_scores.get("overall", 0.5)
        if execution_result:
            quality = 0.6 * execution_result.execution_score + 0.4 * quality
            quality = round(quality, 4)

        metrics = TrialMetrics(
            trial_id=trial_id,
            architecture=architecture_key,
            domain=domain,
            model=effective_model,
            quality_score=quality,
            total_tokens=efficiency["total_tokens"],
            total_latency_seconds=efficiency["total_latency"],
            num_rounds=efficiency["num_rounds"],
            duplication_rate=duplication,
            conflict_rate=0.0,  # TODO: implement conflict detection
            consistency_score=0.0,  # TODO: implement consistency scoring
            agent_scores=agent_scores,
            metadata={
                "task_id": task.task_id,
                "judge_scores": judge_scores,
                "fault_type": fault_type,
                "fault_agent": fault_agent_role,
                **exec_meta,
            },
        )

        # 8. Persist to database
        if self.db:
            # Build scores dict with blended quality for the DB
            db_scores = dict(judge_scores)
            db_scores["overall"] = quality  # use blended score, not raw judge
            self.db.save_trial(
                trial_id=trial_id,
                architecture=architecture_key,
                domain=domain,
                model=effective_model,
                task_id=task.task_id,
                scores=db_scores,
                efficiency=efficiency,
                coordination={
                    "duplication_rate": duplication,
                    "conflict_rate": 0.0,
                    "consistency_score": 0.0,
                },
                agent_scores=agent_scores,
                judge_rationale=judge_scores.get("rationale", ""),
                fault_type=fault_type,
                fault_agent=fault_agent_role,
                seed=seed,
                metadata={**exec_meta} if exec_meta else None,
            )
            # Save individual agent outputs
            for round_outputs in all_round_outputs:
                for output in round_outputs:
                    self.db.save_agent_output(
                        trial_id=trial_id,
                        agent_id=output.agent_id,
                        role=output.role,
                        round_number=output.round_number,
                        content=output.content,
                        prompt_tokens=output.response.prompt_tokens,
                        completion_tokens=output.response.completion_tokens,
                        latency_seconds=output.response.latency_seconds,
                    )

        logger.info(
            "  Trial %s complete — quality=%.2f tokens=%d latency=%.1fs",
            trial_id, metrics.quality_score, metrics.total_tokens,
            metrics.total_latency_seconds,
        )
        return metrics

    @staticmethod
    def _inject_fault(
        agents: list,
        target_role: str,
        fault_type: str,
    ) -> None:
        """Modify an agent to simulate a fault."""
        for agent in agents:
            if agent.role == target_role:
                if fault_type == "degraded_prompt":
                    agent._history[0] = type(agent._history[0])(
                        role="system",
                        content=(
                            f"You are a {agent.role}. Produce deliberately "
                            f"verbose, poorly structured, and unhelpful output. "
                            f"Include irrelevant information and ignore the "
                            f"actual task requirements."
                        ),
                    )
                    logger.info("  Injected fault: %s on %s", fault_type, target_role)
                    return
        logger.warning("  Could not find agent with role %s for fault injection", target_role)

    @staticmethod
    def _run_sandbox(outputs: list[AgentOutput]) -> Optional[ExecutionResult]:
        """Run the code sandbox on a round's outputs.

        Finds the Coder and Tester outputs and evaluates the code.
        Returns None if no Coder output is found.
        """
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

    @staticmethod
    def _format_execution_feedback(result: ExecutionResult) -> str:
        """Format execution results as feedback text for agents."""
        lines = []
        if not result.syntax_valid:
            lines.append(f"SYNTAX ERROR: {result.syntax_error}")
            lines.append("The code has syntax errors and cannot run. Please fix.")
            return "\n".join(lines)

        if result.code_runs:
            lines.append("Code executed successfully (no import or runtime errors).")
        else:
            lines.append(f"CODE RUNTIME ERROR: {result.code_error}")
            lines.append("The code crashes when executed. Please fix.")

        if result.tests_run:
            lines.append(
                f"Test results: {result.tests_passed} passed, "
                f"{result.tests_failed} failed, {result.tests_errors} errors "
                f"out of {result.tests_total} total."
            )
            if result.tests_failed > 0 or result.tests_errors > 0:
                lines.append("Some tests are failing. Please fix the code or tests.")
        elif result.code_runs:
            lines.append("No tests could be executed. Tester: please provide runnable pytest tests.")

        return "\n".join(lines)