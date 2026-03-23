"""CrewAI experiment orchestrator.

Mirrors TrialRunner from masid.orchestrator but uses CrewAI for agent execution.
Key difference: per-round crew rebuild instead of persistent agent history.
"""

from __future__ import annotations

import logging
import time
import uuid

from masid.agents import AgentOutput
from masid.agents.roles import get_roles
from masid.domains import TaskSpec
from masid.evaluation import TrialMetrics, compute_duplication_rate
from masid.models import LLMClient
from masid.storage import ExperimentDB
from masid_crewai.architectures import (
    build_crewai_llm,
    build_iamd_crew,
    build_irm_crew,
    build_jro_crew,
)
from masid_crewai.config import CrewAIConfig
from masid_crewai.evaluation import (
    build_scorecards,
    evaluate_crew_output,
    extract_agent_outputs,
)
from masid_crewai.tasks import get_task_for_experiment

logger = logging.getLogger(__name__)


class CrewAITrialRunner:
    """Runs a single CrewAI experiment trial.

    Parameters
    ----------
    config:
        CrewAI adapter configuration.
    db:
        Database for recording results. If None, results are returned
        but not persisted.
    """

    def __init__(
        self,
        config: CrewAIConfig,
        db: ExperimentDB | None = None,
    ) -> None:
        self.config = config
        self.db = db

    def run(
        self,
        architecture_key: str,
        domain: str,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> TrialMetrics:
        """Execute a single trial and return metrics.

        Parameters
        ----------
        architecture_key:
            One of "irm", "jro", "iamd".
        domain:
            Task domain key (e.g. "software_dev").
        task_id:
            Specific task to run. Defaults to the config's task_ids mapping.
        seed:
            Random seed recorded with the trial for reproducibility tracking.
        """
        trial_id = str(uuid.uuid4())[:12]
        effective_task_id = task_id or self.config.task_ids.get(domain)
        if not effective_task_id:
            raise ValueError(f"No task_id configured for domain {domain!r}")

        model_name = self.config.agent_model.model
        logger.info(
            "Trial %s: arch=%s domain=%s model=%s task=%s",
            trial_id, architecture_key, domain, model_name, effective_task_id,
        )

        # 1. Build CrewAI LLM and judge LLMClient
        llm = build_crewai_llm(self.config.agent_model)
        judge_client = LLMClient(
            provider=self.config.judge.judge_provider,
            model_name=self.config.judge.judge_model,
            base_url=self.config.judge.judge_base_url,
            temperature=0.1,
            max_tokens=512,
            timeout=120,
        )

        # 2. Get roles and task
        roles = get_roles(domain)
        role_names = [r.role for r in roles]
        task = get_task_for_experiment(domain, effective_task_id)

        # 3. Run rounds
        all_round_outputs: list[list[AgentOutput]] = []
        prev_outputs: dict[str, str] = {}   # role → content from last round
        scorecards: dict[str, str] = {}     # role → scorecard (IAMD only)
        max_rounds = self.config.experiment.max_rounds
        trial_start = time.monotonic()

        for round_num in range(max_rounds):
            logger.info("  Round %d/%d", round_num + 1, max_rounds)

            # Build and execute crew for this round
            crew = self._build_crew(
                architecture_key, roles, task, round_num, prev_outputs, scorecards, llm
            )

            crew_output = crew.kickoff()

            # Extract outputs into MASID format
            round_outputs = extract_agent_outputs(crew_output, role_names, round_num)
            all_round_outputs.append(round_outputs)

            # Update prev_outputs for next round
            prev_outputs = {o.role: o.content for o in round_outputs}

            # IAMD: generate scorecards between rounds (the mechanism)
            if architecture_key == "iamd" and round_num < max_rounds - 1:
                logger.info("  Generating IAMD scorecards for round %d", round_num + 1)
                scorecards = build_scorecards(round_outputs, task, domain, judge_client)

        trial_elapsed = time.monotonic() - trial_start

        # 4. Evaluate final outputs
        final_outputs = all_round_outputs[-1]
        judge_scores, execution_result = evaluate_crew_output(
            final_outputs, task, domain, judge_client
        )

        # 5. Compute blended quality score
        quality = judge_scores.get("overall", 0.5)
        if execution_result:
            quality = 0.6 * execution_result.execution_score + 0.4 * quality
            quality = round(quality, 4)

        # 6. Compute coordination metrics
        duplication = compute_duplication_rate(final_outputs)

        # 7. Per-agent scores (all receive collective score, matching MASID convention)
        agent_scores = {o.agent_id: quality for o in final_outputs}

        # 8. Build metadata
        exec_meta: dict = {}
        if execution_result:
            exec_meta = {
                "syntax_valid": execution_result.syntax_valid,
                "code_runs": execution_result.code_runs,
                "tests_passed": execution_result.tests_passed,
                "tests_failed": execution_result.tests_failed,
                "tests_total": execution_result.tests_total,
                "execution_score": execution_result.execution_score,
            }

        metrics = TrialMetrics(
            trial_id=trial_id,
            architecture=architecture_key,
            domain=domain,
            model=model_name,
            quality_score=quality,
            total_tokens=0,             # CrewAI does not expose per-task token counts
            total_latency_seconds=round(trial_elapsed, 3),
            num_rounds=max_rounds,
            duplication_rate=duplication,
            conflict_rate=0.0,
            consistency_score=0.0,
            agent_scores=agent_scores,
            metadata={
                "task_id": effective_task_id,
                "judge_scores": judge_scores,
                "framework": "crewai",
                "token_counts_available": False,
                **exec_meta,
            },
        )

        # 9. Persist
        if self.db:
            db_scores = dict(judge_scores)
            db_scores["overall"] = quality
            self.db.save_trial(
                trial_id=trial_id,
                architecture=architecture_key,
                domain=domain,
                model=model_name,
                task_id=effective_task_id,
                scores=db_scores,
                efficiency={
                    "total_tokens": 0,
                    "total_latency": round(trial_elapsed, 3),
                    "num_rounds": max_rounds,
                },
                coordination={
                    "duplication_rate": duplication,
                    "conflict_rate": 0.0,
                    "consistency_score": 0.0,
                },
                agent_scores=agent_scores,
                judge_rationale=judge_scores.get("rationale", ""),
                fault_type=None,
                fault_agent=None,
                seed=seed,
                metadata={"framework": "crewai", **exec_meta} if exec_meta else {"framework": "crewai"},
            )
            for round_outputs in all_round_outputs:
                for output in round_outputs:
                    self.db.save_agent_output(
                        trial_id=trial_id,
                        agent_id=output.agent_id,
                        role=output.role,
                        round_number=output.round_number,
                        content=output.content,
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_seconds=0.0,
                    )

        logger.info(
            "  Trial %s complete — quality=%.2f latency=%.1fs",
            trial_id, metrics.quality_score, metrics.total_latency_seconds,
        )
        return metrics

    def _build_crew(
        self,
        architecture_key: str,
        roles: list,
        task: TaskSpec,
        round_num: int,
        prev_outputs: dict[str, str],
        scorecards: dict[str, str],
        llm: object,
    ) -> object:
        """Dispatch to the right crew builder based on architecture."""
        if architecture_key == "irm":
            return build_irm_crew(roles, task, round_num, prev_outputs, llm)
        elif architecture_key == "jro":
            return build_jro_crew(roles, task, round_num, prev_outputs, llm)
        elif architecture_key == "iamd":
            return build_iamd_crew(roles, task, round_num, prev_outputs, scorecards, llm)
        else:
            raise ValueError(
                f"Unknown architecture {architecture_key!r}. "
                f"Available: irm, jro, iamd"
            )
