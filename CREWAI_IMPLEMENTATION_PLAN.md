# CrewAI Adapter for MASID — Implementation Plan

## Context

MASID benchmarks three game-theoretically grounded incentive architectures (IRM, JRO, IAMD) for LLM multi-agent systems. The existing experiment uses a custom agent framework with Groq-hosted Llama 3.3 70B. This adapter replicates the experiment on CrewAI, a production multi-agent framework, using a **local vLLM endpoint** (`http://localhost:8000/v1` serving Llama 3.3 70B) to validate that MASID's core finding — architecture effects are task-dependent — holds on a different framework.

**Scale:** 3 architectures × 3 domains × 1 task × 10 trials = 90 trials.

---

## Key Design Decisions

### 1. Multi-round execution via per-round crew rebuild
CrewAI has no native multi-round support. Run `crew.kickoff()` 3 times (matching MASID's 3-round default), rebuilding task descriptions each round with outputs from the previous round. This mirrors `TrialRunner.run()`'s round loop.

### 2. IAMD scorecard injection via task descriptions
Since CrewAI tasks are constructed before execution, scorecards are generated *between* rounds and prepended to task descriptions in the next round's crew build. No task_callback needed — the round-rebuild pattern handles this cleanly.

### 3. JRO full transparency via explicit context injection (NOT memory=True)
CrewAI's `memory=True` uses embedding-based RAG, which would require an embedding model and introduces non-determinism. Instead, JRO injects ALL agent outputs into each task's description, matching the original JRO's explicit full-transparency mechanism. `allow_delegation=False` for all architectures to keep the controlled variable clean.

### 4. Reuse MASID evaluation, storage, roles, and tasks
Import directly from `masid.evaluation`, `masid.storage`, `masid.agents.roles`, `masid.domains`. Separate DB file (`data/crewai_experiments.db`) with identical schema.

### 5. Info separation for research_synthesis
Use existing `TaskSpec.get_description_for_role(role)` — Researcher_A/B get only their assigned sources.

---

## File Plan (implementation order)

### Step 1: Package scaffolding
**`pyproject.toml`** — Add crewai optional dependency group and entry point:
```toml
crewai = ["crewai>=0.80", "crewai-tools>=0.14"]
all = ["masid[dev,analysis,crewai]"]
# scripts: masid-crewai = "masid_crewai.cli:main"
```

**`src/masid_crewai/__init__.py`** — Package init, export `CrewAITrialRunner`.

### Step 2: Configuration
**`src/masid_crewai/config.py`** (~60 lines)
- `CrewAIModelConfig`: model, base_url, api_key, temperature, max_tokens
- `CrewAIConfig`: agent_model, judge (reuse `masid.config.EvaluationConfig`), experiment (reuse `masid.config.ExperimentConfig`), storage, architectures, domains, task_ids mapping
- `load_crewai_config(path)` loader

**`configs/crewai_experiment.yaml`**
```yaml
agent_model:
  model: "openai/meta-llama/Llama-3.3-70B-Instruct"
  base_url: "http://localhost:8000/v1"
  api_key: "dummy"
  temperature: 0.7
  max_tokens: 2048
experiment:
  trials_per_cell: 10
  max_rounds: 3
architectures: [irm, jro, iamd]
domains: [software_dev, research_synthesis, project_planning]
task_ids: {software_dev: sw_001, research_synthesis: rs_001, project_planning: pp_001}
judge:
  judge_provider: openai
  judge_model: gpt-4.1-mini
storage:
  db_path: data/crewai_experiments.db
```

### Step 3: Task adaptation
**`src/masid_crewai/tasks.py`** (~80 lines)
- `get_task_for_experiment(domain, task_id)` — retrieve specific TaskSpec from MASID registry
- `format_task_description(task, role, round_num, upstream_outputs)` — build description with upstream context
- `format_expected_output(role)` — derive expected_output string from RoleSpec

Imports: `masid.domains.registry.get_tasks`, `masid.agents.roles.get_roles`

### Step 4: Architecture builders (core)
**`src/masid_crewai/architectures.py`** (~250 lines)

Three builder functions:
- `build_irm_crew(roles, task, round_num, prev_outputs, llm) -> Crew`
  - Agents: `memory=False`, `allow_delegation=False`, self-oriented goals
  - Tasks: upstream-only context
- `build_jro_crew(roles, task, round_num, prev_outputs, llm) -> Crew`
  - Agents: `memory=False`, `allow_delegation=False`, team-oriented goals
  - Tasks: ALL agent outputs injected as context (full transparency)
  - Crew: `memory=False` (explicit injection instead of RAG)
- `build_iamd_crew(roles, task, round_num, prev_outputs, scorecards, llm) -> Crew`
  - Agents: `memory=False`, `allow_delegation=False`, self-oriented goals + scorecard awareness
  - Tasks: upstream context + scorecard from previous round prepended

System prompts replicate the originals from `masid.architectures.{irm,jro,iamd}`.

Helper: `build_crewai_llm(config) -> crewai.LLM` constructs the LLM object.

### Step 5: Evaluation bridge
**`src/masid_crewai/evaluation.py`** (~120 lines)
- `extract_agent_outputs(crew_output, round_number) -> list[AgentOutput]` — map CrewAI `TaskOutput` to MASID `AgentOutput`
- `evaluate_crew_output(crew_output, task_spec, domain, judge_client) -> tuple[dict, ExecutionResult|None]` — calls `masid.evaluation.judge.judge_trial_output()` and `masid.evaluation.sandbox.evaluate_agent_code()` for software_dev
- `build_scorecards(agent_outputs, task_spec, domain, judge_client) -> dict[str, str]` — generate IAMD scorecards (replicates logic from `masid.orchestrator.TrialRunner`)

### Step 6: Orchestrator
**`src/masid_crewai/runner.py`** (~200 lines)

`CrewAITrialRunner`:
```
run(architecture, domain, task_id, seed) -> TrialMetrics:
  1. Build crewai.LLM from config
  2. Get roles via masid.agents.roles.get_roles(domain)
  3. Get task via tasks.get_task_for_experiment(domain, task_id)
  4. Round loop (3 rounds):
     a. Build crew via build_{arch}_crew()
     b. crew.kickoff()
     c. Extract agent outputs
     d. IAMD only: generate scorecards for next round
  5. Evaluate final outputs (judge + sandbox for sw)
  6. Compute metrics (tokens, latency, duplication)
  7. Save to ExperimentDB with metadata={"framework": "crewai"}
  8. Return TrialMetrics
```

### Step 7: CLI
**`src/masid_crewai/cli.py`** (~100 lines)
- `masid-crewai smoke-test` — test vLLM endpoint
- `masid-crewai run --architecture irm --domain software_dev` — single trial
- `masid-crewai batch --config configs/crewai_experiment.yaml` — full batch with resume
- `masid-crewai results` — display results

### Step 8: Tests
**`tests/test_crewai_architectures.py`** (~150 lines)
- Test IRM crew has correct agent properties (no memory, no delegation, self-oriented goals)
- Test JRO crew injects all outputs into task descriptions
- Test IAMD crew injects scorecards into next-round tasks
- Test research_synthesis info separation (Researcher_A sees only Source A)
- Test `extract_agent_outputs` mapping
- `@pytest.mark.integration` test for end-to-end single trial

---

## Critical files to reference during implementation

| File | Why |
|------|-----|
| `src/masid/orchestrator/__init__.py` | Round loop, scorecard generation, evaluation flow, DB persistence |
| `src/masid/architectures/iamd.py` | IAMD system prompts, upstream context building, scorecard injection |
| `src/masid/architectures/irm.py` | IRM system prompts, upstream-only info flow |
| `src/masid/architectures/jro.py` | JRO system prompts, full transparency context building |
| `src/masid/evaluation/judge.py` | `judge_trial_output()` — reused directly |
| `src/masid/evaluation/sandbox.py` | `evaluate_agent_code()` — reused directly |
| `src/masid/agents/roles.py` | `RoleSpec`, `get_roles()` — reused directly |
| `src/masid/domains/registry.py` | `get_tasks()` — reused directly |
| `src/masid/storage/__init__.py` | `ExperimentDB` — reused directly |
| `src/masid/config.py` | Config models to reuse |

---

## Context Management

**On execution start:**
1. Delete `CREWAI_EXPERIMENT_PLAN.md` (superseded by this plan)
2. Save this plan as `CREWAI_IMPLEMENTATION_PLAN.md` in repo root
3. Create `TODO.md` (checkbox list only, ~40 lines) — updated in real-time as tasks complete

**Files loaded per conversation:** `CLAUDE.md` (auto) + `TODO.md` (for progress) = ~120 lines of context overhead. The implementation plan is referenced only when needed, not auto-loaded.

---

## TODO Breakdown

### Step 1: Package scaffolding
- [ ] Add `crewai` optional dep group to `pyproject.toml`
- [ ] Add `masid-crewai` script entry point to `pyproject.toml`
- [ ] Add `masid_crewai` to setuptools package discovery
- [ ] Create `src/masid_crewai/__init__.py`
- [ ] `pip install -e ".[crewai]"` and verify import

### Step 2: Configuration
- [ ] Create `src/masid_crewai/config.py` with `CrewAIModelConfig` and `CrewAIConfig`
- [ ] Create `configs/crewai_experiment.yaml`
- [ ] Test config loading

### Step 3: Task adaptation
- [ ] Create `src/masid_crewai/tasks.py` with `get_task_for_experiment()`
- [ ] Implement `format_task_description()` with upstream context injection
- [ ] Implement `format_expected_output()`
- [ ] Verify info separation works for research_synthesis roles

### Step 4: Architecture builders
- [ ] Implement `build_crewai_llm()` helper
- [ ] Implement `build_irm_crew()` — self-oriented, upstream-only, no feedback
- [ ] Implement `build_jro_crew()` — team-oriented, full context injection
- [ ] Implement `build_iamd_crew()` — self-oriented + scorecard injection
- [ ] Verify system prompts match originals in `masid.architectures.{irm,jro,iamd}`

### Step 5: Evaluation bridge
- [ ] Implement `extract_agent_outputs()` — CrewAI TaskOutput → MASID AgentOutput
- [ ] Implement `evaluate_crew_output()` — judge + sandbox bridge
- [ ] Implement `build_scorecards()` — IAMD feedback generation

### Step 6: Orchestrator
- [ ] Implement `CrewAITrialRunner.__init__()` with config and DB
- [ ] Implement `CrewAITrialRunner.run()` round loop
- [ ] Wire IAMD scorecard generation between rounds
- [ ] Wire evaluation and DB persistence
- [ ] Test single trial end-to-end (smoke test level)

### Step 7: CLI
- [ ] Implement `smoke-test` command
- [ ] Implement `run` command (single trial)
- [ ] Implement `batch` command with resume support
- [ ] Implement `results` command

### Step 8: Tests
- [ ] Unit tests for architecture builder properties
- [ ] Unit tests for task description formatting and info separation
- [ ] Unit tests for output extraction
- [ ] Integration test (marked, requires live endpoint)

---

## Verification

1. **Smoke test:** `masid-crewai smoke-test` confirms vLLM endpoint responds
2. **Single trial:** `masid-crewai run --architecture irm --domain software_dev` produces a TrialMetrics with non-zero scores
3. **All 3 architectures:** Run one trial each for IRM/JRO/IAMD on sw_001 and verify different behavior (JRO tasks contain all-agent context, IAMD tasks contain scorecards)
4. **Unit tests:** `pytest tests/test_crewai_architectures.py -v -m "not integration"` passes
5. **Full batch:** `masid-crewai batch --config configs/crewai_experiment.yaml` runs 90 trials with resume support
6. **Data compatibility:** Results queryable alongside original experiment in analysis notebooks
