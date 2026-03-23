# CrewAI Adapter — TODO

## Step 1: Package scaffolding
- [x] Add `crewai` optional dep group to `pyproject.toml`
- [x] Add `masid-crewai` script entry point to `pyproject.toml`
- [x] Add `masid_crewai` to setuptools package discovery
- [x] Create `src/masid_crewai/__init__.py`
- [x] `pip install -e ".[crewai]"` and verify import

## Step 2: Configuration
- [x] Create `src/masid_crewai/config.py`
- [x] Create `configs/crewai_experiment.yaml`
- [x] Test config loading

## Step 3: Task adaptation
- [x] Create `src/masid_crewai/tasks.py` with `get_task_for_experiment()`
- [x] Implement `format_task_description()` with upstream context injection
- [x] Implement `format_expected_output()`
- [x] Verify info separation for research_synthesis roles

## Step 4: Architecture builders
- [x] Implement `build_crewai_llm()` helper
- [x] Implement `build_irm_crew()`
- [x] Implement `build_jro_crew()`
- [x] Implement `build_iamd_crew()`
- [x] Verify system prompts match originals

## Step 5: Evaluation bridge
- [x] Implement `extract_agent_outputs()`
- [x] Implement `evaluate_crew_output()`
- [x] Implement `build_scorecards()`

## Step 6: Orchestrator
- [x] Implement `CrewAITrialRunner.__init__()`
- [x] Implement `CrewAITrialRunner.run()` round loop
- [x] Wire IAMD scorecard generation between rounds
- [x] Wire evaluation and DB persistence
- [ ] Test single trial end-to-end (requires live vLLM)

## Step 7: CLI
- [x] Implement `smoke-test` command
- [x] Implement `run` command
- [x] Implement `batch` command with resume
- [x] Implement `results` command

## Step 8: Tests
- [x] Unit tests for architecture builder properties
- [x] Unit tests for task description formatting
- [x] Unit tests for output extraction
- [x] Integration test (requires live endpoint) — marked @pytest.mark.integration

---

## Run Experiment (when vLLM is up)

```bash
# 1. Verify endpoint is responding
masid-crewai smoke-test

# 2. Run full 90-trial batch (resumable)
masid-crewai batch --config configs/crewai_experiment.yaml

# 3. View results
masid-crewai results --config configs/crewai_experiment.yaml

# Optional: run a single trial to sanity-check before full batch
masid-crewai run --architecture irm --domain project_planning

# Optional: run integration test suite against live endpoint
pytest tests/test_crewai_architectures.py -v -m integration
```
