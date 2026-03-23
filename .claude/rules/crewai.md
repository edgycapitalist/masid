# CrewAI Experiment Plan

## Goal
Validate MASID's core finding (architecture effects are task-dependent) 
on CrewAI, a production multi-agent framework, using local inference on DGX Spark.

## Architecture Mapping

### IRM on CrewAI
- Process: Sequential
- Agent goals: Self-oriented ("produce highest quality output for YOUR role")
- memory=False, allow_delegation=False
- Context: upstream task output only (CrewAI default in sequential)
- No task_callback feedback

### JRO on CrewAI  
- Process: Sequential with memory=True (shared crew memory)
- Agent goals: Team-oriented ("maximize overall project quality")  
- allow_delegation=True
- All agents see shared memory state
- No explicit feedback, but memory provides implicit sharing

### IAMD on CrewAI
- Process: Sequential
- Agent goals: Self-oriented (same as IRM) + "you will receive performance feedback"
- memory=False, allow_delegation=False
- task_callback generates scorecard after each task
- Scorecard injected into next task's context

## Scale
- 3 architectures × 3 domains × 1 task each × 10 trials = 90 trials
- Tasks: sw_001 (software dev), rs_001 (research synthesis), pp_001 (project planning)

## Files to create
- src/masid_crewai/__init__.py
- src/masid_crewai/architectures.py — IRM, JRO, IAMD crew builders
- src/masid_crewai/tasks.py — task specifications adapted for CrewAI format
- src/masid_crewai/evaluation.py — scoring using same judge as primary experiment
- src/masid_crewai/runner.py — experiment runner with trial logging
- src/masid_crewai/cli.py — CLI for running experiments
- tests/test_crewai_architectures.py
- configs/crewai_experiment.yaml