# CrewAI Experiment Speed Options

## Timing baseline

One trial on local Ollama (llama3.3:70b) = **~74 minutes** (12 sequential calls).

---

## Option A: Groq cloud inference (Recommended for speed)

**Config file:** `configs/crewai_groq.yaml`

**Timing:** ~85s/trial → **~2 hours** for 90 trials (10/cell), **~40 min** for 27 trials (3/cell)

**Prerequisites:**
1. Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=gsk_...
   ```
   Get a key at https://console.groq.com

2. Results are stored separately in `data/crewai_groq_experiments.db`

**To run (fresh start, 10 trials/cell = 90 trials total):**
```bash
tmux new -s masid-crewai
masid-crewai batch --config configs/crewai_groq.yaml
# Ctrl+B, D  →  detach (safe to close terminal)
# tmux attach -t masid-crewai  →  reconnect later
```

**To resume if interrupted:**
```bash
tmux attach -t masid-crewai
# or start a new session — same command, skips completed cells:
masid-crewai batch --config configs/crewai_groq.yaml
```

**Rate limits (Groq free tier):**
- 30 req/min, 6k tokens/min for llama-3.3-70b-versatile
- Paid tier is higher — if you hit limits, LiteLLM will retry automatically

---

## Option B: Reduce trials_per_cell to 3 (local Ollama)

**Config file:** `configs/crewai_experiment.yaml` (change `trials_per_cell: 3`)

**Timing:** ~33 hours total (26 remaining after the 1 already done)

Change in `configs/crewai_experiment.yaml`:
```yaml
experiment:
  trials_per_cell: 3   # was 10
  max_rounds: 3
```

The 1 existing trial (`irm × project_planning × pp_001`) counts toward the new target —
so only 2 more run for that cell.

**To run:**
```bash
tmux new -s masid-crewai
masid-crewai batch --config configs/crewai_experiment.yaml
```

---

## Option C: Switch to smaller local model (llama3.1:8b)

**Timing:** ~5 min/trial → **~7.5 hours** for 90 trials, **~2 hours** for 27 trials (3/cell)

```bash
ollama pull llama3.1:8b
```

Then update `configs/crewai_experiment.yaml`:
```yaml
agent_model:
  model: "ollama/llama3.1:8b"
  base_url: "http://localhost:11434"
  api_key: "dummy"
storage:
  db_path: data/crewai_experiments_8b.db   # separate DB to keep results clean
```

---

## Summary table

| Option | Model | Trials | Est. time |
|---|---|---|---|
| Groq (10/cell) | llama-3.3-70b-versatile | 90 | ~2 hours |
| Groq (3/cell) | llama-3.3-70b-versatile | 27 | ~40 min |
| Local 8b (10/cell) | llama3.1:8b | 90 | ~7.5 hours |
| Local 8b (3/cell) | llama3.1:8b | 27 | ~2 hours |
| Local 70b (3/cell) | llama3.3:70b | 27 | ~33 hours |
| Local 70b (10/cell) | llama3.3:70b | 90 | ~110 hours |

---

## Checking results after the batch

```bash
masid-crewai results --config configs/crewai_groq.yaml         # Groq results
masid-crewai results --config configs/crewai_experiment.yaml   # local Ollama results
```
