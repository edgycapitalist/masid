import sqlite3
import json
import csv
from pathlib import Path

conn = sqlite3.connect("data/experiments.db")
conn.row_factory = sqlite3.Row

# Export 1: Trial summary CSV
trials = conn.execute("SELECT * FROM trials ORDER BY architecture, domain, task_id").fetchall()
with open("data/trials_export.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trial_id", "architecture", "domain", "model", "task_id", "seed",
        "quality_score", "correctness", "completeness", "coherence", "integration",
        "total_tokens", "total_latency_seconds", "num_rounds",
        "duplication_rate", "judge_rationale", "metadata_json"
    ])
    for t in trials:
        writer.writerow([
            t["trial_id"], t["architecture"], t["domain"], t["model"],
            t["task_id"], t["seed"],
            t["quality_score"], t["correctness"], t["completeness"],
            t["coherence"], t["integration"],
            t["total_tokens"], t["total_latency_seconds"], t["num_rounds"],
            t["duplication_rate"], t["judge_rationale"], t["metadata_json"],
        ])

# Export 2: Full agent outputs
outputs = conn.execute("""
    SELECT ao.trial_id, t.architecture, t.domain, t.task_id,
           ao.agent_id, ao.role, ao.round_number, 
           ao.content, ao.prompt_tokens, ao.completion_tokens, ao.latency_seconds
    FROM agent_outputs ao
    JOIN trials t ON ao.trial_id = t.trial_id
    ORDER BY t.architecture, t.domain, t.task_id, ao.trial_id, ao.round_number, ao.role
""").fetchall()

with open("data/agent_outputs_export.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trial_id", "architecture", "domain", "task_id",
        "agent_id", "role", "round_number",
        "content", "prompt_tokens", "completion_tokens", "latency_seconds"
    ])
    for o in outputs:
        writer.writerow([
            o["trial_id"], o["architecture"], o["domain"], o["task_id"],
            o["agent_id"], o["role"], o["round_number"],
            o["content"], o["prompt_tokens"], o["completion_tokens"], o["latency_seconds"],
        ])

# Export 3: Summary statistics
print("\n" + "=" * 70)
print("SUMMARY BY ARCHITECTURE × DOMAIN")
print("=" * 70)

stats = conn.execute("""
    SELECT architecture, domain, 
           COUNT(*) as n_trials,
           ROUND(AVG(quality_score), 3) as avg_quality,
           ROUND(MIN(quality_score), 3) as min_quality,
           ROUND(MAX(quality_score), 3) as max_quality,
           ROUND(AVG(total_tokens), 0) as avg_tokens,
           ROUND(AVG(total_latency_seconds), 1) as avg_latency,
           ROUND(AVG(correctness), 3) as avg_correctness,
           ROUND(AVG(completeness), 3) as avg_completeness,
           ROUND(AVG(coherence), 3) as avg_coherence,
           ROUND(AVG(integration), 3) as avg_integration
    FROM trials
    GROUP BY architecture, domain
    ORDER BY domain, architecture
""").fetchall()

print(f"{'Arch':<6} {'Domain':<22} {'N':>3} {'Avg Q':>7} {'Min':>6} {'Max':>6} {'Tokens':>8} {'Lat':>6} {'Corr':>6} {'Comp':>6} {'Cohr':>6} {'Intg':>6}")
print("-" * 100)
for s in stats:
    print(f"{s['architecture']:<6} {s['domain']:<22} {s['n_trials']:>3} {s['avg_quality']:>7.3f} {s['min_quality']:>6.3f} {s['max_quality']:>6.3f} {s['avg_tokens']:>8.0f} {s['avg_latency']:>6.1f} {s['avg_correctness']:>6.3f} {s['avg_completeness']:>6.3f} {s['avg_coherence']:>6.3f} {s['avg_integration']:>6.3f}")

# Per-task breakdown
print("\n" + "=" * 70)
print("SUMMARY BY ARCHITECTURE × DOMAIN × TASK")
print("=" * 70)

task_stats = conn.execute("""
    SELECT architecture, domain, task_id,
           COUNT(*) as n,
           ROUND(AVG(quality_score), 3) as avg_q,
           ROUND(AVG(total_tokens), 0) as avg_tok
    FROM trials
    GROUP BY architecture, domain, task_id
    ORDER BY domain, task_id, architecture
""").fetchall()

print(f"{'Arch':<6} {'Domain':<22} {'Task':<8} {'N':>3} {'Avg Q':>7} {'Tokens':>8}")
print("-" * 60)
for s in task_stats:
    print(f"{s['architecture']:<6} {s['domain']:<22} {s['task_id']:<8} {s['n']:>3} {s['avg_q']:>7.3f} {s['avg_tok']:>8.0f}")

print("\nExported to data/trials_export.csv and data/agent_outputs_export.csv")
conn.close()