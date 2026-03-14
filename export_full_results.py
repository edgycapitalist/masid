"""Export complete trial data from MASID experiments.

Exports:
1. trials_summary.csv — scores and metadata per trial
2. agent_outputs_full.csv — every agent response from every round
3. trial_details/ — one text file per trial with full readable output
4. summary_stats.txt — aggregate statistics
"""

import sqlite3
import json
import csv
import os
from collections import defaultdict
from pathlib import Path

DB_PATH = "data/experiments.db"
EXPORT_DIR = Path("data/export")


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "trial_details").mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # 1. Trial summary CSV
    # ------------------------------------------------------------------
    trials = conn.execute("SELECT * FROM trials ORDER BY architecture, domain, task_id").fetchall()
    print(f"Total trials: {len(trials)}")

    with open(EXPORT_DIR / "trials_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "architecture", "domain", "model", "task_id", "seed",
            "quality_score", "correctness", "completeness", "coherence", "integration",
            "total_tokens", "total_latency_seconds", "num_rounds",
            "duplication_rate", "fault_type", "fault_agent",
            "judge_rationale", "metadata_json",
        ])
        for t in trials:
            writer.writerow([
                t["trial_id"], t["architecture"], t["domain"], t["model"],
                t["task_id"], t["seed"],
                t["quality_score"], t["correctness"], t["completeness"],
                t["coherence"], t["integration"],
                t["total_tokens"], t["total_latency_seconds"], t["num_rounds"],
                t["duplication_rate"], t["fault_type"], t["fault_agent"],
                t["judge_rationale"], t["metadata_json"],
            ])

    print(f"  Exported: {EXPORT_DIR / 'trials_summary.csv'}")

    # ------------------------------------------------------------------
    # 2. Full agent outputs CSV
    # ------------------------------------------------------------------
    outputs = conn.execute("""
        SELECT ao.trial_id, t.architecture, t.domain, t.task_id, t.seed,
               ao.agent_id, ao.role, ao.round_number,
               ao.content, ao.prompt_tokens, ao.completion_tokens, ao.latency_seconds
        FROM agent_outputs ao
        JOIN trials t ON ao.trial_id = t.trial_id
        ORDER BY t.architecture, t.domain, t.task_id, ao.trial_id, ao.round_number, ao.role
    """).fetchall()

    with open(EXPORT_DIR / "agent_outputs_full.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "architecture", "domain", "task_id", "seed",
            "agent_id", "role", "round_number",
            "content", "prompt_tokens", "completion_tokens", "latency_seconds",
        ])
        for o in outputs:
            writer.writerow([
                o["trial_id"], o["architecture"], o["domain"], o["task_id"], o["seed"],
                o["agent_id"], o["role"], o["round_number"],
                o["content"], o["prompt_tokens"], o["completion_tokens"], o["latency_seconds"],
            ])

    print(f"  Exported: {EXPORT_DIR / 'agent_outputs_full.csv'}")

    # ------------------------------------------------------------------
    # 3. Per-trial detail files (human readable)
    # ------------------------------------------------------------------
    for t in trials:
        tid = t["trial_id"]
        filename = f"{t['architecture']}_{t['domain']}_{t['task_id']}_{tid}.txt"
        filepath = EXPORT_DIR / "trial_details" / filename

        trial_outputs = conn.execute("""
            SELECT role, round_number, content, prompt_tokens, completion_tokens, latency_seconds
            FROM agent_outputs
            WHERE trial_id = ?
            ORDER BY round_number, role
        """, (tid,)).fetchall()

        with open(filepath, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"TRIAL: {tid}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Architecture:  {t['architecture']}\n")
            f.write(f"Domain:        {t['domain']}\n")
            f.write(f"Task:          {t['task_id']}\n")
            f.write(f"Model:         {t['model']}\n")
            f.write(f"Seed:          {t['seed']}\n")
            f.write(f"\n")
            f.write(f"SCORES:\n")
            f.write(f"  Quality (blended): {t['quality_score']}\n")
            f.write(f"  Correctness:       {t['correctness']}\n")
            f.write(f"  Completeness:      {t['completeness']}\n")
            f.write(f"  Coherence:         {t['coherence']}\n")
            f.write(f"  Integration:       {t['integration']}\n")
            f.write(f"\n")
            f.write(f"EFFICIENCY:\n")
            f.write(f"  Total tokens:  {t['total_tokens']}\n")
            f.write(f"  Latency:       {t['total_latency_seconds']}s\n")
            f.write(f"  Rounds:        {t['num_rounds']}\n")
            f.write(f"  Duplication:   {t['duplication_rate']}\n")
            f.write(f"\n")
            f.write(f"JUDGE RATIONALE:\n")
            f.write(f"  {t['judge_rationale'] or '(empty)'}\n")
            f.write(f"\n")

            if t["metadata_json"]:
                try:
                    meta = json.loads(t["metadata_json"])
                    if "execution_score" in meta:
                        f.write(f"CODE EXECUTION:\n")
                        f.write(f"  Execution score: {meta.get('execution_score')}\n")
                        f.write(f"  Syntax valid:    {meta.get('syntax_valid')}\n")
                        f.write(f"  Code runs:       {meta.get('code_runs')}\n")
                        f.write(f"  Tests passed:    {meta.get('tests_passed')}/{meta.get('tests_total')}\n")
                        f.write(f"\n")
                except (json.JSONDecodeError, TypeError):
                    pass

            # Group outputs by round
            rounds = defaultdict(list)
            for o in trial_outputs:
                rounds[o["round_number"]].append(o)

            for round_num in sorted(rounds.keys()):
                f.write(f"\n{'#' * 80}\n")
                f.write(f"# ROUND {round_num + 1}\n")
                f.write(f"{'#' * 80}\n")

                for o in rounds[round_num]:
                    f.write(f"\n{'-' * 60}\n")
                    f.write(f"{o['role']} (round {round_num + 1})\n")
                    f.write(f"Tokens: {o['prompt_tokens']} in / {o['completion_tokens']} out | ")
                    f.write(f"Latency: {o['latency_seconds']}s\n")
                    f.write(f"{'-' * 60}\n")
                    f.write(o["content"])
                    f.write(f"\n")

    print(f"  Exported: {len(trials)} trial detail files to {EXPORT_DIR / 'trial_details/'}")

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    with open(EXPORT_DIR / "summary_stats.txt", "w") as f:
        f.write("MASID EXPERIMENT RESULTS — SUMMARY STATISTICS\n")
        f.write(f"Total trials: {len(trials)}\n")
        f.write(f"{'=' * 90}\n\n")

        # By architecture × domain
        f.write(f"{'Arch':<6} {'Domain':<22} {'N':>3} {'Avg Q':>7} {'Min':>6} {'Max':>6} "
                f"{'StdDev':>7} {'Avg Tok':>8} {'Avg Lat':>8}\n")
        f.write("-" * 90 + "\n")

        groups = defaultdict(list)
        for t in trials:
            groups[(t["architecture"], t["domain"])].append(t)

        for (arch, domain) in sorted(groups.keys(), key=lambda x: (x[1], x[0])):
            g = groups[(arch, domain)]
            scores = [t["quality_score"] for t in g]
            n = len(scores)
            avg = sum(scores) / n
            mn = min(scores)
            mx = max(scores)
            std = (sum((s - avg) ** 2 for s in scores) / n) ** 0.5
            avg_tok = sum(t["total_tokens"] for t in g) / n
            avg_lat = sum(t["total_latency_seconds"] for t in g) / n
            f.write(f"{arch:<6} {domain:<22} {n:>3} {avg:>7.3f} {mn:>6.3f} {mx:>6.3f} "
                    f"{std:>7.3f} {avg_tok:>8.0f} {avg_lat:>8.1f}\n")

        # Per-task breakdown
        f.write(f"\n{'=' * 90}\n")
        f.write("PER-TASK COMPARISON\n")
        f.write(f"{'=' * 90}\n\n")

        task_groups = defaultdict(list)
        for t in trials:
            task_groups[(t["domain"], t["task_id"])].append(t)

        for (domain, task_id) in sorted(task_groups.keys()):
            f.write(f"\n  {domain} / {task_id}:\n")
            arch_scores = defaultdict(list)
            for t in task_groups[(domain, task_id)]:
                arch_scores[t["architecture"]].append(t["quality_score"])

            for arch in ["irm", "jro", "iamd"]:
                scores = arch_scores.get(arch, [])
                if scores:
                    avg = sum(scores) / len(scores)
                    score_str = " ".join(f"{s:.2f}" for s in scores)
                    f.write(f"    {arch:<6}: {score_str:>30s}  avg={avg:.3f}\n")

        # Domain winners
        f.write(f"\n{'=' * 90}\n")
        f.write("DOMAIN WINNERS\n")
        f.write(f"{'=' * 90}\n\n")

        for domain in ["software_dev", "research_synthesis", "project_planning"]:
            avgs = {}
            tokens = {}
            for arch in ["irm", "jro", "iamd"]:
                g = groups.get((arch, domain), [])
                if g:
                    avgs[arch] = sum(t["quality_score"] for t in g) / len(g)
                    tokens[arch] = sum(t["total_tokens"] for t in g) / len(g)
            if avgs:
                winner = max(avgs, key=avgs.get)
                f.write(f"  {domain}:\n")
                for arch in sorted(avgs, key=avgs.get, reverse=True):
                    marker = " ← BEST" if arch == winner else ""
                    f.write(f"    {arch:<6} avg={avgs[arch]:.3f}  tokens={tokens[arch]:,.0f}{marker}\n")
                f.write("\n")

    print(f"  Exported: {EXPORT_DIR / 'summary_stats.txt'}")

    conn.close()
    print(f"\nAll exports complete in {EXPORT_DIR}/")


if __name__ == "__main__":
    main()