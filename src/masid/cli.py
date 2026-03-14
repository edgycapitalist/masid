"""Command-line interface for MASID.

Usage:
    python -m masid.cli smoke-test
    python -m masid.cli run --architecture irm --domain software_dev
    python -m masid.cli batch --config configs/pilot_batch.yaml
    python -m masid.cli results --format table
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from masid.config import load_config
from masid.domains.registry import get_tasks, list_domains
from masid.orchestrator import TrialRunner
from masid.storage import ExperimentDB

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@click.group()
def main() -> None:
    """MASID — Multi-Agent System Incentive Design Benchmark."""


@main.command(name="smoke-test")
def smoke_test() -> None:
    """Run a minimal smoke test to verify the setup works."""
    _setup_logging("INFO")
    console.print("[bold green]MASID Smoke Test[/bold green]")
    console.print("Loading default config...")

    config = load_config()
    console.print(f"  Model: {config.model.provider}/{config.model.name}")
    console.print(f"  Base URL: {config.model.base_url}")

    # Quick LLM connectivity check
    from masid.models import LLMClient

    client = LLMClient(
        provider=config.model.provider,
        model_name=config.model.name,
        base_url=config.model.base_url,
        temperature=0.7,
        max_tokens=64,
        timeout=30,
    )
    console.print("Testing LLM connection...")
    try:
        resp = client.chat([{"role": "user", "content": "Say 'hello' and nothing else."}])
        console.print(f"  [green]OK[/green] — Response: {resp.content[:80]}")
        console.print(f"  Tokens: {resp.total_tokens}, Latency: {resp.latency_seconds}s")
    except Exception as e:
        console.print(f"  [red]FAILED[/red] — {e}")
        console.print(
            "\n[yellow]Make sure Ollama is running (ollama serve) and the model "
            "is pulled (ollama pull llama3.1:8b)[/yellow]"
        )
        sys.exit(1)

    console.print("\n[bold green]Smoke test passed![/bold green]")


@main.command()
@click.option("--architecture", "-a", required=True, type=click.Choice(["irm", "jro", "iamd"]))
@click.option("--domain", "-d", required=True, type=click.Choice(list_domains()))
@click.option("--model", "-m", default=None, help="Override model name")
@click.option("--config", "config_path", default=None, help="Path to config YAML")
@click.option("--task-index", default=0, help="Index of the task within the domain")
def run(
    architecture: str,
    domain: str,
    model: str | None,
    config_path: str | None,
    task_index: int,
) -> None:
    """Run a single trial."""
    _setup_logging("INFO")
    config = load_config(config_path)
    db = ExperimentDB(config.storage.db_path)

    tasks = get_tasks(domain)
    if task_index >= len(tasks):
        console.print(f"[red]Task index {task_index} out of range (max {len(tasks) - 1})[/red]")
        sys.exit(1)

    task = tasks[task_index]
    runner = TrialRunner(config=config, db=db)

    console.print(f"[bold]Running trial:[/bold] {architecture} × {domain} × {task.title}")
    metrics = runner.run(
        architecture_key=architecture,
        domain=domain,
        task=task,
        model_name=model,
        seed=config.experiment.seed,
    )
    console.print(f"[green]Done![/green] Quality: {metrics.quality_score:.2f} "
                  f"Tokens: {metrics.total_tokens} Latency: {metrics.total_latency_seconds:.1f}s")


@main.command()
@click.option("--config", "config_path", required=True, help="Path to batch config YAML")
def batch(config_path: str) -> None:
    """Run a batch of trials from a config file.

    Safely resumes: checks the database for existing trials and only
    runs the remaining trials needed per cell. Seeds start above the
    highest seed already in the database to avoid duplicates.
    """
    _setup_logging("INFO")
    config = load_config(config_path)
    db = ExperimentDB(config.storage.db_path)
    runner = TrialRunner(config=config, db=db)

    # Start seeds above anything already in the DB
    next_seed = db.max_seed() + 1
    if next_seed < 42:
        next_seed = 42  # Default starting seed

    target = config.experiment.trials_per_cell
    total = 0
    skipped = 0
    errors = 0

    for arch in config.architectures:
        for domain in config.domains:
            tasks = get_tasks(domain)
            for task in tasks:
                existing = db.count_trials_for_cell(arch, domain, task.task_id)
                needed = max(0, target - existing)

                if needed == 0:
                    skipped += target
                    console.print(
                        f"[dim]  Skipping {arch} × {domain} × {task.task_id} "
                        f"({existing}/{target} already done)[/dim]"
                    )
                    continue

                if existing > 0:
                    console.print(
                        f"[dim]  {arch} × {domain} × {task.task_id}: "
                        f"{existing}/{target} done, running {needed} more[/dim]"
                    )

                for trial_num in range(needed):
                    seed = next_seed
                    next_seed += 1

                    console.print(
                        f"[dim]Trial {total + 1}: {arch} × {domain} × {task.task_id} "
                        f"(trial {existing + trial_num + 1}/{target}, "
                        f"seed={seed})[/dim]"
                    )
                    try:
                        runner.run(
                            architecture_key=arch,
                            domain=domain,
                            task=task,
                            seed=seed,
                        )
                    except Exception as e:
                        console.print(f"  [red]ERROR: {e}[/red]")
                        errors += 1
                    total += 1

    console.print(
        f"\n[bold green]Batch complete: {total} trials run, "
        f"{skipped} skipped ({errors} errors).[/bold green]"
    )
    console.print(f"Results saved to: {config.storage.db_path}")


@main.command()
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
@click.option("--config", "config_path", default=None, help="Path to config YAML")
def results(fmt: str, config_path: str | None) -> None:
    """Display experiment results."""
    config = load_config(config_path)
    db = ExperimentDB(config.storage.db_path)

    trials = db.get_all_trials()
    if not trials:
        console.print("[yellow]No trials recorded yet.[/yellow]")
        return

    if fmt == "json":
        import json

        console.print_json(json.dumps(trials, default=str, indent=2))
    else:
        table = Table(title=f"MASID Results ({len(trials)} trials)")
        table.add_column("Trial ID", style="dim")
        table.add_column("Architecture")
        table.add_column("Domain")
        table.add_column("Model")
        table.add_column("Quality", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Latency", justify="right")

        for t in trials:
            table.add_row(
                t["trial_id"],
                t["architecture"],
                t["domain"],
                t["model"],
                f"{t['quality_score']:.2f}",
                str(t["total_tokens"]),
                f"{t['total_latency_seconds']:.1f}s",
            )
        console.print(table)


@main.command()
@click.argument("trial_id")
@click.option("--config", "config_path", default=None, help="Path to config YAML")
def inspect(trial_id: str, config_path: str | None) -> None:
    """Inspect a trial's details including judge rationale and agent outputs."""
    config = load_config(config_path)
    db = ExperimentDB(config.storage.db_path)

    # Find trial (support partial ID match)
    trials = db.get_all_trials()
    match = [t for t in trials if t["trial_id"].startswith(trial_id)]
    if not match:
        console.print(f"[red]No trial found matching '{trial_id}'[/red]")
        return

    t = match[0]
    console.print(f"\n[bold]Trial {t['trial_id']}[/bold]")
    console.print(f"  Architecture: {t['architecture']}")
    console.print(f"  Domain:       {t['domain']}")
    console.print(f"  Model:        {t['model']}")
    console.print(f"  Task:         {t['task_id']}")
    console.print()
    console.print("[bold]Scores:[/bold]")
    console.print(f"  Overall:     {t['quality_score']:.2f}")
    console.print(f"  Correctness: {t['correctness']:.2f}")
    console.print(f"  Completeness:{t['completeness']:.2f}")
    console.print(f"  Coherence:   {t['coherence']:.2f}")
    console.print(f"  Integration: {t['integration']:.2f}")
    console.print()
    console.print("[bold]Efficiency:[/bold]")
    console.print(f"  Tokens:  {t['total_tokens']}")
    console.print(f"  Latency: {t['total_latency_seconds']:.1f}s")
    console.print(f"  Rounds:  {t['num_rounds']}")
    console.print()
    console.print("[bold]Judge Rationale:[/bold]")
    console.print(f"  {t['judge_rationale'] or '(empty)'}")

    # Show execution results if present (software_dev domain)
    if t.get("metadata_json"):
        import json

        meta = json.loads(t["metadata_json"]) if isinstance(t["metadata_json"], str) else {}
        if "execution_score" in meta:
            console.print()
            console.print("[bold]Code Execution (objective):[/bold]")
            console.print(f"  Execution score: {meta['execution_score']:.2f}")
            console.print(f"  Syntax valid:    {meta.get('syntax_valid', 'N/A')}")
            console.print(f"  Code runs:       {meta.get('code_runs', 'N/A')}")
            console.print(
                f"  Tests:           {meta.get('tests_passed', 0)} passed / "
                f"{meta.get('tests_total', 0)} total"
            )


if __name__ == "__main__":
    main()
