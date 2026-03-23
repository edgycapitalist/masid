"""Command-line interface for the MASID CrewAI adapter.

Usage:
    masid-crewai smoke-test
    masid-crewai run --architecture irm --domain software_dev
    masid-crewai batch --config configs/crewai_experiment.yaml
    masid-crewai results --config configs/crewai_experiment.yaml
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from masid.domains.registry import list_domains
from masid.storage import ExperimentDB
from masid_crewai.config import load_crewai_config
from masid_crewai.runner import CrewAITrialRunner

console = Console()

_DEFAULT_CONFIG = "configs/crewai_experiment.yaml"


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "crewai", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@click.group()
def main() -> None:
    """MASID CrewAI Adapter — validate incentive architecture findings on CrewAI."""


@main.command(name="smoke-test")
@click.option("--config", "config_path", default=_DEFAULT_CONFIG, help="Path to config YAML")
def smoke_test(config_path: str) -> None:
    """Test LLM endpoint connectivity via CrewAI LLM."""
    _setup_logging("INFO")
    console.print("[bold green]CrewAI Adapter Smoke Test[/bold green]")

    try:
        config = load_crewai_config(config_path)
    except FileNotFoundError:
        config = load_crewai_config()
        console.print(f"[yellow]Config not found at {config_path!r}, using defaults[/yellow]")

    console.print(f"  Model: {config.agent_model.model}")
    console.print(f"  Endpoint: {config.agent_model.base_url}")

    # Test via CrewAI LLM
    from masid_crewai.architectures import build_crewai_llm

    llm = build_crewai_llm(config.agent_model)
    console.print("Testing LLM endpoint via CrewAI LLM...")
    try:
        response = llm.call([{"role": "user", "content": "Say 'hello' and nothing else."}])
        preview = str(response)[:80] if response else "(empty)"
        console.print(f"  [green]OK[/green] — Response: {preview}")
    except Exception as e:
        console.print(f"  [red]FAILED[/red] — {e}")
        console.print(
            "\n[yellow]Make sure Ollama is running at "
            f"{config.agent_model.base_url}[/yellow]"
        )
        sys.exit(1)

    console.print("\n[bold green]Smoke test passed![/bold green]")


@main.command()
@click.option("--architecture", "-a", required=True, type=click.Choice(["irm", "jro", "iamd"]))
@click.option("--domain", "-d", required=True, type=click.Choice(list_domains()))
@click.option("--task-id", default=None, help="Specific task ID (e.g. sw_001)")
@click.option("--config", "config_path", default=_DEFAULT_CONFIG, help="Path to config YAML")
def run(
    architecture: str,
    domain: str,
    task_id: str | None,
    config_path: str,
) -> None:
    """Run a single trial."""
    _setup_logging("INFO")

    try:
        config = load_crewai_config(config_path)
    except FileNotFoundError:
        config = load_crewai_config()
        console.print(f"[yellow]Config not found at {config_path!r}, using defaults[/yellow]")

    db = ExperimentDB(config.storage.db_path)
    runner = CrewAITrialRunner(config=config, db=db)

    effective_task_id = task_id or config.task_ids.get(domain)
    console.print(
        f"[bold]Running CrewAI trial:[/bold] {architecture} × {domain} × {effective_task_id}"
    )

    try:
        metrics = runner.run(
            architecture_key=architecture,
            domain=domain,
            task_id=task_id,
            seed=config.experiment.trials_per_cell,
        )
        console.print(
            f"[green]Done![/green] Quality: {metrics.quality_score:.2f} "
            f"Latency: {metrics.total_latency_seconds:.1f}s"
        )
    except Exception as e:
        console.print(f"[red]Trial failed: {e}[/red]")
        raise


@main.command()
@click.option("--config", "config_path", required=True, help="Path to config YAML")
def batch(config_path: str) -> None:
    """Run a full batch of trials with resume support.

    Iterates: architectures × domains × task_ids × trials_per_cell.
    Skips cells that already have enough trials in the database.
    """
    _setup_logging("INFO")
    config = load_crewai_config(config_path)
    db = ExperimentDB(config.storage.db_path)
    runner = CrewAITrialRunner(config=config, db=db)

    # Start seeds above existing DB content
    next_seed = db.max_seed() + 1
    if next_seed < 42:
        next_seed = 42

    target = config.experiment.trials_per_cell
    total = 0
    skipped = 0
    errors = 0

    for arch in config.architectures:
        for domain in config.domains:
            task_id = config.task_ids.get(domain)
            if not task_id:
                console.print(f"[yellow]No task_id for domain {domain!r}, skipping[/yellow]")
                continue

            existing = db.count_trials_for_cell(arch, domain, task_id)
            needed = max(0, target - existing)

            if needed == 0:
                skipped += target
                console.print(
                    f"[dim]  Skipping {arch} × {domain} × {task_id} "
                    f"({existing}/{target} already done)[/dim]"
                )
                continue

            if existing > 0:
                console.print(
                    f"[dim]  {arch} × {domain} × {task_id}: "
                    f"{existing}/{target} done, running {needed} more[/dim]"
                )

            for trial_num in range(needed):
                seed = next_seed
                next_seed += 1
                console.print(
                    f"[dim]Trial {total + 1}: {arch} × {domain} × {task_id} "
                    f"(trial {existing + trial_num + 1}/{target}, seed={seed})[/dim]"
                )
                try:
                    runner.run(
                        architecture_key=arch,
                        domain=domain,
                        task_id=task_id,
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
@click.option("--config", "config_path", default=_DEFAULT_CONFIG, help="Path to config YAML")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]))
def results(config_path: str, fmt: str) -> None:
    """Display CrewAI experiment results."""
    try:
        config = load_crewai_config(config_path)
    except FileNotFoundError:
        config = load_crewai_config()

    db = ExperimentDB(config.storage.db_path)
    trials = db.get_all_trials()

    if not trials:
        console.print("[yellow]No trials recorded yet.[/yellow]")
        return

    if fmt == "json":
        import json
        console.print_json(json.dumps(trials, default=str, indent=2))
    else:
        table = Table(title=f"CrewAI Results ({len(trials)} trials)")
        table.add_column("Trial ID", style="dim")
        table.add_column("Architecture")
        table.add_column("Domain")
        table.add_column("Task")
        table.add_column("Quality", justify="right")
        table.add_column("Latency", justify="right")

        for t in trials:
            table.add_row(
                t["trial_id"],
                t["architecture"],
                t["domain"],
                t.get("task_id", ""),
                f"{t['quality_score']:.2f}",
                f"{t['total_latency_seconds']:.1f}s",
            )
        console.print(table)


if __name__ == "__main__":
    main()
