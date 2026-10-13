"""Typer app exposing the 7 subcommands.

Use as a script: ``swebench-dind <command> [options]`` (installed entry
point) or ``python -m swebench_dind <command>``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .config import (
    ALL_AGENTS,
    DEFAULT_AGENTS,
    DEFAULT_CASES,
    JOBS_DIR,
    LOGS_DIR,
    NEW_CASES,
)

app = typer.Typer(
    name="swebench-dind",
    help="SWE-bench DinD multi-agent trial orchestration CLI",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"swebench-dind {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """SWE-bench DinD CLI root."""


# === orchestrator sub-app ===
orch_app = typer.Typer(help="DinD container lifecycle")
app.add_typer(orch_app, name="orchestrator")


@orch_app.command("start")
def orch_start(
    recreate: bool = typer.Option(False, "--recreate", help="Remove existing container and recreate."),
) -> None:
    """Start the DinD orchestrator container."""
    from .container import start
    start(recreate=recreate)


@orch_app.command("stop")
def orch_stop(
    remove: bool = typer.Option(False, "--remove", help="Also remove the container (data preserved on host)."),
) -> None:
    """Stop the orchestrator container."""
    from .container import stop
    stop(remove=remove)


@orch_app.command("status")
def orch_status() -> None:
    """Show orchestrator container status."""
    from .container import print_status
    print_status()


# === build sub-app ===
build_app = typer.Typer(help="Build L1/L2 baked images")
app.add_typer(build_app, name="build")


@build_app.command("l1")
def build_l1_cmd(
    case: list[str] = typer.Option(..., "--case", help="Case number(s), e.g. 11099"),
    force: bool = typer.Option(False, "--force", help="Rebuild even if image exists."),
) -> None:
    """Build L1 case-base images."""
    from .builder import build_l1
    for c in case:
        build_l1(c, force=force)


@build_app.command("l2")
def build_l2_cmd(
    agent: list[str] = typer.Option(..., "--agent", help="Agent name(s)"),
    case: list[str] = typer.Option(DEFAULT_CASES, "--case", help="Case number(s)"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build L2 agent-baked images."""
    from .builder import build_l2
    for a in agent:
        for c in case:
            build_l2(c, a, force=force)


@build_app.command("all")
def build_all_cmd(
    case: list[str] = typer.Option(DEFAULT_CASES + NEW_CASES, "--case"),
    agent: list[str] = typer.Option(DEFAULT_AGENTS, "--agent"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build all L1 + L2 baked images."""
    from .builder import build_l1_all, build_l2_all
    build_l1_all(case, force=force)
    build_l2_all(case, agent, force=force)


# === launch sub-app ===
launch_app = typer.Typer(help="Launch trials")
app.add_typer(launch_app, name="launch")


@launch_app.command("trial")
def launch_trial_cmd(
    case: str = typer.Option(..., "--case"),
    agent: str = typer.Option(..., "--agent"),
    job_name: Optional[str] = typer.Option(None, "--job-name"),
    n: int = typer.Option(1, "-n"),
    model: str = typer.Option("openai/Qwen/Qwen3-Coder-30B-A3B-Instruct", "-m"),
    api_base: str = typer.Option("https://api.siliconflow.cn/v1", "--api-base"),
    wait: bool = typer.Option(False, "--wait", help="Block until job finishes."),
    timeout_min: int = typer.Option(120, "--timeout-min"),
) -> None:
    """Launch a single (case, agent) trial."""
    from .launcher import launch_trial
    launch_trial(
        case, agent,
        job_name=job_name, n=n, model=model, api_base=api_base,
        wait=wait, timeout_min=timeout_min,
    )


@launch_app.command("3x3")
def launch_3x3_cmd(
    wait: bool = typer.Option(False, "--wait"),
    timeout_min: int = typer.Option(120, "--timeout-min"),
) -> None:
    """Launch the default 3×3 matrix (3 cases × 3 agents)."""
    from .launcher import launch_3x3
    launch_3x3(wait=wait, timeout_min=timeout_min)


@launch_app.command("matrix")
def launch_matrix_cmd(
    case: list[str] = typer.Option(DEFAULT_CASES, "--case"),
    agent: list[str] = typer.Option(DEFAULT_AGENTS, "--agent"),
    wait: bool = typer.Option(False, "--wait"),
    timeout_min: int = typer.Option(120, "--timeout-min"),
) -> None:
    """Launch an N×M matrix of trials."""
    from .launcher import launch_matrix
    launch_matrix(case, agent, wait=wait, timeout_min=timeout_min)


# === watch (top-level) ===
@app.command()
def watch(job_name: str = typer.Argument(...)) -> None:
    """Block until a job's result.json is finished."""
    from .launcher import wait_for_job
    try:
        data = wait_for_job(job_name, timeout_min=24 * 60)
        console.print(f"[green]✅ {job_name} done: {data.get('finished_at')}[/green]")
    except TimeoutError as e:
        console.print(f"[red]❌ {e}[/red]")
        raise typer.Exit(1)


# === summarize (top-level) ===
@app.command()
def summarize(
    jobs_dir: Path = typer.Option(JOBS_DIR, "--jobs-dir"),
    output_dir: Path = typer.Option(LOGS_DIR, "--output-dir"),
    include: Optional[str] = typer.Option(None, "--include", help="Comma-separated substring filter."),
) -> None:
    """Aggregate results.json → summary-<ts>.{md,csv,json}."""
    from .summarizer import summarize as _summarize
    inc = [s.strip() for s in (include or "").split(",") if s.strip()] or None
    out = _summarize(jobs_dir=jobs_dir, output_dir=output_dir, include=inc)
    console.print(f"[green]✅ {len(out['rows'])} jobs summarized[/green]")
    console.print(f"  → {out['md']}")
    console.print(f"  → {out['csv']}")
    console.print(f"  → {out['json']}")


# === patch sub-app ===
patch_app = typer.Typer(help="Patch harbor for idempotent installs")
app.add_typer(patch_app, name="patch")


@patch_app.command("harbor")
def patch_harbor_cmd(
    agent: list[str] = typer.Option(["qwen-code"], "--agent", help="Agent(s) to patch."),
) -> None:
    """Patch harbor's installed agent module to add idempotent install probe."""
    from .patcher import patch_agent
    for a in agent:
        patch_agent(a)


# === aisbench sub-app ===
aisbench_app = typer.Typer(help="AISBench integration (P1)")
app.add_typer(aisbench_app, name="aisbench")


@aisbench_app.command("install")
def aisbench_install() -> None:
    """Symlink the aisbench_adapter into aisbench's runtime/."""
    from .aisbench_adapter import install as _install
    _install()


@aisbench_app.command("run")
def aisbench_run(
    config: Path = typer.Option(..., "--config", help="Path to ais_bench config.py"),
) -> None:
    """Run the AISBench CLI with the installed swebench_dind task adapter."""
    import subprocess
    console.print(f"[bold]ais_bench[/bold] {config}")
    subprocess.run(["ais_bench", str(config)], check=False)


@aisbench_app.command("result-format")
def aisbench_result_format() -> None:
    """Print the AISBench-compatible result schema docstring."""
    from .aisbench_adapter.result_writer import SCHEMA_DOC
    console.print(SCHEMA_DOC)


if __name__ == "__main__":
    app()