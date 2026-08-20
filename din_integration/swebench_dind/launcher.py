"""Trial launcher.

Wraps the legacy ``launch_*.sh`` scripts. Each trial becomes a
``harbor jobs start`` invocation inside the orchestrator container.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rich.console import Console

from .config import (
    AGENT_AE,
    COMMON_AE_KEYS,
    CONTAINER_JOBS_DIR,
    CONTAINER_NAME,
    DEFAULT_API_BASE,
    DEFAULT_BARE_MODEL,
    DEFAULT_MODEL,
    DEFAULT_MULTIPLIERS,
    HARBOR_AGENT_FLAG,
    JOBS_DIR,
    LOGS_DIR,
    container_task_path,
    substitute_ae,
)
from .container import exec_in_orchestrator

console = Console()


@dataclass
class LaunchSpec:
    case: str
    agent: str
    job_name: str
    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    n: int = 1
    multipliers: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MULTIPLIERS))
    extra_ae: list[str] = field(default_factory=list)


def _load_api_key() -> str:
    """Read OPENAI_API_KEY from ``scripts/api_key.env``.

    Tolerates both ``OPENAI_API_KEY=sk-...`` and
    ``export OPENAI_API_KEY="sk-..."`` forms (the latter is what the
    legacy file uses when sourced from a shell).
    """
    env_path = Path("/home/zengziyu/mini_matrix/scripts/api_key.env")
    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} missing; copy from another host")
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip optional `export ` prefix
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "OPENAI_API_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not set in api_key.env")


def _build_harbor_args(spec: LaunchSpec, api_key: str) -> list[str]:
    """Build the full ``harbor jobs start`` command-line as a list."""
    args = [
        "harbor", "jobs", "start",
        "--job-name", spec.job_name,
        "--path", container_task_path(spec.case, spec.agent),
        "-a", HARBOR_AGENT_FLAG[spec.agent],
        "-m", spec.model,
        "--jobs-dir", CONTAINER_JOBS_DIR,
        "-n", str(spec.n),
        "--agent-setup-timeout-multiplier", str(spec.multipliers["agent_setup"]),
        "--agent-timeout-multiplier", str(spec.multipliers["agent"]),
        "--verifier-timeout-multiplier", str(spec.multipliers["verifier"]),
    ]

    # Common AE (model + api base + api key)
    for k in COMMON_AE_KEYS:
        if k == "OPENAI_API_KEY":
            v = api_key
        elif k == "OPENAI_API_BASE":
            v = spec.api_base
        elif k == "LLM_BASE_URL":
            v = spec.api_base
        elif k == "LLM_MODEL":
            v = spec.model
        else:
            v = api_key
        args += ["--ae", f"{k}={v}"]

    # Agent-specific AE
    extras = AGENT_AE.get(spec.agent, [])
    for raw in extras:
        for s in substitute_ae([raw], api_key=api_key, model=spec.model, api_base=spec.api_base):
            args += ["--ae", s]

    # User extras
    for e in spec.extra_ae:
        args += ["--ae", e.strip()]
    return args


def _cleanup_job_dir(job_name: str) -> None:
    """Remove stale trial subdirs and lock.json to allow re-runs."""
    exec_in_orchestrator(
        "bash", "-c",
        f"""
        if [ -d /opt/swebench/jobs/{job_name} ]; then
          find /opt/swebench/jobs/{job_name} -mindepth 1 -maxdepth 1 -type d -exec rm -rf {{}} +
        fi
        rm -f /opt/swebench/jobs/{job_name}/lock.json 2>/dev/null
        """,
        check=False,
    )


def launch_trial(
    case: str,
    agent: str,
    *,
    job_name: str | None = None,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    n: int = 1,
    api_key: str | None = None,
    extra_ae: list[str] | None = None,
    wait: bool = False,
    timeout_min: int = 120,
    verbose: bool = True,
) -> str:
    """Launch a single (case, agent) trial. Returns the job name.

    Prints each framework step to stdout so the user can see what's
    happening in real time.
    """
    from rich.panel import Panel
    from rich.table import Table

    if verbose:
        console.print(Panel.fit(
            f"[bold]case[/bold] {case}   [bold]agent[/bold] {agent}   "
            f"[bold]model[/bold] {model.split('/')[-1]}",
            border_style="cyan", title="SWE-bench DinD trial",
        ))

    with console.status("[cyan]loading API key…[/cyan]") if verbose else _nullctx():
        api_key = api_key or _load_api_key()

    job_name = job_name or f"{agent}-{case}"
    if verbose:
        console.print(f"  [green]✓[/green] job name: [bold]{job_name}[/bold]")

    spec = LaunchSpec(
        case=case,
        agent=agent,
        job_name=job_name,
        model=model,
        api_base=api_base,
        n=n,
        extra_ae=extra_ae or [],
    )

    with console.status(f"[cyan]cleaning stale job dir {job_name}…[/cyan]") if verbose else _nullctx():
        _cleanup_job_dir(job_name)
    if verbose:
        console.print(f"  [green]✓[/green] cleaned stale state in /opt/swebench/jobs/{job_name}")

    with console.status("[cyan]building harbor args…[/cyan]") if verbose else _nullctx():
        args = _build_harbor_args(spec, api_key)
    if verbose:
        # show a small table of the harbor flags
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column(style="dim")
        t.add_column()
        for i in range(0, len(args), 2):
            if args[i].startswith("--"):
                t.add_row(args[i], args[i + 1] if i + 1 < len(args) else "")
            else:
                t.add_row(args[i], "")
        console.print(f"  [green]✓[/green] harbor command ({len(args)} tokens):")
        console.print(t)

    cmd = [
        "docker", "exec",
        "-e", f"OPENAI_API_KEY={api_key}",
        "-e", f"OPENAI_API_BASE={spec.api_base}",
        CONTAINER_NAME, *args,
    ]
    if verbose:
        console.print(f"  [yellow]→[/yellow] docker exec into [bold]{CONTAINER_NAME}[/bold]")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if verbose:
        console.print(f"  [green]✓[/green] harbor PID {proc.pid} (returns immediately, runs in container)")

    if wait:
        result = wait_for_job(job_name, timeout_min=timeout_min, verbose=verbose)
        if verbose:
            _print_trial_result(result)
        return result
    return job_name


class _nullctx:
    """Stand-in for ``console.status`` when ``verbose=False``."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _print_trial_result(result: dict) -> None:
    from rich.panel import Panel
    stats = result.get("stats", {})
    evals = stats.get("evals", {})
    if isinstance(evals, dict):
        n_total = len(evals)
        scores = [
            (e.get("metrics", [{}])[0].get("mean", 0) if isinstance(e, dict) else 0)
            for e in evals.values()
        ]
    else:  # legacy list form
        n_total = len(evals)
        scores = [
            (e.get("metrics", [{}])[0].get("mean", 0) if isinstance(e, dict) else 0)
            for e in evals
        ]
    n_pass = sum(1 for s in scores if s > 0.5)
    score = scores[0] if scores else 0.0
    color = "green" if score > 0.5 else "red"
    verdict = "PASS" if score > 0.5 else "FAIL"
    n_err = stats.get("n_errored_trials", 0)
    n_done = stats.get("n_completed_trials", 0)
    started = result.get("started_at", "?")
    finished = result.get("finished_at", "?")
    console.print(Panel(
        f"[bold {color}]{verdict}[/bold {color}]   pass_rate={n_pass}/{n_total}   "
        f"score={score:.4f}   trials_done={n_done}   trials_err={n_err}",
        border_style=color, title="trial verdict",
    ))
    console.print(f"  [dim]started  {started}[/dim]")
    console.print(f"  [dim]finished {finished}[/dim]")


def wait_for_job(job_name: str, *, timeout_min: int = 120, poll_sec: int = 30, verbose: bool = True) -> dict:
    """Poll ``jobs/<name>/result.json`` until DONE or timeout. Returns the parsed JSON."""
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    deadline = time.time() + timeout_min * 60
    last = None
    last_update = 0.0
    spinner = Spinner("dots", text=Text(f"waiting for {job_name}…", style="cyan"))

    if verbose:
        with Live(spinner, console=console, refresh_per_second=4, transient=False) as live:
            while time.time() < deadline:
                elapsed = int(time.time() - (deadline - timeout_min * 60))
                remaining = max(0, int(deadline - time.time()))
                try:
                    content = exec_in_orchestrator(
                        "cat", f"{CONTAINER_JOBS_DIR}/{job_name}/result.json", check=False
                    ).stdout
                except subprocess.CalledProcessError:
                    content = ""
                if content.strip():
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        data = None
                    if data and data.get("finished_at"):
                        # Final tick — replace spinner with a checkmark before exit
                        live.update(Text(f"✅ {job_name} done after {elapsed}s", style="bold green"))
                        return data
                    last = data
                    if data and time.time() - last_update > 5:
                        # Re-render spinner with current elapsed
                        spinner.text = Text(
                            f"waiting for {job_name}…  elapsed={elapsed}s  remaining={remaining}s",
                            style="cyan",
                        )
                        last_update = time.time()
                else:
                    if time.time() - last_update > 5:
                        spinner.text = Text(
                            f"waiting for {job_name}…  (no result.json yet)  elapsed={elapsed}s",
                            style="cyan",
                        )
                        last_update = time.time()
                time.sleep(poll_sec)
    else:
        while time.time() < deadline:
            try:
                content = exec_in_orchestrator(
                    "cat", f"{CONTAINER_JOBS_DIR}/{job_name}/result.json", check=False
                ).stdout
            except subprocess.CalledProcessError:
                content = ""
            if content.strip():
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = None
                if data and data.get("finished_at"):
                    return data
                last = data
            time.sleep(poll_sec)
    raise TimeoutError(f"Job {job_name!r} did not finish within {timeout_min} min")


def launch_matrix(
    cases: list[str],
    agents: list[str],
    *,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    parallel: int | None = None,
    wait: bool = False,
    timeout_min: int = 120,
) -> list[str]:
    """Launch N×M trials in parallel. Returns the list of job names launched."""
    api_key = _load_api_key()
    jobs = []
    for case in cases:
        for agent in agents:
            name = f"{agent}-{case}"
            jobs.append(launch_trial(
                case, agent,
                job_name=name,
                model=model,
                api_base=api_base,
                api_key=api_key,
                wait=False,
            ))
    console.print(f"[green]launched {len(jobs)} trials[/green]")
    if wait:
        for name in jobs:
            try:
                wait_for_job(name, timeout_min=timeout_min)
                console.print(f"[green]✅ {name} done[/green]")
            except TimeoutError as e:
                console.print(f"[red]❌ {e}[/red]")
    return jobs


def launch_3x3(*, wait: bool = False, timeout_min: int = 120) -> list[str]:
    """Launch the default 3×3 matrix (DEFAULT_CASES × DEFAULT_AGENTS)."""
    from .config import DEFAULT_CASES, DEFAULT_AGENTS
    return launch_matrix(DEFAULT_CASES, DEFAULT_AGENTS, wait=wait, timeout_min=timeout_min)