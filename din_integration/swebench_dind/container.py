"""Orchestrator (DinD container) lifecycle management.

Wraps the logic of legacy ``scripts/start_orchestrator.sh`` and
``scripts/stop_orchestrator.sh`` into Python functions.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from rich.console import Console

from .config import (
    AGENT_PATCHES_DIR,
    API_KEY_ENV,
    CONFIG_DIR,
    CONTAINER_NAME,
    DATA_CONTAINER_NAME,
    DATA_IMAGE,
    ENTRYPOINT_SH,
    JOBS_DIR,
    LOGS_DIR,
    ORCHESTRATOR_IMAGE,
    TASKS_DIR,
)

console = Console()


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _container_exists(name: str) -> bool:
    out = _docker("ps", "-a", "--format", "{{.Names}}", check=False).stdout
    return name in out.splitlines()


def _container_running(name: str) -> bool:
    out = _docker("ps", "--format", "{{.Names}}", check=False).stdout
    return name in out.splitlines()


def status() -> dict:
    """Report orchestrator container + DinD dockerd health."""
    exists = _container_exists(CONTAINER_NAME)
    running = _container_running(CONTAINER_NAME)
    dockerd_ready = False
    if running:
        try:
            _docker("exec", CONTAINER_NAME, "docker", "info", check=True)
            dockerd_ready = True
        except subprocess.CalledProcessError:
            dockerd_ready = False
    return {
        "container": CONTAINER_NAME,
        "exists": exists,
        "running": running,
        "dockerd_ready": dockerd_ready,
    }


def print_status() -> None:
    s = status()
    console.print(f"[bold]{s['container']}[/bold]")
    console.print(f"  exists:    {s['exists']}")
    console.print(f"  running:   {s['running']}")
    console.print(f"  dockerd:   {'✅ ready' if s['dockerd_ready'] else '❌ not ready'}")


def _load_api_key() -> str:
    if not API_KEY_ENV.exists():
        raise FileNotFoundError(f"{API_KEY_ENV} missing; copy from another host")
    for line in API_KEY_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "OPENAI_API_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError("OPENAI_API_KEY not found in api_key.env")


def _ensure_dirs() -> None:
    for d in (JOBS_DIR, TASKS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def start(recreate: bool = False) -> None:
    """Start the DinD orchestrator container.

    Equivalent to ``bash scripts/start_orchestrator.sh [--recreate]``.
    Idempotent: if container already running, prints status and exits.
    """
    _ensure_dirs()

    if _container_exists(CONTAINER_NAME):
        if not recreate:
            s = status()
            if s["running"] and s["dockerd_ready"]:
                console.print(f"[green]✅ {CONTAINER_NAME} already running[/green]")
                return
            console.print(f"[yellow]⚠ {CONTAINER_NAME} exists but not ready[/yellow]")
            console.print("  Use --recreate to remove and recreate.")
            return
        console.print(f"[yellow]Removing old container {CONTAINER_NAME}[/yellow]")
        _docker("rm", "-f", CONTAINER_NAME, check=False)

    if not _container_exists(DATA_CONTAINER_NAME):
        console.print(f"Creating data container {DATA_CONTAINER_NAME} from {DATA_IMAGE}")
        _docker("create", "--name", DATA_CONTAINER_NAME, DATA_IMAGE)

    api_key = _load_api_key()
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")

    console.print(f"Starting [bold]{CONTAINER_NAME}[/bold] from {ORCHESTRATOR_IMAGE}")
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--hostname", "orchestrator",
        "--privileged",
        "--cgroupns=host",
        "--restart", "unless-stopped",
        "--volumes-from", f"{DATA_CONTAINER_NAME}:ro",
        "-v", f"{JOBS_DIR}:/opt/swebench/jobs:rw",
        "-v", f"{TASKS_DIR}:/opt/swebench/data/tasks:rw",
        "-v", f"{CONFIG_DIR}:/opt/swebench/config:ro",
        "-v", f"{API_KEY_ENV}:/opt/swebench/api_key.env:ro",
        "-v", f"{ENTRYPOINT_SH}:/opt/swebench/scripts/entrypoint.sh:ro",
        "-v", f"{AGENT_PATCHES_DIR}:/opt/swebench/agent-patches:ro",
        "-v", f"{LOGS_DIR}:/opt/swebench/logs:rw",
        "-e", f"OPENAI_API_KEY={api_key}",
        "-e", f"OPENAI_API_BASE={api_base}",
        ORCHESTRATOR_IMAGE,
        "bash", "-c", "tail -f /dev/null",
    ]
    subprocess.run(cmd, check=True)

    # Wait for dockerd
    console.print("Waiting for DinD dockerd to be ready...")
    for _ in range(60):
        try:
            _docker("exec", CONTAINER_NAME, "docker", "info", check=True)
            break
        except subprocess.CalledProcessError:
            time.sleep(2)
    else:
        raise RuntimeError("dockerd did not become ready in 120s")

    console.print(f"[green]✅ {CONTAINER_NAME} ready[/green]")


def stop(remove: bool = False) -> None:
    """Stop the orchestrator container (data is on host, safe).

    Equivalent to ``bash scripts/stop_orchestrator.sh [stop|rm]``.
    """
    if not _container_exists(CONTAINER_NAME):
        console.print(f"[yellow]{CONTAINER_NAME} does not exist[/yellow]")
        return
    if remove:
        _docker("rm", "-f", CONTAINER_NAME)
        console.print(f"[green]Removed {CONTAINER_NAME}[/green]")
    else:
        _docker("stop", CONTAINER_NAME)
        console.print(f"[green]Stopped {CONTAINER_NAME}[/green]")


def exec_in_orchestrator(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command inside the orchestrator container."""
    return _docker("exec", CONTAINER_NAME, *args, check=check)