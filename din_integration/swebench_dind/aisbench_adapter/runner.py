"""Subprocess entrypoint for SwebenchDindTask.

Invoked by ``aisbench.LocalRunner`` as:
    python -m swebench_dind.aisbench_adapter.runner --config <cfg> --work-dir <dir>

Reads the cfg, runs one harbor trial (via ``launcher.launch_trial``),
waits for result.json, converts it to AISBench schema via
``result_writer.write_result``, and writes:
    <work_dir>/results/<model_abbr>/<dataset_abbr>.json
    <work_dir>/results/<model_abbr>/<dataset_abbr>/details/
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from rich.console import Console

from ..config import DEFAULT_API_BASE, DEFAULT_MODEL, JOBS_DIR
from ..launcher import _build_harbor_args, _cleanup_job_dir, _load_api_key, wait_for_job
from .result_writer import write_result

console = Console()


def main(cfg: dict | None = None, task_state_manager=None) -> None:
    """Entry point used by both BaseTask.run() (subprocess) and direct CLI."""
    parser = argparse.ArgumentParser(prog="swebench_dind.aisbench_adapter.runner")
    parser.add_argument("--config", required=True, help="Path to ais_bench config.py")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--dataset-index", type=int, default=0)
    args = parser.parse_args()

    # Lazy import mmengine (only needed in subprocess context)
    from mmengine.config import Config
    cfg_obj = Config.fromfile(args.config)

    models = cfg_obj.get("models", [])
    datasets = cfg_obj.get("datasets", [])
    if not models or not datasets:
        console.print("[red]config must define models and datasets[/red]")
        sys.exit(1)

    model_cfg = models[args.model_index]
    dataset_cfg = datasets[args.dataset_index]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Translate cfg → launch args
    agent = model_cfg.get("agent_name", "aider")
    model_name = (model_cfg.get("model_names") or [DEFAULT_MODEL])[0]
    api_key = model_cfg.get("api_key") or _load_api_key()
    api_base = cfg_obj.get("api_base", DEFAULT_API_BASE)

    dataset_args = dataset_cfg.get("args", {})
    task_path = Path(dataset_args.get("path", ""))
    # Parse path: .../django__django-{case}-{agent}
    case, _, agent_in_path = task_path.name.partition("django__django-")[2].partition("-")
    if not case:
        console.print(f"[red]could not parse case from path {task_path}[/red]")
        sys.exit(1)
    # Honor agent from path if not in cfg
    if agent_in_path and not model_cfg.get("agent_name"):
        agent = agent_in_path

    job_name = f"aisbench-{agent}-{case}"
    _cleanup_job_dir(job_name)

    multipliers = {
        "agent_setup": 4,
        "agent": 4,
        "verifier": 4,
        "environment_build": 4,
    }

    from ..launcher import LaunchSpec  # local import (avoids circular)
    spec = LaunchSpec(
        case=case,
        agent=agent,
        job_name=job_name,
        model=model_name,
        api_base=api_base,
        n=1,
        multipliers=multipliers,
        extra_ae=[],
    )
    args_list = _build_harbor_args(spec, api_key)

    console.print(f"[bold]launch[/bold] {job_name}")
    import subprocess
    subprocess.Popen(
        ["docker", "exec", "-e", "OPENAI_API_KEY", "-e", "OPENAI_API_BASE",
         spec.container_name, *args_list] if hasattr(spec, "container_name")
        else ["docker", "exec", "-e", "OPENAI_API_KEY", "-e", "OPENAI_API_BASE",
              "swebench-orchestrator", *args_list],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    if task_state_manager is not None:
        task_state_manager.update_task_state({"status": "running", "finish_count": 0})

    try:
        data = wait_for_job(job_name, timeout_min=24 * 60)
    except TimeoutError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Write AISBench result.json
    harbor_result_path = JOBS_DIR / job_name / "result.json"
    model_abbr = model_cfg.get("summarizer_abbr") or model_cfg.get("abbr", "default")
    dataset_abbr = dataset_cfg.get("abbr", task_path.name)
    out_path = work_dir / "results" / model_abbr / f"{dataset_abbr}.json"
    write_result(harbor_result_path, out_path, model_abbr=model_abbr, dataset_abbr=dataset_abbr)

    # Copy details (Harbor convention)
    details_src = JOBS_DIR / job_name
    details_dst = work_dir / "results" / model_abbr / dataset_abbr / "details"
    if details_src.exists():
        details_dst.mkdir(parents=True, exist_ok=True)
        for f in details_src.iterdir():
            if f.is_file():
                shutil.copy2(f, details_dst / f.name)

    console.print(f"[green]✅ {job_name} → {out_path}[/green]")

    if task_state_manager is not None:
        task_state_manager.update_task_state({"status": "done", "finish_count": 1})


if __name__ == "__main__":
    main()