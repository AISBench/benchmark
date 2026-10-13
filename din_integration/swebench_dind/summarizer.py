"""Aggregate ``jobs/*/result.json`` → md/csv/json summary.

Equivalent to ``scripts/summarize.py`` but with a programmatic API.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .config import JOBS_DIR, LOGS_DIR


def _load_jobs(jobs_dir: Path, include: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        if include and not any(inc in job_dir.name for inc in include):
            continue
        result = job_dir / "result.json"
        if not result.exists():
            continue
        try:
            data = json.loads(result.read_text())
        except json.JSONDecodeError:
            continue
        config = {}
        cfg_path = job_dir / "config.json"
        if cfg_path.exists():
            try:
                config = json.loads(cfg_path.read_text())
            except json.JSONDecodeError:
                pass
        rows.append(_parse_job(job_dir.name, data, config))
    return rows


def _parse_job(name: str, data: dict, config: dict) -> dict:
    agents = config.get("agents", [])
    tasks = config.get("tasks", [])
    datasets = config.get("datasets", [])
    agent_name = agents[0]["name"] if agents else "?"
    if tasks:
        bench = Path(tasks[0]["path"]).name
    elif datasets:
        bench = Path(datasets[0]["path"]).name
    else:
        bench = "?"
    stats = data.get("stats", {})
    evals = stats.get("evals", {})
    rewards: list[float] = []
    for _, eval_data in evals.items():
        buckets = eval_data.get("reward_stats", {}).get("reward", {})
        for r_str, trial_ids in buckets.items():
            try:
                r_val = float(r_str)
            except (ValueError, TypeError):
                continue
            n = len(trial_ids) if isinstance(trial_ids, list) else 1
            rewards.extend([r_val] * n)
    n_pass = sum(1 for r in rewards if r == 1.0)
    return {
        "job_name": name,
        "agent": agent_name,
        "benchmark": bench,
        "n_total": stats.get("n_total_trials", 0),
        "n_completed": stats.get("n_completed_trials", 0),
        "n_errored": stats.get("n_errored_trials", 0),
        "n_running": stats.get("n_running_trials", 0),
        "n_pending": stats.get("n_pending_trials", 0),
        "n_pass": n_pass,
        "pass_at_1": n_pass / max(len(rewards), 1),
        "finished_at": data.get("finished_at"),
        "started_at": data.get("started_at"),
    }


def _write_markdown(rows: list[dict], path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["benchmark"], r["agent"])].append(r)
    with path.open("w") as f:
        f.write("# Multi-Bench × Multi-Agent Summary\n\n")
        f.write(f"_Generated: {datetime.now().isoformat()}_\n\n")
        f.write(f"**Total jobs**: {len(rows)}\n\n")
        f.write("## Pass@1 by (Benchmark, Agent)\n\n")
        f.write("| Benchmark | Agent | n_jobs | n_pass | pass@1 |\n")
        f.write("|---|---|---|---|---|\n")
        for (bench, agent), items in sorted(grouped.items()):
            n = len(items)
            n_p = sum(it["n_pass"] for it in items)
            f.write(f"| {bench} | {agent} | {n} | {n_p} | {n_p / max(n, 1):.1%} |\n")
        f.write("\n## All Jobs\n\n")
        f.write("| Job | Agent | Bench | n_total | n_done | n_err | pass@1 | finished |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda x: (x["benchmark"], x["agent"], x["job_name"])):
            done = "✅" if r["finished_at"] else "🔄"
            f.write(
                f"| `{r['job_name']}` | {r['agent']} | {r['benchmark']} | "
                f"{r['n_total']} | {r['n_completed']} | {r['n_errored']} | "
                f"{r['pass_at_1']:.0%} | {r['finished_at'] or '—'} {done} |\n"
            )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    *,
    jobs_dir: Path = JOBS_DIR,
    output_dir: Path = LOGS_DIR,
    include: list[str] | None = None,
) -> dict:
    """Aggregate all jobs → write summary-<ts>.{md,csv,json}. Returns paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_jobs(jobs_dir, include=include)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md = output_dir / f"summary-{ts}.md"
    csv_p = output_dir / f"summary-{ts}.csv"
    json_p = output_dir / f"summary-{ts}.json"
    _write_markdown(rows, md)
    _write_csv(rows, csv_p)
    json_p.write_text(json.dumps(rows, indent=2, default=str))
    return {"rows": rows, "md": md, "csv": csv_p, "json": json_p}


def watch(job_name: str, *, poll_sec: int = 30) -> dict:
    """Live-watch a job until finished. Yields status dicts (one per poll)."""
    from .launcher import wait_for_job
    return wait_for_job(job_name, timeout_min=24 * 60, poll_sec=poll_sec)