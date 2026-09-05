#!/usr/bin/env python3
"""
summarize.py — 扫 jobs/<name>/result.json → pass@1 by (bench, agent) 表

输出:
  logs/summary-<timestamp>.md
  logs/summary-<timestamp>.csv
  logs/summary-<timestamp>.json  (raw)

移植自 mini_matrix/scripts/summarize.py,适配 PR #410 runtime 容器内场景:
  - 读 harbor 0.20.x 输出的 result.json schema
  - 写到 /opt/swebench/logs/(host bind mount 可读)
  - 算法一致(stats.evals.reward_stats.reward → reward=1.0 的 trial 数)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_jobs(jobs_dir: Path, include: list[str] | None = None):
    """Load all jobs/<name>/result.json + config.json. Optionally filter by name substring."""
    rows = []
    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        result = job_dir / "result.json"
        if not result.exists():
            continue
        if include and not any(inc in job_dir.name for inc in include):
            continue
        try:
            data = json.loads(result.read_text())
        except json.JSONDecodeError:
            print(f"[warn] {result} 不是有效 JSON,跳过", file=sys.stderr)
            continue

        # Try to load config.json (harbor 0.20.x 与 result.json 分开)
        config = {}
        cfg_path = job_dir / "config.json"
        if cfg_path.exists():
            try:
                config = json.loads(cfg_path.read_text())
            except json.JSONDecodeError:
                pass
        rows.append(parse_job(job_dir.name, data, config))
    return rows


def parse_job(job_name: str, data: dict, config: dict) -> dict:
    """Extract benchmark / agent / reward from harbor JobResult."""
    agents = config.get("agents", [])
    tasks = config.get("tasks", [])
    datasets = config.get("datasets", [])

    # Single agent (矩阵常见情况)
    agent_name = agents[0]["name"] if agents else "?"

    # Task or dataset path → benchmark 标识
    if tasks:
        bench = Path(tasks[0]["path"]).name
    elif datasets:
        bench = Path(datasets[0]["path"]).name
    else:
        bench = "?"

    stats = data.get("stats", {})
    n_total = stats.get("n_total_trials", 0)
    n_completed = stats.get("n_completed_trials", 0)
    n_errored = stats.get("n_errored_trials", 0)
    n_running = stats.get("n_running_trials", 0)
    n_pending = stats.get("n_pending_trials", 0)
    finished = data.get("finished_at")

    # Reward from evals: reward_stats.reward is {reward_value: [trial_ids]}
    evals = stats.get("evals", {})
    rewards = []
    for eval_key, eval_data in evals.items():
        reward_buckets = eval_data.get("reward_stats", {}).get("reward", {})
        for r_str, trial_ids in reward_buckets.items():
            try:
                r_val = float(r_str)
            except (ValueError, TypeError):
                continue
            n_with_reward = len(trial_ids) if isinstance(trial_ids, list) else 1
            rewards.extend([r_val] * n_with_reward)

    n_pass = sum(1 for r in rewards if r == 1.0)
    pass_at_1 = n_pass / max(len(rewards), 1) if rewards else 0.0

    return {
        "job_name": job_name,
        "agent": agent_name,
        "benchmark": bench,
        "n_total": n_total,
        "n_completed": n_completed,
        "n_errored": n_errored,
        "n_running": n_running,
        "n_pending": n_pending,
        "n_pass": n_pass,
        "pass_at_1": pass_at_1,
        "finished_at": finished,
        "started_at": data.get("started_at"),
    }


def write_markdown(rows: list[dict], path: Path):
    """Write a Markdown summary."""
    # group by (bench, agent)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["benchmark"], r["agent"])].append(r)

    with path.open("w") as f:
        f.write("# Multi-Bench × Multi-Agent Summary\n\n")
        f.write(f"_Generated: {datetime.now().isoformat()}_\n\n")
        f.write(f"**Total jobs**: {len(rows)}\n\n")

        # Pass@1 by (bench, agent)
        f.write("## Pass@1 by (Benchmark, Agent)\n\n")
        f.write("| Benchmark | Agent | n_jobs | n_pass | pass@1 |\n")
        f.write("|---|---|---|---|---|\n")
        for (bench, agent), items in sorted(grouped.items()):
            n_jobs = len(items)
            n_pass = sum(it["n_pass"] for it in items)
            p1 = n_pass / max(n_jobs, 1)
            f.write(f"| {bench} | {agent} | {n_jobs} | {n_pass} | {p1:.1%} |\n")

        # Detailed rows
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


def write_csv(rows: list[dict], path: Path):
    """Write CSV summary."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", required=True, type=Path)
    p.add_argument("--include", default=None,
                   help="Comma-separated substrings to filter jobs")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    include = [s.strip() for s in (args.include or "").split(",") if s.strip()] or None
    rows = load_jobs(args.jobs_dir, include=include)
    if not rows:
        print("[summarize] No jobs found.", file=sys.stderr)
        sys.exit(0)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = args.output_dir / f"summary-{ts}.md"
    csv_path = args.output_dir / f"summary-{ts}.csv"
    json_path = args.output_dir / f"summary-{ts}.json"

    write_markdown(rows, md_path)
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2, default=str))

    print(f"[summarize] {len(rows)} jobs summarized")
    print(f"  → {md_path}")
    print(f"  → {csv_path}")
    print(f"  → {json_path}")


if __name__ == "__main__":
    main()