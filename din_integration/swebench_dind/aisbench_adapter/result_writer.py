"""Translate harbor ``result.json`` → AISBench-compatible schema.

AISBench (via HarborSummarizer / AgentContribSummarizer) reads
``<work_dir>/results/<model_abbr>/<dataset_abbr>.json`` with the schema
documented in ``SCHEMA_DOC``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_DOC = """
AISBench results/<model>/<dataset>.json schema:
{
  "total_count":       int,        # total number of trials
  "n_errors":          int,        # trials that errored out
  "avg_score":         float,      # mean reward (4 decimals)
  "reward_keys":       list[str],  # optional, for multi-key verifiers
  "per_key_avg_score": dict,       # optional
  "reward_distribution": [{"score": float, "count": int}, ...],
  "exception_distribution": [{"exception_type": str, "count": int}, ...],
  "n_total_trials":    int,
  "pass_at_k":         dict[int, float]   # optional
}
"""


def read_harbor_result(path: Path) -> dict[str, Any]:
    """Read a harbor ``result.json``."""
    return json.loads(path.read_text())


def _flatten_rewards(evals: dict[str, Any]) -> list[float]:
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
    return rewards


def write_result(
    harbor_result_path: Path,
    output_path: Path,
    *,
    model_abbr: str,
    dataset_abbr: str,
) -> Path:
    """Convert harbor result.json → AISBench result JSON. Returns output path."""
    data = read_harbor_result(harbor_result_path)
    stats = data.get("stats", {})
    evals = stats.get("evals", {})
    rewards = _flatten_rewards(evals)
    n_total = stats.get("n_total_trials", 0)
    n_errors = stats.get("n_errored_trials", 0)
    avg_score = round(sum(rewards) / max(len(rewards), 1), 4) if rewards else 0.0

    # Reward distribution
    reward_counter = Counter(rewards)
    reward_distribution = [
        {"score": float(s), "count": int(c)} for s, c in sorted(reward_counter.items())
    ]

    # Exception distribution
    exception_counter: Counter[str] = Counter()
    for _, eval_data in evals.items():
        for ex_type, trial_ids in eval_data.get("exception_stats", {}).items():
            n = len(trial_ids) if isinstance(trial_ids, list) else 1
            exception_counter[ex_type] += n
    exception_distribution = [
        {"exception_type": t, "count": int(c)} for t, c in exception_counter.most_common()
    ]

    # pass@1
    pass_at_k = {}
    if rewards:
        pass_at_k[1] = round(sum(1 for r in rewards if r == 1.0) / len(rewards), 4)

    out = {
        "total_count": n_total,
        "n_errors": n_errors,
        "avg_score": avg_score,
        "reward_keys": ["reward"] if rewards else [],
        "per_key_avg_score": {"reward": avg_score} if rewards else {},
        "reward_distribution": reward_distribution,
        "exception_distribution": exception_distribution,
        "n_total_trials": n_total,
        "pass_at_k": pass_at_k,
        # AISBench extras (not used by HarborSummarizer but useful for debugging)
        "_meta": {
            "model_abbr": model_abbr,
            "dataset_abbr": dataset_abbr,
            "harbor_finished_at": data.get("finished_at"),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return output_path


def write_results_for_all_jobs(
    jobs_dir: Path,
    output_dir: Path,
    *,
    model_abbr: str = "default-model",
    dataset_abbr_finder=None,
) -> list[Path]:
    """Iterate ``jobs_dir/*/result.json`` and write AISBench result files.

    ``dataset_abbr_finder(job_dir) -> str`` lets callers derive the
    AISBench dataset abbr from the job directory (default: job_dir.name).
    """
    outputs: list[Path] = []
    finder = dataset_abbr_finder or (lambda p: p.name)
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        result = job_dir / "result.json"
        if not result.exists():
            continue
        ds = finder(job_dir)
        out = output_dir / model_abbr / f"{ds}.json"
        try:
            write_result(result, out, model_abbr=model_abbr, dataset_abbr=ds)
            outputs.append(out)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [skip] {job_dir.name}: {e}")
    return outputs