"""Harbor monitor: periodic on-disk status snapshots + HTTP status service.

The monitor builds a two-level snapshot for every running harbor task:

- task-level: subprocess status (from the AISBench ``status_tmp`` file) plus
  harbor ``JobStats`` progress read from ``job_dir/result.json``;
- case-level: per-trial execution status and success/failure reasons derived
  from the on-disk ``trial_*/`` files (layout per harbor 0.21.0
  ``harbor.models.trial.paths.TrialPaths``):

    trial_*/result.json            TrialResult (exception_info, rewards, timings)
    trial_*/exception.txt          exception message text
    trial_*/verifier/reward.json   reward detail
    trial_*/verifier/test-stdout.txt / test-stderr.txt
    trial_*/verifier/ctrf.json     per-test-case pass/fail + message
    trial_*/agent/trajectory.json  agent trajectory presence
    trial_*/trial.log              trial log tail

Only stdlib is used so the monitor carries no extra dependencies. File reads
are cached by mtime so the refresh loop stays cheap on large jobs.
"""

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

CANCELLED_ERROR_TYPE = "CancelledError"

_DEFAULT_TAIL_LINES = 50
_DEFAULT_TAIL_CHARS = 4096


def _tail(path: Path, max_lines: int = _DEFAULT_TAIL_LINES, max_chars: int = _DEFAULT_TAIL_CHARS) -> str | None:
    """Return the tail of a text file, truncated, or None if unreadable."""
    if not Path(path).exists():
        return None
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return None
    text = "\n".join(lines[-max_lines:])
    return text[-max_chars:] or None


def _read_json(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _fromiso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class HarborMonitor:
    """Thread-safe monitor that aggregates per-task snapshots from disk."""

    def __init__(self, work_dir: str, refresh_interval: float = 0.5) -> None:
        self.work_dir = Path(work_dir)
        self.refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = {}
        self._snapshots: dict[str, dict] = {}
        self._job_result_cache: dict[str, tuple[float, dict]] = {}
        self._case_cache: dict[str, tuple[float, dict]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def register_task(self, task_name: str, status_file: str, job_dir: str | None = None) -> None:
        with self._lock:
            self._tasks[task_name] = {
                "status_file": status_file,
                "job_dir": job_dir,
            }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:
                # a scan failure must never kill the monitor thread
                pass
            self._stop.wait(self.refresh_interval)

    # ------------------------------------------------------------------
    # snapshot access (thread-safe)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        with self._lock:
            for task_name, info in list(self._tasks.items()):
                self._snapshots[task_name] = self._build_task_snapshot(task_name, info)

    def snapshot(self, task_name: str | None = None) -> list[dict] | dict | None:
        with self._lock:
            if task_name is not None:
                return self._snapshots.get(task_name)
            return list(self._snapshots.values())

    def cases(self, task_name: str) -> list[dict]:
        snap = self.snapshot(task_name)
        if not isinstance(snap, dict):
            return []
        return snap.get("cases") or []

    def jobs(self) -> list[dict]:
        jobs = []
        for snap in self.snapshot() or []:
            harbor = snap.get("harbor") or {}
            cases = snap.get("cases") or []
            status_counts: dict[str, int] = {}
            for case in cases:
                status = case.get("status")
                status_counts[status] = status_counts.get(status, 0) + 1
            jobs.append(
                {
                    "task_name": snap.get("task_name"),
                    "job_dir": snap.get("job_dir"),
                    "harbor": harbor,
                    "case_status_counts": status_counts,
                }
            )
        return jobs

    # ------------------------------------------------------------------
    # task-level snapshot
    # ------------------------------------------------------------------

    def _build_task_snapshot(self, task_name: str, info: dict) -> dict:
        status_file = info.get("status_file")
        job_dir = info.get("job_dir")
        snap: dict = {
            "task_name": task_name,
            "status": "not start",
            "process_id": None,
            "finish_count": None,
            "total_count": None,
            "progress_description": None,
            "start_time": None,
            "log_path": None,
            "job_dir": str(job_dir) if job_dir else None,
            "harbor": None,
            "cases": [],
            "extra": {},
        }
        if status_file and Path(status_file).exists():
            snap.update(self._read_status_file(status_file))
        if job_dir and Path(job_dir).is_dir():
            harbor_snap, cases = self._scan_job_dir(Path(job_dir))
            snap["harbor"] = harbor_snap
            snap["cases"] = cases
        return snap

    def _read_status_file(self, status_file: str) -> dict:
        """The status file is a JSON list of status dicts appended over time;
        merge them in order so the latest write wins per field."""
        try:
            data = json.loads(Path(status_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return {}
        if not isinstance(data, list):
            return {}
        merged: dict = {}
        for entry in data:
            if isinstance(entry, dict):
                merged.update(entry)
        return merged

    # ------------------------------------------------------------------
    # harbor job dir scanning
    # ------------------------------------------------------------------

    def _scan_job_dir(self, job_dir: Path) -> tuple[dict | None, list[dict]]:
        harbor_snap = None
        result_path = job_dir / "result.json"
        if result_path.exists():
            mtime = result_path.stat().st_mtime
            cached = self._job_result_cache.get(str(result_path))
            if cached is not None and cached[0] == mtime:
                harbor_snap = cached[1]
            else:
                data = _read_json(result_path)
                if data is not None:
                    stats = data.get("stats") or {}
                    harbor_snap = {
                        "n_total_trials": data.get("n_total_trials"),
                        "n_running_trials": stats.get("n_running_trials"),
                        "n_completed_trials": stats.get("n_completed_trials"),
                        "n_errored_trials": stats.get("n_errored_trials"),
                        "n_pending_trials": stats.get("n_pending_trials"),
                        "n_cancelled_trials": stats.get("n_cancelled_trials"),
                        "n_retries": stats.get("n_retries"),
                        "evals": stats.get("evals") or {},
                    }
                    self._job_result_cache[str(result_path)] = (mtime, harbor_snap)
        cases = self._scan_trials(job_dir)
        return harbor_snap, cases

    def _scan_trials(self, job_dir: Path) -> list[dict]:
        trial_dirs = sorted(p for p in job_dir.glob("trial_*") if p.is_dir())
        return [self._scan_trial(trial_dir) for trial_dir in trial_dirs]

    # ------------------------------------------------------------------
    # per-case (trial) snapshot
    # ------------------------------------------------------------------

    def _scan_trial(self, trial_dir: Path) -> dict:
        result_path = trial_dir / "result.json"
        if result_path.exists():
            mtime = result_path.stat().st_mtime
            cache_key = str(result_path)
            cached = self._case_cache.get(cache_key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            result = _read_json(result_path)
            if result is not None:
                case = self._case_from_result(trial_dir, result)
                self._case_cache[cache_key] = (mtime, case)
                return case
        return self._case_from_files(trial_dir)

    def _case_from_result(self, trial_dir: Path, result: dict) -> dict:
        exception_info = result.get("exception_info")
        verifier_result = result.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}

        status = "completed"
        exception = None
        if exception_info:
            exception = {
                "type": exception_info.get("exception_type"),
                "message": exception_info.get("exception_message"),
                "occurred_at": exception_info.get("occurred_at"),
            }
            if exception_info.get("exception_type") == CANCELLED_ERROR_TYPE:
                status = "cancelled"
            else:
                status = "errored"

        return {
            "trial_name": result.get("trial_name") or trial_dir.name,
            "task_name": result.get("task_name") or self._task_name_from_config(trial_dir),
            "status": status,
            "reward": rewards.get("reward") if isinstance(rewards, dict) else None,
            "rewards": rewards or None,
            "exception": exception,
            "verifier": self._verifier_info(trial_dir),
            "timings": self._timings(result),
            "agent": self._agent_info(result, trial_dir),
            "extra": {},
        }

    def _case_from_files(self, trial_dir: Path) -> dict:
        """Trial dir exists but result.json is not written yet (still running)."""
        exception = None
        exception_txt = trial_dir / "exception.txt"
        if exception_txt.exists():
            try:
                text = exception_txt.read_text(encoding="utf-8").strip()
                if text:
                    exception = {"type": "Exception", "message": text[:1000]}
            except (OSError, UnicodeDecodeError):
                pass
        return {
            "trial_name": trial_dir.name,
            "task_name": self._task_name_from_config(trial_dir),
            "status": "running",
            "reward": None,
            "rewards": None,
            "exception": exception,
            "verifier": self._verifier_info(trial_dir),
            "timings": {
                "started_at": _dir_mtime_iso(trial_dir),
                "finished_at": None,
                "agent_execution_sec": None,
                "verifier_sec": None,
            },
            "agent": {
                "name": None,
                "version": None,
                "model": None,
                "has_trajectory": (trial_dir / "agent" / "trajectory.json").exists(),
                "tokens": None,
            },
            "extra": {},
        }

    def _task_name_from_config(self, trial_dir: Path) -> str | None:
        data = _read_json(trial_dir / "config.json")
        if data is None:
            return None
        task = data.get("task") or {}
        path = task.get("path")
        if path:
            return Path(path).name
        return task.get("name")

    # ------------------------------------------------------------------
    # field builders
    # ------------------------------------------------------------------

    def _verifier_info(self, trial_dir: Path) -> dict:
        verifier_dir = trial_dir / "verifier"
        info: dict = {
            "has_reward_json": False,
            "reward_json": None,
            "stdout_tail": None,
            "stderr_tail": None,
            "ctrf": None,
        }
        reward_json = verifier_dir / "reward.json"
        if reward_json.exists():
            info["has_reward_json"] = True
            info["reward_json"] = _read_json(reward_json)
        info["stdout_tail"] = _tail(verifier_dir / "test-stdout.txt")
        info["stderr_tail"] = _tail(verifier_dir / "test-stderr.txt")
        info["ctrf"] = self._ctrf_summary(verifier_dir / "ctrf.json")
        return info

    @staticmethod
    def _ctrf_summary(ctrf_path: Path) -> dict | None:
        data = _read_json(ctrf_path)
        if data is None:
            return None
        results = data.get("results")
        if not isinstance(results, list):
            return None
        passed = failed = skipped = 0
        failures = []
        for item in results:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if status == "passed":
                passed += 1
            elif status == "failed":
                failed += 1
                failures.append(
                    {"name": item.get("name"), "message": item.get("message")}
                )
            elif status in ("skipped", "pending"):
                skipped += 1
        return {"passed": passed, "failed": failed, "skipped": skipped, "failures": failures}

    @staticmethod
    def _timings(result: dict) -> dict:
        def _duration(timing) -> float | None:
            if not isinstance(timing, dict):
                return None
            start = _fromiso(timing.get("started_at"))
            end = _fromiso(timing.get("finished_at"))
            if start is None or end is None:
                return None
            return round((end - start).total_seconds(), 3)

        return {
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "agent_execution_sec": _duration(result.get("agent_execution")),
            "verifier_sec": _duration(result.get("verifier")),
        }

    @staticmethod
    def _agent_info(result: dict, trial_dir: Path) -> dict:
        agent_info = result.get("agent_info") or {}
        model_info = agent_info.get("model_info") or {}
        return {
            "name": agent_info.get("name"),
            "version": agent_info.get("version"),
            "model": model_info.get("name"),
            "has_trajectory": (trial_dir / "agent" / "trajectory.json").exists(),
            "tokens": HarborMonitor._tokens_from_result(result),
        }

    @staticmethod
    def _tokens_from_result(result: dict) -> dict | None:
        """Aggregate token/cost totals (mirrors TrialResult.compute_token_cost_totals)."""
        contexts = []
        agent_result = result.get("agent_result")
        if isinstance(agent_result, dict):
            contexts.append(agent_result)
        step_results = result.get("step_results")
        if isinstance(step_results, list):
            for step in step_results:
                if isinstance(step, dict) and isinstance(step.get("agent_result"), dict):
                    contexts.append(step["agent_result"])
        if not contexts:
            return None
        n_input = n_cache = n_output = 0
        cost = 0.0
        for ctx in contexts:
            n_input += ctx.get("n_input_tokens") or 0
            n_cache += ctx.get("n_cache_tokens") or 0
            n_output += ctx.get("n_output_tokens") or 0
            cost += ctx.get("cost_usd") or 0.0
        return {"input": n_input, "cache": n_cache, "output": n_output, "cost_usd": cost}


def _dir_mtime_iso(trial_dir: Path) -> str | None:
    try:
        ts = trial_dir.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts).isoformat()


class HarborMonitorServer:
    """Stdlib HTTP server exposing live monitor snapshots.

    Read-only endpoints (no write support, no CORS)::

        GET /api/health
        GET /api/tasks
        GET /api/tasks/{task_name}
        GET /api/tasks/{task_name}/cases
        GET /api/jobs
    """

    def __init__(self, monitor: HarborMonitor, host: str = "127.0.0.1", port: int = 0) -> None:
        self.monitor = monitor
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int | None:
        """Start serving in a daemon thread. Returns the bound port, or None
        when the port is 0 (server disabled)."""
        if not self.port:
            return None

        monitor = self.monitor

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    if path == "/api/health":
                        self._json({"status": "ok"})
                    elif path == "/api/tasks":
                        self._json({"tasks": monitor.snapshot() or []})
                    elif path == "/api/jobs":
                        self._json({"jobs": monitor.jobs()})
                    elif path.startswith("/api/tasks/"):
                        self._handle_task(path)
                    else:
                        self._json({"error": "not found"}, status=404)
                except Exception:  # noqa: BLE001
                    self._json({"error": "internal error"}, status=500)

            def _handle_task(self, path: str) -> None:
                name = unquote(path[len("/api/tasks/") :])
                if name.endswith("/cases"):
                    name = name[: -len("/cases")]
                    self._json({"task_name": name, "cases": monitor.cases(name)})
                    return
                snap = monitor.snapshot(name)
                if not isinstance(snap, dict):
                    self._json({"error": "task not found"}, status=404)
                    return
                self._json(snap)

            def _json(self, data, status: int = 200) -> None:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # noqa: ARG002
                pass

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
