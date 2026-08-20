from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .artifacts import read_jsonl, validate_artifacts, write_json
from .errors import PrefixCacheError, RuntimeCapabilityError
from .metrics import MetricSnapshot, diff_metrics, metrics_to_dict, parse_metrics, snapshot_to_dict
from .pipeline import prepare_scenario
from .scenario import Scenario, load_scenario


class VLLMClient:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.config = scenario.section("service")
        self.timeout = float(self.config.get("timeout_seconds", 30))
        self.base_headers = {"Content-Type": "application/json"}
        if self.config.get("api_key"):
            self.base_headers["Authorization"] = f"Bearer {self.config['api_key']}"

    def _request(self, url: str, method: str = "GET", body: dict[str, Any] | None = None, dp_rank: int | None = None) -> bytes:
        headers = dict(self.base_headers)
        if dp_rank is not None:
            headers["X-data-parallel-rank"] = str(dp_rank)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeCapabilityError(f"vLLM request failed: {method} {url}: {exc}") from exc

    def send_completion(self, prompt: str, max_tokens: int, dp_rank: int | None = None) -> dict[str, Any]:
        body = {"model": self.config["model"], "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": False}
        raw = self._request(self.config["inference_url"], "POST", body, dp_rank)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeCapabilityError("vLLM completion returned invalid JSON") from exc

    def snapshot(self) -> MetricSnapshot:
        text = self._request(self.config["metrics_url"]).decode("utf-8")
        return parse_metrics(text, self.scenario.dp_size, self.config.get("engine_label_map"))

    def precheck(self) -> dict[str, Any]:
        for rank in range(self.scenario.dp_size):
            self.send_completion(f"prefix-cache-capability-probe-{rank}", 1, rank if self.scenario.dp_size > 1 else None)
        snapshot = self.snapshot()
        return {"ok": True, "ranks": sorted(snapshot.by_rank), "metric_names": snapshot.metric_names}

    def reset(self) -> list[dict[str, Any]]:
        reset_url = self.config.get("reset_url")
        if not reset_url:
            if self.config.get("assume_empty_cache"):
                return [{"code": "ASSUME_EMPTY_CACHE", "message": "reset_url is not configured"}]
            raise RuntimeCapabilityError("reset_url is required unless assume_empty_cache=true")
        try:
            self._request(reset_url, "POST", {})
            return []
        except RuntimeCapabilityError:
            if self.config.get("assume_empty_cache"):
                return [{"code": "ASSUME_EMPTY_CACHE", "message": "reset failed; continuing by explicit configuration"}]
            raise

    def warm_every_group_rank(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        expected = {(item["group_id"], item["dp_rank"]) for item in plan}
        required = {(group, rank) for group in sorted({item["group_id"] for item in plan}) for rank in range(self.scenario.dp_size)}
        if expected != required:
            raise RuntimeCapabilityError("warmup plan does not cover every Prefix Group × DP rank")
        for item in sorted(plan, key=lambda value: (value["group_id"], value["dp_rank"])):
            started = time.perf_counter()
            rank = int(item["dp_rank"]) if self.scenario.dp_size > 1 else None
            self.send_completion(item["prompt"], int(item.get("max_tokens", 1)), rank)
            results.append({"group_id": item["group_id"], "dp_rank": item["dp_rank"], "success": True, "elapsed_seconds": time.perf_counter() - started})
        return results


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_scenario(scenario_path: Path | str, aisbench_config: Path | str | None = None) -> dict[str, Any]:
    scenario = load_scenario(scenario_path)
    manifest_path = scenario.output_dir / f"{scenario.run_id}.manifest.json"
    analysis_path = scenario.output_dir / f"{scenario.run_id}.analysis.json"
    if not manifest_path.exists():
        prepare_scenario(scenario.source_path)
    validate_artifacts(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("scenario_sha256") != hashlib.sha256(scenario.source_path.read_bytes()).hexdigest():
        if scenario.section("run").get("overwrite"):
            prepare_scenario(scenario.source_path, overwrite=True)
            manifest = _read_json(manifest_path)
        else:
            raise PrefixCacheError("existing artifacts were generated from a different scenario; set run.overwrite=true or run prepare --overwrite")
    analysis = _read_json(analysis_path)
    runtime: dict[str, Any] = {"phases": []}
    client = VLLMClient(scenario)
    runtime["precheck"] = client.precheck()
    runtime["phases"].append("precheck")
    warnings = list(analysis.get("warnings", []))
    warnings.extend(client.reset())
    runtime["phases"].append("reset")
    if scenario.cache_mode == "warmup":
        runtime["warmup"] = client.warm_every_group_rank(manifest["warmup"]["plan"])
        runtime["phases"].append("warmup")
    baseline = client.snapshot()
    runtime["metrics_baseline"] = snapshot_to_dict(baseline)
    runtime["phases"].append("baseline")
    config_value = aisbench_config or scenario.section("aisbench").get("config")
    if not config_value:
        raise RuntimeCapabilityError("AISBench config path is required for run")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = (scenario.source_path.parent / config_path).resolve()
    env = os.environ.copy()
    env["AISBENCH_PREFIX_CACHE_SCENARIO"] = str(scenario.source_path)
    work_dir = scenario.section("aisbench").get("work_dir")
    if work_dir:
        work_dir_path = Path(work_dir)
        if not work_dir_path.is_absolute():
            work_dir_path = (scenario.source_path.parent / work_dir_path).resolve()
        env["AISBENCH_PREFIX_CACHE_WORK_DIR"] = str(work_dir_path)
    command = [sys.executable, "-m", "ais_bench.benchmark.cli.main", str(config_path), "--mode", "perf"]
    command.extend(map(str, scenario.section("aisbench").get("extra_args", [])))
    completed = subprocess.run(command, check=False, env=env)
    runtime["aisbench_exit_code"] = completed.returncode
    runtime["phases"].append("formal")
    if completed.returncode != 0:
        raise PrefixCacheError(f"AISBench failed with exit code {completed.returncode}")
    after = client.snapshot()
    runtime["metrics_after"] = snapshot_to_dict(after)
    runtime["phases"].append("after")
    actual = diff_metrics(baseline, after)
    actual_dict = metrics_to_dict(actual)
    theory_rate = float(analysis["theoretical_hit_rate"])
    difference_pp = abs((actual.global_hit_rate or 0.0) - theory_rate) * 100 if actual.global_hit_rate is not None else None
    if difference_pp is not None and difference_pp > scenario.section("validation")["actual_warning_pp"]:
        warnings.append({"code": "ACTUAL_DEVIATION", "difference_pp": difference_pp})
    analysis.update({"status": "complete", "runtime": runtime, "actual": actual_dict, "theory_actual_difference_pp": difference_pp, "warnings": warnings})
    write_json(analysis_path, analysis, overwrite=True)
    return analysis


def analyze_snapshots(manifest_path: Path | str, baseline_path: Path | str, after_path: Path | str) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    validate_artifacts(manifest_file)
    manifest = _read_json(manifest_file)
    effective = manifest["effective_config"]
    service = effective["service"]
    baseline = parse_metrics(Path(baseline_path).read_text(encoding="utf-8"), int(service["dp_size"]), service.get("engine_label_map"))
    after = parse_metrics(Path(after_path).read_text(encoding="utf-8"), int(service["dp_size"]), service.get("engine_label_map"))
    actual = diff_metrics(baseline, after)
    analysis_path = manifest_file.parent / manifest["artifacts"]["analysis"]["name"]
    analysis = _read_json(analysis_path)
    theory_rate = float(analysis["theoretical_hit_rate"])
    difference_pp = abs((actual.global_hit_rate or 0.0) - theory_rate) * 100 if actual.global_hit_rate is not None else None
    warnings = list(analysis.get("warnings", []))
    if difference_pp is not None and difference_pp > effective["validation"]["actual_warning_pp"]:
        warnings.append({"code": "ACTUAL_DEVIATION", "difference_pp": difference_pp})
    analysis.update({
        "status": "analyzed",
        "runtime": {
            "metrics_baseline": snapshot_to_dict(baseline),
            "metrics_after": snapshot_to_dict(after),
        },
        "actual": metrics_to_dict(actual),
        "theory_actual_difference_pp": difference_pp,
        "warnings": warnings,
    })
    write_json(analysis_path, analysis, overwrite=True)
    return analysis
