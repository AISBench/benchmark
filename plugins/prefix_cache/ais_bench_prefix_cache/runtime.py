from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .artifacts import artifact_paths, validate_artifacts, write_json
from .errors import PrefixCacheError, RuntimeCapabilityError
from .metrics import MetricSnapshot, diff_metrics, metrics_to_dict, parse_metrics, snapshot_to_dict, summarize_kv_usage
from .pipeline import prepare_scenario
from .scenario import Scenario, load_scenario, new_execution_timestamp, with_execution_timestamp


class VLLMClient:
    """封装对 vLLM 推理服务/指标服务的 HTTP 调用。

    负责发送探活、正式 completion、抓取/重置缓存指标、warmup 预热等，
    支持按 DP rank 路由（通过 X-data-parallel-rank 请求头）。
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.config = scenario.section("service")
        self.timeout = float(self.config.get("timeout_seconds", 30))
        self.base_headers = {"Content-Type": "application/json"}
        if self.config.get("api_key"):
            self.base_headers["Authorization"] = f"Bearer {self.config['api_key']}"

    def _request(self, url: str, method: str = "GET", body: dict[str, Any] | None = None, dp_rank: int | None = None) -> bytes:
        """执行一次 HTTP 请求；可选携带 DP rank 头，失败统一转 RuntimeCapabilityError。"""
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
        """发送一条非流式 completion 请求并解析 JSON 响应。"""
        body = {"model": self.config["model"], "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": False}
        raw = self._request(self.config["inference_url"], "POST", body, dp_rank)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeCapabilityError("vLLM completion returned invalid JSON") from exc

    def snapshot(self) -> MetricSnapshot:
        """抓取当前 Prefix Cache 指标快照（用于基线/结束对比）。"""
        text = self._request(self.config["metrics_url"]).decode("utf-8")
        return parse_metrics(text, self.scenario.dp_size, self.config.get("engine_label_map"))

    def precheck(self) -> dict[str, Any]:
        """运行前能力探测：向每个 DP rank 发探针请求并确认指标可达。"""
        for rank in range(self.scenario.dp_size):
            self.send_completion(f"prefix-cache-capability-probe-{rank}", 1, rank if self.scenario.dp_size > 1 else None)
        snapshot = self.snapshot()
        return {"ok": True, "ranks": sorted(snapshot.by_rank), "metric_names": snapshot.metric_names}

    def reset(self) -> list[dict[str, Any]]:
        """重置服务端 Prefix Cache 计数器。

        若未配置 reset_url，则在 assume_empty_cache=true 时跳过并返回说明，
        否则视为能力缺失。
        """
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
        """按 warmup 计划预热：覆盖每个 Prefix Group × DP rank 组合，并记录耗时。

        校验计划是否完备（不重不漏），再依次发送预热请求把缓存前缀写满。
        """
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
    """读取 JSON 文件并解析为字典。"""
    return json.loads(path.read_text(encoding="utf-8"))


class _ConfigTypeRef:
    """Marker for a class reference; renders as an import alias expression."""

    def __init__(self, expression: str):
        self.expression = expression

    def __repr__(self) -> str:
        return self.expression


def _render_config_value(value: Any, imports: list[str], refs: dict[tuple[str, str], str]) -> Any:
    """把用户配置里的 Python 值递归渲染为可写文本（类引用转为 import 别名）。"""
    if isinstance(value, type):
        # 遇到类引用：登记 import 别名，返回 _ConfigTypeRef 以输出为引用表达式。
        key = (value.__module__, value.__qualname__)
        alias = refs.get(key)
        if alias is None:
            parts = value.__qualname__.split(".")
            alias = f"_ref{len(refs)}"
            imports.append(f"from {value.__module__} import {parts[0]} as {alias}")
            refs[key] = alias
        return _ConfigTypeRef(alias + "".join(f".{part}" for part in value.__qualname__.split(".")[1:]))
    if isinstance(value, dict):
        return {key: _render_config_value(item, imports, refs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_render_config_value(item, imports, refs) for item in value]
    return value


def render_aisbench_config(config_path: Path, scenario: Scenario) -> Path:
    """Execute the user's AISBench config as plain Python and render a static config file.

    AISBench loads configs through mmengine with lazy imports, so config files
    cannot call functions or read os.environ at parse time. Executing the user
    config here - with the plugin environment variables set - and writing a
    fully static file keeps user configs expressive and AISBench-compatible.
    """
    manifest_path = artifact_paths(scenario.output_dir, scenario.run_id).manifest
    env_values = {
        "AISBENCH_PREFIX_CACHE_SCENARIO": str(scenario.source_path),
        "AISBENCH_PREFIX_CACHE_MANIFEST": str(manifest_path),
    }
    work_dir = scenario.section("aisbench").get("work_dir")
    if work_dir:
        work_dir_path = Path(work_dir)
        if not work_dir_path.is_absolute():
            work_dir_path = (scenario.source_path.parent / work_dir_path).resolve()
        env_values["AISBENCH_PREFIX_CACHE_WORK_DIR"] = str(work_dir_path)
    previous = {key: os.environ.get(key) for key in env_values}
    os.environ.update(env_values)
    try:
        namespace: dict[str, Any] = {"__file__": str(config_path)}
        try:
            exec(compile(config_path.read_text(encoding="utf-8"), str(config_path), "exec"), namespace)
        except Exception as exc:
            raise PrefixCacheError(f"failed to execute AISBench config {config_path}: {exc}") from exc
        required = ("datasets", "models", "infer")
        missing = [key for key in required if key not in namespace]
        if missing:
            raise PrefixCacheError(f"AISBench config {config_path} is missing required keys: {', '.join(missing)}")
        imports: list[str] = []
        refs: dict[tuple[str, str], str] = {}
        rendered: dict[str, Any] = {key: _render_config_value(namespace[key], imports, refs) for key in required}
        extras = [
            key for key, value in namespace.items()
            if key not in required
            and key != "work_dir"
            and not key.startswith("_")
            and not isinstance(value, type)
            and not callable(value)
            and isinstance(value, (dict, list, tuple))
        ]
        rendered.update({key: _render_config_value(namespace[key], imports, refs) for key in extras})
        rendered["work_dir"] = _render_config_value(namespace.get("work_dir", "outputs/prefix_cache"), imports, refs)
        lines = ["# Generated by ais_bench_prefix_cache; do not edit.", *imports]
        lines.extend(f"{key} = {value!r}" for key, value in rendered.items())
        generated_dir = Path(tempfile.mkdtemp(prefix="aisbench-prefix-cache-config-"))
        generated = generated_dir / "config.py"
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return generated
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_aisbench_with_polling(
    command: list[str],
    env: dict[str, str],
    client: VLLMClient,
    poll_interval_seconds: float,
) -> tuple[int, list[tuple[float, dict[int, float | None]]]]:
    """以子进程运行 AISBench，期间周期性轮询 KV 用量。

    返回 (退出码, 样本列表)；样本为 (相对开始时间的秒数, {rank: 用量占比})。
    轮询尽力而为：单次抓取失败只跳过，不中断正式跑分。
    """
    process = subprocess.Popen(command, env=env)
    if poll_interval_seconds <= 0:
        return process.wait(), []
    started = time.perf_counter()
    samples: list[tuple[float, dict[int, float | None]]] = []
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        time.sleep(poll_interval_seconds)
        try:
            snapshot = client.snapshot()
        except RuntimeCapabilityError:
            continue
        elapsed = time.perf_counter() - started
        samples.append((elapsed, {rank: row.kv_cache_usage for rank, row in snapshot.by_rank.items()}))
    return returncode, samples


def run_scenario(
    scenario_path: Path | str,
    aisbench_config: Path | str | None = None,
    *,
    execution_timestamp: str | None = None,
    progress=None,
) -> dict[str, Any]:
    """执行一次完整的前缀缓存基准测试（precheck→reset→warmup→正式跑→对比）。

    先生成/校验工件，再抓取基线指标，然后以子进程方式运行 AISBench perf，
    结束后抓取指标求差，得出真实的 Prefix Cache 命中率并回写 analysis.json。
    """
    base_scenario = load_scenario(scenario_path)
    timestamp = execution_timestamp or new_execution_timestamp()
    scenario = with_execution_timestamp(base_scenario, timestamp)
    paths = artifact_paths(scenario.output_dir, scenario.run_id)
    manifest_path = paths.manifest
    analysis_path = paths.analysis
    # 若尚未 prepare 则自动补齐，再校验工件有效性。
    if not manifest_path.exists():
        prepare_scenario(
            base_scenario.source_path,
            progress=progress,
            execution_timestamp=timestamp,
        )
    validate_artifacts(manifest_path)
    manifest = _read_json(manifest_path)
    # 场景文件指纹不一致时：允许覆盖则重跑 prepare，否则报错提示用户。
    if manifest.get("scenario_sha256") != hashlib.sha256(scenario.source_path.read_bytes()).hexdigest():
        if base_scenario.section("run").get("overwrite"):
            prepare_scenario(
                base_scenario.source_path,
                overwrite=True,
                progress=progress,
                execution_timestamp=timestamp,
            )
            manifest = _read_json(manifest_path)
        else:
            raise PrefixCacheError("existing artifacts were generated from a different scenario; set run.overwrite=true or run prepare --overwrite")
    analysis = _read_json(analysis_path)
    runtime: dict[str, Any] = {"phases": []}
    client = VLLMClient(scenario)
    runtime["precheck"] = client.precheck()
    runtime["phases"].append("precheck")
    # 继承 prepare 阶段的理论告警，并叠加 reset 的说明。
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
    # 渲染静态 AISBench 配置，并以子进程运行 perf 模式。
    generated = render_aisbench_config(config_path, scenario)
    env = os.environ.copy()
    env["AISBENCH_PREFIX_CACHE_SCENARIO"] = str(scenario.source_path)
    env["AISBENCH_PREFIX_CACHE_MANIFEST"] = str(manifest_path)
    work_dir = scenario.section("aisbench").get("work_dir")
    if work_dir:
        work_dir_path = Path(work_dir)
        if not work_dir_path.is_absolute():
            work_dir_path = (scenario.source_path.parent / work_dir_path).resolve()
        env["AISBENCH_PREFIX_CACHE_WORK_DIR"] = str(work_dir_path)
    command = [sys.executable, "-m", "ais_bench.benchmark.cli.main", str(generated), "--mode", "perf"]
    command.extend(map(str, scenario.section("aisbench").get("extra_args", [])))
    poll_interval = float(scenario.section("service")["poll_interval_seconds"])
    returncode, kv_samples = run_aisbench_with_polling(command, env, client, poll_interval)
    runtime["aisbench_exit_code"] = returncode
    runtime["phases"].append("formal")
    if returncode != 0:
        raise PrefixCacheError(f"AISBench failed with exit code {returncode}")
    after = client.snapshot()
    runtime["metrics_after"] = snapshot_to_dict(after)
    runtime["phases"].append("after")
    actual = diff_metrics(baseline, after)
    actual_dict = metrics_to_dict(actual)
    # 跑分期间轮询采样的 KV 用量：峰值/均值合并进 actual，原始样本留在 runtime 供审计。
    kv_summary = summarize_kv_usage([row for _, row in kv_samples])
    runtime["kv_cache_polling"] = {
        "interval_seconds": poll_interval,
        "count": len(kv_samples),
        "summary": kv_summary,
        "samples": [
            {"elapsed_seconds": round(elapsed, 3), "by_dp": {str(rank): value for rank, value in row.items()}}
            for elapsed, row in kv_samples
        ],
    }
    for rank, stats in kv_summary["by_dp"].items():
        actual_dict["by_dp"][rank]["kv_cache_usage_peak"] = stats["peak"]
        actual_dict["by_dp"][rank]["kv_cache_usage_avg"] = stats["avg"]
    actual_dict["global_kv_cache_usage_peak"] = kv_summary["global_peak"]
    actual_dict["global_kv_cache_usage_avg"] = kv_summary["global_avg"]
    theory_rate = float(analysis["theoretical_hit_rate"])
    # 真实命中率与理论命中率做差（百分点），超阈值则追加 ACTUAL_DEVIATION 告警。
    signed_difference_pp = ((actual.global_hit_rate or 0.0) - theory_rate) * 100 if actual.global_hit_rate is not None else None
    difference_pp = abs(signed_difference_pp) if signed_difference_pp is not None else None
    if difference_pp is not None and difference_pp > scenario.section("validation")["actual_warning_pp"]:
        warnings.append({"code": "ACTUAL_DEVIATION", "difference_pp": difference_pp})
    validation = dict(analysis.get("validation", {}))
    validation["actual_status"] = "PASS" if difference_pp is None or difference_pp <= scenario.section("validation")["actual_warning_pp"] else "PASS_WITH_WARNING"
    validation["status"] = "PASS" if not warnings else "PASS_WITH_WARNING"
    analysis.update({
        "status": "complete",
        "runtime": runtime,
        "actual": actual_dict,
        "theory_actual_difference_pp": difference_pp,
        "theory_actual_signed_difference_pp": signed_difference_pp,
        "theory_actual_absolute_difference_pp": difference_pp,
        "validation": validation,
        "warnings": warnings,
    })
    write_json(analysis_path, analysis, overwrite=True)
    return analysis


def analyze_snapshots(manifest_path: Path | str, baseline_path: Path | str, after_path: Path | str) -> dict[str, Any]:
    """离线分析：直接用两个 Prometheus 指标文件对比真实命中率并回写 analysis.json。

    不发送任何请求，只读 manifest 与指标文件，适合事后复算或 CI 校验。
    """
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
    signed_difference_pp = ((actual.global_hit_rate or 0.0) - theory_rate) * 100 if actual.global_hit_rate is not None else None
    difference_pp = abs(signed_difference_pp) if signed_difference_pp is not None else None
    warnings = list(analysis.get("warnings", []))
    if difference_pp is not None and difference_pp > effective["validation"]["actual_warning_pp"]:
        warnings.append({"code": "ACTUAL_DEVIATION", "difference_pp": difference_pp})
    validation = dict(analysis.get("validation", {}))
    validation["actual_status"] = "PASS" if difference_pp is None or difference_pp <= effective["validation"]["actual_warning_pp"] else "PASS_WITH_WARNING"
    validation["status"] = "PASS" if not warnings else "PASS_WITH_WARNING"
    analysis.update({
        "status": "analyzed",
        "runtime": {
            "metrics_baseline": snapshot_to_dict(baseline),
            "metrics_after": snapshot_to_dict(after),
        },
        "actual": metrics_to_dict(actual),
        "theory_actual_difference_pp": difference_pp,
        "theory_actual_signed_difference_pp": signed_difference_pp,
        "theory_actual_absolute_difference_pp": difference_pp,
        "validation": validation,
        "warnings": warnings,
    })
    write_json(analysis_path, analysis, overwrite=True)
    return analysis
