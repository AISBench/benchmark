from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .artifacts import artifact_paths, validate_artifacts, write_json
from .errors import PrefixCacheError, RuntimeCapabilityError
from .metrics import MetricSnapshot, diff_metrics, metrics_to_dict, parse_metrics, snapshot_to_dict, summarize_kv_usage
from .pipeline import prepare_scenario
from .scenario import Scenario, load_scenario, new_execution_timestamp, with_execution_timestamp


logger = logging.getLogger(__name__)


def _safe_url(value: str | None) -> str | None:
    """Return a log-safe URL without credentials, query parameters, or fragments."""
    if not value:
        return value
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid-url>"


def _safe_command(command: list[str]) -> list[str]:
    """Redact values of command-line options that commonly carry credentials."""
    sensitive = ("key", "token", "secret", "password", "authorization")
    result: list[str] = []
    redact_next = False
    for raw in command:
        value = str(raw)
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if "=" in value:
            name, _ = value.split("=", 1)
            if any(marker in name.lower() for marker in sensitive):
                result.append(f"{name}=<redacted>")
                continue
        result.append(value)
        if value.startswith("-") and any(marker in value.lower() for marker in sensitive):
            redact_next = True
    return result


def _snapshot_summary(snapshot: MetricSnapshot) -> dict[str, Any]:
    """Build a compact snapshot summary without logging raw Prometheus text."""
    return {
        "metric_names": snapshot.metric_names,
        "by_dp": {
            str(rank): {
                "queries": row.queries,
                "hits": row.hits,
                "kv_cache_usage": row.kv_cache_usage,
            }
            for rank, row in snapshot.by_rank.items()
        },
    }


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
        logger.info(
            "[runtime] VLLMClient initialized inference_url=%s metrics_url=%s reset_url=%s model=%s dp_size=%d timeout_seconds=%.3f api_key_configured=%s",
            _safe_url(self.config.get("inference_url")),
            _safe_url(self.config.get("metrics_url")),
            _safe_url(self.config.get("reset_url")),
            self.config.get("model"),
            scenario.dp_size,
            self.timeout,
            bool(self.config.get("api_key")),
        )

    def _request(self, url: str, method: str = "GET", body: dict[str, Any] | None = None, dp_rank: int | None = None) -> bytes:
        """执行一次 HTTP 请求；可选携带 DP rank 头，失败统一转 RuntimeCapabilityError。"""
        headers = dict(self.base_headers)
        if dp_rank is not None:
            headers["X-data-parallel-rank"] = str(dp_rank)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        started = time.perf_counter()
        logger.info(
            "[runtime] HTTP request start method=%s url=%s dp_rank=%s body_keys=%s body_bytes=%s",
            method,
            _safe_url(url),
            dp_rank,
            sorted(body) if body is not None else [],
            len(data) if data is not None else 0,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                logger.info(
                    "[runtime] HTTP request complete method=%s url=%s dp_rank=%s status=%s response_bytes=%d elapsed_seconds=%.6f",
                    method,
                    _safe_url(url),
                    dp_rank,
                    getattr(response, "status", None),
                    len(raw),
                    time.perf_counter() - started,
                )
                return raw
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.error(
                "[runtime] HTTP request failed method=%s url=%s dp_rank=%s elapsed_seconds=%.6f error_type=%s",
                method,
                _safe_url(url),
                dp_rank,
                time.perf_counter() - started,
                type(exc).__name__,
            )
            raise RuntimeCapabilityError(
                f"vLLM request failed: {method} {_safe_url(url)}: {type(exc).__name__}"
            ) from exc

    def send_completion(self, prompt: str, max_tokens: int, dp_rank: int | None = None) -> dict[str, Any]:
        """发送一条非流式 completion 请求并解析 JSON 响应。"""
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        logger.info(
            "[runtime] completion start dp_rank=%s prompt_chars=%d prompt_sha256=%s max_tokens=%d",
            dp_rank,
            len(prompt),
            prompt_sha256,
            max_tokens,
        )
        body = {"model": self.config["model"], "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": False}
        raw = self._request(self.config["inference_url"], "POST", body, dp_rank)
        try:
            result = json.loads(raw)
            logger.info(
                "[runtime] completion complete dp_rank=%s prompt_sha256=%s response_keys=%s",
                dp_rank,
                prompt_sha256,
                sorted(result) if isinstance(result, dict) else [],
            )
            return result
        except json.JSONDecodeError as exc:
            logger.error(
                "[runtime] completion invalid_json dp_rank=%s prompt_sha256=%s response_bytes=%d",
                dp_rank,
                prompt_sha256,
                len(raw),
            )
            raise RuntimeCapabilityError("vLLM completion returned invalid JSON") from exc

    def snapshot(self) -> MetricSnapshot:
        """抓取当前 Prefix Cache 指标快照（用于基线/结束对比）。"""
        logger.info("[runtime] metrics snapshot start url=%s", _safe_url(self.config["metrics_url"]))
        text = self._request(self.config["metrics_url"]).decode("utf-8")
        snapshot = parse_metrics(text, self.scenario.dp_size, self.config.get("engine_label_map"))
        logger.info("[runtime] metrics snapshot complete summary=%s", _snapshot_summary(snapshot))
        return snapshot

    def precheck(self) -> dict[str, Any]:
        """运行前能力探测：向每个 DP rank 发探针请求并确认指标可达。"""
        logger.info("[runtime] phase=precheck start dp_size=%d", self.scenario.dp_size)
        for rank in range(self.scenario.dp_size):
            logger.info("[runtime] phase=precheck probe rank=%d routed=%s", rank, self.scenario.dp_size > 1)
            self.send_completion(f"prefix-cache-capability-probe-{rank}", 1, rank if self.scenario.dp_size > 1 else None)
        snapshot = self.snapshot()
        result = {"ok": True, "ranks": sorted(snapshot.by_rank), "metric_names": snapshot.metric_names}
        logger.info("[runtime] phase=precheck complete result=%s", result)
        return result

    def reset(self) -> list[dict[str, Any]]:
        """重置服务端 Prefix Cache 计数器。

        若未配置 reset_url，则在 assume_empty_cache=true 时跳过并返回说明，
        否则视为能力缺失。
        """
        reset_url = self.config.get("reset_url")
        logger.info(
            "[runtime] phase=reset start reset_url=%s assume_empty_cache=%s",
            _safe_url(reset_url),
            self.config.get("assume_empty_cache"),
        )
        if not reset_url:
            if self.config.get("assume_empty_cache"):
                result = [{"code": "ASSUME_EMPTY_CACHE", "message": "reset_url is not configured"}]
                logger.warning("[runtime] phase=reset skipped warnings=%s", result)
                return result
            raise RuntimeCapabilityError("reset_url is required unless assume_empty_cache=true")
        try:
            self._request(reset_url, "POST", {})
            logger.info("[runtime] phase=reset complete")
            return []
        except RuntimeCapabilityError:
            if self.config.get("assume_empty_cache"):
                result = [{"code": "ASSUME_EMPTY_CACHE", "message": "reset failed; continuing by explicit configuration"}]
                logger.warning("[runtime] phase=reset failed_but_allowed warnings=%s", result)
                return result
            raise

    def warm_every_group_rank(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 warmup 计划预热：覆盖每个 Prefix Group × DP rank 组合，并记录耗时。

        校验计划是否完备（不重不漏），再依次发送预热请求把缓存前缀写满。
        """
        logger.info("[runtime] phase=warmup start plan_items=%d dp_size=%d", len(plan), self.scenario.dp_size)
        results = []
        expected = {(item["group_id"], item["dp_rank"]) for item in plan}
        required = {(group, rank) for group in sorted({item["group_id"] for item in plan}) for rank in range(self.scenario.dp_size)}
        if expected != required:
            logger.error("[runtime] phase=warmup invalid_plan expected=%s required=%s", sorted(expected), sorted(required))
            raise RuntimeCapabilityError("warmup plan does not cover every Prefix Group × DP rank")
        for item in sorted(plan, key=lambda value: (value["group_id"], value["dp_rank"])):
            started = time.perf_counter()
            rank = int(item["dp_rank"]) if self.scenario.dp_size > 1 else None
            logger.info(
                "[runtime] phase=warmup request_start group_id=%s dp_rank=%s routed_rank=%s input_tokens=%s shared_prefix_tokens=%s max_tokens=%s",
                item["group_id"],
                item["dp_rank"],
                rank,
                item.get("input_tokens"),
                item.get("shared_prefix_tokens"),
                item.get("max_tokens", 1),
            )
            self.send_completion(item["prompt"], int(item.get("max_tokens", 1)), rank)
            result = {"group_id": item["group_id"], "dp_rank": item["dp_rank"], "success": True, "elapsed_seconds": time.perf_counter() - started}
            results.append(result)
            logger.info("[runtime] phase=warmup request_complete result=%s", result)
        logger.info("[runtime] phase=warmup complete succeeded=%d", len(results))
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
    logger.info(
        "[runtime] render_aisbench_config start source_config=%s scenario=%s manifest=%s run_id=%s",
        config_path,
        scenario.source_path,
        manifest_path,
        scenario.run_id,
    )
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
            logger.exception("[runtime] render_aisbench_config execute_failed source_config=%s", config_path)
            raise PrefixCacheError(f"failed to execute AISBench config {config_path}: {exc}") from exc
        required = ("datasets", "models", "infer")
        missing = [key for key in required if key not in namespace]
        if missing:
            logger.error("[runtime] render_aisbench_config missing_keys=%s", missing)
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
        logger.info(
            "[runtime] render_aisbench_config complete generated=%s datasets=%d models=%d extra_sections=%s imports=%d work_dir=%s",
            generated,
            len(rendered.get("datasets", [])),
            len(rendered.get("models", [])),
            extras,
            len(imports),
            rendered.get("work_dir"),
        )
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
    log_path: Path | None = None,
) -> tuple[int, list[tuple[float, dict[int, float | None]]]]:
    """以子进程运行 AISBench，期间周期性轮询 KV 用量。

    返回 (退出码, 样本列表)；样本为 (相对开始时间的秒数, {rank: 用量占比})。
    轮询尽力而为：单次抓取失败只跳过，不中断正式跑分。log_path 存在时，
    AISBench 子进程 stdout/stderr 全部追加写入该文件，不回显到 CLI 终端。
    """
    logger.info(
        "[runtime] phase=formal launch command=%s poll_interval_seconds=%.3f child_log_path=%s",
        _safe_command(command),
        poll_interval_seconds,
        log_path,
    )
    child_log = None
    popen_kwargs: dict[str, Any] = {"env": env}
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_log = log_path.open("a", encoding="utf-8")
        popen_kwargs.update({"stdout": child_log, "stderr": subprocess.STDOUT})
    try:
        process = subprocess.Popen(command, **popen_kwargs)
        logger.info("[runtime] phase=formal process_started pid=%s", getattr(process, "pid", None))
        if poll_interval_seconds <= 0:
            returncode = process.wait()
            logger.info("[runtime] phase=formal process_complete exit_code=%d kv_polling=disabled", returncode)
            return returncode, []
        started = time.perf_counter()
        samples: list[tuple[float, dict[int, float | None]]] = []
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(poll_interval_seconds)
            try:
                snapshot = client.snapshot()
            except RuntimeCapabilityError as exc:
                logger.warning("[runtime] phase=formal kv_poll_failed elapsed_seconds=%.3f error=%s", time.perf_counter() - started, exc)
                continue
            elapsed = time.perf_counter() - started
            sample = {rank: row.kv_cache_usage for rank, row in snapshot.by_rank.items()}
            samples.append((elapsed, sample))
            logger.info(
                "[runtime] phase=formal kv_poll sample_index=%d elapsed_seconds=%.3f by_dp=%s",
                len(samples),
                elapsed,
                sample,
            )
        logger.info("[runtime] phase=formal process_complete exit_code=%d kv_samples=%d", returncode, len(samples))
        return returncode, samples
    finally:
        if child_log is not None:
            child_log.close()


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
    logger.info(
        "[runtime] run_scenario start scenario_path=%s aisbench_config_override=%s execution_timestamp=%s",
        scenario_path,
        aisbench_config,
        execution_timestamp,
    )
    base_scenario = load_scenario(scenario_path)
    timestamp = execution_timestamp or new_execution_timestamp()
    scenario = with_execution_timestamp(base_scenario, timestamp)
    paths = artifact_paths(scenario.output_dir, scenario.run_id)
    manifest_path = paths.manifest
    analysis_path = paths.analysis
    request_cfg = scenario.section("requests")
    pc_cfg = scenario.section("prefix_cache")
    service_cfg = scenario.section("service")
    logger.info(
        "[runtime] execution_context run_id=%s timestamp=%s output_dir=%s result_dir=%s cache_mode=%s target_hit_rate=%.6f requests=%d block_size=%d dp_size=%d inference_url=%s metrics_url=%s reset_url=%s assume_empty_cache=%s",
        scenario.run_id,
        timestamp,
        scenario.output_dir,
        manifest_path.parent,
        scenario.cache_mode,
        float(pc_cfg["target_hit_rate"]),
        int(request_cfg["count"]),
        scenario.block_size,
        scenario.dp_size,
        _safe_url(service_cfg.get("inference_url")),
        _safe_url(service_cfg.get("metrics_url")),
        _safe_url(service_cfg.get("reset_url")),
        service_cfg.get("assume_empty_cache"),
    )
    # 若尚未 prepare 则自动补齐，再校验工件有效性。
    if not manifest_path.exists():
        logger.info("[runtime] artifacts missing action=auto_prepare manifest=%s", manifest_path)
        prepare_scenario(
            base_scenario.source_path,
            progress=progress,
            execution_timestamp=timestamp,
        )
    else:
        logger.info("[runtime] artifacts found action=reuse manifest=%s", manifest_path)
    validation_result = validate_artifacts(manifest_path)
    logger.info("[runtime] artifacts validated result=%s", validation_result)
    manifest = _read_json(manifest_path)
    logger.info(
        "[runtime] manifest loaded status=%s run_id=%s groups=%s warmup_enabled=%s artifact_keys=%s",
        manifest.get("status"),
        manifest.get("run_id"),
        sorted(manifest.get("groups", {})),
        manifest.get("warmup", {}).get("enabled"),
        sorted(manifest.get("artifacts", {})),
    )
    # 场景文件指纹不一致时：允许覆盖则重跑 prepare，否则报错提示用户。
    if manifest.get("scenario_sha256") != hashlib.sha256(scenario.source_path.read_bytes()).hexdigest():
        logger.warning("[runtime] scenario_sha256 mismatch manifest=%s overwrite=%s", manifest_path, base_scenario.section("run").get("overwrite"))
        if base_scenario.section("run").get("overwrite"):
            prepare_scenario(
                base_scenario.source_path,
                overwrite=True,
                progress=progress,
                execution_timestamp=timestamp,
            )
            manifest = _read_json(manifest_path)
            logger.info("[runtime] artifacts regenerated after scenario mismatch manifest=%s", manifest_path)
        else:
            raise PrefixCacheError("existing artifacts were generated from a different scenario; set run.overwrite=true or run prepare --overwrite")
    analysis = _read_json(analysis_path)
    runtime: dict[str, Any] = {"phases": []}
    client = VLLMClient(scenario)
    logger.info("[runtime] phase=precheck dispatch")
    runtime["precheck"] = client.precheck()
    runtime["phases"].append("precheck")
    # 继承 prepare 阶段的理论告警，并叠加 reset 的说明。
    warnings = list(analysis.get("warnings", []))
    logger.info("[runtime] prepare_analysis loaded theoretical_hit_rate=%s existing_warnings=%s", analysis.get("theoretical_hit_rate"), warnings)
    warnings.extend(client.reset())
    runtime["phases"].append("reset")
    if scenario.cache_mode == "warmup":
        logger.info("[runtime] phase=warmup dispatch plan_items=%d", len(manifest["warmup"]["plan"]))
        runtime["warmup"] = client.warm_every_group_rank(manifest["warmup"]["plan"])
        runtime["phases"].append("warmup")
    else:
        logger.info("[runtime] phase=warmup skipped cache_mode=%s", scenario.cache_mode)
    logger.info("[runtime] phase=baseline start completed_phases=%s", runtime["phases"])
    baseline = client.snapshot()
    runtime["metrics_baseline"] = snapshot_to_dict(baseline)
    runtime["phases"].append("baseline")
    logger.info("[runtime] phase=baseline complete summary=%s", _snapshot_summary(baseline))
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
    logger.info(
        "[runtime] phase=formal prepared source_config=%s generated_config=%s work_dir=%s extra_args=%s poll_interval_seconds=%.3f",
        config_path,
        generated,
        work_dir_path if work_dir else None,
        _safe_command(list(map(str, scenario.section("aisbench").get("extra_args", [])))),
        poll_interval,
    )
    run_log_path = scenario.output_dir / "log" / f"{scenario.run_id}.run.log"
    returncode, kv_samples = run_aisbench_with_polling(
        command,
        env,
        client,
        poll_interval,
        log_path=run_log_path,
    )
    runtime["aisbench_exit_code"] = returncode
    runtime["phases"].append("formal")
    if returncode != 0:
        logger.error("[runtime] phase=formal failed exit_code=%d", returncode)
        raise PrefixCacheError(f"AISBench failed with exit code {returncode}")
    logger.info("[runtime] phase=after start")
    after = client.snapshot()
    runtime["metrics_after"] = snapshot_to_dict(after)
    runtime["phases"].append("after")
    logger.info("[runtime] phase=after complete summary=%s", _snapshot_summary(after))
    actual = diff_metrics(baseline, after)
    actual_dict = metrics_to_dict(actual)
    logger.info("[runtime] actual_metrics delta=%s", actual_dict)
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
    logger.info("[runtime] kv_cache_polling summary=%s", kv_summary)
    theory_rate = float(analysis["theoretical_hit_rate"])
    # 真实命中率与理论命中率做差（百分点），超阈值则追加 ACTUAL_DEVIATION 告警。
    signed_difference_pp = ((actual.global_hit_rate or 0.0) - theory_rate) * 100 if actual.global_hit_rate is not None else None
    difference_pp = abs(signed_difference_pp) if signed_difference_pp is not None else None
    if difference_pp is not None and difference_pp > scenario.section("validation")["actual_warning_pp"]:
        warnings.append({"code": "ACTUAL_DEVIATION", "difference_pp": difference_pp})
        logger.warning(
            "[runtime] actual deviation theory_rate=%.6f actual_rate=%s signed_difference_pp=%s absolute_difference_pp=%s threshold_pp=%s",
            theory_rate,
            actual.global_hit_rate,
            signed_difference_pp,
            difference_pp,
            scenario.section("validation")["actual_warning_pp"],
        )
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
    logger.info(
        "[runtime] run_scenario complete status=%s phases=%s actual_hit_rate=%s theory_hit_rate=%.6f difference_pp=%s warnings=%s analysis_path=%s",
        analysis["status"],
        runtime["phases"],
        actual.global_hit_rate,
        theory_rate,
        difference_pp,
        warnings,
        analysis_path,
    )
    return analysis


def analyze_snapshots(manifest_path: Path | str, baseline_path: Path | str, after_path: Path | str) -> dict[str, Any]:
    """离线分析：直接用两个 Prometheus 指标文件对比真实命中率并回写 analysis.json。

    不发送任何请求，只读 manifest 与指标文件，适合事后复算或 CI 校验。
    """
    logger.info(
        "[runtime] analyze_snapshots start manifest=%s baseline=%s after=%s",
        manifest_path,
        baseline_path,
        after_path,
    )
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
    logger.info(
        "[runtime] analyze_snapshots complete status=%s actual_hit_rate=%s theory_hit_rate=%.6f difference_pp=%s warnings=%s analysis_path=%s",
        analysis["status"],
        actual.global_hit_rate,
        theory_rate,
        difference_pp,
        warnings,
        analysis_path,
    )
    return analysis
