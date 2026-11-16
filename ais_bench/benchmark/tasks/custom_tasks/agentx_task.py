"""AgentX (InferenceX AgentX 1.0) 性能测评任务。

Trace DAG、会话回放、warmup 和 profiling 全部由独立环境中的
``agentx-harness==1.0.13`` 完成；本模块只负责配置、数据溯源、进程和结果。
"""

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import platform
import re
import select
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mmengine.config import Config
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.tasks.custom_tasks.agentx_spec import (
    DATASETS,
    normalize_base_url,
    resolve_cooldown,
    resolve_endpoint,
    resolve_variant,
    validate_agentx_args,
)
from ais_bench.benchmark.tasks.custom_tasks.agentx_result import (
    HARNESS_BRAND_PREFIX_RE,
    HARNESS_SUMMARY_CSV,
    HARNESS_SUMMARY_JSON,
    SUPPORTED_HARNESS_VERSION,
    SUPPORTED_RAW_SCHEMA,
    atomic_write_json,
    normalize_agentx_result,
    redact_text,
)
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg
from ais_bench.benchmark.utils.logging import AISLogger

AGENTX_PACKAGE = "agentx-harness"
AGENTX_SCENARIO = "agentx"
DEFAULT_HARNESS_VERSION = SUPPORTED_HARNESS_VERSION
IMAGE_HARNESS_PYTHON = "/opt/venvs/agentx113/bin/python"
SMOKE_DEFAULT_DURATION = 60.0
SMOKE_DEFAULT_TRACE_LIMIT = 4
# agentx scenario preset 的官方默认口径；manifest 用于记录未显式覆盖时的生效值。
PRESET_BENCHMARK_DURATION = 3600.0
PRESET_WARMUP_REQUESTS_PER_LANE = 10
PRESET_RANDOM_SEED = 42
ENGINE_PROBE_TIMEOUT = 5.0
MANIFEST_SCHEMA_VERSION = "1.0"

_REDACT_NEXT_FLAGS = ("--api-key",)
_REDACT_HEADER_NAMES = ("authorization", "proxy-authorization", "x-api-key", "x-auth-token")
_SECRET_ENV_RE = re.compile(r"(?:key|token|secret|password|credential|authorization)", re.I)
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")

PROCESS_TIMEOUT_ENV = "AGENTX_PROCESS_TIMEOUT"


def _process_timeout(env: Dict[str, str]) -> Optional[float]:
    """读取 Harness 看门狗超时（秒）；未设置时返回 None 表示不启用。"""
    raw = (env or {}).get(PROCESS_TIMEOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{PROCESS_TIMEOUT_ENV} must be a positive number of seconds, got {raw!r}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{PROCESS_TIMEOUT_ENV} must be a positive number of seconds, got {raw!r}.")
    return value


def _sha256_and_line_count(path: str) -> Tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    last_byte = b""
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        lines += 1
    return digest.hexdigest(), lines


def resolve_smoke_overrides(
    run_mode: str, duration: Optional[float], trace_limit: Optional[int]
) -> Tuple[Optional[float], Optional[int]]:
    """smoke 未显式传值时补短跑默认值；benchmark 只透传显式值，否则用 harness 预设。"""
    if run_mode != "smoke":
        return duration, trace_limit
    return (
        SMOKE_DEFAULT_DURATION if duration is None else duration,
        SMOKE_DEFAULT_TRACE_LIMIT if trace_limit is None else trace_limit,
    )


def _split_extra_args(extra_args: Optional[str]) -> List[str]:
    """把用户写的一段 agentx 原始参数拆成 argv 片段；只做分词，不解释参数含义。"""
    if not extra_args:
        return []
    text = extra_args.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"Invalid AgentX extra_args: {exc}") from exc


def _option_name(token: str) -> Optional[str]:
    if not token.startswith("-") or token in ("-", "--"):
        return None
    return token.split("=", 1)[0]


def _apply_extra_args(command: List[str], extra_tokens: List[str]) -> List[str]:
    """extra args 优先级最高：移除基础命令中的同名选项后，再整体追加到末尾。"""
    overridden = {name for name in map(_option_name, extra_tokens) if name}
    if not overridden:
        return command + extra_tokens
    merged: List[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        name = _option_name(token)
        if name in overridden:
            index += 1
            # 与 _argv_parameters 同一口径：下一个 token 不是选项才视为本选项的取值。
            # 无值开关（如 --unsafe-override）因此不会连带吞掉后面的选项，
            # 新增无值开关时也无需维护名单。
            if "=" not in token and index < len(command) and _option_name(command[index]) is None:
                index += 1
            continue
        merged.append(token)
        index += 1
    return merged + extra_tokens


def _argv_parameters(argv: List[str]) -> Dict[str, Any]:
    """从最终 argv 还原各选项的实际取值，写入 manifest 供审计。"""
    parameters: Dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        name = _option_name(token)
        if name is None:
            index += 1
            continue
        if "=" in token:
            parameters[name] = token.split("=", 1)[1]
            index += 1
            continue
        if index + 1 < len(argv) and _option_name(argv[index + 1]) is None:
            parameters[name] = argv[index + 1]
            index += 2
            continue
        parameters[name] = True
        index += 1
    return parameters


def build_agentx_command(
    agentx_bin: List[str], *, url: Optional[str] = None, model: Optional[str] = None,
    concurrency: int, dataset_alias: Optional[str] = None, artifact_dir: str,
    dataset_dir: Optional[str] = None, endpoint: Optional[str] = None,
    tokenizer_path: Optional[str] = None, tokenizer_trust_remote_code: bool = False,
    api_key: Optional[str] = None, extra_args: Optional[str] = None,
    run_mode: str = "benchmark", duration: Optional[float] = None,
    trace_limit: Optional[int] = None, raw: bool = False,
) -> List[str]:
    """唯一的真实/dry-run 命令生成器；benchmark 默认沿用 preset，显式 duration 透传。

    ``raw`` 为真时不要求基础参数齐全：url/model/语料全部由 ``extra_args`` 描述。
    """
    command = list(agentx_bin) + ["profile", "--scenario", "agentx"]
    if url:
        endpoint = resolve_endpoint(endpoint)
        command += ["--url", normalize_base_url(url, endpoint), "--endpoint", endpoint]
    if model:
        command += ["--model", model]
    command += [
        "--concurrency", str(concurrency), "--output-artifact-dir", artifact_dir,
    ]
    if dataset_dir:
        command += ["--hf-weka-dataset", dataset_dir, "--unsafe-override"]
    elif dataset_alias:
        command += ["--public-dataset", dataset_alias]
    elif not raw:
        raise ValueError("AgentX public dataset alias is required when dataset_dir is absent.")
    if tokenizer_path:
        command += ["--tokenizer", osp.expanduser(tokenizer_path)]
        if tokenizer_trust_remote_code:
            command += ["--tokenizer-trust-remote-code"]
    if api_key:
        command += ["--api-key", api_key]
    if duration is not None:
        command += ["--benchmark-duration", str(duration)]
    if run_mode == "smoke":
        if "--unsafe-override" not in command:
            command.append("--unsafe-override")
        if trace_limit is not None:
            command += ["--num-dataset-entries", str(trace_limit)]
    if extra_args:
        command = _apply_extra_args(command, _split_extra_args(extra_args))
    return command


def redact_command(command: List[str]) -> List[str]:
    redacted: List[str] = []
    redact_next = False
    for token in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if token in _REDACT_NEXT_FLAGS:
            # 旗标后若没有取值（位于末尾），不能凭空补一个 <redacted> 值。
            redacted.append(token)
            redact_next = True
            continue
        matched_secret = False
        for flag in _REDACT_NEXT_FLAGS:
            if token.startswith(flag + "="):
                redacted.append(flag + "=<redacted>")
                matched_secret = True
                break
        if matched_secret:
            continue
        if ":" in token:
            name, _, _ = token.partition(":")
            if name.strip().lower() in _REDACT_HEADER_NAMES:
                redacted.append(f"{name}:<redacted>")
                continue
        redacted.append(re.sub(r"\$\{[^}]*\}", "<redacted>", token))
    return redacted


def resolve_harness_python(args: Dict[str, Any], environ: Optional[Dict[str, str]] = None) -> str:
    """Resolve the explicitly supplied interpreter or the image/current interpreter."""
    if environ is None:
        environ = os.environ
    explicit_python = args.get("agentx_python") or environ.get("AISBENCH_AGENTX_PYTHON")
    if explicit_python:
        return osp.expanduser(explicit_python)
    if osp.isfile(IMAGE_HARNESS_PYTHON):
        return IMAGE_HARNESS_PYTHON
    return sys.executable


def probe_harness_environment(python_exe: str, env: Dict[str, str],
                              agentx_bin: Optional[str] = None) -> Dict[str, Any]:
    """在目标解释器内检查 Python、distribution、版本及 aiperf 冲突。"""
    code = (
        "import json,sys,importlib.metadata as m;"
        "print(json.dumps({'python_version':sys.version.split()[0],"
        "'agentx_version':m.version('agentx-harness'),"
        "'upstream_aiperf':any((d.metadata.get('Name') or '').lower()=='aiperf' "
        "for d in m.distributions())}))"
    )
    try:
        completed = subprocess.run([python_exe, "-c", code], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, timeout=120, env=env,
                                   encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"AGENTX_ENV_INVALID: cannot query {python_exe!r}: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"AGENTX_ENV_INVALID: agentx-harness=={DEFAULT_HARNESS_VERSION} is unavailable "
            f"in {python_exe!r}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        metadata = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"AGENTX_ENV_INVALID: invalid environment probe output: {completed.stdout!r}") from exc
    major, minor = tuple(int(item) for item in metadata["python_version"].split(".")[:2])
    if (major, minor) not in ((3, 11), (3, 12), (3, 13)):
        raise RuntimeError(
            f"AGENTX_ENV_INVALID: Harness requires Python 3.11~3.13, "
            f"found {metadata['python_version']}."
        )
    if metadata["agentx_version"] != DEFAULT_HARNESS_VERSION:
        raise RuntimeError(
            f"AGENTX_VERSION_UNSUPPORTED: required agentx-harness=={DEFAULT_HARNESS_VERSION}, "
            f"found {metadata['agentx_version']}. Interpreter: {python_exe}. No benchmark process was started."
        )
    if metadata.get("upstream_aiperf"):
        raise RuntimeError(
            "AGENTX_ENV_INVALID: upstream PyPI aiperf conflicts with agentx-harness; "
            "use an isolated environment."
        )
    command = resolve_agentx_bin(agentx_bin, python_exe)
    try:
        help_result = subprocess.run(command + ["--help"], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, timeout=30, env=env,
                                     encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"AGENTX_ENV_INVALID: agentx entry self-check failed: {exc}") from exc
    if help_result.returncode != 0:
        raise RuntimeError(f"AGENTX_ENV_INVALID: agentx entry self-check failed: {help_result.stdout[-2000:]}")
    return dict(metadata, distribution=AGENTX_PACKAGE, version=metadata["agentx_version"],
                python=python_exe, entry=command, platform=platform.platform())


def resolve_agentx_bin(agentx_bin: Optional[str], python_exe: Optional[str]) -> List[str]:
    if agentx_bin:
        # 校验阶段按 expanduser 判定可执行文件存在，执行阶段必须同样展开。
        return [osp.expanduser(agentx_bin)]
    candidates: List[str] = []
    if python_exe:
        bin_dir = Path(python_exe).resolve().parent
        candidates.append(str(bin_dir / "agentx"))
    for candidate in candidates:
        if osp.isfile(candidate):
            return [candidate]
    if python_exe:
        return [python_exe, "-m", "agentx_cli"]
    raise RuntimeError("Cannot locate agentx; set agentx_bin or agentx_python.")


def _safe_component(value: Any) -> str:
    component = _SAFE_COMPONENT_RE.sub("_", str(value or "unknown")).strip("._-")
    return component[:80] or "unknown"


def _screen_line(text: str) -> str:
    """控制台回显与 agentx_stdout 日志落盘前去掉上游品牌前缀；原始产物不受影响。"""
    return HARNESS_BRAND_PREFIX_RE.sub("", text)


def run_dir_name(variant: str, concurrency: int) -> str:
    """每个 batch size一个全新目录：``agentx_<variant>_batchsize_<N>``。"""
    return f"agentx_{_safe_component(variant)}_batchsize_{int(concurrency)}"


def _http_json(url: str, timeout: float = ENGINE_PROBE_TIMEOUT) -> Tuple[Optional[Any], Optional[str]]:
    """读取引擎探测端点；失败只记录错误，不中断测评。"""
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as exc:
        # IncompleteRead 等 HTTPException 逃逸会吞掉整档结果，探测失败一律降级为错误记录。
        return None, str(exc)


def _extra_arg_secrets(extra_args: Optional[str]) -> List[str]:
    """把 extra args 中已知敏感选项（如 ``--api-key``）的取值纳入脱敏名单。"""
    tokens = _split_extra_args(extra_args)
    secrets: List[str] = []
    for index, token in enumerate(tokens):
        for flag in _REDACT_NEXT_FLAGS:
            if token == flag and index + 1 < len(tokens):
                secrets.append(tokens[index + 1])
            elif token.startswith(flag + "="):
                secrets.append(token.split("=", 1)[1])
    return [secret for secret in secrets if secret]


@TASKS.register_module()
class AgentXTask(BaseTask):
    """一次 Task 对应一个并发档，并逐档串行执行。"""

    name_prefix = "AgentXTask"
    log_subdir = "logs"
    _PROGRESS_INTERVAL = 2.0

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # 支持配置文件的单层 datasets，以及 NaivePartitioner 历史嵌套形状。
        if isinstance(self.dataset_cfgs, dict):
            self.dataset_cfgs = [self.dataset_cfgs]
        self.dataset_cfg = self.dataset_cfgs[0]
        self.args: Dict[str, Any] = dict(self.dataset_cfg.get("args") or {})
        self.task_state_manager: Optional[TaskStateManager] = None
        self._last_state_ts = 0.0

    def get_command(self, cfg_path, template) -> str:
        sys.path.append(os.getcwd())
        return f"{sys.executable} {__file__} {cfg_path}"

    def run(self, task_state_manager: Optional[TaskStateManager] = None) -> Dict[str, Any]:
        self.task_state_manager = task_state_manager
        plans = self._prepare(clear_existing=self._reuse_enabled())
        results: List[Dict[str, Any]] = []
        for index, plan in enumerate(plans):
            self._update_state({"status": "start", "stage": "agentx", "run_id": plan["run_id"]})
            result = self._execute_plan(plan)
            self.logger.info(
                "AgentX run_id=%s concurrency=%s status=%s parse=%s valid=%s scope=%s manifest=%s",
                plan["run_id"], plan["concurrency"], result["run"]["status"],
                result["run"]["parse_status"], result["run"]["benchmark_valid"],
                result["run"]["comparison_scope"], plan["manifest_path"],
            )
            results.append(result)
            if result["run"]["status"] != "completed" or result["run"]["parse_status"] != "success":
                self._update_state({"status": "error", "run_id": plan["run_id"]})
                message = (
                    f"AgentX run {plan['run_id']} failed: {result['run']['status']}/"
                    f"{result['run']['parse_status']}; see {result['artifacts']['harness_log']}"
                )
                skipped = [item["concurrency"] for item in plans[index + 1:]]
                if skipped:
                    message += f"; skipped concurrency levels {skipped} after failure"
                raise RuntimeError(message)
            if index + 1 < len(plans) and plan["cooldown"] > 0:
                self.logger.info(
                    "AgentX cooldown %.3fs before next concurrency level", plan["cooldown"]
                )
                self._adapter_log(
                    plan, f"cooldown {plan['cooldown']:g}s before next concurrency level"
                )
                time.sleep(plan["cooldown"])
        self._update_state({"status": "finish", "run_id": plans[-1]["run_id"]})
        return results[0] if len(results) == 1 else {"runs": results}

    def dry_run(self) -> Dict[str, Any]:
        """完成离线校验并写脱敏计划；不启动 Harness、不访问模型服务。"""
        plans = self._prepare()
        payload = {"runs": [self._redacted_plan(plan) for plan in plans]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        self._adapter_log(
            plans[0],
            f"dry-run completed for {len(plans)} concurrency level(s); "
            "no Harness process or model request was started",
        )
        return payload

    def _reuse_enabled(self) -> bool:
        """--reuse 在 AgentX 模式下只复用时间戳目录，重跑结果覆盖原产物。"""
        return bool((self.cfg.get("cli_args") or {}).get("reuse"))

    def _prepare(self, clear_existing: bool = False) -> List[Dict[str, Any]]:
        levels = validate_agentx_args(self.model_cfg, self.args)
        cooldown = resolve_cooldown(self.args.get("cooldown"))
        run_mode = self.args.get("run_mode") or "benchmark"
        duration, trace_limit = resolve_smoke_overrides(
            run_mode, self.args.get("duration"), self.args.get("trace_limit")
        )
        dataset = self._resolve_dataset(run_mode)
        if dataset.get("num_dataset_entries") is None and trace_limit is not None:
            dataset["num_dataset_entries"] = trace_limit
        harness_env = self._harness_env()
        try:
            # 非法超时属于配置错误，必须在启动 Harness 前按 AGENTX_CONFIG_INVALID 报出，
            # 不能落到进程启动失败分类里。
            _process_timeout(harness_env)
        except ValueError as exc:
            raise RuntimeError(f"AGENTX_CONFIG_INVALID: {exc}") from exc
        harness_python = resolve_harness_python(self.args, harness_env)
        try:
            harness = probe_harness_environment(harness_python, harness_env, self.args.get("agentx_bin"))
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}\nAgentX environment search order: agentx_python/AISBENCH_AGENTX_PYTHON, "
                f"image interpreter {IMAGE_HARNESS_PYTHON}, current interpreter. "
                "Install agentx-harness==1.0.13 in an isolated Python 3.11~3.13 "
                "environment and set --agentx-python to its interpreter."
            ) from exc
        api_key = (self.args.get("api_key") or self.model_cfg.get("api_key")
                   or os.environ.get("AISBENCH_AGENTX_API_KEY"))
        secrets = [api_key] if api_key else []
        secrets.extend(
            value for key, value in (self.args.get("env") or {}).items()
            if _SECRET_ENV_RE.search(key) and value
        )
        secrets.extend(_extra_arg_secrets(self.args.get("extra_args")))
        plans = []
        for concurrency in levels:
            # raw 模式的任务目录固定用 raw 命名；真实语料口径记录在 dataset 里。
            dir_variant = "raw" if self.args.get("raw_mode") else dataset["variant"]
            run_id = run_dir_name(dir_variant, concurrency)
            # 根目录固定 configs/、logs/、performances/，任务以 run_id 分子目录。
            run_dir = osp.abspath(osp.join(self.work_dir, "performances", run_id))
            logs_dir = osp.abspath(osp.join(self.work_dir, "logs", run_id))
            default_artifact_dir = osp.abspath(osp.join(run_dir, "agentx_raw"))
            command = build_agentx_command(
                harness["entry"], url=self.model_cfg.get("url") or self.args.get("url"),
                model=self.model_cfg.get("model") or "", endpoint=self.model_cfg.get("endpoint"),
                concurrency=concurrency, dataset_alias=dataset.get("alias"),
                dataset_dir=dataset.get("path"), artifact_dir=default_artifact_dir,
                tokenizer_path=self.model_cfg.get("path"),
                tokenizer_trust_remote_code=self.model_cfg.get("trust_remote_code", False),
                api_key=api_key, extra_args=self.args.get("extra_args"), run_mode=run_mode,
                duration=duration, trace_limit=trace_limit,
                raw=bool(self.args.get("raw_mode")),
            )
            # 结果收集以最终命令里的 --output-artifact-dir 为准（extra args 可覆盖默认值）。
            parameters = _argv_parameters(command)
            artifact_dir = self._effective_artifact_dir(parameters)
            if clear_existing:
                # reuse 采用覆盖语义：不删除目录，只清理上一轮结果文件，避免旧结果被当成本轮产物。
                self._clear_previous_results(run_dir, artifact_dir)
            elif osp.exists(artifact_dir) and os.listdir(artifact_dir):
                raise RuntimeError(
                    f"AGENTX_OUTPUT_WRITE_FAILED: artifact directory is not empty: {artifact_dir}"
                )
            for directory in (logs_dir, run_dir, artifact_dir):
                mkdir_or_exist(directory)
            plan = {
                "run_id": run_id, "run_mode": run_mode, "concurrency": concurrency,
                "duration": duration, "trace_limit": trace_limit, "cooldown": cooldown,
                "argv": command, "env": harness_env, "cwd": osp.abspath(self.work_dir),
                "run_dir": run_dir, "artifact_dir": artifact_dir,
                "performance_dir": run_dir, "logs_dir": logs_dir,
                "summary_path": osp.join(artifact_dir, HARNESS_SUMMARY_JSON),
                "summary_csv_path": osp.join(artifact_dir, HARNESS_SUMMARY_CSV),
                "log_path": osp.join(logs_dir, "agentx_stdout.log"),
                "adapter_log": osp.join(logs_dir, "agentx_adapter.log"),
                "plan_path": osp.join(run_dir, "execution_plan.redacted.json"),
                "manifest_path": osp.join(run_dir, "manifest.json"),
                "harness": harness, "dataset": dataset, "api_key": api_key,
                "parameters": parameters, "secrets": secrets,
            }
            atomic_write_json(plan["plan_path"], self._redacted_plan(plan))
            self._adapter_log(
                plan,
                f"prepared run_id={run_id} concurrency={concurrency} "
                f"command={' '.join(redact_command(command))}",
            )
            plans.append(plan)
        return plans

    def _redacted_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        # 只记录用户显式传入的环境，不把宿主全部环境落盘。
        public_keys = set((self.args.get("env") or {}).keys())
        public_env = {}
        for key in public_keys:
            if key in plan["env"]:
                public_env[key] = plan["env"][key] if not _SECRET_ENV_RE.search(key) else "<redacted>"
        harness = {key: value for key, value in plan["harness"].items() if key != "entry"}
        harness["entry"] = redact_command(plan["harness"]["entry"])
        return {
            "run_id": plan["run_id"], "run_mode": plan["run_mode"],
            "concurrency": plan["concurrency"], "cooldown": plan["cooldown"],
            "duration": plan["duration"], "trace_limit": plan["trace_limit"],
            "argv": redact_command(plan["argv"]),
            "env": public_env, "cwd": plan["cwd"], "run_dir": plan["run_dir"],
            "artifact_dir": plan["artifact_dir"],
            "summary_path": plan["summary_path"], "summary_csv_path": plan["summary_csv_path"],
            "log_path": plan["log_path"], "adapter_log": plan["adapter_log"],
            "plan_path": plan["plan_path"],
            "manifest_path": plan["manifest_path"],
            "harness": harness, "dataset": plan["dataset"],
        }

    @staticmethod
    def _effective_artifact_dir(parameters: Dict[str, Any]) -> str:
        """结果收集用最终命令里的 ``--output-artifact-dir``（extra args 覆盖后以此为准）。"""
        value = parameters.get("--output-artifact-dir")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                "AGENTX_COMMAND_INVALID: --output-artifact-dir is missing from the "
                "final agentx command."
            )
        return osp.abspath(osp.expanduser(value.strip()))

    @staticmethod
    def _clear_previous_results(run_dir: str, artifact_dir: str) -> None:
        """reuse 覆盖重跑前清理上一轮结果文件，保留目录和目录内其他文件。"""
        stale = [osp.join(run_dir, "manifest.json")]
        try:
            stale.extend(
                str(path) for path in Path(artifact_dir).rglob("profile_export*")
                if path.is_file()
            )
        except OSError:
            # 产物目录暂不可读时交给后续 mkdir/执行报错，不阻断清理。
            pass
        for path in stale:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(
                    f"AGENTX_OUTPUT_WRITE_FAILED: cannot overwrite stale result {path}: {exc}"
                ) from exc

    def _resolve_dataset(self, run_mode: str) -> Dict[str, Any]:
        if self.args.get("raw_mode"):
            return self._resolve_raw_dataset()
        variant = resolve_variant(self.args.get("variant"))
        spec = DATASETS[variant]
        dataset_path = self.args.get("dataset_path")
        expected = spec["sha256"].lower()
        base = {
            "variant": variant, "registry_alias": spec["alias"], "repo": spec["hf_repo"],
            "revision": spec["hf_revision"], "expected_sha256": expected,
            "actual_sha256": None, "source": "public", "alias": spec["alias"],
            "verification": ["not_verified"], "sha256_mismatch": False,
            "num_dataset_entries": None,
        }
        if not dataset_path:
            return base
        started = time.monotonic()
        trace_file = osp.realpath(osp.join(dataset_path, "traces.jsonl"))
        actual, num_entries = _sha256_and_line_count(trace_file)
        mismatch = actual.lower() != expected
        if mismatch and run_mode != "smoke":
            raise ValueError(
                f"AGENTX_DATASET_HASH_MISMATCH: expected {expected}, got {actual}. "
                "Only smoke mode may continue with a mismatched local corpus."
            )
        stat = os.stat(trace_file)
        base.update({
            "source": "local", "alias": None, "path": osp.realpath(dataset_path),
            "trace_file": trace_file, "actual_sha256": actual, "sha256_mismatch": mismatch,
            "verification": ["not_verified"] if mismatch else ["local_content_hash_verified"],
            "file_size": stat.st_size, "file_mtime": stat.st_mtime,
            "num_dataset_entries": num_entries,
            "hash_elapsed_seconds": round(time.monotonic() - started, 3),
        })
        return base

    def _resolve_raw_dataset(self) -> Dict[str, Any]:
        """raw 模式的溯源信息全部取自最终命令的 ``--public-dataset`` / ``--hf-weka-dataset``。

        已知公共别名映射到内置 registry（variant/repo/revision/expected_sha256）；
        本地目录只记录路径，不做哈希校验（raw 是直通模式，不额外读盘）。
        两者都没有时保持 ``source='raw'``，benchmark 会因 dataset_provenance_unverified 判无效。
        """
        parameters = _argv_parameters(_split_extra_args(self.args.get("extra_args")))
        alias = parameters.get("--public-dataset")
        dataset_dir = parameters.get("--hf-weka-dataset")
        base: Dict[str, Any] = {
            "variant": "raw", "registry_alias": None, "repo": None, "revision": None,
            "expected_sha256": None, "actual_sha256": None, "source": "raw",
            "alias": None, "verification": ["not_verified"], "sha256_mismatch": False,
            "num_dataset_entries": None,
        }
        entries = parameters.get("--num-dataset-entries")
        if isinstance(entries, str) and entries.strip().isdigit():
            base["num_dataset_entries"] = int(entries.strip())
        if isinstance(alias, str) and alias.strip():
            alias = alias.strip()
            registry = {spec["alias"]: (variant, spec) for variant, spec in DATASETS.items()}
            variant, spec = registry.get(alias, (None, None))
            base.update({"source": "public", "alias": alias, "provenance_origin": "--public-dataset"})
            if spec is None:
                base["registry_alias"] = alias
            else:
                base.update({
                    "variant": variant, "registry_alias": spec["alias"],
                    "repo": spec["hf_repo"], "revision": spec["hf_revision"],
                    "expected_sha256": spec["sha256"].lower(),
                })
            return base
        if isinstance(dataset_dir, str) and dataset_dir.strip():
            base.update({
                "source": "local",
                "path": osp.realpath(osp.expanduser(dataset_dir.strip())),
                "provenance_origin": "--hf-weka-dataset",
            })
        return base

    def _harness_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(self.args.get("env") or {})
        return env

    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        started_wall = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        return_code, cancelled, start_error, timeout_error = self._execute(
            plan["argv"], plan["log_path"], plan["env"], plan["run_id"],
            plan.get("secrets") or [], plan["adapter_log"],
        )
        raw, parse_status, parse_error = self._read_summary(plan["artifact_dir"], plan["summary_path"])
        dataset = dict(plan["dataset"])
        entries = self._dataset_entries(dataset, raw)
        if entries is not None:
            dataset["num_dataset_entries"] = entries
        try:
            engine, probe_errors = self._probe_engine(plan)
        except Exception as exc:
            # 探测只是附加信息：兜底降级为失败记录，确保 manifest 必定落盘。
            engine = None
            probe_errors = [{
                "code": "AGENTX_ENGINE_PROBE_FAILED",
                "message": f"engine probe crashed: {exc}",
            }]
        was_cancelled = cancelled or bool(raw.get("was_cancelled"))
        if start_error:
            status = "start_failed"
        elif was_cancelled:
            status = "cancelled"
        elif return_code != 0 or parse_status != "success":
            status = "failed"
        else:
            status = "completed"
        errors = list(probe_errors)
        if start_error:
            errors.append({"code": "AGENTX_PROCESS_START_FAILED", "message": start_error})
        if timeout_error:
            errors.append({"code": "AGENTX_PROCESS_TIMEOUT", "message": timeout_error})
        elif return_code not in (0, None) and not was_cancelled:
            # 取消路径的退出码来自我方 kill，不是 Harness 故障，不重复记 PROCESS_FAILED。
            errors.append({"code": "AGENTX_PROCESS_FAILED", "message": f"Harness exited with {return_code}."})
        if parse_error:
            code = "AGENTX_RESULT_SCHEMA_UNSUPPORTED" if parse_status == "unsupported_schema" else (
                "AGENTX_RESULT_MISSING" if parse_status == "missing" else "AGENTX_RESULT_INVALID")
            errors.append({"code": code, "message": parse_error})
        if was_cancelled:
            errors.append({"code": "AGENTX_CANCELLED", "message": "Run was cancelled; partial artifacts retained."})
        secrets = plan.get("secrets") or []
        errors = [{**item, "message": redact_text(str(item["message"]), secrets)} for item in errors]
        parameters = plan.get("parameters") or {}
        identity = {
            "model": parameters.get("--model") or self.model_cfg.get("model"),
            "model_abbr": self.model_cfg.get("abbr"),
            "dataset_abbr": self.dataset_cfg["abbr"], "variant": plan["dataset"]["variant"],
            "concurrency": plan["concurrency"],
        }
        run = {
            "run_id": plan["run_id"], "mode": plan["run_mode"], "status": status,
            "parse_status": parse_status, "start_time": started_wall,
            "end_time": datetime.now(timezone.utc).isoformat(), "return_code": return_code,
            "was_cancelled": was_cancelled, "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        artifacts = {
            "raw_summary": plan["summary_path"],
            "raw_csv": plan["summary_csv_path"],
            "raw_records": osp.join(plan["artifact_dir"], "profile_export.jsonl"),
            "raw_artifacts": plan["artifact_dir"],
            "harness_log": plan["log_path"], "adapter_log": plan["adapter_log"],
            "execution_plan": plan["plan_path"],
            "manifest": plan["manifest_path"],
        }
        harness = dict(plan["harness"])
        harness.pop("entry", None)
        result = normalize_agentx_result(
            raw, run=run, identity=identity, harness=harness, dataset=dataset,
            command=redact_command(plan["argv"]), artifacts=artifacts,
            errors=errors,
        )
        manifest = self._build_manifest(plan, result, raw, engine)
        manifest = self._finalize_run(plan, result, manifest)
        self._adapter_log(
            plan,
            f"finished status={status} parse={parse_status} return_code={return_code} "
            f"manifest={plan['manifest_path']}",
        )
        return result

    def _finalize_run(self, plan: Dict[str, Any], result: Dict[str, Any],
                      manifest: Dict[str, Any]) -> Dict[str, Any]:
        """复制上游 summary 到任务目录并写入 manifest（单任务即落盘）。"""
        from ais_bench.benchmark.summarizers.agentx import AgentXSummarizer

        summarizer = AgentXSummarizer({"work_dir": self.work_dir})
        manifest = summarizer.finalize_run(
            plan["run_dir"], manifest, plan["artifact_dir"], plan["logs_dir"]
        )
        manifest_result = manifest.get("result") or {}
        result["artifacts"]["manifest"] = plan["manifest_path"]
        for key in ("summary_json", "summary_csv"):
            relative = manifest_result.get(key)
            result["artifacts"][key] = (
                osp.join(plan["run_dir"], relative) if relative else None
            )
        return manifest

    def _dataset_entries(self, dataset: Dict[str, Any], raw: Dict[str, Any]) -> Optional[int]:
        if dataset.get("num_dataset_entries") is not None:
            return dataset["num_dataset_entries"]
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        harness_dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
        if isinstance(harness_dataset, dict):
            for key in ("num_entries", "num_dataset_entries", "total_entries", "records"):
                value = harness_dataset.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
        return None

    def _probe_engine(self, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """只读探测引擎版本与模型信息；探测目标取自最终命令，失败不中断测评。"""
        parameters = plan.get("parameters") or {}
        base_url_source = parameters.get("--url")
        if not isinstance(base_url_source, str) or not base_url_source.strip():
            # 命令里没有 --url：AISBench 侧没有可探测的服务地址（如 raw 且未传 --url）。
            return {
                "type": None, "version": None, "served_model": None, "max_model_len": None,
                "base_url": None, "endpoint": None,
                "probe": {"skipped": "no --url in the generated agentx command"},
            }, []
        endpoint = resolve_endpoint(parameters.get("--endpoint"))
        base_url = normalize_base_url(base_url_source.strip(), endpoint)
        model_name = parameters.get("--model")
        version_url = base_url + "/version"
        models_url = base_url + "/v1/models"
        version_payload, version_error = _http_json(version_url)
        models_payload, models_error = _http_json(models_url)
        version = version_payload.get("version") if isinstance(version_payload, dict) else None
        served_model = max_model_len = None
        entries = models_payload.get("data") if isinstance(models_payload, dict) else None
        if isinstance(entries, list):
            candidates = [item for item in entries if isinstance(item, dict)]
            chosen = next(
                (item for item in candidates if item.get("id") == model_name),
                candidates[0] if candidates else None,
            )
            if chosen is not None:
                served_model = chosen.get("id")
                value = chosen.get("max_model_len")
                if isinstance(value, int) and not isinstance(value, bool):
                    max_model_len = value
        errors: List[Dict[str, Any]] = []
        if version_error:
            errors.append({
                "code": "AGENTX_ENGINE_PROBE_FAILED",
                "message": f"GET {version_url} failed: {version_error}",
            })
        if models_error:
            errors.append({
                "code": "AGENTX_ENGINE_PROBE_FAILED",
                "message": f"GET {models_url} failed: {models_error}",
            })
        engine = {
            "type": "vllm" if version is not None else None,
            "version": version,
            "served_model": served_model,
            "max_model_len": max_model_len,
            "base_url": base_url,
            "endpoint": endpoint,
            "probe": {
                "version": {
                    "url": version_url,
                    "status": "ok" if version_error is None else "error",
                    "error": version_error,
                },
                "models": {
                    "url": models_url,
                    "status": "ok" if models_error is None else "error",
                    "error": models_error,
                },
            },
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        return engine, errors

    def _build_manifest(self, plan: Dict[str, Any], result: Dict[str, Any],
                        raw: Dict[str, Any], engine: Dict[str, Any]) -> Dict[str, Any]:
        run = result.get("run") or {}
        identity = result.get("identity") or {}
        harness = result.get("harness") or {}
        dataset = result.get("dataset") or {}
        validity = result.get("validity") or {}
        primary_reason = validity.get("primary_invalid_reason")
        invalid_reasons = [primary_reason] if primary_reason else []
        all_invalid_reasons = list(validity.get("all_invalid_reasons") or [])
        redacted_argv = redact_command(plan["argv"])
        # scenario 允许被 extra args 覆盖，记录最终命令里的真实值。
        scenario = _argv_parameters(redacted_argv).get("--scenario") or AGENTX_SCENARIO
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "upstream": {
                "framework": AGENTX_PACKAGE,
                "version": harness.get("version"),
                "aiperf_version": harness.get("aiperf_version"),
                "python_version": harness.get("python_version"),
                "platform": harness.get("platform"),
            },
            "run_params": {
                "run_mode": plan["run_mode"],
                "benchmark_id": f"{identity.get('model_abbr')}/{identity.get('dataset_abbr')}",
                "command": harness.get("command") or redacted_argv,
                "command_note": (
                    "duration/seed/warmup 未显式覆盖时沿用 agentx scenario preset 默认口径"
                    if scenario == AGENTX_SCENARIO else
                    f"scenario={scenario} 由 extra args 覆盖；AISBench 仅透传与记录，"
                    "汇总/有效性口径仍按 agentx 场景假定"
                ),
                "started_at": run.get("start_time"),
                "ended_at": run.get("end_time"),
            },
            # 实际执行的 agentx 最终生效参数；extra args 覆盖后的值以这里为准。
            "agentx_command": {
                "parameters": _argv_parameters(redacted_argv),
                "extra_args": redact_command(_split_extra_args(self.args.get("extra_args"))),
            },
            "dataset": {
                "abbr": identity.get("dataset_abbr"),
                "variant": dataset.get("variant"),
                "source": dataset.get("source"),
                "registry_alias": dataset.get("registry_alias"),
                "repo": dataset.get("repo"),
                "revision": dataset.get("revision"),
                "expected_sha256": dataset.get("expected_sha256"),
                "actual_sha256": dataset.get("actual_sha256"),
                "sha256_mismatch": dataset.get("sha256_mismatch"),
                "trace_file": dataset.get("trace_file"),
                "path": dataset.get("path"),
                "provenance_origin": dataset.get("provenance_origin"),
                "file_size": dataset.get("file_size"),
                "num_dataset_entries": dataset.get("num_dataset_entries"),
            },
            "model": {
                "name": identity.get("model"),
                "abbr": identity.get("model_abbr"),
                "path": (plan.get("parameters") or {}).get("--tokenizer") or self.model_cfg.get("path"),
            },
            "engine": engine,
            "test_params": self._test_params(plan, raw),
            "result": {
                "status": run.get("status"),
                "parse_status": run.get("parse_status"),
                "data_status": run.get("data_status"),
                "benchmark_valid": run.get("benchmark_valid"),
                "comparison_scope": run.get("comparison_scope"),
                "return_code": run.get("return_code"),
                "was_cancelled": run.get("was_cancelled"),
                "harness_submission_valid": validity.get("harness_submission_valid"),
                "invalid_reasons": invalid_reasons,
                "all_invalid_reasons": all_invalid_reasons,
                "aisbench_invalid_reasons": validity.get("aisbench_invalid_reasons"),
                "errors": result.get("errors") or [],
                "summary_json": None,
                "summary_csv": None,
                "raw_artifacts": None,
            },
        }

    def _test_params(self, plan: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
        metadata = raw.get("metadata") if isinstance(raw, dict) else None
        config = metadata.get("config") if isinstance(metadata, dict) else None

        def pick(*keys):
            for source in (config, metadata):
                if isinstance(source, dict):
                    for key in keys:
                        if source.get(key) is not None:
                            return source[key]
            return None

        # 以 extra args 覆盖后的最终生效 argv 推导，保证与 agentx_command.parameters 一致；
        # argv 未显式给出时回退上游 summary 报告的实际值，再回退 preset 默认。
        parameters = _argv_parameters(plan["argv"])

        duration = parameters.get("--benchmark-duration")
        if duration is None:
            duration = pick("benchmark_duration", "benchmark_duration_s", "duration")
        if duration is None:
            duration = PRESET_BENCHMARK_DURATION
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError:
                pass

        seed = parameters.get("--random-seed") or parameters.get("--seed")
        if seed is None:
            seed = pick("random_seed", "seed")
        if seed is None:
            seed = PRESET_RANDOM_SEED
        if isinstance(seed, str):
            try:
                seed = int(seed)
            except ValueError:
                pass

        warmup = parameters.get("--num-warmup-requests")
        if warmup is None:
            warmup = pick("warmup_requests_per_lane", "num_warmup_requests", "warmup_requests")
        if warmup is None:
            warmup = PRESET_WARMUP_REQUESTS_PER_LANE
        if isinstance(warmup, str):
            try:
                warmup = int(warmup)
            except ValueError:
                pass
        return {
            "batch_size": plan["concurrency"],
            "benchmark_duration_s": duration,
            "warmup_requests_per_lane": warmup,
            "random_seed": seed,
            "scenario": parameters.get("--scenario") or AGENTX_SCENARIO,
            "trace_limit": plan.get("trace_limit"),
        }

    def _read_summary(self, artifact_dir: str, expected_path: str) -> Tuple[Dict[str, Any], str, Optional[str]]:
        matches = sorted(str(path) for path in Path(artifact_dir).rglob(HARNESS_SUMMARY_JSON) if path.is_file())
        if len(matches) > 1:
            return {}, "invalid_json", f"Multiple {HARNESS_SUMMARY_JSON} files found in the fresh run directory."
        if not osp.isfile(expected_path):
            detail = f"Expected summary is missing: {expected_path}"
            if matches:
                detail += f"; unexpected alternate locations: {matches}"
            return {}, "missing", detail
        try:
            with open(expected_path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except json.JSONDecodeError as exc:
            return {}, "invalid_json", f"Invalid summary JSON: {exc}"
        except OSError as exc:
            return {}, "io_error", f"Cannot read summary: {exc}"
        if not isinstance(raw, dict):
            return {}, "invalid_json", "AgentX summary top level must be an object."
        if raw.get("schema_version") != SUPPORTED_RAW_SCHEMA:
            return raw, "unsupported_schema", f"Unsupported AgentX schema_version: {raw.get('schema_version')!r}."
        return raw, "success", None

    def _execute(self, command: List[str], log_path: str, env: Dict[str, str], run_id: str,
                 secrets: Optional[List[str]] = None, adapter_log: Optional[str] = None
                 ) -> Tuple[Optional[int], bool, Optional[str], Optional[str]]:
        """流式消费日志；取消、超时和异常路径均回收整个进程树。"""
        secrets = secrets or []
        try:
            timeout = _process_timeout(env)
        except ValueError as exc:
            # _prepare 已提前校验；保留同样的配置错误分类，避免被记成 start_failed。
            raise RuntimeError(f"AGENTX_CONFIG_INVALID: {exc}") from exc
        popen_kwargs: Dict[str, Any] = {"start_new_session": True}
        process: Optional[subprocess.Popen] = None
        cancelled = False
        finished = threading.Event()
        timed_out = threading.Event()
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"AISBench AgentX run_id={run_id}\n")
                log_file.flush()
                try:
                    process = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=env,
                        cwd=osp.dirname(osp.dirname(osp.dirname(log_path))),
                        encoding="utf-8", errors="replace",
                        **popen_kwargs,
                    )
                except OSError as exc:
                    return None, False, str(exc), None
                if adapter_log:
                    self._append_log(adapter_log, f"started run_id={run_id} pid={process.pid}")
                assert process.stdout is not None
                if timeout is not None:
                    # 看门狗避免 Harness hang 死时 AISBench 永久阻塞在读行上。
                    threading.Thread(
                        target=self._watchdog, args=(process, timeout, finished, timed_out),
                        name=f"agentx-watchdog-{run_id}", daemon=True,
                    ).start()
                try:
                    for line in process.stdout:
                        safe_line = _screen_line(redact_text(line, secrets))
                        log_file.write(safe_line)
                        log_file.flush()
                        self.logger.info("%s", safe_line.rstrip())
                        self._update_state({"stage": "agentx", "run_id": run_id})
                    return_code = process.wait()
                    finished.set()
                    timeout_error = None
                    if timed_out.is_set():
                        timeout_error = (
                            f"Harness did not exit within {PROCESS_TIMEOUT_ENV}={timeout:g}s "
                            "and was terminated."
                        )
                    return return_code, cancelled, None, timeout_error
                except KeyboardInterrupt:
                    cancelled = True
                    if adapter_log:
                        self._append_log(adapter_log, f"cancellation requested run_id={run_id}")
                    # 先非阻塞排空管道里已到达的尾部输出（常含取消原因），再回收进程组。
                    self._drain_stdout(process, log_file, secrets)
                    code = self._terminate(process)
                    self._drain_stdout(process, log_file, secrets)
                    return code, True, None, None
                except BaseException:
                    self._drain_stdout(process, log_file, secrets)
                    self._terminate(process)
                    raise
        finally:
            finished.set()
            if process is not None and process.poll() is None:
                self._terminate(process)
            if process is not None and process.stdout is not None:
                process.stdout.close()

    @staticmethod
    def _watchdog(process: subprocess.Popen, timeout: float,
                  finished: threading.Event, timed_out: threading.Event) -> None:
        """超时未结束时回收整个进程组，从而解除主线程的阻塞读。"""
        if finished.wait(timeout):
            return
        timed_out.set()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _drain_stdout(self, process: subprocess.Popen, log_file, secrets: List[str]) -> None:
        """非阻塞读取已到达的 stdout，避免取消/超时时丢掉管道缓冲的日志尾部。"""
        stream = process.stdout
        if stream is None:
            return
        try:
            fd = stream.fileno()
        except (OSError, ValueError):
            return
        while True:
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                return
            line = stream.readline()
            if not line:
                return
            safe_line = _screen_line(redact_text(line, secrets))
            log_file.write(safe_line)
            log_file.flush()
            self.logger.info("%s", safe_line.rstrip())

    @staticmethod
    def _terminate(process: subprocess.Popen) -> int:
        """POSIX 进程组分级回收。"""
        if process.poll() is not None:
            return process.returncode or 0
        try:
            os.killpg(process.pid, signal.SIGINT)
            return process.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return process.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()

    def _adapter_log(self, plan: Dict[str, Any], message: str) -> None:
        self._append_log(plan["adapter_log"], redact_text(message, plan.get("secrets") or []))

    @staticmethod
    def _append_log(path: str, message: str) -> None:
        mkdir_or_exist(osp.dirname(path))
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    def _update_state(self, payload: Dict[str, Any]) -> None:
        if self.task_state_manager is None:
            return
        now = time.time()
        if now - self._last_state_ts < self._PROGRESS_INTERVAL and payload.get("status") is None:
            return
        self._last_state_ts = now
        self.task_state_manager.update_task_state({"other_kwargs": payload})


def parse_args():
    parser = argparse.ArgumentParser(description="AISBench AgentX performance task")
    parser.add_argument("config", help="Config file path")
    return parser.parse_args()


if __name__ == "__main__":
    logger = AISLogger(__name__)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg), is_debug=cfg["cli_args"]["debug"],
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state({"status": "start", "task_log_path": AgentXTask.log_subdir})
    started = time.perf_counter()
    try:
        AgentXTask(cfg).run(task_state_manager)
    except BaseException:
        task_state_manager.update_task_state({"status": "error"})
        raise
    task_state_manager.update_task_state({"status": "finish"})
    logger.info("AgentX task time elapsed: %.2fs", time.perf_counter() - started)
    manager_t.join()
