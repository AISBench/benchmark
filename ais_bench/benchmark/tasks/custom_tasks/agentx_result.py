"""AgentX 1.0.13 原始产物的运行事实判定与安全落盘工具。

指标本体不再做 AISBench 式规范化转换：上游 summary 逐字节保存在
``performances/<run_id>/agentx_summary.json|csv``，本模块只负责解析状态、
有效性判定和原子写。
"""

import json
import math
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

RESULT_SCHEMA_VERSION = "aisbench_agentx_result/v1"
SUPPORTED_RAW_SCHEMA = "1.4"
SUPPORTED_HARNESS_VERSION = "1.0.13"

# Harness 产物文件名与日志品牌前缀：task 与 summarizer 共用，避免字面量两处漂移。
HARNESS_SUMMARY_JSON = "profile_export_aiperf.json"
HARNESS_SUMMARY_CSV = "profile_export_aiperf.csv"
HARNESS_BRAND_PREFIX_RE = re.compile(r"NVIDIA\s+AIPerf\s*\|\s*")


def json_safe(value: Any) -> Any:
    """递归移除非标准 NaN/Infinity；缺失和非有限值统一写成 null。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: str, payload: Any) -> None:
    """同目录临时文件 + os.replace，避免 Viz 读到半个 JSON。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".agentx-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(json_safe(payload), file, indent=2, ensure_ascii=False,
                      allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def redact_text(text: str, secrets: List[str]) -> str:
    """文本脱敏：先按长度倒序替换已知密钥，再兜底处理 header 与 ``--api-key``。"""
    result = text
    # 长值优先替换：api_key="secret" 是 "--agentx-extra-arg --api-key extra-secret"
    # 的子串，先替换短值会残留 "extra-<redacted>"，导致回显并非完整脱敏。
    for secret in sorted({str(item) for item in secrets if item}, key=len, reverse=True):
        result = result.replace(secret, "<redacted>")
    result = re.sub(r"(?i)(authorization|proxy-authorization|x-api-key|x-auth-token)\s*[:=]\s*\S+",
                    r"\1:<redacted>", result)
    result = re.sub(r"(--api-key(?:=|\s+))\S+", r"\1<redacted>", result)
    return result


def _has_metrics(raw: Dict[str, Any]) -> bool:
    """上游 summary 顶层是否存在任一满足 ``{unit: str}`` 契约的指标。"""
    return any(
        isinstance(value, dict) and isinstance(value.get("unit"), str)
        for value in raw.values()
    )


def _verification_recorded(verification: Any) -> bool:
    """验证方式是否已记录：verification 列表须含 *_verified 状态标记。

    ``not_verified`` 与 ``local_content_hash_verified`` 一样是显式记录的验证状态；
    缺失、空列表或未按 *_verified 命名的项视为验证方式未记录。
    """
    return isinstance(verification, list) and any(
        isinstance(item, str) and item.endswith("_verified") for item in verification
    )


def normalize_agentx_result(
    raw: Optional[Dict[str, Any]],
    *,
    run: Dict[str, Any],
    identity: Dict[str, Any],
    harness: Dict[str, Any],
    dataset: Dict[str, Any],
    command: List[str],
    artifacts: Dict[str, Any],
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """生成运行事实判定；指标本体保留在 performances/<run_id>/，不在此复制。"""
    raw = raw if isinstance(raw, dict) else {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    raw_schema = raw.get("schema_version")
    parse_status = run.get("parse_status") or "missing"
    if raw and raw_schema != SUPPORTED_RAW_SCHEMA:
        parse_status = "unsupported_schema"

    if not _has_metrics(raw):
        data_status = "unavailable"
    elif run.get("status") == "completed" and parse_status == "success":
        data_status = "complete"
    else:
        data_status = "partial"

    submission_valid = metadata.get("submission_valid")
    if not isinstance(submission_valid, bool):
        submission_valid = None
    harness_reasons = metadata.get("submission_invalid_reasons") or []
    if not isinstance(harness_reasons, list):
        harness_reasons = [harness_reasons]
    # 原因项统一转字符串：上游给出非字符串（如对象）时也能安全去重，不阻断 manifest 落盘。
    harness_reasons = [str(item) for item in harness_reasons]

    aisbench_reasons: List[str] = []
    is_benchmark = run.get("mode") == "benchmark"
    if run.get("status") != "completed":
        aisbench_reasons.append(f"run_status:{run.get('status')}")
    if parse_status != "success":
        aisbench_reasons.append(f"parse_status:{parse_status}")
    if run.get("was_cancelled"):
        aisbench_reasons.append("run_cancelled")
    if harness.get("version") != SUPPORTED_HARNESS_VERSION:
        aisbench_reasons.append("harness_version_unsupported")
    if raw_schema != SUPPORTED_RAW_SCHEMA:
        aisbench_reasons.append("raw_schema_unsupported")
    if not is_benchmark:
        # smoke 不是正式结果，这是该模式下最优先的口径根因。
        aisbench_reasons.append("run_mode_not_benchmark")
    elif submission_valid is not True:
        # 上游给出的具体提交原因优先作为根因；缺失时才用概括标记。
        aisbench_reasons.append(harness_reasons[0] if harness_reasons else "harness_submission_not_valid")
    if data_status != "complete":
        # 数据不可用/不完整时没有可判有效的正式结果；run/parse/上游提交原因优先作根因。
        aisbench_reasons.append(f"data_status:{data_status}")
    if dataset.get("source") not in ("public", "local") or not _verification_recorded(dataset.get("verification")):
        aisbench_reasons.append("dataset_provenance_unverified")
    if dataset.get("sha256_mismatch"):
        aisbench_reasons.append("dataset_sha256_mismatch")

    # 去重但保持原因顺序，便于屏幕、日志和自动化结果稳定比较。
    aisbench_reasons = list(dict.fromkeys(aisbench_reasons))
    all_invalid_reasons = list(dict.fromkeys(list(harness_reasons) + aisbench_reasons))
    # 只对外报一条根因（列表首项），其余根因保留在 all_invalid_reasons 中供审计。
    primary_invalid_reason = aisbench_reasons[0] if aisbench_reasons else None
    # benchmark 下上游提交无效的具体原因已并入无效原因，此处仍须据此判定无效。
    benchmark_valid = not aisbench_reasons and (not is_benchmark or submission_valid is True)
    if run.get("mode") == "smoke":
        comparison_scope = "smoke"
    elif dataset.get("source") == "local":
        comparison_scope = "internal_only"
    elif benchmark_valid:
        comparison_scope = "official_agentx"
    else:
        comparison_scope = "invalid"

    actual_dataset = metadata.get("dataset")
    dataset_result = dict(dataset)
    if actual_dataset is not None:
        dataset_result["harness_provenance"] = json_safe(actual_dataset)

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run": dict(run, parse_status=parse_status, data_status=data_status,
                    comparison_scope=comparison_scope, benchmark_valid=benchmark_valid),
        "identity": dict(identity),
        "harness": dict(harness, raw_schema_version=raw_schema,
                        aiperf_version=raw.get("aiperf_version"), command=list(command)),
        "dataset": dataset_result,
        "validity": {
            "harness_submission_valid": submission_valid,
            "harness_invalid_reasons": json_safe(harness_reasons),
            "aisbench_invalid_reasons": aisbench_reasons,
            "primary_invalid_reason": primary_invalid_reason,
            "all_invalid_reasons": all_invalid_reasons,
        },
        "artifacts": dict(artifacts),
        "errors": json_safe(errors or []),
    }
    return json_safe(result)
