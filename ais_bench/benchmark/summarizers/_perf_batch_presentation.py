"""Presentation helpers for batch performance reports."""

from collections import defaultdict
from typing import Dict, List, Tuple


def _format_value(value) -> str:
    """统一格式化（CSV 与 HTML 共用，保证数值逐单元格一致）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _comparability_key(row: dict) -> Tuple:
    """可比性分组：不同输入规格/口径不混画。"""
    return (
        row.get("stream"),
        row.get("ttft_valid"),
        row.get("use_timestamp"),
        row.get("request_size"),
        row.get("prefix_len"),
    )


def _sorted_groups(rows: List[dict]) -> List[Tuple[Tuple, List[dict]]]:
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "success":
            continue
        groups[_comparability_key(row)].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: (item.get("batch_size") or 0))
    return sorted(groups.items(), key=lambda item: str(item[0]))


def _group_label(row: dict) -> str:
    """可比分组标签：与容量曲线图的前缀保持一致。"""
    return "{}/{} (size={}, prefix={})".format(
        _display(row.get("model_abbr")),
        _display(row.get("dataset_abbr")),
        _display(row.get("request_size")),
        _display(row.get("prefix_len")),
    )


def _best_configs(rows: List[dict]) -> List[dict]:
    """最佳配置：按可比分组分别取最高吞吐与最低平均 TTFT，组间不混算。"""
    groups = _sorted_groups(rows)
    if not groups:
        # 全部失败时保留一张空卡片，与分组前的展示一致
        return [{"label": "", "throughput": None, "ttft": None}]
    multi_group = len(groups) > 1
    best_groups: List[dict] = []
    for _, group_rows in groups:
        best_throughput = None
        for row in group_rows:
            value = row.get("throughput_tok_s")
            if value is None:
                continue
            if best_throughput is None or value > best_throughput["throughput_tok_s"]:
                best_throughput = row
        best_ttft = None
        for row in group_rows:
            if not row.get("ttft_valid"):
                continue
            value = row.get("ttft_avg")
            if value is None:
                continue
            if best_ttft is None or value < best_ttft["ttft_avg"]:
                best_ttft = row
        best_groups.append({
            "label": _group_label(group_rows[0]) if multi_group else "",
            "throughput": best_throughput,
            "ttft": best_ttft,
        })
    return best_groups


def _saturation_note(rows: List[dict]) -> Tuple[str, str]:
    """按吞吐-并发曲线给出 未饱和/接近饱和/已过载 的粗判（中英双语）。"""
    ok_rows = [row for row in rows
               if row.get("status") == "success" and row.get("throughput_tok_s") is not None]
    if len(ok_rows) < 2:
        return ("可用任务不足 2 个，无法判定吞吐-并发曲线形态。",
                "Fewer than 2 usable tasks; cannot judge the throughput-concurrency curve.")
    ok_rows.sort(key=lambda item: item.get("batch_size") or 0)
    last, prev = ok_rows[-1], ok_rows[-2]
    last_throughput = last.get("throughput_tok_s") or 0.0
    prev_throughput = prev.get("throughput_tok_s") or 0.0
    last_success = last.get("success_rate")
    if last_success is not None and last_success < 99:
        return ("已过载：最大并发档成功率下降（{}%），吞吐指标不可直接比较。".format(
                    _format_value(last_success)),
                "Overloaded: success rate drops at the largest batch size ({}%); "
                "throughput metrics are not directly comparable.".format(
                    _format_value(last_success)))
    if prev_throughput > 0 and last_throughput <= prev_throughput:
        return ("已饱和：最大并发档吞吐不再随并发上升，继续加大 batch_size 收益有限。",
                "Saturated: throughput no longer grows with concurrency; "
                "further increasing batch_size yields limited gain.")
    if prev_throughput > 0 and (last_throughput - prev_throughput) / prev_throughput < 0.05:
        return ("接近饱和：最大并发档吞吐增幅不足 5%，已接近服务上限。",
                "Near saturation: throughput gain at the largest batch size is under 5%; "
                "the service limit is close.")
    return ("未饱和：吞吐仍随并发上升，可继续增大 batch_size 观察拐点。",
            "Not saturated: throughput still grows with concurrency; "
            "keep increasing batch_size to locate the knee.")


def _saturation_notes(rows: List[dict]) -> List[Tuple[str, str]]:
    """按可比分组分别给出饱和判定，多组时在文案中标注所属组。"""
    groups = _sorted_groups(rows)
    if not groups:
        return [_saturation_note(rows)]
    multi_group = len(groups) > 1
    notes: List[Tuple[str, str]] = []
    for _, group_rows in groups:
        zh, en = _saturation_note(group_rows)
        if multi_group:
            label = _group_label(group_rows[0])
            zh, en = f"[{label}] {zh}", f"[{label}] {en}"
        notes.append((zh, en))
    return notes


def _display(value) -> str:
    text = _format_value(value)
    return text if text else "—"


def _run_summary(row: dict) -> List[Tuple[str, str]]:
    """逐组合 Run Summary 的键值对（对齐 EvalScope 的 stat-grid）。"""
    fields = [
        ("Total Requests", "total_requests"),
        ("Success Requests", "completed_requests"),
        ("Failed Requests", "failed_requests"),
        ("Concurrency", "concurrency"),
        ("Max Concurrency", "max_concurrency"),
        ("Time Taken (s)", "duration_s"),
        ("Request Throughput (req/s)", "request_throughput"),
        ("Avg Latency (s)", "e2el_avg_s"),
        ("Output Tok Throughput (tok/s)", "throughput_tok_s"),
        ("Total Tok Throughput (tok/s)", "total_throughput_tok_s"),
        ("Avg TTFT (ms)", "ttft_avg"),
        ("Avg TPOT (ms)", "tpot_avg"),
        ("Avg ITL (ms)", "itl_avg"),
        ("Avg Input Tokens", "avg_input_tokens"),
        ("Avg Output Tokens", "avg_output_tokens"),
        ("Success Rate", "success_rate_text"),
    ]
    return [(label, _display(row.get(key))) for label, key in fields]


def _basic_info(rows: List[dict], first_combo: dict) -> List[Tuple[str, str, str]]:
    """报告头部的 Basic Information（manifest + 汇总合计），返回 (中文, English, 值)。"""
    model_cfg = first_combo.get("model_config") or {}
    ok_rows = [row for row in rows if row.get("status") == "success"]

    def total(key: str):
        values = [row.get(key) for row in ok_rows if row.get(key) is not None]
        return round(sum(values), 4) if values else None

    fields = [
        ("模型", "Model",
         model_cfg.get("model_name")),
        ("模型任务", "Model Task",
         model_cfg.get("abbr") or first_combo.get("model_abbr")),
        ("数据集", "Dataset", first_combo.get("dataset_abbr")),
        ("总请求数", "Total Requests", total("total_requests")),
        ("成功请求数", "Success Requests", total("completed_requests")),
        ("总测试时长 (s)", "Total Test Time (s)", total("duration_s")),
        ("输出 Token 总数", "Total Output Tokens", total("total_output_tokens")),
    ]
    return [(zh, en, _display(value)) for zh, en, value in fields]


def _recommendations(rows: List[dict], best_groups: List[dict]) -> List[Tuple[str, str]]:
    """性能建议：按组饱和判定 + 最佳档位 + 时延增长 + 失败组合提示（中英双语）。"""
    notes = list(_saturation_notes(rows))
    for group in best_groups:
        best_throughput = group.get("throughput")
        if not best_throughput:
            continue
        prefix = f"[{group['label']}] " if group.get("label") else ""
        batch_size = _display(best_throughput.get("batch_size"))
        tok_s = _display(best_throughput.get("throughput_tok_s"))
        rps = _display(best_throughput.get("request_throughput"))
        success = _display(best_throughput.get("success_rate_text"))
        notes.append(
            (prefix + "建议并发档：batch_size={}，输出吞吐 {} token/s，RPS {}，成功率 {}。".format(
                batch_size, tok_s, rps, success),
             prefix + "Recommended concurrency: batch_size={}, output throughput {} token/s, "
             "RPS {}, success rate {}.".format(batch_size, tok_s, rps, success))
        )
    groups = _sorted_groups(rows)
    multi_group = len(groups) > 1
    for _, group_rows in groups:
        latencies = sorted(
            (row.get("batch_size") or 0, row.get("e2el_avg_s"))
            for row in group_rows
            if row.get("e2el_avg_s") is not None
        )
        if len(latencies) < 2:
            continue
        first_batch, first_latency = latencies[0]
        last_batch, last_latency = latencies[-1]
        if not first_latency:
            continue
        prefix = f"[{_group_label(group_rows[0])}] " if multi_group else ""
        ratio = round(last_latency / first_latency, 2)
        notes.append(
            (prefix + "端到端平均时延从 batch_size={} 的 {} s 增长到 batch_size={} 的 {} s（{}×）。".format(
                first_batch, _display(first_latency), last_batch,
                _display(last_latency), ratio),
             prefix + "Average end-to-end latency grows from {} s at batch_size={} "
             "to {} s at batch_size={} ({}x).".format(
                _display(first_latency), first_batch,
                _display(last_latency), last_batch, ratio))
        )
    failed = [row for row in rows if row.get("status") == "fail"]
    if failed:
        notes.append(
            ("{} 个任务未产出结果（Task Status=fail），不参与曲线与最佳配置比较。".format(
                len(failed)),
             "{} task(s) produced no result (Task Status=fail) and are excluded "
             "from curves and best-configuration comparison.".format(len(failed)))
        )
    return notes
