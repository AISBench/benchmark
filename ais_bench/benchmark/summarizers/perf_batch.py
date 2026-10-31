"""多组合并发批量压测的汇总与报告渲染。

`aggregate_batch_perf` 读取配置加载阶段落盘的 `combos_manifest.json`，逐组合
汇总 `performances/<model_abbr>/<dataset_abbr>.json|.csv`，先落盘
`batch_perf_summary.csv`，再以同一份 rows 渲染 `batch_perf_report.html`。

单组合（manifest 缺失或组合数 ≤ 1）时直接跳过，零开销；个别组合结果缺失时
该行标记 fail，不阻塞其余行；HTML 渲染失败只 WARNING，不影响 CSV 与退出码。
"""

import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ais_bench.benchmark import __version__
from ais_bench.benchmark.cli.sweep import MANIFEST_FILENAME
from ais_bench.benchmark.utils.logging.logger import AISLogger

logger = AISLogger()

SUMMARY_CSV_NAME = "batch_perf_summary.csv"
REPORT_HTML_NAME = "batch_perf_report.html"
REPORT_MARKER = "<!-- ais-bench-perf-render:v1 -->"
TEMPLATE_NAME = "batch_perf_report.html.j2"
# 与 EvalScope PLOTLY_CDN_URL 对齐：单文件报告只引一次 CDN，不内联 JS
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.27.0.min.js"
# 图表文字色取中间灰：报告默认深色主题，但切到浅色主题后仍需可读
CHART_TEXT_COLOR = "#7f8a9e"

# HTML 概览宽表：一个组合一行，列口径对齐 EvalScope 的 Detailed Performance Metrics
OVERVIEW_COLUMNS = [
    ("Batch Size", "batch_size"),
    ("Num Prompts", "num_prompts"),
    ("RPS", "request_throughput"),
    ("Avg Lat.(s)", "e2el_avg_s"),
    ("P99 Lat.(s)", "e2el_p99_s"),
    ("Avg TTFT(ms)", "ttft_avg"),
    ("P99 TTFT(ms)", "ttft_p99"),
    ("Avg TPOT(ms)", "tpot_avg"),
    ("P99 TPOT(ms)", "tpot_p99"),
    ("Gen. tok/s", "throughput_tok_s"),
    ("Success Rate", "success_rate_text"),
    ("Task Status", "status"),
]
OVERVIEW_HEADERS = [label for label, _ in OVERVIEW_COLUMNS]

# batch_perf_summary.csv 与单任务 synthetic.csv 对齐：指标逐行展开，
# 身份列逐行重复填充，保证可排序、可筛选、可直接透视
# CSV 表头使用可读名称，右侧为内部字段名
IDENTITY_COLUMNS = [
    ("Model Abbr", "model_abbr"),
    ("Dataset Abbr", "dataset_abbr"),
    ("Batch Size", "batch_size"),
    ("Num Prompts", "num_prompts"),
    ("Task Status", "status"),
]
IDENTITY_HEADERS = [label for label, _ in IDENTITY_COLUMNS]

PERF_METRIC_HEADERS = [
    "Performance Parameters",
    "Stage",
    "Average",
    "Min",
    "Max",
    "Median",
    "P75",
    "P90",
    "P99",
    "N",
]

CSV_HEADERS = IDENTITY_HEADERS + PERF_METRIC_HEADERS

# CSV 汇总区（synthetic.json 的 common res）：一个组合一行，指标名作表头、取值保留原单位
RUN_SUMMARY_METRICS = [
    "Benchmark Duration",
    "Total Requests",
    "Failed Requests",
    "Success Requests",
    "Concurrency",
    "Max Concurrency",
    "Request Throughput",
    "Total Input Tokens",
    "Total Generated Tokens",
    "Input Token Throughput",
    "Output Token Throughput",
    "Total Token Throughput",
]
RUN_SUMMARY_HEADERS = IDENTITY_HEADERS + RUN_SUMMARY_METRICS

# 指标统计折线图：横轴为百分位点 Min→Max（不含 Average/Median）
PERCENTILE_STATS = ["Min", "P75", "P90", "P99", "Max"]
# 指标行内统计列下标（与 PERF_METRIC_HEADERS 一一对应）
PERF_STAT_INDEX = {label: index for index, label in enumerate(PERF_METRIC_HEADERS)}

# 指标统计图：(中文标题, 英文标题, y 轴标题, [(指标名, 图例名, 颜色)])
METRIC_CHART_SPECS = [
    ("时延", "E2EL", "Latency (ms)",
     [("E2EL", "E2EL", "#63b3ed")]),
    ("首Token/每Token/Token间时延", "TTFT/TPOT/ITL", "Latency (ms)",
     [("TTFT", "TTFT", "#63b3ed"),
      ("TPOT", "TPOT", "#a78bfa"),
      ("ITL", "ITL", "#34d399")]),
    ("输入/输出Token数", "Input/Output Tokens", "Tokens",
     [("InputTokens", "Input Tokens", "#63b3ed"),
      ("OutputTokens", "Output Tokens", "#34d399")]),
    ("输入/输出Token吞吐", "Prefill/Output Token Throughput", "Tokens/s",
     [("OutputTokenThroughput", "Output Token Throughput", "#34d399"),
      ("PrefillTokenThroughput", "Prefill Token Throughput", "#a78bfa")]),
]

# 逐请求曲线超过该点数时只画线不画点，避免密集遮挡
DETAIL_MARKER_LIMIT = 200

_NUMBER_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


@dataclass
class BatchPerfArtifacts:
    """汇总产物路径；跳过或失败时为 None。"""
    csv_path: Optional[Path] = None
    html_path: Optional[Path] = None


def _parse_number(value) -> Optional[float]:
    """把 `"123.4 ms"` / `"1.2e+03"` / 数字统一解析为 float。"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = _NUMBER_PATTERN.match(str(value))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _load_perf_json(path: Path) -> Optional[dict]:
    """读取 common res（`{metric: {stage: "value unit"}}`）。"""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError) as exc:
        logger.warning(f"Failed to load perf result {path}: {exc}")
        return None
    return data if isinstance(data, dict) else None


def _common_metric(common: dict, name: str) -> Optional[float]:
    """取 common res 指标的 total 阶段值（无 total 时取第一个阶段）。"""
    stage_map = common.get(name)
    if not isinstance(stage_map, dict) or not stage_map:
        return None
    if "total" in stage_map:
        return _parse_number(stage_map["total"])
    return _parse_number(next(iter(stage_map.values())))


def _load_perf_csv(path: Path) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
    """读取逐请求指标统计（`[param, stage, stat...]`）。"""
    table: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    if not path.exists():
        return table
    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                headers = next(reader)
            except StopIteration:
                return table
            stats = headers[2:]
            for raw_row in reader:
                if len(raw_row) < 2:
                    continue
                param, stage = raw_row[0], raw_row[1]
                values = raw_row[2:]
                table[(param, stage)] = {
                    stat: _parse_number(values[index] if index < len(values) else None)
                    for index, stat in enumerate(stats)
                }
    except OSError as exc:
        logger.warning(f"Failed to load perf csv {path}: {exc}")
    return table


def _csv_stat(table, param: str, stat: str,
              stage: str = "total") -> Optional[float]:
    stage_map = table.get((param, stage))
    if not stage_map:
        return None
    return stage_map.get(stat)


def _manifest_path(work_dir: Path) -> Path:
    return work_dir / MANIFEST_FILENAME


def _load_manifest(work_dir: Path) -> Optional[dict]:
    path = _manifest_path(work_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, ValueError) as exc:
        logger.warning(f"Failed to load sweep manifest {path}: {exc}")
        return None
    return manifest if isinstance(manifest, dict) else None


def _is_truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _build_row(work_dir: Path, combo: dict) -> dict:
    """单个组合的汇总行（数值态，格式化在落盘/渲染时统一做）。"""
    model_abbr = combo.get("model_abbr")
    dataset_abbr = combo.get("dataset_abbr")
    model_cfg = combo.get("model_config") or {}
    dataset_cfg = combo.get("dataset_config") or {}
    stream = _is_truthy(model_cfg.get("stream", False))
    ttft_valid = _is_truthy(model_cfg.get("ttft_valid", False))

    row = {
        "index": combo.get("index"),
        "model_abbr": model_abbr,
        "dataset_abbr": dataset_abbr,
        "batch_size": combo.get("batch_size"),
        "num_prompts": combo.get("num_prompts"),
        "request_count": combo.get("request_count"),
        "stream": stream,
        "ttft_valid": ttft_valid,
        "use_timestamp": _is_truthy(model_cfg.get("use_timestamp", False)),
        "prefix_len": dataset_cfg.get("PrefixLen"),
        "request_size": dataset_cfg.get("RequestSize"),
        "output_len": model_cfg.get("max_out_len"),
        "dataset_type": dataset_cfg.get("Type"),
        "status": "fail",
        "throughput_tok_s": None,
        "total_throughput_tok_s": None,
        "input_throughput_tok_s": None,
        "request_throughput": None,
        "e2el_avg": None,
        "e2el_p99": None,
        "e2el_avg_s": None,
        "e2el_p99_s": None,
        "ttft_avg": None,
        "ttft_p90": None,
        "ttft_p99": None,
        "tpot_avg": None,
        "tpot_p99": None,
        "itl_avg": None,
        "itl_p99": None,
        "success_rate": None,
        "success_rate_text": "",
        "completed_requests": None,
        "total_requests": None,
        "failed_requests": None,
        "concurrency": None,
        "max_concurrency": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "avg_input_tokens": None,
        "avg_output_tokens": None,
        "duration_s": None,
    }

    perf_dir = work_dir / "performances" / str(model_abbr)
    common = _load_perf_json(perf_dir / f"{dataset_abbr}.json")
    if common is None:
        return row
    metrics = _load_perf_csv(perf_dir / f"{dataset_abbr}.csv")

    row["status"] = "success"
    row["throughput_tok_s"] = _common_metric(common, "Output Token Throughput")
    row["total_throughput_tok_s"] = _common_metric(common, "Total Token Throughput")
    row["input_throughput_tok_s"] = _common_metric(common, "Input Token Throughput")
    row["request_throughput"] = _common_metric(common, "Request Throughput")
    row["concurrency"] = _common_metric(common, "Concurrency")
    row["max_concurrency"] = _common_metric(common, "Max Concurrency")
    row["total_input_tokens"] = _common_metric(common, "Total Input Tokens")
    row["total_output_tokens"] = _common_metric(common, "Total Generated Tokens")

    total_requests = _common_metric(common, "Total Requests")
    success_requests = _common_metric(common, "Success Requests")
    failed_requests = _common_metric(common, "Failed Requests")
    if total_requests is not None:
        row["total_requests"] = int(total_requests)
    if success_requests is not None:
        row["completed_requests"] = int(success_requests)
    if failed_requests is not None:
        row["failed_requests"] = int(failed_requests)
    if total_requests:
        row["success_rate"] = round(success_requests / total_requests * 100, 2) \
            if success_requests is not None else None
        if row["success_rate"] is not None:
            row["success_rate_text"] = f"{_format_value(row['success_rate'])}%"
    duration_ms = _common_metric(common, "Benchmark Duration")
    if duration_ms is not None:
        row["duration_s"] = round(duration_ms / 1000, 4)

    row["e2el_avg"] = _csv_stat(metrics, "E2EL", "Average")
    row["e2el_p99"] = _csv_stat(metrics, "E2EL", "P99")
    if row["e2el_avg"] is not None:
        row["e2el_avg_s"] = round(row["e2el_avg"] / 1000, 3)
    if row["e2el_p99"] is not None:
        row["e2el_p99_s"] = round(row["e2el_p99"] / 1000, 3)
    row["avg_input_tokens"] = _csv_stat(metrics, "InputTokens", "Average")
    row["avg_output_tokens"] = _csv_stat(metrics, "OutputTokens", "Average")
    if ttft_valid:
        row["ttft_avg"] = _csv_stat(metrics, "TTFT", "Average")
        row["ttft_p90"] = _csv_stat(metrics, "TTFT", "P90")
        row["ttft_p99"] = _csv_stat(metrics, "TTFT", "P99")
        row["tpot_avg"] = _csv_stat(metrics, "TPOT", "Average")
        row["tpot_p99"] = _csv_stat(metrics, "TPOT", "P99")
        row["itl_avg"] = _csv_stat(metrics, "ITL", "Average")
        row["itl_p99"] = _csv_stat(metrics, "ITL", "P99")
    return row


def _format_value(value) -> str:
    """统一格式化（CSV 与 HTML 共用，保证数值逐单元格一致）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _escape_csv_text(value) -> str:
    """CSV 公式注入转义：仅作用于文本列。"""
    text = "" if value is None else str(value)
    if text[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


def _format_row(row: dict) -> dict:
    """HTML 展示用的格式化行（自动转义由模板负责，这里只做数值格式化）。"""
    return {key: _format_value(value) for key, value in row.items()}


def _identity_values(row: dict) -> List[str]:
    """汇总 CSV 的身份列取值（文本列做公式注入转义）。"""
    values = []
    for _, key in IDENTITY_COLUMNS:
        if key in ("model_abbr", "dataset_abbr", "status"):
            values.append(_escape_csv_text(row.get(key)))
        else:
            values.append(_format_value(row.get(key)))
    return values


def _read_metric_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """原样读取单组合 `synthetic.csv`：返回 (表头, 指标行)，值不做二次格式化。"""
    if not path.exists():
        return [], []
    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                return [], []
            raw_rows = [row for row in reader if len(row) >= 2]
    except OSError as exc:
        logger.warning(f"Failed to load perf csv {path}: {exc}")
        return [], []
    width = len(header)
    return header, [(row + [""] * width)[:width] for row in raw_rows]


def _metric_rows_for_row(row: dict, work_dir: Path) -> Tuple[List[str], List[List[str]]]:
    """读取单组合的指标行；非流式组合的逐 token 时延口径置空。"""
    if row.get("status") != "success":
        return [], []
    perf_dir = work_dir / "performances" / str(row.get("model_abbr"))
    header, metric_rows = _read_metric_rows(
        perf_dir / f"{row.get('dataset_abbr')}.csv")
    normalized: List[List[str]] = []
    for metric_row in metric_rows:
        if not row.get("ttft_valid") and metric_row[0] in ("TTFT", "TPOT", "ITL"):
            # 非流式组合的逐 token 时延口径无效：保留指标名、清空统计值
            metric_row = metric_row[:2] + [""] * (len(header) - 2)
        normalized.append(metric_row)
    return header, normalized


def _expand_summary_rows(rows: List[dict], work_dir: Path) -> List[List[str]]:
    """把"一组合一行"展开为「身份列 + 指标行」，身份列每行重复填充。"""
    expanded: List[List[str]] = []
    for row in rows:
        identity = _identity_values(row)
        _, metric_rows = _metric_rows_for_row(row, work_dir)
        if not metric_rows:
            expanded.append(identity + [""] * len(PERF_METRIC_HEADERS))
            continue
        for metric_row in metric_rows:
            # CSV 表头固定为 PERF_METRIC_HEADERS，逐行对齐列宽
            metric_row = (metric_row + [""] * len(PERF_METRIC_HEADERS))[
                :len(PERF_METRIC_HEADERS)]
            expanded.append(identity + metric_row)
    return expanded


def _build_run_summary_row(row: dict, work_dir: Path) -> List[str]:
    """汇总区一行：身份列 + synthetic.json 原文取值（含单位），缺失的组合留空。"""
    perf_dir = work_dir / "performances" / str(row.get("model_abbr"))
    common = _load_perf_json(
        perf_dir / f"{row.get('dataset_abbr')}.json") or {}
    values: List[str] = []
    for name in RUN_SUMMARY_METRICS:
        stage_map = common.get(name)
        if isinstance(stage_map, dict) and stage_map:
            stage = "total" if "total" in stage_map else next(iter(stage_map))
            values.append(_format_value(stage_map.get(stage)))
        else:
            values.append("")
    return _identity_values(row) + values


def _write_summary_csv(path: Path, rows: List[dict], work_dir: Path) -> None:
    """写汇总 CSV：先按组合展开指标行，再空一行追加 synthetic.json 汇总区。"""
    expanded = _expand_summary_rows(rows, work_dir)
    run_rows = [_build_run_summary_row(row, work_dir) for row in rows]
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
            writer.writerows(expanded)
            if run_rows:
                # 空一行后是独立表头 + 每组合一行的单值汇总区
                writer.writerow([])
                writer.writerow(RUN_SUMMARY_HEADERS)
                writer.writerows(run_rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except OSError:
        try:
            if tmp_path.exists():
                os.remove(tmp_path)
        except OSError:
            pass
        raise


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


def _figure_html(figure) -> str:
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _style_figure(figure, x_title: str, y_title: str, height: int = 380) -> None:
    """统一图表样式：透明背景、横向图例、宽度随容器自适应。"""
    figure.update_layout(
        font={"color": CHART_TEXT_COLOR, "size": 12},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1,
                "font": {"color": CHART_TEXT_COLOR}},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#16163a", "bordercolor": "rgba(129,109,248,0.4)",
                    "font": {"color": "#e2e8f0"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": {"text": x_title},
               "tickfont": {"color": CHART_TEXT_COLOR},
               "gridcolor": "rgba(129,109,248,0.12)",
               "zerolinecolor": "rgba(129,109,248,0.2)"},
        yaxis={"title": {"text": y_title},
               "tickfont": {"color": CHART_TEXT_COLOR},
               "gridcolor": "rgba(129,109,248,0.12)",
               "zerolinecolor": "rgba(129,109,248,0.2)"},
    )


def _stat_value(metric_rows: List[List[str]], metric: str,
                stat: str) -> Optional[float]:
    """从指标行取某指标的统计值；同一指标有多个 stage 时优先 total。"""
    index = PERF_STAT_INDEX.get(stat)
    if index is None:
        return None
    fallback = None
    for metric_row in metric_rows:
        if len(metric_row) <= index or metric_row[0] != metric:
            continue
        value = _parse_number(metric_row[index])
        if value is None:
            continue
        if len(metric_row) > 1 and metric_row[1] == "total":
            return value
        if fallback is None:
            fallback = value
    return fallback


def _build_metric_charts(metric_rows: List[List[str]]) -> List[dict]:
    """指标统计折线图：横轴为百分位点 Min→P99→Max，仅画连线。"""
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # plotly 缺失时降级为无图表
        logger.warning(f"plotly is unavailable, skip metric percentile charts: {exc}")
        return []

    charts: List[dict] = []
    for title_zh, title_en, y_title, series in METRIC_CHART_SPECS:
        figure = go.Figure()
        has_data = False
        for metric, label, color in series:
            line_points = [(stat, _stat_value(metric_rows, metric, stat))
                           for stat in PERCENTILE_STATS]
            line_points = [(stat, value) for stat, value in line_points
                           if value is not None]
            if not line_points:
                continue
            has_data = True
            figure.add_trace(
                go.Scatter(
                    x=[stat for stat, _ in line_points],
                    y=[value for _, value in line_points],
                    mode="lines+markers",
                    name=label,
                    line={"color": color, "width": 2},
                    marker={"size": 7},
                    hovertemplate="%{x}: %{y}<extra>" + label + "</extra>",
                )
            )
        if not has_data:
            continue
        _style_figure(figure, "Percentile", y_title)
        figure.update_xaxes(type="category", categoryorder="array",
                            categoryarray=PERCENTILE_STATS)
        charts.append({"title_zh": title_zh, "title_en": title_en,
                       "html": _figure_html(figure)})
    return charts


def _build_request_charts(detail_rows: List[dict]) -> List[dict]:
    """请求明细折线图：延迟 / TTFT-TPOT-ITL / 输入输出 token / 成功失败。"""
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # plotly 缺失时降级为无图表
        logger.warning(f"plotly is unavailable, skip request charts: {exc}")
        return []

    indexes = list(range(1, len(detail_rows) + 1))
    charts: List[dict] = []

    def add(title_zh: str, title_en: str, y_title: str, series) -> None:
        figure = go.Figure()
        has_data = False
        for field, label, color in series:
            # 明细字段可能是 list（多模态输入）或缺失，统一走 _parse_number 只保留数值点
            points = [(index, _parse_number(item.get(field)))
                      for index, item in zip(indexes, detail_rows)]
            points = [(index, value) for index, value in points
                      if value is not None]
            if not points:
                continue
            has_data = True
            figure.add_trace(
                go.Scatter(
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                    mode="lines+markers" if len(points) <= DETAIL_MARKER_LIMIT else "lines",
                    name=label,
                    line={"color": color, "width": 2},
                    marker={"size": 5},
                )
            )
        if not has_data:
            return
        _style_figure(figure, "Request Index", y_title)
        charts.append({"title_zh": title_zh, "title_en": title_en,
                       "html": _figure_html(figure)})

    add("时延", "Latency", "Latency (ms)",
        [("latency", "Latency", "#63b3ed")])
    add("首Token/每Token/Token间时延", "TTFT/TPOT/ITL", "Latency (ms)",
        [("ttft", "TTFT", "#63b3ed"),
         ("tpot", "TPOT", "#a78bfa"),
         ("itl", "ITL", "#34d399")])
    add("输入/输出Token数", "Input/Output Tokens", "Tokens",
        [("input_len", "Input Tokens", "#63b3ed"),
         ("output_len", "Output Tokens", "#34d399")])

    status_points = [(index, 1 if item.get("success") else 0)
                     for index, item in zip(indexes, detail_rows)]
    if status_points:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in status_points],
                y=[point[1] for point in status_points],
                mode="lines+markers",
                name="Status",
                line={"shape": "hv", "width": 1.5, "color": CHART_TEXT_COLOR},
                marker={
                    "size": 7,
                    "color": ["#34d399" if point[1] else "#f87171"
                              for point in status_points],
                },
                hovertemplate="Request %{x}: %{y}<extra></extra>",
            )
        )
        _style_figure(figure, "Request Index", "Status")
        figure.update_yaxes(tickvals=[0, 1], ticktext=["fail", "success"],
                            range=[-0.2, 1.2])
        charts.append({"title_zh": "请求状态", "title_en": "Request Status",
                       "html": _figure_html(figure)})
    return charts


def _build_charts(rows: List[dict]) -> Dict[str, List[dict]]:
    """容量曲线：延迟分析（Latency/TTFT/TPOT）与吞吐分析（RPS/Token/成功率）。"""
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # plotly 缺失时降级为空图表
        logger.warning(f"plotly is unavailable, skip capacity charts: {exc}")
        return {}

    charts: Dict[str, List[dict]] = {"latency": [], "throughput": []}
    groups = _sorted_groups(rows)
    multi_group = len(groups) > 1

    def group_prefix(row: dict) -> str:
        if not multi_group:
            return ""
        return (f"{row.get('model_abbr')}/{row.get('dataset_abbr')} "
                f"(size={row.get('request_size')}, prefix={row.get('prefix_len')}) ")

    def add(category: str, title_zh: str, title_en: str,
            traces: List[dict], y_title: str) -> None:
        traces = [trace for trace in traces
                  if any(point[1] is not None for point in trace["points"])]
        if not traces:
            return
        figure = go.Figure()
        for trace in traces:
            points = [point for point in trace["points"] if point[1] is not None]
            figure.add_trace(
                go.Scatter(
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                    mode="lines+markers",
                    name=trace["name"],
                    line={
                        "color": trace["color"],
                        "width": 2,
                        "dash": trace.get("dash"),
                    },
                    marker={"size": 7},
                    fill="tozeroy" if trace.get("fill") else None,
                    fillcolor=trace.get("fillcolor"),
                )
            )
        _style_figure(figure, "Concurrency (batch_size)", y_title)
        # 图内不再重复画标题：Tab 名称与纵坐标标题已表达
        charts[category].append({"title_zh": title_zh,
                                 "title_en": title_en,
                                 "html": _figure_html(figure)})

    latency_pairs = [
        ("Avg Latency", "e2el_avg_s", "P99 Latency", "e2el_p99_s",
         "时延", "Latency", "Latency (s)"),
        ("Avg TTFT", "ttft_avg", "P99 TTFT", "ttft_p99",
         "首Token时延", "TTFT", "TTFT (ms)"),
        ("Avg TPOT", "tpot_avg", "P99 TPOT", "tpot_p99",
         "每Token时延", "TPOT", "TPOT (ms)"),
    ]
    for avg_label, avg_key, p99_label, p99_key, title_zh, title_en, y_title in latency_pairs:
        traces: List[dict] = []
        for _, group_rows in groups:
            if avg_key != "e2el_avg_s" and not group_rows[0].get("ttft_valid"):
                continue
            prefix = group_prefix(group_rows[0])
            traces.append(
                {
                    "name": f"{prefix}{avg_label}",
                    "color": "#63b3ed",
                    "points": [(row.get("batch_size"), row.get(avg_key))
                               for row in group_rows],
                }
            )
            traces.append(
                {
                    "name": f"{prefix}{p99_label}",
                    "color": "#a78bfa",
                    "dash": "dash",
                    "points": [(row.get("batch_size"), row.get(p99_key))
                               for row in group_rows],
                }
            )
        add("latency", title_zh, title_en, traces, y_title)

    rps_traces: List[dict] = []
    for _, group_rows in groups:
        rps_traces.append(
            {
                "name": f"{group_prefix(group_rows[0])}RPS",
                "color": "#63b3ed",
                "fill": True,
                "fillcolor": "rgba(99,179,237,0.08)",
                "points": [(row.get("batch_size"), row.get("request_throughput"))
                           for row in group_rows],
            }
        )
    add("throughput", "请求吞吐量", "Request Throughput", rps_traces,
        "requests/s")

    token_traces: List[dict] = []
    for _, group_rows in groups:
        prefix = group_prefix(group_rows[0])
        token_traces.append(
            {
                "name": f"{prefix}Output tok/s",
                "color": "#34d399",
                "fill": True,
                "fillcolor": "rgba(52,211,153,0.08)",
                "points": [(row.get("batch_size"), row.get("throughput_tok_s"))
                           for row in group_rows],
            }
        )
        token_traces.append(
            {
                "name": f"{prefix}Total tok/s",
                "color": "#a78bfa",
                "dash": "dash",
                "points": [(row.get("batch_size"),
                            row.get("total_throughput_tok_s"))
                           for row in group_rows],
            }
        )
    add("throughput", "Token吞吐量", "Token Throughput", token_traces,
        "tokens/s")

    success_traces: List[dict] = []
    for _, group_rows in groups:
        success_traces.append(
            {
                "name": f"{group_prefix(group_rows[0])}Success Rate",
                "color": "#fbbf24",
                "fill": True,
                "fillcolor": "rgba(251,191,36,0.08)",
                "points": [(row.get("batch_size"), row.get("success_rate"))
                           for row in group_rows],
            }
        )
    add("throughput", "成功率", "Success Rate", success_traces,
        "Success Rate (%)")

    return charts


def _resolve_db_data(perf_dir: Path, db_name: str):
    """按 PerfViz 同规则定位并读取 numpy_store。"""
    from ais_bench.benchmark.openicl.icl_inferencer.output_handler.db_utils import (
        init_db,
        load_all_numpy_from_db,
    )

    for candidate in (
        perf_dir / "db_data" / db_name,
        perf_dir / "tmp" / f"tmp_{db_name}",
    ):
        if not candidate.exists():
            continue
        try:
            conn = init_db(str(candidate))
            try:
                return load_all_numpy_from_db(conn)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(f"Failed to read perf db {candidate}: {exc}")
    return {}


def _load_detail_rows(perf_dir: Path, dataset_abbr: str) -> List[dict]:
    """读取逐请求明细并还原 time_points，用于逐组合的请求明细表。"""
    path = perf_dir / f"{dataset_abbr}_details.jsonl"
    if not path.exists():
        return []
    grouped = defaultdict(list)
    try:
        with open(path, "rb") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                db_name = record.get("db_name")
                if db_name:
                    grouped[db_name].append(record)
    except OSError as exc:
        logger.warning(f"Failed to load detail rows {path}: {exc}")
        return []

    rows: List[dict] = []
    for db_name, records in grouped.items():
        numpy_data = _resolve_db_data(perf_dir, db_name)
        for record in records:
            input_tokens = record.get("input_tokens")
            if not record.get("success", True):
                rows.append({"success": False, "latency": None, "ttft": None,
                             "tpot": None, "itl": None, "input_len": input_tokens,
                             "output_len": None})
                continue
            time_points = record.get("time_points")
            if isinstance(time_points, dict) and "__db_ref__" in time_points:
                time_points = numpy_data.get(time_points["__db_ref__"])
            if time_points is None:
                rows.append({"success": False, "latency": None, "ttft": None,
                             "tpot": None, "itl": None, "input_len": input_tokens,
                             "output_len": None})
                continue
            time_points = np.asarray(time_points, dtype=float).ravel()
            if time_points.size < 2:
                rows.append({"success": False, "latency": None, "ttft": None,
                             "tpot": None, "itl": None, "input_len": input_tokens,
                             "output_len": None})
                continue
            latency = float(time_points[-1] - time_points[0])
            ttft = float(time_points[1] - time_points[0])
            output_tokens = record.get("output_tokens")
            tpot = None
            if isinstance(output_tokens, int) and output_tokens > 1:
                tpot = (latency - ttft) / (output_tokens - 1) * 1000
            # 平均 ITL：首个 token 之后相邻 token 的间隔均值（与 EvalScope 的 avg inter-token latency 同口径）
            itl = None
            gaps = np.diff(time_points[1:])
            if gaps.size:
                itl = float(gaps.mean()) * 1000
            rows.append(
                {
                    "success": True,
                    "latency": latency * 1000,
                    "ttft": ttft * 1000,
                    "tpot": tpot,
                    "itl": itl,
                    "input_len": input_tokens,
                    "output_len": output_tokens,
                }
            )
    return rows


def _load_template():
    """惰性导入 jinja2 并加载模板（依赖缺失返回 None）。"""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except Exception as exc:
        logger.warning(f"jinja2 is unavailable, skip HTML report: {exc}")
        return None
    template_dir = Path(__file__).resolve().parent / "templates"
    if not (template_dir / TEMPLATE_NAME).exists():
        logger.warning(f"Report template not found: {template_dir / TEMPLATE_NAME}")
        return None
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "htm", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(TEMPLATE_NAME)


def _best_configs(rows: List[dict]) -> dict:
    """最佳配置：最高吞吐与最低平均 TTFT 两个口径分别标注。"""
    ok_rows = [row for row in rows if row.get("status") == "success"]
    best_throughput = None
    for row in ok_rows:
        value = row.get("throughput_tok_s")
        if value is None:
            continue
        if best_throughput is None or value > best_throughput["throughput_tok_s"]:
            best_throughput = row
    best_ttft = None
    for row in ok_rows:
        if not row.get("ttft_valid"):
            continue
        value = row.get("ttft_avg")
        if value is None:
            continue
        if best_ttft is None or value < best_ttft["ttft_avg"]:
            best_ttft = row
    return {"throughput": best_throughput, "ttft": best_ttft}


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


def _recommendations(rows: List[dict], best: dict) -> List[Tuple[str, str]]:
    """性能建议：饱和判定 + 最佳档位 + 延迟增长 + 失败组合提示（中英双语）。"""
    notes = [_saturation_note(rows)]
    best_throughput = best.get("throughput")
    if best_throughput:
        batch_size = _display(best_throughput.get("batch_size"))
        tok_s = _display(best_throughput.get("throughput_tok_s"))
        rps = _display(best_throughput.get("request_throughput"))
        success = _display(best_throughput.get("success_rate_text"))
        notes.append(
            ("建议并发档：batch_size={}，输出吞吐 {} token/s，RPS {}，成功率 {}。".format(
                batch_size, tok_s, rps, success),
             "Recommended concurrency: batch_size={}, output throughput {} token/s, "
             "RPS {}, success rate {}.".format(batch_size, tok_s, rps, success))
        )
    latencies = sorted(
        (row.get("batch_size") or 0, row.get("e2el_avg_s"))
        for row in rows
        if row.get("status") == "success" and row.get("e2el_avg_s") is not None
    )
    if len(latencies) >= 2:
        first_batch, first_latency = latencies[0]
        last_batch, last_latency = latencies[-1]
        if first_latency:
            ratio = round(last_latency / first_latency, 2)
            notes.append(
                ("端到端平均时延从 batch_size={} 的 {} s 增长到 batch_size={} 的 {} s（{}×）。".format(
                    first_batch, _display(first_latency), last_batch,
                    _display(last_latency), ratio),
                 "Average end-to-end latency grows from {} s at batch_size={} "
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


def render_batch_perf_report(rows: List[dict], manifest: dict,
                             warnings: List[str], out_dir: Path) -> Optional[Path]:
    """以与 CSV 同源的 rows 渲染 `batch_perf_report.html`；异常只告警。"""
    template = _load_template()
    if template is None:
        return None
    try:
        charts = _build_charts(rows)
        report_rows = []
        for row in rows:
            formatted = _format_row(row)
            report_rows.append(
                {
                    "status": row.get("status"),
                    "cells": [formatted.get(key) or "—"
                              for _, key in OVERVIEW_COLUMNS],
                }
            )
        details = []
        for row in rows:
            if row.get("status") != "success":
                continue
            perf_dir = out_dir / "performances" / str(row.get("model_abbr"))
            detail_rows = _load_detail_rows(perf_dir, str(row.get("dataset_abbr")))
            metric_headers, metric_rows = _metric_rows_for_row(row, out_dir)
            details.append(
                {
                    "label": "BatchSize {} · NumPrompts {}".format(
                        _display(row.get("batch_size")),
                        _display(row.get("num_prompts")
                                 if row.get("num_prompts") is not None
                                 else row.get("total_requests")),
                    ),
                    "badge_zh": "{} 个请求".format(
                        _display(row.get("total_requests"))),
                    "badge_en": "{} requests".format(
                        _display(row.get("total_requests"))),
                    "stats": _run_summary(row),
                    "metric_headers": metric_headers,
                    "metric_table": metric_rows,
                    "metric_charts": _build_metric_charts(metric_rows),
                    "request_charts": _build_request_charts(detail_rows),
                    "request_total": len(detail_rows),
                }
            )

        combos = manifest.get("combos") or []
        first_combo = combos[0] if combos else {}
        first_model_cfg = first_combo.get("model_config") or {}
        best = {
            key: (_format_row(value) if value else None)
            for key, value in _best_configs(rows).items()
        }
        context = {
            "marker": REPORT_MARKER,
            "plotly_cdn": PLOTLY_CDN_URL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "ais_bench_version": __version__,
            "schema_version": manifest.get("schema_version"),
            "combo_count": len(combos),
            "model_label": _display(
                first_model_cfg.get("model_name")
                or first_model_cfg.get("abbr")
                or first_combo.get("model_abbr")),
            "dataset_label": _display(first_combo.get("dataset_abbr")),
            "basic_info": _basic_info(rows, first_combo),
            "overview_headers": OVERVIEW_HEADERS,
            "report_rows": report_rows,
            "charts": charts,
            "best": best,
            "saturation": _saturation_note(rows)[0],
            "recommendations": _recommendations(rows, best),
            "details": details,
            "warnings": warnings or [],
        }
        html = template.render(**context)
        if not html.startswith("<!DOCTYPE html>"):
            html = "<!DOCTYPE html>\n" + html
        path = out_dir / REPORT_HTML_NAME
        tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as file:
                file.write(html)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, path)
        except OSError:
            try:
                if tmp_path.exists():
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
        return path
    except Exception as exc:
        logger.warning(f"Failed to render batch perf report: {exc}")
        return None


def aggregate_batch_perf(work_dir, manifest: Optional[dict] = None,
                         warnings: Optional[List[str]] = None) -> BatchPerfArtifacts:
    """聚合 manifest 内各组合的性能结果，产出 CSV 与 HTML 报告。

    manifest 不存在或组合数 ≤ 1 时跳过（单组合零开销）；个别组合结果缺失时该行
    标记 fail，不阻塞其余行；HTML 渲染失败只告警，CSV 正常返回。
    """
    work_path = Path(work_dir)
    manifest = manifest if manifest is not None else _load_manifest(work_path)
    if not manifest:
        logger.debug("No sweep manifest found, skip batch perf aggregation.")
        return BatchPerfArtifacts()
    combos = manifest.get("combos") or []
    if len(combos) <= 1:
        logger.debug("Single combo run, skip batch perf aggregation.")
        return BatchPerfArtifacts()

    rows = [_build_row(work_path, combo) for combo in combos]
    # 行序按 batch_size 升序，"并发-吞吐曲线"可直接从表读出
    rows.sort(key=lambda row: (row.get("batch_size") or 0))
    failed = [row.get("model_abbr") for row in rows if row.get("status") == "fail"]
    if failed:
        logger.warning(
            "Batch perf results missing for combo(s): %s (marked as fail rows).",
            ", ".join(str(item) for item in failed),
        )

    csv_path = work_path / SUMMARY_CSV_NAME
    try:
        _write_summary_csv(csv_path, rows, work_path)
    except OSError as exc:
        logger.warning(f"Failed to write {csv_path}: {exc}")
        return BatchPerfArtifacts()
    logger.info(f"Batch perf summary written to {csv_path}")

    html_path = render_batch_perf_report(rows, manifest, list(warnings or []), work_path)
    if html_path is not None:
        logger.info(f"Batch perf report written to {html_path}")
    return BatchPerfArtifacts(csv_path=csv_path, html_path=html_path)
