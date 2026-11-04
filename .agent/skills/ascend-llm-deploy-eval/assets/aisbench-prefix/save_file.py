import os
import re
import logging
import shutil
import traceback
import pandas as pd
from datetime import datetime
from runtime_config import MODEL_PATH
logging.getLogger().setLevel(logging.INFO)

DEFAULT_METRICS = {
    "current_time": None,
    "input_len": 99999,
    "output_len": 99999,
    "total_req": 99999,
    "max_cc": 99999,
    "cc": 99999,
    "rr": 0,
    "repeat_rate": 0,
    "TTFT avg": 99999,
    "TTFT P90": 99999,
    "TPOT avg": 99999,
    "TPOT SLO_P90": 99999,
    "E2E_time": 99999,
    "output_throughput": 99999,
    "single_output_throughput": 99999,
    "E2E_throughput": 9999,
    "single_E2E_throughput": 9999,
    "qps": 9999,
    "qpm": 9999,
    "input_token_throughput": 9999,
    "prefill_token_throughput": 9999,
    "E2EL avg": 9999,
    "E2EL P90": 9999
}

_METRIC_RULES = [
    ("E2EL",                      r'(\d+\.\d+)', 5, 1.0,    None),
    ("TTFT",                      r'(\d+\.\d+)', 5, 1.0,    None),
    ("TPOT",                      r'(\d+\.\d+)', 5, 1.0,    None),
    ("Benchmark Duration",        r'(\d+\.\d+)', 0, 1/1000, None),
    ("Concurrency",               r'(\d+\.\d+)', 0, 1.0,    None),
    ("Output Token Throughput",   r'(\d+\.\d+)', 0, 1.0,    lambda v, n: {
        "output_throughput": v, "single_output_throughput": v / n}),
    ("Input Token Throughput",    r'(\d+\.\d+)', 0, 1.0,    None),
    ("Total Token Throughput",    r'(\d+\.\d+)', 0, 1.0,    lambda v, n: {
        "E2E_throughput": v, "single_E2E_throughput": v / n}),
    ("InputTokens",               r'(\d+\.?\d*)', 0, 1.0,   None),
    ("OutputTokens",              r'(\d+\.?\d*)', 0, 1.0,   None),
    ("Total Requests",            r'(\d+\.?\d*)', 0, 1.0,   None),
    ("Request Throughput",        r'(\d+\.\d+)', 0, 1.0,    lambda v, n: {
        "qps": v, "qpm": v * 60}),
    ("Prefill Token Throughput",  r'(\d+\.\d+)', 0, 1.0,    None),
]


def _extract_value(line, pattern, index, scale):
    matches = re.findall(pattern, line)
    if not matches or index >= len(matches):
        return None
    try:
        return float(matches[index]) * scale
    except (ValueError, TypeError):
        return None


def _get_field_name(keyword):
    name_map = {
        "E2EL": ["E2EL P90", "E2EL avg"],
        "TTFT": ["TTFT P90", "TTFT avg"],
        "TPOT": ["TPOT SLO_P90", "TPOT avg"],
        "Benchmark Duration": ["E2E_time"],
        "Concurrency": ["cc"],
        "Max Concurrency": ["max_cc"],
        "Output Token Throughput": ["output_throughput"],
        "Input Token Throughput": ["input_token_throughput"],
        "Total Token Throughput": ["E2E_throughput"],
        "InputTokens": ["input_len"],
        "OutputTokens": ["output_len"],
        "Total Requests": ["total_req"],
        "Request Throughput": ["qps"],
        "Prefill Token Throughput": ["prefill_token_throughput"],
    }
    return name_map.get(keyword, [])


def get_data(aisbench_log, req_rate, npu_num, repeat_rate):
    log_dir = ""
    metrics = dict(DEFAULT_METRICS)
    try:
        with open(aisbench_log, 'r') as f_streaming:
            txt = f_streaming.readlines()
            for line in txt:
                if "Current exp folder" in line:
                    matches = re.search(r"Current exp folder:\s*(.+)$", line)
                    if matches:
                        log_dir = matches.group(1).strip()
                if "Max Concurrency" in line:
                    matches = re.findall(r"[\w']+", line)
                    if matches:
                        metrics["max_cc"] = matches[-1]
                    continue
                for keyword, pattern, index, scale, extra_fn in _METRIC_RULES:
                    if keyword in line:
                        raw_value = _extract_value(line, pattern, index, scale)
                        if raw_value is not None:
                            if extra_fn:
                                metrics.update(extra_fn(raw_value, npu_num))
                            else:
                                if keyword in ("E2EL", "TTFT", "TPOT"):
                                    p90_key = f"{keyword} {'SLO_' if keyword == 'TPOT' else ''}P90"
                                    avg_key = f"{keyword} avg"
                                    if index == 5:
                                        metrics[p90_key] = raw_value
                                    avg_value = _extract_value(line, pattern, 0, scale)
                                    if avg_value is not None:
                                        metrics[avg_key] = avg_value
                                else:
                                    field_names = _get_field_name(keyword)
                                    if field_names:
                                        metrics[field_names[0]] = raw_value
                        break
        metrics["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["rr"] = req_rate
        metrics["repeat_rate"] = repeat_rate
    except Exception as e:
        logging.warning(traceback.format_exc())
    logging.info(metrics)
    return metrics, log_dir


def save_log(aisbench_log, log_dir):
    # 保存到 aisbench 的 Current exp folder 路径
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copy2(aisbench_log, log_dir)
        logging.info(f"aisbench.log copied to: {log_dir}")
    # aisbench.log 本身就在当前运行 python 命令的目录（无需额外复制）
    # aisbench_all.log 也保存在当前目录（由脚本直接写入）
    logging.info(f"aisbench.log saved at current directory: {os.getcwd()}")


def save_csv(ans, filename):
    file_exists = os.path.exists(filename)
    # 从 config.py 的 MODEL_PATH 中取最后一段路径作为 model_name，作为 CSV 第一列
    model_name = MODEL_PATH.split("/")[-1]
    ans = {"model_name": model_name, **ans}
    df_new = pd.DataFrame([ans])
    try:
        if file_exists:
            df_existing = pd.read_csv(filename)
            logging.info("文件已存在，读取现有数据")
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated = df_updated.reindex(columns=df_existing.columns.union(df_new.columns, sort=False))
            # 确保 model_name 始终是第一列
            df_updated = df_updated.reindex(
                columns=["model_name"] + [c for c in df_updated.columns if c != "model_name"])
            df_updated.to_csv(filename, index=False)
            logging.info("成功追加新行")
        else:
            df_new.to_csv(filename, index=False)
            logging.info("创建新文件并写入数据")
    except Exception as e:
        logging.error(f"操作失败: {e}")
