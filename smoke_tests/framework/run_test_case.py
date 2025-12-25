import json
import os
import subprocess
import sys
import re
import traceback
import time
from typing import List, Dict, Any, Tuple, Optional, Union

import yaml
from run_log import log

class RunCaseException(Exception):
    pass


class CleanCaseException(Exception):
    pass


class Config(dict):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.__dict__ = self


def read_yaml_config(test_case_yml: str, default_config: Dict):
    config = default_config
    with open(test_case_yml) as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
        config.update(yaml_config)
    return config


def get_test_case_config(test_case_yml):
    config = Config(
        case_type="default",
        case_group="default",
        case_tags=[],
        case_name=test_case_yml,
        script=Config(
            start="run.sh",
            clean="clean.sh",
        ),
        enable=True,
        metric="metric.json",
        metric_check_rule=None,
        timeout=3,
        rank_size=1,
        yml_path=test_case_yml
    )
    config = read_yaml_config(test_case_yml, config)
    config["case_group"] = sorted(config.get("case_group", []))
    # 创建空集合存储标签
    case_tags = set()
    # 智能添加字段，处理单值和多值场景
    def safe_add(value):
        if isinstance(value, list):
            case_tags.update(value)
        else:
            case_tags.add(value)  # 处理字符串、数字等单值
    
    safe_add(config.get("case_group", []))
    safe_add(config.get("case_type", "default_type"))
    safe_add(config.get("case_name", test_case_yml))
    config["case_tags"] = set(sorted(case_tags))
    return config


def exec_script_file(
    script_path: str, 
    log_file: Optional[str] = None,
    timeout_minutes: Optional[int] = None,
) -> int:
    """
    执行脚本文件（支持Python和Bash），并设置超时时间
    
    参数:
        script_path: 脚本文件路径
        timeout_minutes: 超时时间（分钟）
        log_file: 日志文件路径，如果提供，则重定向输出到此文件
    
    返回:
        返回码: 0表示成功，非0表示失败
    
    异常:
        超时时抛出RunCaseException
    """
    # 根据文件扩展名确定执行命令
    if script_path.endswith(".py"):
        python_command = sys.executable or "python3"
        args = [python_command, script_path]
    elif script_path.endswith(".sh"):
        args = ["bash", script_path]
    else:
        raise RunCaseException(f"Unsupported script type: {script_path}")
    
    process:subprocess.Popen = None
    try:
        # 处理日志重定向
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        if log_file:
            stdout_dest = open(log_file, 'w')
            stderr_dest = subprocess.STDOUT
        
        # 创建子进程执行命令
        process = subprocess.Popen(
            args,
            stdout=stdout_dest,
            stderr=stderr_dest,
            text=True
        )

        try:
            if timeout_minutes is not None:
                # 有超时设置
                process.wait(timeout=timeout_minutes * 60)
            else:
                # 无超时设置
                process.wait()
                
            return process.returncode
        
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as ex:
            # 超时处理 - 先尝试终止
            process.terminate()
            try:
                # 等待3秒正常退出
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # 如果3秒后还没退出，强制杀死
                process.kill()
            finally:
                if timeout_minutes and isinstance(ex, subprocess.TimeoutExpired):
                    raise RunCaseException(f"Timeout after {timeout_minutes} minutes for script: {script_path}")
    except Exception as e:
        print(e)
        raise RunCaseException(f"Error executing script {script_path}: {str(e)}")
    finally:
        # 确保文件句柄关闭
        if log_file and stdout_dest != subprocess.DEVNULL:
            stdout_dest.close()


def pre_processing(config, test_case_workspace_path, res_path):
    pass


def post_processing(config, test_case_workspace_path):
    # 调用clean.sh
    clean_script = config.get("script", {}).get("clean", "clean.sh")
    clean_script_path = os.path.join(test_case_workspace_path, clean_script)
    if os.path.exists(clean_script_path):
        exec_script_file(clean_script_path)


def record_and_analysis_metric(config, test_case_workspace_path, result_record):
    # record result
    metric_json_file = config.get("metric", "metric.json")
    result_file_path = os.path.join(test_case_workspace_path, metric_json_file)
    if not os.path.exists(result_file_path):
        return

    try:
        with open(result_file_path, "r") as json_file:
            metric_result = json.load(json_file)
    except Exception as ex:
        raise RunCaseException(ex) from ex

    result_record["metric_result"] = metric_result

    metric_check_rule: Dict = config.get("metric_check_rule", None)
    result_record["metric_check_rule"] = metric_check_rule
    if not metric_check_rule:
        return

    try:
        failed_message = ""
        metric_check_result = {}
        for metric_key, metric_rule in metric_check_rule.items():
            if metric_key not in metric_result:
                failed_message += f"metric result has no key {metric_key}."
                metric_check_result[metric_key] = False
                continue
            metric_value = float(metric_result[metric_key])
            if len(metric_rule) < 2:
                raise RunCaseException("The metric check rule is incorrectly configured. It must contain two elements.")
            min_metric_value = float(metric_rule[0])
            max_metric_value = float(metric_rule[1])

            if min_metric_value is not None and metric_value < min_metric_value:
                failed_message += f"{metric_key} value {metric_value} less than the minimum value {min_metric_value}."
                metric_check_result[metric_key] = False
                continue

            if max_metric_value is not None and metric_value > max_metric_value:
                failed_message += f"{metric_key} value {metric_value} more than the maximum  value {max_metric_value}."
                metric_check_result[metric_key] = False
                continue
            metric_check_result[metric_key] = True

        result_record["metric_check_result"] = metric_check_result
        if not all(metric_check_result.values()):
            raise RunCaseException(failed_message)
    except Exception as ex:
        raise RunCaseException(ex) from ex


def get_metric_key(path:str, start_from: str = "test-case"):
    normalized = re.sub(r"\\", "/", path).rstrip("/")
    parts = normalized.split("/")

    # 查找 start_from 位置
    try:
        start_index = parts.index(start_from)
    except ValueError:
        raise ValueError(f"路径中必须包含 {start_from} 目录")
    
    # 不包含 start_from
    target_parts = parts[start_index + 1:]
    return target_parts


def get_last_lines(file_path: str, last_lines:int = 2) -> list[str]:
    """返回文件最后 last_lines 行（默认2行）的列表，跳过空白行"""
    try:
        with open(file_path, 'rb') as f:
            # 智能决定读取窗口大小（最大10KB）
            size = min(10240, os.path.getsize(file_path))
            f.seek(-size, os.SEEK_END)
            
            # 读取最后部分并过滤非空白行
            lines = [
                line.decode(errors='ignore').rstrip()
                for line in f
                if line.strip()
            ]
            
            # 返回最后两个非空白行
            return lines[-last_lines:] if len(lines) >= last_lines else lines
    
    except Exception:
        # 回退到标准方法
        with open(file_path) as f:
            return [line.rstrip() for line in f if line.strip()][-last_lines:]

def run_test_case(config, test_case_workspace_path, res_path):
    result_record = dict(
        case_name=config.get("case_name", config.get("yml_path", "default")), success=False, message="未执行",
        metric_key=get_metric_key(test_case_workspace_path),
        metric_result={}, metric_check_rule=None, metric_check_result={},
        log_file_path="", execution_time=None
    )
    try:
        try:
            # 运行前处理
            # pre_processing(config, test_case_workspace_path, res_path)

            # 运行
            run_script = config.get("script", {}).get("start", "run.sh")
            run_script_path = os.path.join(test_case_workspace_path, run_script)
            log_path = os.path.join(test_case_workspace_path, "result.log")
            result_record["log_file_path"] = log_path
            if not os.path.exists(run_script_path):
                raise RunCaseException(f"run script path not exists. {run_script_path}")
            start_time = time.perf_counter()
            ret_code = exec_script_file(run_script_path,
                                        log_file=log_path,
                                        timeout_minutes=config.get("timeout", 3)
                                        )
            if ret_code != 0:
                result_log_info = "Last 2 rows in result.log: \n```\n" + str('\n'.join(get_last_lines(log_path))) + "\n```"
                error_info = f"run script path failed. {run_script_path}\n{result_log_info}"
                raise RunCaseException(error_info)
            # 记录&分析结果
            record_and_analysis_metric(config, test_case_workspace_path, result_record)
        except RunCaseException as ex:
            result_record["success"] = False
            result_record["message"] = str(ex)
            log(f"\033[1;31m", f"[{config.get('case_name')}] has ERROR:", str(ex), "\033[0m", print_enable=False)
        except KeyboardInterrupt:
            result_record["success"] = False
            result_record["message"] = "Has been interrupted"
            log(f"\033[1;31m", f"[{config.get('case_name')}] has been interrupted.\033[0m", print_enable=False)
        else:
            result_record["success"] = True
            result_record["message"] = ""
            result_record["execution_time"] = time.perf_counter() - start_time


        # 后处理
        try:
            post_processing(config, test_case_workspace_path)
        except CleanCaseException as ex:
            log(f"\033[5;33m", f"[{config.get('case_name')}] has WARRING:", str(ex), "\033[0m", print_enable=False)
    except Exception:
        log(f"\033[5;31m", f"[{config.get('case_name')}] has ERROR:", str(ex), "\033[0m", print_enable=False)

    return config.get("case_type"), config.get("case_name"), result_record
