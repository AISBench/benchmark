import os
import sys
import json
import time
import signal
import psutil
import builtins
import argparse
import traceback
import threading
import subprocess
import itertools

from io import StringIO
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed


from run_log import initialize_logging, log, Logger
from record import record, get_default_record_path
from run_test_case import run_test_case, get_test_case_config
from utils import (WorkspaceChange, print_json, change_all_target_key_to_target_value,
                   have_common_same_key_values, find_all_values, get_spinner_frames)
from create_excel import create_excel
from schedule_task import TaskScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_path', dest="project_path", help='project path', required=True)
    parser.add_argument('--test-case', dest="test_case_path", help='test case path', required=True)
    parser.add_argument('--workspace-path', dest="workspace_path", help='workspace path', required=True)
    parser.add_argument('--resource-path', dest="res_path", help='resource path', required=True)
    parser.add_argument('--case-tags', dest="case_tags", nargs='*',
                        help='case tags or type, split by space, run cases which contain all tags', default=None)
    parser.add_argument('--case-label', dest="case_label", nargs='*', help='case label with value: key:value', default=None)
    parser.add_argument('--dev-mode', type=int, dest="dev_mode", help='dev mode', default='0')
    parser.add_argument('--send-email', type=int, dest="send_email", help='send-email', default='1')
    parser.add_argument('--rerun-mode', type=int, dest="rerun_failed", help='rerun-failed-cases', default='0')

    args = parser.parse_args()
    return args


def set_test_env(args):
    project_path = args.project_path
    os.environ["PROJECT_PATH"] = project_path

    workspace_path = args.workspace_path
    os.makedirs(workspace_path, exist_ok=True)
    os.environ["WORKSPACE_PATH"] = workspace_path

    res_path = args.res_path
    os.environ["RESOURCE_PATH"] = res_path


def capture_system_info(args):
    """
    收集和记录系统环境信息

    包括:
    - 系统umask值
    - Git仓库最近两次提交日志
    - 测试配置信息（用例路径、标签、工作路径等）
    - 系统命令执行结果（系统信息、安装包、环境变量等）

    Args:
        args (argparse.Namespace): 命令行参数对象

    Returns:
        str: 包含所有系统信息的格式化字符串
    """
    results = StringIO()

    # 捕获umask
    mask = oct(os.umask(0)) # 调用 os.umask(0) 在传递 0 作为参数时，只会获取当前的umask 值，而不会修改它。
    results.write(f"umask: {mask}\n")

    # 捕获git日志
    project_path = os.getenv('PROJECT_RESOURCE_PATH')
    if project_path:
        git_dir = os.path.join(project_path, 'code', 'benchmark')
        if os.path.exists(git_dir):
            try:
                git_log = subprocess.check_output(
                    ['git', '-C', git_dir, 'log', '-2'],
                    text=True, stderr=subprocess.STDOUT
                )
                results.write(f"git -C {git_dir} log -2:\n{git_log}\n")
            except subprocess.CalledProcessError as e:
                results.write(f"Git错误:\n{e.output}\n")
        else:
            results.write("Git目录不存在\n")

    # 捕获测试信息
    test_info = {
        'case_path': str(args.test_case_path),
        'tags': str(args.case_tags),
        'label': str(args.case_label),
        'work_path': str(args.workspace_path),
        'project_path': str(args.project_path),
        'resource_path': str(args.res_path)
    }
    results.write(f"运行用例路径: {test_info['case_path']}\n")
    results.write(f"运行用例: [{test_info['tags']}][{test_info['label']}]\n")
    results.write(f"运行工作路径: {test_info['work_path']}\n")
    results.write(f"运行工程路径: {test_info['project_path']}\n")
    results.write(f"运行资源路径: {test_info['resource_path']}\n")

    # 捕获系统命令
    commands = {
        "系统信息": "uname -a",
        "安装包": "pip list",
        "环境变量": "env",
        "cpu型号": "lscpu"  # 根据实际情况调整命令
    }

    for desc, cmd in commands.items():
        try:
            output = subprocess.check_output(cmd.split(),
                                            text=True,
                                            stderr=subprocess.STDOUT)
            results.write(f"{desc}:\n{output}\n")
        except subprocess.CalledProcessError as e:
            results.write(f"{desc}错误:\n{e.output}\n")
        except FileNotFoundError:
            results.write(f"{desc}命令不可用\n")


    return results.getvalue()


def get_test_cases_from_path(test_case_path: str):
    if os.path.isfile(test_case_path):
        if test_case_path.endswith(".yml"):
            yield test_case_path
    elif os.path.isdir(test_case_path):
        files = list(os.listdir(test_case_path))
        if "case.yml" in files:
            yield os.path.join(test_case_path, "case.yml")
            return

        for file_name in files:
            file_path = os.path.join(test_case_path, file_name)
            if not os.path.isdir(file_path):
                continue
            for path in get_test_cases_from_path(file_path):
                yield path


def execute_test_cases(execute_config_list:list[dict] , test_case_result = None):
    """
    执行测试用例的主入口函数

    实现:
    1. 按rank_size分组排序，同rank组内并行执行，当组内数量超过阈值则使用LPT算法进行调度排序
    2. 使用ProcessPoolExecutor管理并行进程
    3. 实时显示进度条(tqdm)
    4. 收集日志和执行结果
    5. 聚合度量指标

    Args:
        execute_config_list (list[dict]): 待执行的测试用例配置列表
        test_case_result (dict): 用于存储所有测试结果的字典
    """
    if test_case_result is None:
        log(f"\033[5;31m", "The test_case_result cannot be None!", "\n\n\033[0m", flush=True)
        return

    res_path = os.getenv("RESOURCE_PATH", default = None)
    if res_path is None:
        log(f"\033[5;31m", "The args.res_path/'RESOURCE_PATH' cannot be empty!", "\n\n\033[0m", flush=True)
        return

    workspace_path = os.getenv("WORKSPACE_PATH", default = None)
    if workspace_path is None:
        log(f"\033[5;31m", "The args.workspace_path/'WORKSPACE_PATH' cannot be empty!", "\n\n\033[0m", flush=True)
        return

    # 设置用例执行顺序相关逻辑
    # 创建调度器实例
    schedule_file = os.path.join(res_path, "others", "execute_time_record.json")
    scheduler = TaskScheduler(schedule_file)

    # 排序逻辑：rank_size -> case_tags (此处case_tags排序无意义, 因为是按照rank_size进行并行执行用例, 所有相同的rank一起执行, 无先后顺序)
    execute_test_cases_config_list = sorted(
            execute_config_list,
            key=lambda e: (e['rank_size'], e['case_tags'])
        )

    # 2. 按rank_size分组
    rank_groups = defaultdict(list)
    for config in execute_test_cases_config_list:
        rank_groups[config['rank_size']].append(config)

    # 3. 按rank_size排序
    sorted_ranks = sorted(rank_groups.keys())

    # 全局记录进程树，用于安全终止
    process_tree = {}
    def _terminate_process_group(group_id):
        """安全终止进程组及其子进程树"""
        if group_id not in process_tree:
            return

        executor = process_tree[group_id]['executor']

        try:
            # 立即关闭执行器，取消所有未开始的任务
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            log(f"\033[33m关闭执行器失败: {str(e)}\033[0m")

        # 终止工作进程及其子进程
        for pid in process_tree[group_id]['processes']:
            try:
                parent = psutil.Process(pid)

                # 1. 递归终止所有子进程
                for child in parent.children(recursive=True):
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # 子进程已退出

                # 2. 终止父进程
                try:
                    parent.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                # 3. 统一等待所有进程退出
                try:
                    _, alive = psutil.wait_procs(
                        [parent] + parent.children(recursive=True),
                        timeout=1.0
                    )
                except Exception:
                    alive = []

                # 4. 强制终止未退出的进程
                for proc in alive:
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # 进程已不存在
            except Exception as e:
                log(f"\033[33m终止进程{pid}失败: {str(e)}\033[0m")
        del process_tree[group_id]

    cases_record = {
        "pending_cases" : find_all_values(execute_test_cases_config_list, "case_name"), # 存储待执行但尚未开始的用例名称
        "running_cases" : [], # 存储已开始但未完成的用例名称
        "finished_cases": {   # 存储已完成的用例名称
            "success" : [],
            "failed"  : []
        }
    }

    result_container = {
        'record': cases_record,
        'interrupted': False
    }

    spinner_clock = itertools.cycle(get_spinner_frames("clock"))
    spinner_pulse = itertools.cycle(get_spinner_frames("pulse"))
    spinner_ripple = itertools.cycle(get_spinner_frames("ripple"))

    MAXIMUM_CONCURRENCY_NUM = 50 # 设置最大并行度

    total_start_time = time.perf_counter()

    # 记录执行开始总时间
    log(f"Test Execution Started at {time.ctime()}\n")
    try:
        # 4. 按rank_size组串行执行
        for rank in sorted_ranks:
            group_configs = rank_groups[rank]
            group_count = len(group_configs)

            # 5. 确定并行度（限制最大并行数量）
            max_workers = min(MAXIMUM_CONCURRENCY_NUM, group_count)

            # 每个rank组内的调度排序
            if group_count > MAXIMUM_CONCURRENCY_NUM:
                group_configs = scheduler.schedule_LPT(group_configs)

            rank_info = {
                "success": 0,
                "failed" : 0
            }
            log(f"\n\033[34mStarting rank group {rank} ({group_count} test cases)\033[0m", flush=True)

            current_start_time = time.perf_counter()

            # 6. 使用进程池执行当前rank组
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                group_id = f"rank_{rank}"
                process_tree[group_id] = {'executor': executor, 'processes': set()}

                stop_spinner = threading.Event()

                # 创建进度条
                pbar = tqdm(
                    total=group_count,
                    desc="",
                    dynamic_ncols=True,  # 自适应宽度
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    leave=False
                )

                def update_spinner_animation():
                    while not stop_spinner.is_set():
                        time.sleep(0.1)
                        try:
                            spinner_info = (
                                               f"| {next(spinner_clock)}"
                                               f" {next(spinner_pulse)}"
                                               f" {next(spinner_ripple)} | "
                                              )
                            cases_info = (
                                            f"Rank Group {rank}"
                                            f" | Success: {rank_info['success']}"
                                            f" | Failed: {rank_info['failed']}"
                                            f" | 进程数：{len(process_tree[group_id]['processes'])}"
                                            f" | "
                                            )
                            mem_usage = psutil.virtual_memory().percent
                            cpu_usage = psutil.cpu_percent()
                            resource_info = f"CPU: {cpu_usage}% | MEM: {mem_usage}%"
                            pbar.set_description(spinner_info + cases_info + resource_info)
                        except:
                            break

                # 启动旋转动画
                spinner_thread = threading.Thread(target=update_spinner_animation, daemon=True)
                spinner_thread.start()

                for config in group_configs:
                    # 获取用例工作目录
                    test_case_workspace_path = os.path.dirname(config.get("yml_path", ""))
                    case_name = config.get('case_name', 'unknown')
                    if not test_case_workspace_path:
                        log(f"\033[33mSkipping {case_name}: Invalid yml_path\033[0m", flush=True)
                        continue

                    # 提交任务到进程池
                    future = executor.submit(
                        run_test_case,
                        config,
                        test_case_workspace_path,
                        res_path
                    )

                    # 更新状态: 从待执行移到执行中
                    if case_name in cases_record["pending_cases"]:
                        cases_record["pending_cases"].remove(case_name)
                        cases_record["running_cases"].append(case_name)

                    futures[future] = config

                start_time = time.perf_counter()
                while len(executor._processes) < max_workers:
                    if time.perf_counter() - start_time > 3.0:
                        log(f"\033[33m警告: rank组 {rank} 工作进程初始化超时\033[0m")
                        break
                    time.sleep(0.1)

                # 获取工作进程PID
                for worker in executor._processes.values():
                    try:
                        pid = worker.pid
                        if pid:  # 确保pid有效
                            process_tree[group_id]['processes'].add(pid)
                    except AttributeError as e:  # worker可能尚未完全初始化
                        pass
                    except Exception as e:
                        log(f"\033[33m获取工作进程PID失败: {str(e)}\033[0m")

                try:
                    # 7. 处理完成的任务
                    for future in as_completed(futures):
                        config = futures[future]
                        case_name = config.get('case_name', 'unknown')
                        try:
                            case_type, case_name, result = future.result()
                            # 从执行中列表移除完成的任务
                            if case_name in cases_record["running_cases"]:
                                cases_record["running_cases"].remove(case_name)
                                curr_list = []
                                if result.get('success'):
                                    curr_list = cases_record["finished_cases"]["success"]
                                    rank_info["success"] += 1
                                else:
                                    curr_list = cases_record['finished_cases']['failed']
                                    rank_info["failed"] += 1
                                curr_list.append(case_name)
                            # 8. 将日志内容追加到主日志文件
                            log_file = result.get('log_file_path', '')
                            result_details = ""
                            if log_file and os.path.exists(log_file):
                                # 追加详细日志内容
                                with open(log_file, 'r') as case_f:
                                    result_details = case_f.read()
                                log(f"\n\n{'=' * 80}\n", print_enable=False)
                                log(f"Test Case: {case_name} running...", print_enable=False)
                                log(f"{'=' * 40}\n", print_enable=False)

                                log(result_details, print_enable=False)

                                # 在主日志中添加分隔标识
                                log(f"{'=' * 40}", print_enable=False)
                                log(f"Test Case: {case_name}", print_enable=False)
                                log(f"Rank Size: {config['rank_size']}", print_enable=False)
                                log(f"Case Tags: {', '.join(config['case_tags'])}", print_enable=False)
                                log(f"Result: {'SUCCESS' if result.get('success') else 'FAILURE'}", print_enable=False)
                                if not result.get('success'):
                                    log(f"Error: {result.get('message', 'No details')}", print_enable=False)
                                log(f"{'=' * 80}", print_enable=False)

                            # 9. 记录摘要
                            if result.get("success"):
                                log(f"\033[4;32m[PASS] {case_name}\033[0m", flush=True, print_enable=False)
                                scheduler.record_case_execution(config, result) # 更新本用例执行时间
                            else:
                                log(f"\033[5;31m[FAILED] {case_name}: {result.get('message')}\033[0m", flush=True, print_enable=False)

                            # 10. 存储结果
                            if case_type not in test_case_result:
                                test_case_result.setdefault(case_type, OrderedDict())
                            update_metrics(test_case_result[case_type], result.get("metric_key", []), result)

                        except Exception as e:
                            log(f"\033[5;31mError processing {case_name}: {str(e)}\033[0m", flush=True, print_enable=False)

                        finally:
                            pbar.update(1)

                except KeyboardInterrupt as e:
                    # 安全终止整个进程组
                    _terminate_process_group(group_id)

                    # 处理Ctrl+C中断时刻的信息
                    log(f"\n\033[1;31m[本层中断] 用户终止了执行 (rank group {rank})\033[0m", flush=True)
                    len_pending_cases = len(cases_record["pending_cases"])
                    len_running_cases = len(cases_record["running_cases"])
                    len_finished_cases = len(cases_record["finished_cases"]["success"] + cases_record["finished_cases"]["failed"])

                    log(f"\033[1;33m待执行的用例: {len_pending_cases}个\033[0m", flush=True)

                    log(f"\n\033[1;33m正在执行中的用例: {len_running_cases}个\033[0m", flush=True)

                    log(f"\n\033[1;33m执行完成的用例: {len_finished_cases}个\033[0m", flush=True)

                    log(f"\n\033[1;31m执行已中止，共中断了{len_pending_cases + len_running_cases}个用例\033[0m", flush=True)
                    # 向上层抛出中断异常
                    raise e
                except Exception as e:
                    # 安全终止整个进程组
                    _terminate_process_group(group_id)

                    log(f"\033[1;31m处理任务时发生意外错误: {str(e)}\033[0m", flush=True)
                    raise e
                finally:
                    stop_spinner.set()
                    spinner_thread.join(timeout=0.5)

                    # 确保进度条总是关闭
                    pbar.close()

            # 11. 当前rank组执行完成后等待
            time_period = time.perf_counter() - current_start_time
            log(f"本轮耗时: {time_period:.6f}秒")
            log(f"\n\033[34mCompleted rank group {rank} - (Finished) \033[1;32mSuccess:{rank_info['success']}\033[0m | \033[1;31mFailed {rank_info['failed']}\033[0m", flush=True)
            if time_period > 3:
                time.sleep(3)  # 给系统时间释放资源

    except KeyboardInterrupt as e:
        result_container['interrupted'] = True
        # 捕获全局中断异常
        log(f"\n\033[1;31m[本轮中断] 本轮测试执行被用户终止\033[0m", flush=True)
    except Exception as e:
        log(f"\033[1;31m[本轮错误] 处理任务时发生意外错误: {str(e)}\033[0m", flush=True)
    else:
        log("\n\033[36m[本轮完成] 本轮所有测试用例均完成执行\033[0m", flush=True)
    finally:
        # 最后保障，再次终止所有存活的进程组
        try:
            for gid in list(process_tree.keys()):
                try:
                    _terminate_process_group(gid)
                except Exception as e:
                    log(f"\033[33m清理进程组 {gid} 失败: {str(e)}\033[0m")
        except Exception as e:
            log(f"\033[33m清理进程树失败: {str(e)}\033[0m")

        try:
            scheduler.save_data()
        except Exception as e:
            log(f"\033[33m保存调度数据失败: {str(e)}\033[0m")

        log(f"共计耗时: {(time.perf_counter() - total_start_time):.6f}秒")
        log(f"\nTest Execution Finished at {time.ctime()}\n")
    return result_container['record'], result_container['interrupted']


def update_metrics(d: Dict, keys: List[Any], value: Any) -> None:
    """
    沿指定键路径更新字典结构中的值

    自动创建嵌套字典路径:
    update_metrics({}, ['a','b','c'], 10) 将创建 {a: {b: {c: 10}}}

    Args:
        d (dict): 目标字典
        keys (list): 键路径列表
        value (any): 要设置的值
    """
    # 验证输入类型
    if not isinstance(d, dict):
        raise TypeError(f"第一个参数必须是字典类型，而不是 {type(d).__name__}")

    # 验证键路径有效性
    if not keys:
        raise ValueError("键路径不能为空")

    # 获取当前指针位置（从根字典开始）
    current = d

    # 遍历所有中间键（除了最后一个）
    for key in keys[:-1]:
        # 如果当前键不存在或对应非字典值，创建新字典
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}

        # 移动到下一层级
        current = current[key]

    # 设置最后一个键的值
    current[keys[-1]] = value


def update_failed_cases_names_list(data: dict) -> list:
    """
    递归遍历测试结果字典，收集所有失败的用例名称

    通过DFS深度优先搜索遍历字典结构，查找所有包含：
    {success: False, case_name: ...} 的节点

    Args:
        data (dict): 测试结果字典

    Returns:
        list: 包含所有失败用例名称的列表
    """
    failed_cases = []
    stack = [data]

    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            # 检查当前层是否包含目标字段
            if "success" in current and "case_name" in current:
                if current["success"] is False:
                    failed_cases.append(current["case_name"])

            # 添加所有值到栈中
            stack.extend(reversed(list(current.values())))

        elif isinstance(current, (list, tuple)):
            # 添加所有元素到栈中
            stack.extend(reversed(current))

    return failed_cases


def record_failed_cases(failed_cases_list: list):
    seg_symbol = " "
    if failed_cases_list:
        workspace_path = os.getenv("WORKSPACE_PATH", os.path.split(os.path.realpath(__file__))[0])
        if not workspace_path:
            log(f"\033[5;31m", "The WORKSPACE_PATH cannot be empty!", "\n\n\033[0m", flush=True)
            return
        failed_cases_record_path = workspace_path + "/failed_cases.txt"
        with open(failed_cases_record_path, "w+") as f:
            f.write(f"{seg_symbol}".join(failed_cases_list))
            log("\033[31m", f"完成记录失败用例 {len(failed_cases_list)} 条 : [", ", ".join(failed_cases_list), "]", "\n\n\033[0m", flush=True)
            log(f"记录于 : {failed_cases_record_path}", flush=True)
    else:
        log("\033[32m", f"无失败用例，所有用例结果均已正确！", "\n\n\033[0m", flush=True)


def update_execute_test_cases_config_list(test_case_path:str, filter_case_tags_set: set,
                                        execute_test_cases_config_list: list, test_case_result: dict,
                                        specific_label_value_dict = None, dev_mode = False) -> list:
    disable_express_list = ['n', 'no', 'not', 'f', 'false']
    for test_case_yml in get_test_cases_from_path(test_case_path):
        if dev_mode:
            log(test_case_yml, flush=True)
        try:
            config = get_test_case_config(test_case_yml)
            if filter_case_tags_set and not (config.get("case_tags") & filter_case_tags_set):
                if dev_mode:
                    log("ignore")
                continue
            if specific_label_value_dict and not have_common_same_key_values(config, specific_label_value_dict):
                if dev_mode:
                    log("ignore")
                continue
            if config.get("enable", "true").lower() in disable_express_list:
                if dev_mode:
                    log("ignore")
                continue
            execute_test_cases_config_list.append(config)
        except Exception as ex:
            test_case_result.setdefault("unexpected", OrderedDict())
            test_case_result["unexpected"][test_case_yml] = dict(success=False, message=str(ex))
            continue


def type_converter(value, default_type=int):
    """自定义类型转换器，使用 's:' 前缀标识字符串类型"""
    # 检查是否用 s: 前缀显式标识字符串
    if value.startswith('s:'):
        return value[2:]  # 返回去除前缀后的字符串

    # 尝试转为default_type
    try:
        return default_type(value)
    except ValueError:
        return value  # 转换失败则保持原始字符串


def dict_converter(value, seg = "="):
    """处理类字典格式的转换，支持 key:value 格式"""
    try:
        parts = value.split(seg, 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: '{value}' (expected key:value)")

        key = parts[0].strip()
        value_part = parts[1].strip()

        # 对值部分进行类型推断
        if value_part.startswith('"') and value_part.endswith('"'):
            value_part = value_part[1:-1]  # 去除引号
        elif value_part.startswith("'") and value_part.endswith("'"):
            value_part = value_part[1:-1]  # 去除引号
        else:
            try:
                value_part = type_converter(value_part)
            except ValueError:
                pass

        return (key, value_part)
    except Exception as e:
        raise e


def validate_and_convert(input_value, label=None):
    """
    类型验证和转换器

    根据标签对应的预期类型验证和转换输入值:
    - "case_group" -> list
    - "case_name" -> str
    - "rank_size" -> int
    - 其他标签默认转换为dict

    Args:
        input_value (any): 需要验证和转换的值
        label (str): 目标标签名称

    Returns:
        any: 转换后的值

    Raises:
        TypeError: 类型转换失败时抛出
    """
    LABEL_TYPE_MAP = {
        "case_group": list,
        "case_name": str,
        "case_type": str,
        "enable": str,
        "rank_size": int,
        "script": dict,
        "timeout": int
    }

    # 获取期望类型
    expected_type = LABEL_TYPE_MAP.get(label, dict)

    # 如果值为None或已经是正确类型，直接返回
    if input_value is None:
        return input_value
    if isinstance(input_value, expected_type):
        if isinstance(input_value, list):
            return [type_converter(each) for each in input_value]
        return type_converter(input_value)

    # 处理特定类型转换场景
    try:
        # 字符串到列表的转换
        if expected_type is list and isinstance(input_value, str):
            return [type_converter(item) for item in input_value.split(",")]

        if expected_type is dict and isinstance(input_value, str):
            return dict({dict_converter(item, seg=":") for item in input_value.split(",")})

        # 字典到列表的转换
        if expected_type is list and isinstance(input_value, dict):
            return list(input_value.keys())

        # 列表到字典的转换
        if expected_type is dict and isinstance(input_value, (list, tuple, set)):
            return dict({dict_converter(item) for item in input_value})

        # 列表到字符串的转换
        if expected_type is str and isinstance(input_value, list):
            return ''.join(input_value)

        # 列表到数值的转换
        if expected_type is int and isinstance(input_value, list):
            ret_value = type_converter(''.join(input_value))
            return int(ret_value)

        # 尝试通用类型转换
        return expected_type(input_value)
    except Exception as e:
        # 提供更友好的错误信息
        raise TypeError(
            f"标签 '{label}' 需要 {expected_type.__name__} 类型, "
            f"但传入的是 {type(input_value).__name__}: {input_value}"
        ) from e


def main():
    """
    主程序执行流程

    步骤:
    1. 解析命令行参数
    2. 初始化环境变量和日志系统
    3. 收集系统信息并记录
    4. 加载和过滤测试用例配置
    5. 执行测试用例（含失败重试）
    6. 分析测试结果并记录失败用例
    7. 生成Excel报告和发送邮件
    """

    # 获取入参
    args = parse_args()

    # 初始化一些环境变量
    set_test_env(args)

    # 初始化日志系统，落盘到$WORKSPACE_PATH/run.log
    initialize_logging(os.getenv("WORKSPACE_PATH", default = None))

    # 记录当前运行环境变量
    log(capture_system_info(args))

    # 初始化变量
    test_case_result = {}
    filter_case_tags_set = set(args.case_tags)
    filter_case_labels_dict = dict([(k, validate_and_convert(str(v), label=k)) for k,v in validate_and_convert(args.case_label).items()])

    execute_test_cases_config_list = []
    failed_cases_names_list = []
    case_record = {}

    # 更新用例
    update_execute_test_cases_config_list(args.test_case_path, filter_case_tags_set,
                                            execute_test_cases_config_list, test_case_result,
                                            specific_label_value_dict = filter_case_labels_dict, dev_mode = args.dev_mode)

    # 执行用例
    try:
        case_record, interrupted = execute_test_cases(execute_test_cases_config_list, test_case_result)
        if interrupted:
            log(f"\033[1;31m[全局中断] 整个冒烟执行被用户终止\033[0m", flush=True)
            sys.exit(130)
    except Exception as e:
        # 处理其他异常
        log(f"\033[1;31m未处理的异常: {str(e)}\033[0m", flush=True)
        sys.exit(1)
    finally:
        if case_record and (case_record["pending_cases"] or case_record["running_cases"] or case_record["finished_cases"]["failed"]):
            print_json(case_record)
            failed_cases_names_list = case_record["pending_cases"] + case_record["running_cases"] + case_record["finished_cases"]["failed"]

    case_names_list = case_record["finished_cases"]["success"] + case_record["finished_cases"]["failed"]

    # # 输出失败用例
    if failed_cases_names_list:
        log(f"[WARNING] There are {len(failed_cases_names_list)} failed test cases: {', '.join(failed_cases_names_list)}")
        # # 若有失败用例, 尝试再执行一遍
        if args.rerun_failed:
            reexecute_list = failed_cases_names_list[:]
            failed_cases_config_list = [case_config for case_config in execute_test_cases_config_list if case_config["case_name"] in reexecute_list]
            log(f"[WARNING] These test cases will be executed again: {', '.join(reexecute_list)}")
            try:
                case_record, interrupted = execute_test_cases(failed_cases_config_list, test_case_result)
                if interrupted:
                    log(f"\033[1;31m[全局中断] 整个冒烟执行被用户终止\033[0m", flush=True)
                    sys.exit(130)
            except Exception as e:
                # 处理其他异常
                log(f"\033[1;31m未处理的异常: {str(e)}\033[0m", flush=True)
                sys.exit(1)
            finally:
                if case_record:
                    if case_record["pending_cases"] or case_record["running_cases"] or case_record["finished_cases"]["failed"]:
                        print_json(case_record)
                    failed_cases_names_list = case_record["pending_cases"] + case_record["running_cases"] + case_record["finished_cases"]["failed"]
                    if failed_cases_names_list:
                        log(f"[WARNING] There are still {len(failed_cases_names_list)} failed test cases: {', '.join(failed_cases_names_list)}")

    # 记录失败用例
    record_failed_cases(failed_cases_names_list)

    # 总结所有记录
    record(test_case_result=test_case_result)
    log("运行工作路径: ", os.path.abspath(os.getenv("WORKSPACE_PATH", default = None)))
    log("评估数值记录：", os.path.abspath(get_default_record_path()))
    log("运行日志记录：", Logger.get_instance().log_path)

    if args.dev_mode:
        return

    # 整理为excel
    with open(get_default_record_path(), 'r') as f:
        test_case_result = json.load(f).get("test_case_result")
        case_types = test_case_result.keys()
        for case_type in case_types:
            case_result = test_case_result.get(case_type)
            # # 整理为excel
            excel_name = "{}-{}-result.xls".format(time.strftime("%Y%m%d_%H%M%S"), case_type)

            create_excel(case_result=case_result,
                            excel_name=excel_name,
                            workspace_path=args.workspace_path,
                            case_type=case_type,
                            case_names=case_names_list)

if __name__ == '__main__':
    main()
