import argparse
import copy
import os
import json
import os.path as osp
import random
import threading
import sys
import time
import re
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

import threading
from pathlib import Path
from tqdm import tqdm

from ais_bench.benchmark.registry import (TASKS)
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.config import ConfigDict
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg, model_abbr_from_cfg, dataset_abbr_from_cfg
from ais_bench.benchmark.tasks.base import BaseTask

# ================= 替换litellm中计费函数 =================
import litellm
import logging

litellm_logger = logging.getLogger("litellm")
litellm_logger.setLevel(logging.CRITICAL)

try:
    from litellm.utils import get_response_cost as litellm_get_response_cost
except ImportError:
    try:
        from litellm.cost_calculator import get_response_cost as litellm_get_response_cost
    except ImportError:
        litellm_get_response_cost = None

def patched_get_response_cost(*args, **kwargs):
    if litellm_get_response_cost is None:
        return 0.0
    try:
        return litellm_get_response_cost(*args, **kwargs)
    except Exception as e:
        if "This model isn't mapped yet" in str(e):
            return 0.0
        raise e

try:
    litellm.utils.get_response_cost = patched_get_response_cost
except AttributeError:
    pass
try:
    litellm.cost_calculator.get_response_cost = patched_get_response_cost
except AttributeError:
    pass
# ================= 替换litellm中计费函数 =================

DEFAULT_FAKE_API_KEY = "fake_api_key"

from tau2.data_model.simulation import RunConfig
from tau2.run import run_domain, get_tasks
from tau2.metrics.agent_metrics import compute_metrics

# ================= 替换tau2中计费函数 =================
import tau2.utils.llm_utils as tau2_llm_utils
from tau2.utils.display import ConsoleDisplay
import loguru

_original_tau2_get_response_cost = tau2_llm_utils.get_response_cost
_original_tau2_logger_error = tau2_llm_utils.logger.error

def _patched_logger_error(message, *args, **kwargs):
    if "This model isn't mapped yet" in str(message):
        return
    _original_tau2_logger_error(message, *args, **kwargs)

tau2_llm_utils.logger.error = _patched_logger_error

# 保存原始的 input 方法
original_input = ConsoleDisplay.console.input

# 猴子补丁：替换 input 方法为总是返回 "y"
def auto_y_input(prompt):
    print(f"自动响应: {prompt.strip()} -> y")
    return "y"

# 应用补丁
ConsoleDisplay.console.input = auto_y_input

# ================= 替换tau2中计费函数 =================

@TASKS.register_module()
class TAU2BenchTask(BaseTask):
    name_prefix = "TAU2BenchTask"
    log_subdir = "logs/eval"
    output_subdir = "results"

    def __init__(self, cfg: ConfigDict) -> None:
        super().__init__(cfg)
        self.captured_metrics = None

    def get_command(self, cfg_path, template) -> str:
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        return f'{python} {script_path} {cfg_path}'

    def run(self, task_state_manager: TaskStateManager):
        self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
        self.task_state_manager: TaskStateManager = task_state_manager

        self._set_api_key()

        self._prepare_out_dir()

        self._refresh_cfg()

        self.run_config: RunConfig = self._construct_run_cfg()

        simulation_results = self._run_with_tqdm()

        self._dump_eval_results(simulation_results)

    def _get_task_count(self, config):
        if config.task_set_name is None:
            task_set_name = config.domain
        else:
            task_set_name = config.task_set_name
        tasks = get_tasks(
            task_set_name=task_set_name,
            task_split_name=config.task_split_name,
            task_ids=config.task_ids,
            num_tasks=config.num_tasks,
        )
        return len(tasks)

    def _run_with_tqdm(self):
        """
        Display the progress bar while running the simulation.
        """
        self.logger.info(f"Pipeline Execute Config: {self.run_config}")
        total_tasks = self._get_task_count(self.run_config) * self.run_config.num_trials
        save_to = f"{self.run_config.save_to}.json"
        pbar = tqdm(total=total_tasks, desc="Running TAU2 Bench", unit="task")
        self.task_state_manager.update_task_state(
            {
                "status": "running",
                "total_count": total_tasks,
                "progress_description": f"Running TAU2 Bench",
                "finish_count": 0,
            }
        )
        completed = 0

        def monitor_file():
            nonlocal completed
            task_id_pattern = re.compile(r'"task_id"\s*:')
            while True:
                if osp.exists(save_to):
                    try:
                        with open(save_to, 'r', encoding='utf-8') as f:
                            content = f.read()
                        new_completed = len(task_id_pattern.findall(content))
                        if new_completed > completed:
                            pbar.update(new_completed - completed)
                            self.task_state_manager.update_task_state(
                                {
                                    "finish_count": new_completed,
                                }
                            )
                            completed = new_completed
                    except (IOError, OSError):
                        pass
                time.sleep(0.5)
                if completed >= total_tasks:
                    pbar.update(completed - pbar.n)
                    break

        monitor_thread = threading.Thread(target=monitor_file, daemon=True)
        monitor_thread.start()

        try:
            results = run_domain(self.run_config)
            monitor_thread.join()
        finally:
            pbar.update(total_tasks - pbar.n)
            self.task_state_manager.update_task_state(
                {
                    "finish_count": total_tasks,
                }
            )
            pbar.close()

        return results

    def _set_api_key(self):
        api_key = self.cfg["models"][0].get("api_key")
        if api_key is None:
            api_key = DEFAULT_FAKE_API_KEY
        os.environ["OPENAI_API_KEY"] = api_key

    def _prepare_out_dir(self):
        self.out_dir = osp.join(self.work_dir, self.output_subdir, self.cfg["models"][0]["abbr"])
        mkdir_or_exist(osp.join(self.out_dir, self.cfg["datasets"][0][0]["abbr"]))
        out_detail_json = osp.join(self.out_dir, self.cfg["datasets"][0][0]["abbr"], "tau2_run_detail")
        if osp.exists(out_detail_json):
            os.remove(out_detail_json)
        self.cfg["datasets"][0][0]["args"]["save_to"] = osp.abspath(out_detail_json)

    def _refresh_cfg(self):
        for key, value in self.cfg["models"][0].items():
            if key == "type":
                continue
            self.cfg["datasets"][0][0]["args"][key] = value

    def _construct_run_cfg(self) -> RunConfig:
        kwargs = {}
        for key, value in self.cfg["datasets"][0][0]["args"].items():
            if value is None:
                continue
            kwargs[key] = value
        self.logger.info(f"Run Config: {kwargs}")
        run_cfg = RunConfig(**kwargs)
        return run_cfg

    def _dump_eval_results(self, simulation_results):
        self.captured_metrics = compute_metrics(simulation_results)
        if self.captured_metrics is None:
            self.logger.error("No metrics captured. Please check the Tau2 run.")
            return
        out_json = osp.join(f"{self.out_dir}", f"{self.cfg['datasets'][0][0]['abbr']}.json")
        results = {
            f"pass@{self.run_config.num_trials}": 100 * self.captured_metrics.avg_reward,
            "total_count": self._get_task_count(self.run_config),
        }
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Evaluation results saved to {out_json}")


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logger = AISLogger(__name__)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg["cli_args"]["debug"],
    )
    manager_t = threading.Thread(
        target=task_state_manager.launch,
        args=()
    )
    manager_t.start()
    task_state_manager.update_task_state(
        {
            "status": "start",
            "task_log_path": os.path.join(TAU2BenchTask.log_subdir, f"{task_abbr_from_cfg(cfg)}.out"),
        }
    )
    start_time = time.perf_counter()
    try:
        inferencer: TAU2BenchTask = TAU2BenchTask(cfg)
        inferencer.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise e

    end_time = time.perf_counter()
    logger.info(f'Local infer task time elapsed: {end_time - start_time:.2f}s')
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()