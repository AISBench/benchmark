import copy
import time
import json
import os
import re
from abc import abstractmethod
from typing import List, Optional

from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.core.abbr import get_infer_output_path, task_abbr_from_cfg
from ais_bench.benchmark.utils.file import write_status
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import TASK_CODES

def extract_role_pred(s: str, begin_str: Optional[str], end_str: Optional[str]) -> str:
    """Extract the role prediction from the full prediction string. The role
    prediction may be the substring between the begin and end string.

    Args:
        s (str): Full prediction string.
        begin_str (str): The beginning string of the role
        end_str (str): The ending string of the role.

    Returns:
        str: The extracted role prediction.
    """
    start = 0
    end = len(s)

    if begin_str and re.match(r"\s*", begin_str) is None:
        begin_idx = s.find(begin_str)
        if begin_idx != -1:
            start = begin_idx + len(begin_str)

    if end_str and re.match(r"\s*", end_str) is None:
        # TODO: Support calling tokenizer for the accurate eos token
        # and avoid such hardcode
        end_idx = s.find(end_str, start)
        if end_idx != -1:
            end = end_idx

    return s[start:end]


class BaseTask:
    """Base class for all tasks. There are two ways to run the task:
    1. Directly by calling the `run` method.
    2. Calling the `get_command` method to get the command,
        and then run the command in the shell.

    Args:
        cfg (ConfigDict): Config dict.
    """

    # The prefix of the task name.
    name_prefix: str = None
    # The subdirectory of the work directory to store the log files.
    log_subdir: str = None
    # The subdirectory of the work directory to store the output files.
    output_subdir: str = None

    def __init__(self, cfg: ConfigDict):
        self.logger = AISLogger()
        cfg = copy.deepcopy(cfg)
        self.cfg = cfg
        if len(cfg["models"]) > 1:
            self.logger.warning(f"One task only supports one model, but got {len(cfg['models'])} models, " +
                                "only use the first model.")
        self.model_cfg = cfg["models"][0]
        self.logger.debug(f"Model config: {self.model_cfg}")
        self.dataset_cfgs = cfg["datasets"][0]
        self.logger.debug(f"Dataset config: {self.dataset_cfgs}")
        self.work_dir = cfg["work_dir"]
        self.logger.debug(f"Work directory: {self.work_dir}")
        self.cli_args = cfg["cli_args"]

    @abstractmethod
    def run(self):
        """Run the task."""

    @abstractmethod
    def get_command(self, cfg_path, template) -> str:
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """

    @property
    def name(self) -> str:
        return self.name_prefix + task_abbr_from_cfg(
            {"models": [self.model_cfg], "datasets": [self.dataset_cfgs]}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.cfg})"

    def get_log_path(self, file_extension: str = "json") -> str:
        """Get the path to the log file.

        Args:
            file_extension (str): The file extension of the log file.
                Default: 'json'.
        """
        return get_infer_output_path(
            self.model_cfg,
            self.dataset_cfgs,
            os.path.join(self.work_dir, self.log_subdir),
            file_extension,
        )

    def get_output_paths(self, file_extension: str = "json") -> List[str]:
        """Get the paths to the output files. Every file should exist if the
        task succeeds.

        Args:
            file_extension (str): The file extension of the output files.
                Default: 'json'.
        """
        output_paths = []
        for model, datasets in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset in datasets:
                output_paths.append(
                    get_infer_output_path(
                        model,
                        dataset,
                        os.path.join(self.work_dir, self.output_subdir),
                        file_extension,
                    )
                )
        return output_paths


class TaskStateManager:
    def __init__(self, tmp_path: str, task_name: str, is_debug: bool, refresh_interval: int = 0.5):
        self.logger = AISLogger()
        self.tmp_file = os.path.join(tmp_path, f"tmp_{task_name.replace('/', '_')}.json")
        if os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)
        with open(self.tmp_file, 'w') as f:
            json.dump([], f)
        self.logger.debug(f"TaskStateManager initialized, temporary file: {self.tmp_file}")

        self.task_state = {"task_name": task_name, "process_id": os.getpid()}
        self.is_debug = is_debug
        self.refresh_interval = refresh_interval

    def launch(self):
        self.task_state["start_time"] = time.time()
        if self.is_debug:
            self.logger.info("Debug mode, print progress directly")
            self._display_task_state()
        else:
            self._post_task_state()

    def update_task_state(self, task_state: dict):
        self.task_state.update(task_state)

    def _post_task_state(self):
        while(True):
            write_status(self.tmp_file, self.task_state)
            if self.task_state.get("status") == "error":
                self.logger.warning("Task state is error, exit loop")
                break
            elif self.task_state.get("status") == "finish":
                self.logger.info("Task state is finish, exit loop")
                break
            time.sleep(self.refresh_interval)

    def _display_task_state(self):
        pass