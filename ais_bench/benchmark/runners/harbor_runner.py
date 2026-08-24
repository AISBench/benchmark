"""Harbor runner: launch multiple harbor agent tasks and monitor them.

Each task is launched as a subprocess (same mechanism as LocalRunner: dump a
param file, build the command via ``task.get_command``, redirect output to a
log file). In parallel, a :class:`HarborMonitor` periodically snapshots the
on-disk status of every running harbor job, and an optional stdlib HTTP
server exposes the live information to external callers.

No harbor imports at module level so this module stays importable in
non-agent AISBench environments.
"""

import os
import os.path as osp
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import mmengine

from ais_bench.benchmark.registry import RUNNERS, TASKS
from ais_bench.benchmark.runners.base import BaseRunner
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg
from ais_bench.benchmark.utils.logging.error_codes import RUNNER_CODES


@RUNNERS.register_module()
class HarborRunner(BaseRunner):
    """Runner that launches multiple harbor agent tasks as subprocesses.

    Args:
        task (ConfigDict): HarborAgentTask type config.
        max_num_workers (int): Max number of tasks to run in parallel.
            Defaults to 1 (harbor jobs parallelize trials internally).
        monitor_port (int): Port of the HTTP monitor service. 0 disables it.
            Defaults to 0.
        refresh_interval (float): Monitor refresh interval in seconds.
        debug (bool): Whether to run in debug mode.
    """

    def __init__(self,
                 task: Dict[str, Any],
                 max_num_workers: int = 1,
                 monitor_port: int = 0,
                 refresh_interval: float = 0.5,
                 jobs_dir: str = None,
                 keep_tmp_file: bool = False,
                 debug: bool = False,
                 **kwargs):
        super().__init__(task=task, debug=debug)
        self.max_num_workers = max_num_workers
        self.monitor_port = monitor_port
        self.refresh_interval = refresh_interval
        self.jobs_dir = jobs_dir
        self.keep_tmp_file = keep_tmp_file
        for k, v in kwargs.items():
            self.logger.warning(f'Ignored argument in {self.__module__}: {k}={v}')

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple harbor agent tasks and monitor them.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """
        from ais_bench.benchmark.runners.harbor_monitor import (
            HarborMonitor,
            HarborMonitorServer,
        )

        self.logger.debug(f"HarborRunner.launch called with {len(tasks)} task(s)")
        work_dir = tasks[0]['work_dir']
        monitor = HarborMonitor(work_dir=work_dir, refresh_interval=self.refresh_interval)
        for task in tasks:
            task_name = task_abbr_from_cfg(task)
            status_file = osp.join(
                work_dir, 'status_tmp', f"tmp_{task_name.replace('/', '_')}.json"
            )
            job_dir = self._task_job_dir(task)
            monitor.register_task(task_name, status_file=status_file, job_dir=job_dir)

        monitor.start()
        server = HarborMonitorServer(monitor, port=self.monitor_port)
        port = server.start()
        if port:
            self.logger.info(
                f"Harbor monitor server started at http://127.0.0.1:{port}"
            )
        try:
            status = self._run_tasks(tasks)
        finally:
            server.stop()
            monitor.stop()
        return status

    def _task_job_dir(self, task: Dict[str, Any]) -> str:
        """Expected harbor job dir for a task (out_detail_dir/details)."""
        model_abbr = task['models'][0]['abbr']
        dataset_abbr = task['datasets'][0][0]['abbr']
        return osp.join(task['work_dir'], 'results', model_abbr, dataset_abbr, 'details')

    def _run_tasks(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        if self.max_num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_num_workers) as executor:
                return list(executor.map(self._launch, tasks))
        return [self._launch(task) for task in tasks]

    def _launch(self, task: Dict[str, Any]) -> Tuple[str, int]:
        """Launch a single harbor agent task in a subprocess."""
        built = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
        task_name = built.name

        pwd = os.getcwd()
        mmengine.mkdir_or_exist('tmp/')
        uuid_str = str(uuid.uuid4())
        param_file = f'{pwd}/tmp/{uuid_str}_params.py'
        try:
            built.cfg.dump(param_file)
            cmd = built.get_command(cfg_path=param_file, template='{task_cmd}')
            out_path = built.get_log_path(file_extension='out')
            mmengine.mkdir_or_exist(osp.split(out_path)[0])
            with open(out_path, 'w', encoding='utf-8') as stdout:
                result = subprocess.run(cmd,
                                        shell=True,
                                        text=True,
                                        stdout=stdout,
                                        stderr=stdout)
            if result.returncode != 0:
                self.logger.error(RUNNER_CODES.TASK_FAILED,
                                  f"{task_name} failed with code {result.returncode}, see\n{out_path}")
        finally:
            if not self.keep_tmp_file and osp.exists(param_file):
                os.remove(param_file)
        return task_name, result.returncode
