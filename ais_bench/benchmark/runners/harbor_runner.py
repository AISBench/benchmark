"""Harbor runner: launch harbor agent tasks and monitor them.

Two execution modes based on the number of tasks:

- single task (len(tasks) == 1): the harbor job runs in-process (no
  subprocess), its logs print directly to the main stdout, and per-case
  details are still served through the monitor HTTP service;
- multiple tasks (len(tasks) > 1): each task runs in a subprocess (same
  mechanism as LocalRunner: dump a param file, build the command via
  ``task.get_command``, redirect output to a log file); the main process
  starts a TasksMonitor board that prints a progress bar per harbor task,
  and the monitor HTTP service serves live per-case info for all tasks.

No harbor imports at module level so this module stays importable in
non-agent AISBench environments.
"""

import os
import os.path as osp
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import mmengine

from ais_bench.benchmark.registry import RUNNERS, TASKS
from ais_bench.benchmark.runners.base import BaseRunner, TasksMonitor
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg
from ais_bench.benchmark.utils.logging.error_codes import RUNNER_CODES


@RUNNERS.register_module()
class HarborRunner(BaseRunner):
    """Runner that launches harbor agent tasks and monitors them.

    Args:
        task (ConfigDict): HarborAgentTask type config.
        max_num_workers (int): Max number of tasks to run in parallel
            (multi-task mode only). Defaults to 1.
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
        """Launch harbor agent tasks and monitor them.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """
        self.logger.debug(f"HarborRunner.launch called with {len(tasks)} task(s)")
        if len(tasks) == 1:
            # Mode A: single task runs in-process with direct log output.
            return [self._launch_inline(tasks[0])]
        # Mode B: multiple tasks run in subprocesses with a main-process board.
        return self._launch_multi(tasks)

    # ------------------------------------------------------------------
    # Mode A: single task, in-process
    # ------------------------------------------------------------------

    def _launch_inline(self, task: Dict[str, Any]) -> Tuple[str, int]:
        """Run a single harbor agent task in the current process.

        The harbor job logs print directly to stdout; per-case details are
        served through the monitor HTTP service (job_dir is the only info
        source since there is no subprocess status file).
        """
        from ais_bench.benchmark.runners.harbor_monitor import (
            HarborMonitor,
            HarborMonitorServer,
        )

        task_name = task_abbr_from_cfg(task)
        monitor = HarborMonitor(
            work_dir=task['work_dir'], refresh_interval=self.refresh_interval
        )
        monitor.register_task(
            task_name, status_file=None, job_dir=self._task_job_dir(task)
        )
        monitor.start()
        server = HarborMonitorServer(monitor, port=self.monitor_port)
        port = server.start()
        if port:
            self.logger.info(
                f"Harbor monitor server started at http://127.0.0.1:{port}"
            )
        try:
            built = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
            self.logger.info(f"Running harbor agent task '{built.name}' in-process")
            built.run(task_state_manager=None)
            return built.name, 0
        except Exception:
            self.logger.exception(
                f"Harbor agent task '{task_name}' failed in-process"
            )
            return task_name, 1
        finally:
            server.stop()
            monitor.stop()

    # ------------------------------------------------------------------
    # Mode B: multiple tasks, subprocesses + main-process board
    # ------------------------------------------------------------------

    def _launch_multi(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        from ais_bench.benchmark.runners.harbor_monitor import (
            HarborMonitor,
            HarborMonitorServer,
        )

        work_dir = tasks[0]['work_dir']
        task_names = [task_abbr_from_cfg(task) for task in tasks]

        monitor = HarborMonitor(work_dir=work_dir, refresh_interval=self.refresh_interval)
        for task in tasks:
            task_name = task_abbr_from_cfg(task)
            status_file = osp.join(
                work_dir, 'status_tmp', f"tmp_{task_name.replace('/', '_')}.json"
            )
            monitor.register_task(
                task_name, status_file=status_file, job_dir=self._task_job_dir(task)
            )
        monitor.start()
        server = HarborMonitorServer(monitor, port=self.monitor_port)
        port = server.start()
        if port:
            self.logger.info(
                f"Harbor monitor server started at http://127.0.0.1:{port}"
            )

        board = self._start_task_board(task_names, work_dir)
        interrupted = False
        try:
            status = self._run_tasks(tasks)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            if interrupted:
                # The terminal SIGINT already reached the harbor subprocesses
                # (same process group); they are recycling containers. Stop
                # the board so the terminal is restored promptly instead of
                # showing a stale "running" state that looks stuck.
                self._stop_board(board)
                self.logger.warning(
                    "Interrupted. Task board stopped; harbor subprocesses are "
                    "recycling their containers and will exit on their own. "
                    "Container cleanup can take a while."
                )
            if board is not None:
                board.join(timeout=2)
            server.stop()
            monitor.stop()
            TasksMonitor.rm_tmp_files(work_dir)
        return status

    def _start_task_board(self, task_names: List[str], work_dir: str) -> threading.Thread | None:
        """Start a TasksMonitor board (daemon thread) printing a progress bar
        per harbor task, reading each subprocess's status_tmp file."""
        if self.debug:
            self.logger.debug("Debug mode, won't launch task state board")
            return None

        holder: Dict[str, Any] = {}

        def _run_board():
            try:
                tasks_monitor = TasksMonitor(
                    task_names, work_dir, self.debug, self.refresh_interval
                )
                holder['monitor'] = tasks_monitor
                tasks_monitor.launch_state_board()
            except Exception:
                self.logger.exception("Harbor task board failed")

        board = threading.Thread(target=_run_board, daemon=True)
        board._holder = holder  # type: ignore[attr-defined]
        board.start()
        return board

    def _stop_board(self, board: threading.Thread | None) -> None:
        """Ask the board loop to exit so the terminal is restored promptly."""
        if board is None:
            return
        holder = getattr(board, '_holder', None)
        if holder and holder.get('monitor') is not None:
            try:
                holder['monitor'].stop_state_board()
            except Exception:  # noqa: BLE001
                self.logger.debug("Failed to stop task board", exc_info=True)

    def _task_job_dir(self, task: Dict[str, Any]) -> str:
        """Expected harbor job dir for a task (out_detail_dir/details)."""
        model_abbr = task['models'][0]['abbr']
        dataset_abbr = task['datasets'][0][0]['abbr']
        return osp.join(task['work_dir'], 'results', model_abbr, dataset_abbr, 'details')

    def _run_tasks(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        if self.max_num_workers <= 1:
            return [self._launch(task) for task in tasks]
        executor = ThreadPoolExecutor(max_workers=self.max_num_workers)
        futures = [executor.submit(self._launch, task) for task in tasks]
        try:
            return [future.result() for future in futures]
        except KeyboardInterrupt:
            # Don't block on in-flight subprocesses here: the terminal SIGINT
            # already reached them and they are recycling containers.
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=False)

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
                # Use Popen + wait() instead of subprocess.run(): on Ctrl+C,
                # subprocess.run would SIGKILL the child and abort harbor's
                # container cleanup. The terminal SIGINT already reached the
                # child (same process group), so wait() lets it recycle its
                # containers and exit on its own.
                popen = subprocess.Popen(cmd,
                                         shell=True,
                                         text=True,
                                         stdout=stdout,
                                         stderr=stdout)
                returncode = popen.wait()
            if returncode != 0:
                self.logger.error(RUNNER_CODES.TASK_FAILED,
                                  f"{task_name} failed with code {returncode}, see\n{out_path}")
        finally:
            if not self.keep_tmp_file and osp.exists(param_file):
                os.remove(param_file)
        return task_name, returncode
