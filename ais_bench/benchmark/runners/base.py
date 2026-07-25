import os
import time
import psutil
import shutil
from tqdm import tqdm
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import curses
from tabulate import tabulate
from datetime import datetime, timedelta
from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.file import read_and_clear_statuses


def create_progress_bar(finished_count=0, total_count=1000, description="", length=30):
    """create progress bar string"""
    if finished_count is None or not total_count:
        return "NA"
    if finished_count < 0:
        finished_count = 0
    if total_count < 0:
        total_count = 0
    if finished_count > total_count:
        finished_count = total_count
    filled = int(float(finished_count) / total_count * length) if total_count > 0 else 0
    empty = length - filled
    return f"[{ '#' * filled }{ ' ' * empty }] {finished_count}/{total_count} {description}"


def format_time(seconds):
    """format time to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))


class TasksMonitor:
    def __init__(self,
        task_names: list,
        output_path: str,
        is_debug: bool = False,
        refresh_interval:float = 0.3,
    ):
        self.logger = AISLogger()
        self.output_path = output_path
        self.tmp_file_path = os.path.join(self.output_path, "status_tmp")
        self.tmp_file_name_list = [f"tmp_{task_name.replace('/', '_')}.json" for task_name in task_names]
        if not os.path.exists(self.tmp_file_path):
            os.makedirs(self.tmp_file_path, mode=0o750)
        self.logger.debug(f"TasksMonitor initialized, temporary file directory: {self.tmp_file_path}")

        self.tasks_state_map = {task_name: {"status": "not start"} for task_name in task_names}
        self.task_end_status_list = {task_name: [] for task_name in task_names}
        self.is_debug = is_debug
        self.refresh_interval = refresh_interval
        self.run_in_background = self.is_running_in_background() if not self.is_debug else True
        self.last_table = None
        self.logger.info(f"Launch TasksMonitor, "
                    f"PID: {os.getpid()}, "
                    f"Refresh interval: {self.refresh_interval}, "
                    f"Run in background: {self.run_in_background}"
                    )

    def is_running_in_background(self):
        try:
            curses.initscr()    # raise if not link to terminal
            curses.curs_set(0)  # raise if terminal not support cursor
            curses.endwin()     # raise when call curses incorrect
        except Exception as e:
            self.logger.warning(f"Can't set cursor because of {e}, running in background mode")
            return True
        return False

    @staticmethod
    def rm_tmp_files(work_dir: str):
        """
        Remove temporary files
        """
        if os.path.exists(os.path.join(work_dir, "status_tmp")):
            shutil.rmtree(os.path.join(work_dir, "status_tmp"))

    def launch_state_board(self):
        if self.is_debug:
            self.logger.debug("Debug mode, won't launch task state board")
            return
        if not self.run_in_background:
            self.logger.info("Start launch task state board ...")
            curses.wrapper(self._display_task_state)
            print(self.last_table)
        else:
            self.logger.debug("Running task progress monitor in background mode")
            self._update_tasks_progress()

    def _is_all_task_done(self):
        unfinished_tasks = []
        for task_name, state in self.tasks_state_map.items():
            status = state.get("status")
            if status not in ("finish", "error", "killed"):
                unfinished_tasks.append((task_name, status))

        if unfinished_tasks:
            return False

        self.logger.debug("All tasks are finished")
        return True

    def _refresh_task_state(self):
        start_time = time.time()
        statuses = read_and_clear_statuses(self.tmp_file_path, self.tmp_file_name_list)

        if len(statuses) == 0:
            # check whether process exist
            for task_name, state in self.tasks_state_map.items():
                if not state.get("process_id"):
                    continue
                if (
                    not psutil.pid_exists(state.get("process_id"))
                    and state.get("status") != "finish"
                    and state.get("status") != "error"
                ):  # killed
                    self.tasks_state_map[task_name]['status'] = "killed"
                else:
                    continue
            return

        for status in statuses:
            # get status information from queue
            task_name = status['task_name']
            if not self.tasks_state_map[task_name].get('start_time'):
                self.tasks_state_map[task_name]['start_time'] = time.time()
                self.tasks_state_map[task_name]['status'] = "start"
            if not self.tasks_state_map[task_name].get('task_log_path'):
                self.tasks_state_map[task_name]['task_log_path'] = status.get('task_log_path')

            self.tasks_state_map[task_name]['process_id'] = status.get('process_id')
            self.tasks_state_map[task_name]['finish_count'] = status.get('finish_count')
            self.tasks_state_map[task_name]['total_count'] = status.get('total_count')
            self.tasks_state_map[task_name]['progress_description'] = status.get('progress_description')

            if status.get('status'):
                self.tasks_state_map[task_name]['status'] = status['status']
            self.tasks_state_map[task_name]['other_kwargs'] = status.get('other_kwargs')
            if time.time() - start_time > 100:
                self.logger.warning("Task monitor refresh time out!")
                break

    def _get_task_states(self):
        data = []
        for task_name, state in self.tasks_state_map.items():
            data.append(
                [
                    task_name, # name
                    state.get("process_id"), # process id
                    create_progress_bar(state.get("finish_count"), state.get("total_count"), state.get("progress_description")), # progress
                    format_time(time.time() - state.get("start_time")) if state.get("start_time") else "NA", # time
                    state.get("status"), # task status
                    state.get("task_log_path"), # log path
                    f"{state.get('other_kwargs')}", # other kwargs
                ]
            )
            if state.get("status") == "finish" or state.get("status") == "error":
                if len(self.task_end_status_list.get(task_name)) == 0:
                    self.task_end_status_list[task_name] = data[-1]
                else:
                    data[-1] = self.task_end_status_list.get(task_name)
        return data

    def _update_tasks_progress(self):
        pbar = tqdm(total=len(self.tasks_state_map), desc="Monitoring tasks progress")
        while True:
            self._refresh_task_state()
            _ = self._get_task_states()
            cur_count = 0
            for _, state in self.tasks_state_map.items():
                if state.get("status") == "finish" or state.get("status") == "error":
                    cur_count += 1

            if cur_count > pbar.n:
                pbar.update(cur_count - pbar.n)
            # break when all the task finished
            if cur_count >= pbar.total or self._is_all_task_done():
                pbar.close()
                break
            time.sleep(self.refresh_interval)

    def _display_task_state(self, stdscr):
        curses.curs_set(0)  # hide cursor
        stdscr.nodelay(1)   # non-blocking input
        stdscr.timeout(200)  # 200ms timeout, check input more frequently, improve response speed

        headers = ["Task Name", "Process", "Progress", "Time Cost", "Status", "Log Path", "Extend Parameters"]
        current_page = 0
        last_refresh_time = time.time()
        page_size = curses.LINES - 10  # reserve space for title and prompt
        stop_screen_refresh = False  # 添加停止屏幕刷新的标志

        try:
            while True:
                # function of key input
                try:
                    key = stdscr.getch()
                    if key != -1:
                        # up and down key to switch page
                        if key == curses.KEY_UP and current_page > 0:
                            current_page -= 1
                        elif key == curses.KEY_DOWN:
                            current_page += 1
                        # press 'p' or 'P' to pause/resume screen refresh
                        elif key in (ord('p'), ord('P')):
                            stop_screen_refresh = not stop_screen_refresh
                except Exception:
                    self.logger.debug("Error handling key input")
                    pass

                # if screen refresh is paused, only check key input
                if stop_screen_refresh:
                    continue

                # check if need refresh data
                current_time = time.time()
                need_refresh_data = (current_time - last_refresh_time >= self.refresh_interval)

                # clear screen
                stdscr.clear()

                # get current time as update time
                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # refresh task state data if need
                if need_refresh_data:
                    self._refresh_task_state()
                    last_refresh_time = current_time

                # get task state data
                data = self._get_task_states()

                # format table
                full_table = tabulate(data, headers=headers, tablefmt="grid")

                # split table to lines
                table_lines = full_table.splitlines()
                total_pages = max(0, (len(table_lines) - 1) // page_size) + 1
                current_page = min(current_page, total_pages - 1)  # make sure the page won't out of

                # get current page table lines
                start_line = current_page * page_size
                end_line = start_line + page_size
                current_table_lines = table_lines[start_line:end_line]
                current_table = '\n'.join(current_table_lines)

                # display table and info
                stdscr.addstr(0, 0, f"Base path of result&log : {self.output_path}")
                stdscr.addstr(1, 0, f"Task Progress Table (Updated at: {current_time_str})")
                stdscr.addstr(2, 0, f"Page: {current_page + 1}/{total_pages}  Total {(int((len(table_lines) - 1) / 2))} rows of data")

                # add screen refresh status display and operation prompt
                stdscr.addstr(3, 0, f"Press Up/Down arrow to page, 'P' to PAUSE/RESUME screen refresh, 'Ctrl + C' to exit")

                try:
                    # display current page table lines, start from line 5
                    for i, line in enumerate(current_table_lines):
                        if i + 5 < curses.LINES:  # make sure the line won't out of screen height
                            stdscr.addstr(i + 5, 0, line)
                except curses.error as e:
                    # handle curses error, prevent program crash
                    self.logger.debug(f"Curses display error (screen may be too small): {e}")
                    pass

                # refresh screen
                stdscr.refresh()

                if self._is_all_task_done():
                    self.last_table = full_table
                    self.logger.debug("All tasks completed, exiting interactive display")
                    break
        except KeyboardInterrupt as e:
            self.logger.debug("Received KeyboardInterrupt, saving current state and exiting")
            self._refresh_task_state()
            data = self._get_task_states()
            full_table = tabulate(data, headers=headers, tablefmt="grid")
            self.last_table = full_table


class BaseRunner:
    """Base class for all runners. A runner is responsible for launching
    multiple tasks.

    Args:
        task (ConfigDict): Task type config.
        debug (bool): Whether to run in debug mode.
        lark_bot_url (str): Lark bot url.
    """

    def __init__(self,
                 task: ConfigDict,
                 debug: bool = False):
        self.logger = AISLogger()
        self.task_cfg = Config(task)
        self.debug = debug

    def __call__(self, tasks: List[Dict[str, Any]]):
        """Launch multiple tasks and summarize the results.

        Args:
            tasks (list[dict]): A list of task configs, usually generated by
                Partitioner.
        """
        status = self.launch(tasks)
        status_list = list(status)  # change into list format
        self.summarize(status_list)

    @abstractmethod
    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks.

        Args:
            tasks (list[dict]): A list of task configs, usually generated by
                Partitioner.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """

    def summarize(self, status: List[Tuple[str, int]]) -> None:
        """Summarize the results of the tasks.

        Args:
            status (list[tuple[str, int]]): A list of (task name, exit code).
        """

        failed_logs = []
        for _task, code in status:
            if code != 0:
                failed_logs.append(_task)
