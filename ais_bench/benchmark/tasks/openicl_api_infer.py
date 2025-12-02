import argparse
import os
import sys
import threading
import time
import json
from typing import Any, List
import asyncio
import multiprocessing as mp
from multiprocessing import Event, Process, Queue, shared_memory, BoundedSemaphore
from typing import Dict
import pickle
from mmengine.config import Config, ConfigDict
from collections import defaultdict

from ais_bench.benchmark.global_consts import WORKERS_NUM
from ais_bench.benchmark.registry import ICL_INFERENCERS, TASKS, ICL_RETRIEVERS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.tasks.utils import (
    update_global_data_index,
    check_virtual_memory_usage,
    create_message_share_memory,
    ProgressBar,
    TokenProducer,
    format_dict_as_table,
)
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg, merge_dataset_abbr_from_cfg
from ais_bench.benchmark.utils.config import build_dataset_from_cfg
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError, AISBenchRuntimeError
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer import MAX_BATCH_SIZE
from ais_bench.benchmark.utils.logging import AISLogger

CONCURRENCY_PER_PROCESS = 500
MAX_WORKERS_NUM = mp.cpu_count() * 0.8
TASK_WAIT_TIME = 30


def run_single_inferencer(
    model_cfg: Config,
    inferencer_cfg: Config,
    shm_name: str,
    message_shm_name: str,
    max_concurrency: int,
    indexes: Dict,
    token_bucket: BoundedSemaphore,
):
    """Run a single inferencer that reads samples from shared memory.

    Args:
        model_cfg: API model configuration
        inferencer_cfg: API inferencer configuration. Must implement `inference_with_shm`
        shm_name: The name of the shared memory block containing pickled samples
        max_concurrency: Maximum concurrent requests in this process
        index_queue: Queue yielding (index, offset, length) for items in shared memory
        token_bucket: Token bucket for rate limiting
    """
    inferencer_cfg["model_cfg"] = model_cfg
    inferencer_cfg["batch_size"] = max_concurrency
    inferencer = ICL_INFERENCERS.build(inferencer_cfg)
    # pressure mode each process has a copy of the data list

    inferencer.inference_with_shm(
        shm_name,
        message_shm_name,
        indexes,
        token_bucket,
    )


@TASKS.register_module()
class OpenICLApiInferTask(BaseTask):
    """OpenICL API Inference Task.

    Runs API inference with one or more inferencer workers in parallel.
    """

    name_prefix = "OpenICLApiInfer"
    log_subdir = "logs/infer"
    output_subdir = "predictions"

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.concurrency = self.model_cfg.get("batch_size", 1)
        self.pressure = self.cli_args.get("pressure", False)
        self.debug = self.cli_args.get("debug", False)
        self.pressure_time = self.cli_args.get("pressure_time")
        if self.pressure:
            try:
                from ais_bench.benchmark.global_consts import PRESSURE_TIME
                self.logger.warning("`PRESSURE_TIME` config in global_consts is deprecated, please set `--pressure_time` in cli args instead.")
                if PRESSURE_TIME > 0:
                    self.pressure_time = PRESSURE_TIME
                else:
                    self.logger.warning(f"PRESSURE_TIME is invalid, using `--pressure_time {PRESSURE_TIME}` of cli args instead.")
            except ImportError:
                pass
        self.warmup_size = self.cli_args.get("num_warmups", 1)
        self.task_mode = self.cli_args.get("mode", "infer") if not self.pressure else "pressure"
        self.inferencer_cfg = self.dataset_cfgs[0]["infer_cfg"]["inferencer"]
        self.inferencer_cfg["model_cfg"] = self.model_cfg
        self.inferencer_cfg["pressure_time"] = self.pressure_time
        self.inferencer_cfg["mode"] = self.task_mode
        self.inferencer_cfg["batch_size"] = self.model_cfg.get("batch_size", 1)
        self.inferencer_cfg["output_json_filepath"] = self.work_dir
        self.logger.debug(f"Inferencer config: {self.inferencer_cfg}")
        # Control switch for async tasks within process
        self.stop_evt = Event()
        self.stop_evt.set()
        self.repeat = self.model_cfg["generation_kwargs"].get("num_return_sequences", 1)
        if self.repeat > 1:
            self.logger.info(f'num_return_sequences is greater than 1, each data will be infer independently {self.repeat} times')

    def get_command(self, cfg_path, template):
        """Build the CLI command to execute this task.

        Args:
            cfg_path (str): Path to the task config file.
            template (str): Template string containing '{task_cmd}' placeholder.
        """
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f"{python} {script_path} {cfg_path}"

        return template.format(task_cmd=command)

    def _get_workers_num(self):
        """Calculate the number of worker processes.

        Returns:
            int: Number of worker processes
        """
        if isinstance(WORKERS_NUM, int):
            if WORKERS_NUM > 0:
                return min(WORKERS_NUM, MAX_WORKERS_NUM)
        workers_num = (self.concurrency - 1) // CONCURRENCY_PER_PROCESS
        workers_num = min(workers_num + 1, MAX_WORKERS_NUM)
        self.logger.debug(f"Workers number: {workers_num}")
        return workers_num

    def _get_data_list(self) -> tuple[List, List]:
        """Retrieve data from the inferencer and return a picklable dataset list.

        Supports datasets with different retrievers and prompt templates.

        Returns:
            List: List of pickled dataset items
        """
        data_list, global_indexes = [], []
        finish_cache_data = {}
        try:
            finish_cache_data = self.inferencer.get_finish_data_list()
        except Exception as e:
            self.logger.warning(f"Failed to get finish data list: {e}, infer cache data will be ignored")
            finish_cache_data = {}
        finish_index_nums, total_data_nums = 0, 0
        for dataset_cfg in self.dataset_cfgs:
            data_abbr = dataset_cfg["abbr"]
            cur_data_cache = finish_cache_data.get(data_abbr, {})
            infer_cfg = dataset_cfg["infer_cfg"]
            dataset = build_dataset_from_cfg(dataset_cfg)
            retriever_cfg = infer_cfg["retriever"].copy()
            retriever_cfg["dataset"] = dataset
            retriever_cfg["prompt_template"] = infer_cfg.get("prompt_template", None)
            retriever_cfg["ice_template"] = infer_cfg.get("ice_template", None)
            retriever = ICL_RETRIEVERS.build(retriever_cfg)
            infer_data_list = self.inferencer.get_data_list(retriever)
            # get all data_list and data_indexes to infer
            cur_data_indexes = [x for x in range(len(infer_data_list)) for _ in range(self.repeat)]
            cur_finish_indexes = [x["id"] for x in cur_data_cache.values()]
            for i in cur_finish_indexes:
                try:
                    cur_data_indexes.remove(i)
                except ValueError: # num-prompts change cause predictions num more than references, ignore it
                    pass
            finish_index_nums += len(cur_finish_indexes)
            data_list += infer_data_list
            global_indexes += [x + total_data_nums for x in cur_data_indexes]
            total_data_nums += len(infer_data_list)

        if finish_index_nums > 0:
            self.logger.info(f"Found {finish_index_nums} completed data in cache, "
                             "run infer task from the last interrupted position")

        # remove finished data in data_list and change indexes accordingly
        picked_data_list = [data_list[i] for i in global_indexes]
        data_list = [data_list[i] for i in set(global_indexes)]
        pos_map = {v['data_abbr'] + '-' + str(v['index']): k for k, v in enumerate(data_list)}
        global_indexes = [pos_map[v['data_abbr'] + '-' + str(v['index'])] for v in picked_data_list]

        return data_list, finish_index_nums, global_indexes

    def _dump_dataset_to_share_memory(self, data_list: List):
        """Dump the serialized dataset into a shared memory block.

        Returns:
            tuple: (dataset_size, dataset_shm, index_queue)
                - dataset_size: Number of items in the dataset
                - dataset_shm: The shared memory region
                - index_queue: Queue yielding (index, offset, length)
        """
        pickled_dataset = [pickle.dumps(data) for data in data_list]
        # Dump dataset to shared memory
        lengths = [len(b) for b in pickled_dataset]
        dataset_bytes = sum(lengths)

        # Check virtual memory usage and raise exception if exceeds 80%
        check_virtual_memory_usage(dataset_bytes=dataset_bytes, threshold_percent=80)

        dataset_shm = shared_memory.SharedMemory(create=True, size=dataset_bytes)

        buf = dataset_shm.buf
        indexes = {}
        index = 0
        offset = 0
        for data, length in zip(pickled_dataset, lengths):
            buf[offset : offset + length] = data
            indexes[index] = (index, offset, length)
            offset += length
            index += 1
        if not self.pressure:
            indexes[index] = None
        return len(pickled_dataset), dataset_shm, indexes

    def _deliver_concurrency_for_workers(self):
        """Split total concurrency across worker processes as evenly as possible.

        Returns:
            List[int]: List of concurrency values for each worker process
        """
        # Allow _get_workers_num to return float, but normalize to positive integer
        workers_num_raw = self._get_workers_num()
        # Convert workers_num to nearest integer and ensure at least 1
        workers_num = int(round(workers_num_raw)) if workers_num_raw is not None else 0
        workers_num = max(1, workers_num)
        # Ensure total concurrency is integer and non-negative
        total_concurrency = int(self.concurrency) if self.concurrency is not None else 0
        if total_concurrency <= 0 or total_concurrency > MAX_BATCH_SIZE:
            raise ParameterValueError(
                TINFER_CODES.CONCURRENCY_ERROR,
                f"Concurrency must be greater than 0 and <= {MAX_BATCH_SIZE}, but got {self.concurrency}",
            )
        q, r = divmod(total_concurrency, workers_num)
        per_worker_concurrency = [q + 1] * r + [q] * (workers_num - r)

        self.logger.info(
            f"Total concurrency: {total_concurrency}, per worker concurrency: {per_worker_concurrency}"
        )
        return per_worker_concurrency

    def _run_debug(
        self,
        dataset_shm: shared_memory.SharedMemory,
        message_shm: shared_memory.SharedMemory,
        indexes: Dict,
        token_bucket: Queue,
    ):
        """Run single-process debug mode; may be insufficient for high concurrency.

        Args:
            dataset_shm: Shared memory containing dataset
            message_shm: Shared memory for message passing
            indexes: Indexes for data
            data_index_value: Value for data index
            token_bucket: Token bucket for rate limiting
        """
        if self.concurrency > CONCURRENCY_PER_PROCESS:
            self.logger.warning(
                f"Concurrency exceeds the default per-process limit ({CONCURRENCY_PER_PROCESS}). "
                "This may limit throughput. Recommend unsetting `--debug` to enable multi-process mode."
            )
        else:
            self.logger.info(f"Debug mode, run with concurrency: {self.concurrency}")
        self.inferencer.inference_with_shm(
            dataset_shm.name,
            message_shm.name,
            indexes,
            token_bucket,
        )

    def _run_multi_process(
        self,
        dataset_shm: shared_memory.SharedMemory,
        indexes: Dict,
        token_bucket: BoundedSemaphore,
        message_shms: List[shared_memory.SharedMemory],
    ):
        """Launch multiple worker processes and create per-worker shared memory.

        Args:
            dataset_shm: Shared memory containing dataset
            indexes: Indexes for data
            data_index_value: Value for data index
            token_bucket: Token bucket for rate limiting
            message_shms: List to store message shared memory objects (mutated)

        Returns:
            List[Process]: List of started worker processes
        """
        per_worker_concurrency = self._deliver_concurrency_for_workers()
        if not per_worker_concurrency:
            return []

        processes = []

        for i, concurrency in enumerate(per_worker_concurrency):
            pid = None
            message_shm = None
            try:
                # Create named shared memory for this worker's message/status
                message_shm = create_message_share_memory()

                # Prepare process arguments
                # NOTE: run_single_inferencer must be importable at module top-level (spawn-safe)
                p = Process(
                    target=run_single_inferencer,
                    args=(
                        self.model_cfg,
                        self.inferencer_cfg,
                        dataset_shm.name,
                        message_shm.name,
                        concurrency,
                        indexes,
                        token_bucket,
                    ),
                )

                p.start()  # may raise
                # p.pid should be set after start()
                pid = p.pid
                message_shms[pid] = message_shm
                processes.append(p)

            except Exception as exc:
                # Any error creating shm or starting process -> clean up message_shm if created
                self.logger.error(TINFER_CODES.FAILED_TO_START_WORKER, f"Failed to start worker {i}: {exc}, " +
                    f"total workers to launch: {len(per_worker_concurrency)}.")
                # Cleanup any shm created for this iteration
                if pid is not None and pid in message_shms and message_shms[pid] is not None:
                    message_shm = message_shms[pid]
                    self._cleanup_shms(message_shm)
                elif message_shm is not None:
                    # If pid is None but message_shm was created, clean it up directly
                    self._cleanup_shms(message_shm)
        return processes

    def warm_up(self, data_list: List, task_state_manager: TaskStateManager):
        """Warm up the inferencer.

        Args:
            data_list: Data list to warm up
        """
        warm_up_inferencer: BaseApiInferencer = ICL_INFERENCERS.build(self.inferencer_cfg)
        # warmup
        if self.warmup_size > 0:
            task_state_manager.update_task_state({"status": "warmup"})
            self.logger.info(f"Start warmup, run with concurrency: {self.concurrency}")
            warmup_results = asyncio.run(
                warm_up_inferencer.warmup(data_list, self.warmup_size, self.concurrency)
            )

            task_state_manager.update_task_state(
                {
                    "status": "warmup finished",
                    "other_kwargs": {
                        "Total Count": self.warmup_size,
                        "Success Count": warmup_results["success"],
                        "Failed Count": warmup_results["failed"],
                        "Failed Reasons": dict(warmup_results["failed_reasons"]),
                    },
                })
            if self.debug:
                warm_up_log = f"Warmup finished "
                warm_up_log += f"Total Count: {self.warmup_size} "
                warm_up_log += f"Success Count: {warmup_results['success']} "
                warm_up_log += f"Failed Count: {warmup_results['failed']} "
                if warmup_results['failed'] > 0:
                    failed_reasons = warmup_results['failed_reasons']
                    if failed_reasons:
                        table_str = format_dict_as_table(
                            failed_reasons,
                            title="Failed Reasons:",
                            key_column_name="Failed Reason",
                            value_column_name="Count",
                        )
                        warm_up_log += "\n" + table_str
                self.logger.info(warm_up_log)
            if warmup_results["success"] == 0:
                task_state_manager.update_task_state({"status": "Warmup failed"})
                raise AISBenchRuntimeError(
                    TINFER_CODES.WARMUP_FAILED,
                    f"Exit task because all warmup requests failed, failed reasons: {dict(warmup_results['failed_reasons'])}"
                )
        else:
            self.logger.info(f"Warmup size is 0, skip...")

    def run(self, task_state_manager: TaskStateManager):
        self.logger.info(f"Task [{task_abbr_from_cfg(self.cfg)}]")
        self.inferencer:BaseApiInferencer = ICL_INFERENCERS.build(self.inferencer_cfg)
        self.clean_failed_results()

        data_list, finish_data_count, global_indexes = self._get_data_list()
        if len(data_list) == 0:
            self.logger.warning(f"Get no data to infer, task finished")
            return
        self.warm_up(data_list, task_state_manager)
        dataset_size, dataset_shm, indexes = self._dump_dataset_to_share_memory(data_list)
        # In pressure mode, treat the first `concurrency` requests as the dataset size
        if self.pressure:
            request_num = self.concurrency
        else:
            request_num = dataset_size

        # Create token producer
        token_producer = TokenProducer(
            self.model_cfg.pop("request_rate", 0),
            self.model_cfg.pop("traffic_cfg", {}),
            request_num,
            self.task_mode,
            os.path.join(self.inferencer.get_output_dir(self.work_dir), merge_dataset_abbr_from_cfg(self.cfg)),
        )
        message_shms = {}
        # Message queue collecting per-process request state; polled periodically

        try:
            processes = []
            if self.debug:
                message_shm = create_message_share_memory()
                message_shms[os.getpid()] = message_shm
                # Create progress bar
                pb = ProgressBar(
                    message_shms,
                    self.stop_evt,
                    len(global_indexes),
                    finish_data_count,
                    self.debug,
                    self.pressure,
                    self.pressure_time,
                )
                # Start display progress
                pb_thread = threading.Thread(
                    target=pb.display, args=(task_state_manager,), daemon=True
                )
                pb_thread.start()
                # Start produce tokens
                token_thread = threading.Thread(
                    target=token_producer.produce_token,
                    args=(self.stop_evt, message_shms),
                    daemon=True,
                )
                token_thread.start()

                global_data_index_process = Process(
                    target=update_global_data_index,
                    args=(
                        list(shm.name for shm in message_shms.values()),
                        len(indexes),
                        global_indexes,
                        self.pressure,
                    ),
                    daemon=True,
                )

                global_data_index_process.start()

                self._run_debug(
                    dataset_shm,
                    message_shm,
                    indexes,
                    token_producer.token_bucket,
                )

            # Run inference with multiple processes
            else:
                processes = self._run_multi_process(
                    dataset_shm,
                    indexes,
                    token_producer.token_bucket,
                    message_shms,
                )

                global_data_index_process = Process(
                    target=update_global_data_index,
                    args=(
                        list(shm.name for shm in message_shms.values()),
                        len(indexes),
                        global_indexes,
                        self.pressure,
                    ),
                    daemon=True,
                )
                global_data_index_process.start()
                # Start ProgressBar after getting process IDs in multi-process mode
                # Create progress bar
                pb = ProgressBar(
                    message_shms,
                    self.stop_evt,
                    len(global_indexes),
                    finish_data_count,
                    self.debug,
                    self.pressure,
                    self.pressure_time,
                )
                # Start display progress
                pb_thread = threading.Thread(
                    target=pb.display, args=(task_state_manager,), daemon=True
                )
                pb_thread.start()
                # Start produce tokens
                token_thread = threading.Thread(
                    target=token_producer.produce_token,
                    args=(self.stop_evt, message_shms),
                    daemon=True,
                )
                token_thread.start()
            if processes:
                while True:
                    alive = any(p.is_alive() for p in processes)
                    if not alive:
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            # Wait for all subprocesses to finish, timeout 1 minute and force terminate
            self.logger.warning(f"Keyboard interrupt!!! Task [{task_abbr_from_cfg(self.cfg)}] will be terminated")
            self.stop_evt.set()
            global_data_index_process.join(timeout=TASK_WAIT_TIME)
            pb_thread.join()
            pb.set_message_flag(1)
            if processes:
                for p in processes:
                    p.join(timeout=TASK_WAIT_TIME)
                # Check if any process is still alive, force terminate
                for p in processes:
                    if p.is_alive():
                        self.logger.warning(
                            f"Process {p.pid} timed out and tried to force terminate."
                        )
                        p.terminate()
                        p.join(timeout=TASK_WAIT_TIME)
        finally:
            self.stop_evt.set()
            global_data_index_process.join(timeout=TASK_WAIT_TIME)
            pb_thread.join()
            pb.set_message_flag(1)
            token_thread.join()
            for pid, shm in message_shms.items():
                self._cleanup_shms(shm)
            self._cleanup_shms(dataset_shm)
            self._summary_failed_results()

    def _cleanup_shms(self, shm: shared_memory.SharedMemory):
        """Clean up shared memory object.

        Args:
            shm: Shared memory object to clean up
        """
        try:
            shm.close()
            shm.unlink()
            self.logger.debug(f"Cleanup shared memory: {shm.name}")
        except (FileNotFoundError, OSError) as e:
            # shared memory already cleaned up or not found
            self.logger.debug(f"Shared memory {shm.name} already cleaned up or not found: {e}")

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        """Set default value for configuration key if not present.

        Args:
            cfg: Configuration dictionary
            key: Configuration key
            value: Default value to set
        """
        if key not in cfg:
            cfg[key] = value

    def clean_failed_results(self):
        """Clean failed results.
        """
        output_dir = self.inferencer.get_output_dir()
        for dataset_cfg in self.dataset_cfgs:
            data_abbr = dataset_cfg["abbr"]
            failed_data_path = os.path.join(output_dir, f"{data_abbr}_failed.jsonl")
            if os.path.exists(failed_data_path):
                os.remove(failed_data_path)
                self.logger.debug(f"Cleaned failed results for dataset {data_abbr}")


    def _summary_failed_results(self):
        """Summary failed results.
        """
        output_dir = self.inferencer.get_output_dir()
        failed_results = defaultdict(int)
        for dataset_cfg in self.dataset_cfgs:
            data_abbr = dataset_cfg["abbr"]
            failed_data_path = os.path.join(output_dir, f"{data_abbr}_failed.jsonl")
            if os.path.exists(failed_data_path):
                with open(failed_data_path, "r") as f:
                    for line in f:
                        json_line = json.loads(line)
                        failed_results[json_line.get("error_info", "Unknown Error")] += 1
        if not failed_results:
            self.logger.debug(f"No failed results")
            return
        table_str = format_dict_as_table(
            failed_results,
            title="Task finished, failed reasons summary:",
            key_column_name="Failed Reason",
            value_column_name="Count",
        )
        self.logger.info(f"Task finished, failed reasons summary: \n{table_str} \n")
        self.logger.info(f"Read {output_dir}*_failed.jsonl for more details")


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inferencer")
    parser.add_argument("config", help="Config file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg["cli_args"]["debug"],
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state(
        {
            "status": "start",
            "process_id": os.getpid(),
            "task_log_path": os.path.join(
                "logs/infer/", f"{task_abbr_from_cfg(cfg)}.out"
            ),
        }
    )
    start_time = time.perf_counter()
    try:
        inferencer: OpenICLApiInferTask = OpenICLApiInferTask(cfg)
        inferencer.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise e

    end_time = time.perf_counter()
    logger.info(f"Api infer task time elapsed: {end_time - start_time:.2f}s")
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
