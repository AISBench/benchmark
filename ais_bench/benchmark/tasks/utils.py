import os
import time
import struct
from typing import Dict, List, Any
from multiprocessing import Event, shared_memory, BoundedSemaphore

import numpy as np
import psutil
from tqdm import tqdm
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    ParameterValueError,
    AISBenchRuntimeError,
)
from ais_bench.benchmark.utils.config.message_constants import (
    STATUS_REPORT_INTERVAL,
    MESSAGE_INFO,
    WAIT_FLAG,
    SYNC_MAIN_PROCESS_INTERVAL,
    MESSAGE_SIZE,
    FMT,
)
from ais_bench.benchmark.utils.visualization.rps_distribution_plot import plot_rps_distribution

MAX_VIRTUAL_MEMORY_USAGE_PERCENT = 80
INDEX_READ_FLAG = -1

FINAL_RPS_MINIMUM_THRESHOLD = 0.1  # minimum acceptable RPS
MIN_RELIABLE_INTERVAL = 0.001  # minimum reliable time interval (1 millisecond)

logger = AISLogger()


def create_message_share_memory():
    """Create shared memory for inter-process communication.

    Returns:
        shared_memory.SharedMemory: Shared memory object for message passing.
    """
    shm = shared_memory.SharedMemory(create=True, size=MESSAGE_SIZE)
    buf = shm.buf
    # Set flag to 2, indicating child process is ready for first batch data deserialization
    buf[:] = struct.pack(FMT, 0, 0, 0, 0, 0, 0)
    return shm


def check_virtual_memory_usage(dataset_bytes: int, threshold_percent: int = MAX_VIRTUAL_MEMORY_USAGE_PERCENT) -> None:
    """Check current virtual memory usage and raise exception if threshold is exceeded.

    Uses psutil library for cross-platform memory monitoring.

    Args:
        dataset_bytes (int): Dataset size in bytes
        threshold_percent (int): Memory usage threshold percentage, default 80%

    Raises:
        AISBenchRuntimeError: When virtual memory usage exceeds threshold
    """

    # Get memory information using psutil
    memory = psutil.virtual_memory()

    # Extract memory information (all values are in bytes)
    total_mem = memory.total
    available_mem = memory.available
    used_mem = memory.used

    # Calculate memory usage after adding dataset
    total_used_after_dataset = used_mem + dataset_bytes
    usage_percent = (total_used_after_dataset / total_mem) * 100 if total_mem > 0 else 0

    # Check if usage exceeds threshold
    if usage_percent > threshold_percent:
        error_msg = (
            f"Virtual memory usage too high: {usage_percent:.2f}% > {threshold_percent}% "
            f"(Total memory: {total_mem / (1024**3):.2f} GB, "
            f"Used: {used_mem / (1024**3):.2f} GB, "
            f"Available: {available_mem / (1024**3):.2f} GB, "
            f"Dataset needed memory size: {dataset_bytes / (1024**2):.8f} MB)"
        )
        raise AISBenchRuntimeError(TINFER_CODES.VIRTUAL_MEMORY_USAGE_TOO_HIGH, error_msg)

    logger.info(f"Dataset needed memory size: {dataset_bytes / (1024**2):.8f} MB")
    logger.info(f"Memory usage check passed: {usage_percent:.2f}% < {threshold_percent}% "
                f"(Available: {available_mem / (1024**3):.2f} GB)")

class ProgressBar:
    """Progress monitor reading per-worker SharedMemory objects.

    Args:
        per_pid_shms: Mapping from worker pid to SharedMemory instance
        stop_event: Event to signal when to stop monitoring
        data_num: Total number of data items to process
        debug: Whether to run in debug mode
        pressure: Whether to run in pressure testing mode
        refresh_interval: Interval for refreshing progress display
    """

    def __init__(
        self,
        per_pid_shms: Dict[int, shared_memory.SharedMemory],
        stop_event: Event,
        data_num: int = -1,
        finish_data_num: int = 0,
        debug: bool = False,
        pressure: bool = False,
        pressure_time: int = 15,
        refresh_interval: float = 1.0,
    ):
        self.logger = AISLogger()
        self.debug = debug
        self.stop_event = stop_event
        self.data_num = data_num
        self.finish_data_num = finish_data_num
        self.total_data_num = data_num + finish_data_num
        self.data_index = -1

        # expected: pid -> SharedMemory instance
        # We copy the mapping so external mutations are allowed but won't break internal dict ops.
        self.per_pid_shms: Dict[int, shared_memory.SharedMemory] = per_pid_shms

        self.pressure = pressure
        self.pressure_time = pressure_time
        self.refresh_interval = refresh_interval

        self.per_pid_stats: Dict[int, Dict[str, int]] = {}
        self.stats = {"post": 0, "recv": 0, "fail": 0, "finish": 0, "case_finish": 0}
        self._keys = ("post", "recv", "fail", "finish", "case_finish")

        self.start_time = time.perf_counter()
        self._last_snapshot_time = self.start_time
        self._last_snapshot_stats = self.stats.copy()

    # ------------------- aggregation logic -------------------
    def _recalc_aggregate(self):
        """Recalculate aggregate statistics from per-pid stats."""
        agg = {k: 0 for k in self._keys}
        for st in self.per_pid_stats.values():
            for k in self._keys:
                agg[k] += int(st.get(k, 0))
        self.stats = agg

    def _read_shared_memory_and_update_per_pid(self) -> bool:
        """Read shared memory and update per-pid statistics.

        Returns:
            bool: True if any per-pid stat changed, False otherwise.
        """
        updated = False
        # Iterate over a snapshot of keys to allow external mapping mutations
        for pid, shm in self.per_pid_shms.items():
            raw = bytes(shm.buf[:MESSAGE_SIZE])
            _, post, recv, fail, finish, case_finish = struct.unpack(FMT, raw)
            normalized = {
                "post": max(0, int(post)),
                "recv": max(0, int(recv)),
                "fail": max(0, int(fail)),
                "finish": max(0, int(finish)),
                "case_finish": max(0, int(case_finish)),
            }
            prev = self.per_pid_stats.get(pid)
            if prev != normalized:
                self.per_pid_stats[pid] = normalized
                updated = True
        if updated:
            self._recalc_aggregate()
        return updated

    # ------------------- rate computations -------------------
    def _compute_rates_since_start(self):
        """Compute rates since the start of monitoring."""
        now = time.perf_counter()
        dt = max(1e-6, now - self.start_time)
        return {k: self.stats.get(k, 0) / dt for k in self._keys}

    def _compute_rates_interval(self):
        """Compute rates for the current interval."""
        now = time.perf_counter()
        dt = now - self._last_snapshot_time
        if dt <= 0:
            return {k: 0.0 for k in self._keys}
        rates = {}
        for k in self._keys:
            rates[k] = (self.stats.get(k, 0) - self._last_snapshot_stats.get(k, 0)) / dt
        self._last_snapshot_time = now
        self._last_snapshot_stats = self.stats.copy()
        return rates

    # ---------- normal: two-line fixed display ----------
    def _draw_progress(self):
        """Draw progress bar with statistics."""
        if self.total_data_num <= 0:
            raise ValueError("Data num must be greater than 0 for progress bar display")
        if self.pressure:
            total = self.pressure_time
            unit = "s"
            self.logger.info(
                f"Starting progress bar Time for pressure testing: {total} s"
            )
        else:
            total = self.total_data_num
            unit = "case"
            self.logger.info(
                f"Starting progress bar Total data num: {total}"
                f" Finished data num: {self.finish_data_num}"
                f" Left data num: {self.data_num}"
            )

        def get_new_count():
            if self.pressure:
                return min(int(time.perf_counter() - start_time), total)
            else:
                return min(
                    int(self.stats.get("case_finish", 0) + self.finish_data_num),
                    self.total_data_num,
                )

        # leave=True ensures final display is retained after closing
        main_bar = tqdm(total=total, desc="Progress", unit=unit, position=0, leave=True)
        if self.finish_data_num > 0:
            main_bar.update(self.finish_data_num)
        info_bar = tqdm(total=1, desc="", bar_format="{desc}", position=1, leave=True)

        try:
            start_time = time.perf_counter()
            initial = min(
                int(self.stats.get("case_finish", 0) or self.stats.get("post", 0)),
                total,
            )
            if initial > 0 and not self.pressure:
                main_bar.update(initial)

            last_update = 0.0
            while main_bar.n <= total and not self.stop_event.is_set():
                updated = self._read_shared_memory_and_update_per_pid()
                if updated:
                    new_count = get_new_count()
                    if new_count > main_bar.n:
                        main_bar.update(new_count - main_bar.n)

                now = time.perf_counter()
                if now - last_update >= self.refresh_interval:
                    rates = self._compute_rates_interval()
                    info = (
                        f"POST={self.stats['post']} ({rates['post']:.1f}/s)  "
                        f"RECV={self.stats['recv']} ({rates['recv']:.1f}/s)  "
                        f"FAIL={self.stats['fail']} ({rates['fail']:.1f}/s)  "
                        f"FIN={self.stats['finish']} ({rates['finish']:.1f}/s)   "
                    )
                    if self.pressure and main_bar.n == total:
                        info += "The time for pressure testing has arrived. Waiting for sent requests to complete..."
                    info_bar.set_description_str(info)
                    info_bar.refresh()
                    last_update = now

                time.sleep(min(0.2, self.refresh_interval))
        except KeyboardInterrupt:
            self.logger.debug(f"Keyboard interrupt detected, stopping progress bar")
            pass
        finally:
            self._read_shared_memory_and_update_per_pid()
            new_count = get_new_count()
            main_bar.update(new_count - main_bar.n)
            rates = self._compute_rates_interval()
            info = (
                f"POST={self.stats['post']} ({rates['post']:.1f}/s)  "
                f"RECV={self.stats['recv']} ({rates['recv']:.1f}/s)  "
                f"FAIL={self.stats['fail']} ({rates['fail']:.1f}/s)  "
                f"FINISH={self.stats['finish']} ({rates['finish']:.1f}/s)   "
            )
            info_bar.set_description_str(info)
            info_bar.refresh()

            main_bar.close()
            info_bar.close()

    def _refresh_task_monitor(self, task_state_manager: TaskStateManager):
        """Refresh task monitor with current statistics.

        Args:
            task_state_manager: Task state manager for updating status
        """
        if not self.pressure:
            task_state_manager.update_task_state(
                {
                    "total_count": self.total_data_num,
                }
            )
        else:
            task_state_manager.update_task_state(
                {
                    "total_count": self.pressure_time,
                }
            )
        start_time = time.perf_counter()
        while not self.stop_event.is_set():
            updated = self._read_shared_memory_and_update_per_pid()
            if updated:
                rates = self._compute_rates_interval()
                finish_rate = round(rates["finish"], 1)
                state = {
                    "status": "inferencing",
                    "finish_count": (
                        self.stats["case_finish"] + self.finish_data_num
                        if not self.pressure
                        else min(
                            self.pressure_time, int(time.perf_counter() - start_time)
                        )
                    ),
                    "other_kwargs": {
                        "POST": self.stats["post"],
                        "RECV": self.stats["recv"],
                        "FINISH": self.stats["finish"],
                        "FAIL": self.stats["fail"],
                    },
                }
                if finish_rate > 0:
                    state["progress_description"] = (
                        f"[{finish_rate} it/s]" if not self.pressure else f"[s]"
                    )
                task_state_manager.update_task_state(state)
            time.sleep(STATUS_REPORT_INTERVAL)
        self._read_shared_memory_and_update_per_pid()
        state = {
            "status": "write cache",
            "finish_count": (
                self.stats["case_finish"] + self.finish_data_num
                if not self.pressure
                else min(self.pressure_time, int(time.perf_counter() - start_time))
            ),
            "other_kwargs": {
                "POST": self.stats["post"],
                "RECV": self.stats["recv"],
                "FINISH": self.stats["finish"],
                "FAIL": self.stats["fail"],
            },
        }
        task_state_manager.update_task_state(state)

    def set_message_flag(self, flag: int):
        """Set message flag for all shared memory objects.

        Args:
            flag: Flag value to set
        """
        self.logger.debug(f"Set all message status to {flag}")
        for _, shm in self.per_pid_shms.items():
            shm.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", flag)

    def display(self, task_state_manager: TaskStateManager):
        """Display progress monitoring.

        Args:
            task_state_manager: Task state manager for updating status
        """
        while self.stop_event.is_set():
            time.sleep(SYNC_MAIN_PROCESS_INTERVAL)
        if not self.debug:
            self._refresh_task_monitor(task_state_manager)
        else:
            self._draw_progress()


class TokenProducer:
    """Token generator for controlling request pacing in multi-process scenarios.

    Produces tokens according to request_rate and optional traffic_cfg to control
    multi-process request pacing.
    """

    def __init__(
        self,
        request_rate: float,
        time_stamps: List[float],
        traffic_cfg: ConfigDict,
        request_num: int = None,
        mode: str = "infer",
        work_dir: str = os.getcwd(),
    ):
        """
        Args:
            request_rate: Desired request rate (RPS) used to pace requests.
            time_stamps: List of timestamps in seconds.
            traffic_cfg: Traffic configuration controlling ramp-up and burstiness.
            request_num: Total number of requests to schedule when known.
            pressure_mode: If True, after generating the first `request_num` tokens
                (used to warm up connections), subsequent tokens are produced without sleep.
            work_dir: Working directory for saving RPS distribution plot.
        """
        self.logger = AISLogger()
        self.request_rate = request_rate
        self.pressure_mode = mode == "pressure"
        if self.pressure_mode:
            try:
                from ais_bench.benchmark.global_consts import CONNECTION_ADD_RATE
                self.logger.warning("`CONNECTION_ADD_RATE` config in global_consts is deprecated, please set `request_rate` in model config instead.")
                if CONNECTION_ADD_RATE >= 0:
                    self.request_rate = CONNECTION_ADD_RATE
                else:
                    self.logger.warning(f"CONNECTION_ADD_RATE is invalid, using `request_rate` = {request_rate} of model config instead.")
            except ImportError:
                pass
        self.perf_mode = self.pressure_mode or mode == "perf"
        self.burstiness = 1.0
        self.work_dir = work_dir
        # When request_rate < 0.1, treat as infinite (no pacing applied here)
        if self.request_rate < FINAL_RPS_MINIMUM_THRESHOLD and not time_stamps:
            self.token_bucket = None
            if self.pressure_mode:
                self.logger.warning("Pressure mode with no request rate applied, concurrency will increase rapidly")
        else:
            self.token_bucket = BoundedSemaphore(request_num + 1)
            # First release all tokens in token_bucket to make it empty
            for _ in range(request_num + 1):
                self.token_bucket.acquire()

        self.interval_lists = []
        # If timestamps are provided, use them directly
        if time_stamps:
            self.interval_lists = time_stamps[:]
            return

        # If `traffic_cfg` is provided, pre-generate `interval_lists` for ramp-up; after
        # exhausting it, fall back to gamma-distributed intervals based on request_rate.
        self.burstiness = float(traffic_cfg.get("burstiness", self.burstiness))
        ramp_up_strategy = traffic_cfg.get("ramp_up_strategy")
        ramp_up_start_rps = traffic_cfg.get("ramp_up_start_rps")
        ramp_up_end_rps = traffic_cfg.get("ramp_up_end_rps")
        if ramp_up_strategy:
            self.logger.info(
                f"Traffic ramp-up strategy: {ramp_up_strategy}. Will increase "
                f"RPS from {ramp_up_start_rps} to {ramp_up_end_rps} RPS over "
                "the duration of the benchmark."
            )
        else:
            self.logger.info(
                f"Traffic request rate: {request_rate} RPS with burstiness {self.burstiness}."
            )
        self.interval_lists = self._generate_interval_lists(
            request_num,
            self.burstiness,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
        )

    def _generate_interval_lists(
        self,
        request_num: int,
        burstiness: float,
        ramp_up_strategy: str,
        ramp_up_start_rps: float,
        ramp_up_end_rps: float,
    ):
        """Generate interval lists for request pacing and optionally draw distribution.

        Returns:
            List[float]: cumulative sleep intervals (seconds) for each request
        """

        # Defensive checks
        if request_num <= 0:
            return []

        # Precompute delays and keep request rates per request for diagnostics
        delay_ts = []
        request_rates = []

        for request_index in range(request_num):
            progress = request_index / max(request_num - 1, 1)

            # Determine current request rate according to ramp strategy
            if ramp_up_strategy == "linear":
                increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
                current_request_rate = ramp_up_start_rps + increase
            elif ramp_up_strategy == "exponential":
                # handle degenerate case where start is zero or negative
                if ramp_up_start_rps <= 0:
                    current_request_rate = ramp_up_end_rps
                else:
                    ratio = ramp_up_end_rps / ramp_up_start_rps
                    current_request_rate = ramp_up_start_rps * (ratio**progress)
            else:
                # treat falsy ramp_up_strategy as "no ramp" (use fixed request rate)
                if not ramp_up_strategy:
                    current_request_rate = float(self.request_rate)
                else:
                    raise ParameterValueError(
                        TINFER_CODES.INVALID_RAMP_UP_STRATEGY,
                        f"Invalid ramp_up_strategy: {ramp_up_strategy} only support 'linear' and 'exponential'",
                    )

            # sanitize current_request_rate
            if current_request_rate is None or current_request_rate < 1e-9:
                # zero or negative rate -> zero interval (send immediately)
                request_rates.append(0.0)
                delay_ts.append(0.0)
                continue

            request_rates.append(float(current_request_rate))

            # Sample the request interval according to burstiness
            if burstiness == 0:
                # deterministic fixed interval
                interval = 1.0 / current_request_rate
            else:
                # Gamma(shape=k, scale=θ) where θ = 1/(λ·k)
                theta = 1.0 / (current_request_rate * burstiness)
                # guard against invalid theta (shouldn't happen if current_request_rate>0 and burstiness>0)
                if theta <= 0 or not np.isfinite(theta):
                    interval = 1.0 / current_request_rate
                else:
                    interval = float(np.random.gamma(shape=burstiness, scale=theta))
            delay_ts.append(interval)

        # Convert to cumulative delays (time from first request)
        if len(delay_ts) == 0:
            return []

        # cumulative in-place
        for i in range(1, len(delay_ts)):
            delay_ts[i] += delay_ts[i - 1]

        # If no ramp-up strategy: normalize to match target total time (stabilize throughput)
        if not ramp_up_strategy and delay_ts[-1] != 0 and float(self.request_rate) > 0:
            target_total_delay_s = request_num / float(self.request_rate)
            normalize_factor = target_total_delay_s / delay_ts[-1]
            # scale in place
            delay_ts = [d * normalize_factor for d in delay_ts]

        # If final RPS (either fixed or ramp end) is extremely low -> treat as simultaneous sends
        rate_to_check = ramp_up_end_rps if ramp_up_strategy else float(self.request_rate)
        if rate_to_check < FINAL_RPS_MINIMUM_THRESHOLD:
            self.logger.info(
                f"Request rate ({float(self.request_rate)}) or ramp end rps ({ramp_up_end_rps}) "
                f"< {FINAL_RPS_MINIMUM_THRESHOLD}, sending all requests simultaneously"
            )
            return []

        # ---------- Diagnostics: detect timing anomalies & burstiness anomalies ----------
        try:
            delays = np.array(delay_ts)
            # compute per-request inter-arrival times from cumulative deltas:
            inter_arrivals = np.empty(len(delays), dtype=float)
            inter_arrivals[0] = delays[0]
            inter_arrivals[1:] = delays[1:] - delays[:-1]

            request_rates_arr = np.array(request_rates, dtype=float)
            non_zero_mask = request_rates_arr > 0

            # expected intervals (without burstiness)
            expected_intervals = np.zeros_like(request_rates_arr)
            expected_intervals[non_zero_mask] = 1.0 / request_rates_arr[non_zero_mask]

            # timing anomalies: intervals below MIN_RELIABLE_INTERVAL
            timing_anomaly_mask = inter_arrivals < MIN_RELIABLE_INTERVAL
            timing_anomaly_indices = np.where(timing_anomaly_mask)[0]

            # interval deviations
            interval_deviations = np.zeros_like(inter_arrivals)
            interval_deviations[non_zero_mask] = np.abs(inter_arrivals[non_zero_mask] - expected_intervals[non_zero_mask]) / expected_intervals[non_zero_mask]

            # burstiness anomalies: deviation > 50%
            burstiness_anomaly_mask = interval_deviations > 0.5
            burstiness_anomaly_indices = np.where(burstiness_anomaly_mask)[0]

            # remove duplicates (timing anomalies take precedence)
            if timing_anomaly_indices.size > 0 and burstiness_anomaly_indices.size > 0:
                timing_set = set(timing_anomaly_indices.tolist())
                burst_set = set(burstiness_anomaly_indices.tolist())
                burst_set = burst_set - timing_set
                burstiness_anomaly_indices = np.array(sorted(list(burst_set)), dtype=np.int64)

            # If burstiness == 0, clear burstiness anomalies
            if burstiness == 0:
                burstiness_anomaly_indices = np.array([], dtype=np.int64)

        except Exception as e:
            self.logger.warning(f"Error during diagnostics calculation: {e}")
            timing_anomaly_indices = np.array([], dtype=np.int64)
            burstiness_anomaly_indices = np.array([], dtype=np.int64)
            inter_arrivals = np.array(delay_ts)

        # ---------- Visualization (only when performance flag enabled) ----------
        if  len(delay_ts) > 0 and self.perf_mode:
            self.logger.info("Begin to draw RPS distribution plot...")
            # build output path (keep existing behavior: append suffix if necessary)
            model_path = os.path.dirname(self.work_dir)
            os.makedirs(model_path, exist_ok=True)
            rps_plot_path = self.work_dir + "_rps_distribution_plot.html"

            # plot_rps_distribution expects cumulative_delays and anomaly indices
            plot_rps_distribution(
                cumulative_delays=np.array(delay_ts),
                timing_anomaly_indices=timing_anomaly_indices,
                burstiness_anomaly_indices=burstiness_anomaly_indices,
                request_rate=float(self.request_rate),
                burstiness=burstiness,
                ramp_up_strategy=ramp_up_strategy,
                ramp_up_start_rps=ramp_up_start_rps,
                ramp_up_end_rps=ramp_up_end_rps,
                output_path=rps_plot_path,
            )
            self.logger.info(f"RPS distribution charts saved to {rps_plot_path}")


        return delay_ts

    def produce_token(self, stop_evt: Event, per_pid_shms: Dict[int, shared_memory.SharedMemory]):
        """Produce tokens for request pacing.

        Args:
            stop_evt: Event to signal when to stop token production
        """

        # Wait for child process to complete first batch data loading
        while stop_evt.is_set():
            need_wait = any(
                struct.unpack_from("I", shm.buf, 0)[0] != WAIT_FLAG
                for shm in per_pid_shms.values()
            )
            if not need_wait:
                self.logger.info(
                    "All subprocesses have finished deserializing the first batch of data"
                )
                stop_evt.clear()
                for shm in per_pid_shms.values():
                    struct.pack_into("I", shm.buf, 0, 0) # set sync flag to 0
                break
            time.sleep(SYNC_MAIN_PROCESS_INTERVAL)
        if not self.token_bucket:
            return
        interval_index = 0
        theta = 1.0 / (self.request_rate * self.burstiness) if self.request_rate > 0 else 0

        start_time = time.perf_counter()
        if len(self.interval_lists) > 0:
            start_time += self.interval_lists[0]

        while not stop_evt.is_set():
            if interval_index < len(self.interval_lists):
                interval = self.interval_lists[interval_index]
                current_time = time.perf_counter()
                sleep_interval = interval - (current_time - start_time)
                if sleep_interval > 0:
                    time.sleep(sleep_interval)
                try:
                    self.token_bucket.release()
                except ValueError as e:
                    # ValueError: semaphore or lock released too many times
                    # Indicates token bucket is full, wait for tokens to be used
                    wait_interval = np.random.gamma(shape=self.burstiness, scale=theta)
                    time.sleep(wait_interval)
                    continue
                interval_index += 1
            elif not self.pressure_mode:
                try:
                    self.token_bucket.release() # release None token to avoid deadlock
                except Exception:
                    pass
                interval = np.random.gamma(shape=self.burstiness, scale=theta)
                time.sleep(interval)
            else:
                try:
                    # After first batch requests are sent, subsequent requests
                    # are not sent according to request rate strategy
                    self.token_bucket.release()

                except Exception as e:
                    # ValueError: semaphore or lock released too many times
                    # Indicates token bucket is full, wait for tokens to be used
                    interval = np.random.gamma(shape=self.burstiness, scale=theta)
                    time.sleep(interval)


def format_dict_as_table(
    data_dict: Dict[str, Any],
    title: str = "",
    key_column_name: str = "Key",
    value_column_name: str = "Value",
    key_align: str = "<",
    value_align: str = ">",
) -> str:
    """Format a dictionary as a structured table with borders.

    Args:
        data_dict: Dictionary to format (key -> value mapping)
        title: Optional title to display above the table
        key_column_name: Name for the key column header
        value_column_name: Name for the value column header
        key_align: Alignment for key column ('<' for left, '>' for right, '^' for center)
        value_align: Alignment for value column ('<' for left, '>' for right, '^' for center)

    Returns:
        str: Formatted table as a multi-line string

    Example:
        >>> data = {"Connection timeout": 5, "Invalid response": 3}
        >>> print(format_dict_as_table(data, title="Failed Reasons:"))
        Failed Reasons:
        +----------------------+-------+
        | Failed Reason        | Count |
        +----------------------+-------+
        | Connection timeout   |   5   |
        | Invalid response     |   3   |
        +----------------------+-------+
    """
    if not data_dict:
        return title if title else ""

    # Calculate column widths
    max_key_len = max(len(str(key)) for key in data_dict.keys())
    max_key_len = max(max_key_len, len(key_column_name))
    max_value_len = max(len(str(value)) for value in data_dict.values())
    max_value_len = max(max_value_len, len(value_column_name))

    # Build table
    table_lines = []
    if title:
        table_lines.append(title)

    # Table header separator
    header_separator = "+" + "-" * (max_key_len + 2) + "+" + "-" * (max_value_len + 2) + "+"
    table_lines.append(header_separator)

    # Table header
    key_header = f"{key_column_name:{key_align}{max_key_len}}"
    value_header = f"{value_column_name:{value_align}{max_value_len}}"
    table_lines.append(f"| {key_header} | {value_header} |")
    table_lines.append(header_separator)

    # Table rows
    for key, value in data_dict.items():
        key_str = f"{str(key):{key_align}{max_key_len}}"
        value_str = f"{str(value):{value_align}{max_value_len}}"
        table_lines.append(f"| {key_str} | {value_str} |")

    table_lines.append(header_separator)

    return "\n".join(table_lines)
