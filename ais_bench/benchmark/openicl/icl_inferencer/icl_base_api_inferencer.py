import asyncio
import concurrent.futures
import os
import pickle
import queue as std_queue
import struct
import threading
import time
import uuid
import copy
import contextlib
from abc import abstractmethod
from collections import defaultdict
from multiprocessing import BoundedSemaphore, Queue, shared_memory, Value
from typing import Any, Dict, Optional, Tuple

import aiohttp
import janus
from tqdm import tqdm

from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.core.valid_global_consts import get_request_time_out, get_max_chunk_size
from ais_bench.benchmark.utils.config.message_constants import STATUS_REPORT_INTERVAL, MESSAGE_INFO, WAIT_FLAG, SYNC_MAIN_PROCESS_INTERVAL
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer import BaseInferencer
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError, ParameterValueError, AISBenchRuntimeError
from ais_bench.benchmark.utils.logging.logger import AISLogger

BLOCK_INTERVAL = 0.005  # Avoid request burst accumulation when RR is not configured
DEFAULT_SAVE_EVERY_FACTOR = 0.1 # default save every factor is 0.1 of batch size


class BaseApiInferencer(BaseInferencer):
    """Base Inferencer class for all evaluation Inferencer.

    Attributes mirroring your original class. This refactor keeps external API
    stable while avoiding blocking the asyncio event loop.
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        pressure_time: Optional[int] = 15,
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        # Base class handles batch_size validation, model construction, output_handler initialization, etc.
        super().__init__(model_cfg, batch_size, output_json_filepath)

        self.save_every = max(save_every, int(batch_size * DEFAULT_SAVE_EVERY_FACTOR))

        # Mode identification
        self.pressure_mode = mode == "pressure"
        self.perf_mode = mode == "perf" or self.pressure_mode
        self.pressure_time = pressure_time
        # status_counter: If perf mode requires additional threads/counters, consider lazy creation to reduce overhead in normal mode.
        self.status_counter = StatusCounter()

    def _monitor_status_thread(
        self,
        stop_event: threading.Event,
        message_share_memory: shared_memory.SharedMemory,
    ) -> None:
        """Monitor status thread for reporting statistics.

        Args:
            stop_event: Event to signal thread termination
            message_share_memory: Shared memory for status communication
        """
        message_buf = message_share_memory.buf

        while not stop_event.is_set():
            post_req = self.status_counter.post_req
            get_req = self.status_counter.get_req
            failed_req = self.status_counter.failed_req
            finish_req = self.status_counter.finish_req
            finish_case_req = self.status_counter.case_finish_req

            # Pack -> bytes, then write back to corresponding slice of shared memory
            packed = struct.pack("5I", post_req, get_req, failed_req, finish_req, finish_case_req)
            # Write to 16 bytes starting from offset=4 (4 unsigned ints)
            flag = struct.unpack_from("I", message_buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]])[0]
            if flag == 1 or stop_event.is_set():
                break
            message_buf[MESSAGE_INFO.POST[0]:MESSAGE_INFO.POST[0] + len(packed)] = packed
            time.sleep(STATUS_REPORT_INTERVAL)
        try:
            post_req = self.status_counter.post_req
            get_req = self.status_counter.get_req
            failed_req = self.status_counter.failed_req
            finish_req = self.status_counter.finish_req
            finish_case_req = self.status_counter.case_finish_req
            packed = struct.pack("5I", post_req, get_req, failed_req, finish_req, finish_case_req)
            message_buf[MESSAGE_INFO.POST[0]:MESSAGE_INFO.POST[0] + len(packed)] = packed
        except Exception as e:
            self.logger.debug(f"Failed to update status counter: {str(e)}")
            pass

    @abstractmethod
    async def do_request(
        self, data: Any, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession
    ) -> Any:
        """Call model to do request, return output. Must be async in your implementation.

        Args:
            data: Request data
            token_bucket: Semaphore for rate limiting
            session: HTTP session for the request

        Returns:
            Model output

        Raises:
            NotImplementedError: If not implemented in subclass
        """
        pass

    async def warmup(self, data_list: list, warmup_times: int = 1, concurrency: int = 1):
        """Warmup the inferencer.

        Args:
            data_list: Data list to warmup
            warmup_times: Warmup times
            concurrency: Concurrency
        """
        warmup_semaphore = (
            asyncio.Semaphore(concurrency)
            if concurrency
            else contextlib.nullcontext()
        )
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=concurrency + 1),
            timeout=aiohttp.ClientTimeout(total=get_request_time_out()),
            max_line_size=get_max_chunk_size(),
        )
        warmup_tasks = []
        async def warmup_limited_request_func(data: dict):
            async with warmup_semaphore:
                await self.do_request(
                    data=copy.deepcopy(data),
                    token_bucket=None,
                    session=session,
                )
        try:
            for i in range(warmup_times):
                data = data_list[i % len(data_list)]
                request_task = asyncio.create_task(warmup_limited_request_func(data))
                warmup_tasks.append(request_task)

            with tqdm(total=warmup_times, desc="Warmup", unit="case") as pbar:
                for coro in asyncio.as_completed(warmup_tasks):
                    await coro
                    pbar.update(1)
        finally:
            await session.close()

        warmup_results = dict(
            success=0,
            failed=0,
            failed_reasons=defaultdict(int),
        )

        while not self.output_handler.cache_queue.async_q.empty():
            try:
                res = await asyncio.wait_for(self.output_handler.cache_queue.async_q.get(), timeout=1)
            except Exception as e:
                raise AISBenchRuntimeError(ICLI_CODES.WARMUP_GET_RESULT_FAILED,
                                    f"Get result from cache queue failed: {str(e)}")
            data_id, data_abbr, input, output, gold = res
            if not isinstance(output, Output):
                raise AISBenchRuntimeError(ICLI_CODES.UNKNOWN_ERROR, f"Warmup expected Output object, but got {type(output)}")
            if not output.success:
                warmup_results["failed"] += 1
                warmup_results["failed_reasons"][output.error_info] += 1
                continue
            warmup_results["success"] += 1

            self.logger.debug(
                f"Warmup success: data_id: {data_id}, "
                f"data_abbr: {data_abbr}, "
                f"input: {input}, "
                f"output: {output.get_prediction()}, "
                f"gold: {gold}"
            )
        return warmup_results

    def _read_and_unpickle(
        self, buf: memoryview, index_data: Tuple[int, int, int]
    ) -> Any:
        """Read and unpickle data from shared memory.

        Args:
            buf: Memory buffer
            index_data: Tuple of (index, offset, length)

        Returns:
            Unpickled data object
        """
        _, offset, length = index_data
        raw_bytes = buf[offset : offset + length]
        data_bytes = bytes(raw_bytes)
        return pickle.loads(data_bytes)

    def _get_single_data(
        self,
        share_memory: shared_memory.SharedMemory,
        indexes: Dict,
        message_share_memory: shared_memory.SharedMemory,
        stop_event: threading.Event,
    ) -> Optional[Any]:
        """Attempt to consume one token (if configured) and one index entry.

        All blocking operations are dispatched to a thread via asyncio.to_thread so the
        event loop is never blocked.
        Returns the deserialized data or None if there's no data / should stop.

        Args:
            share_memory: Shared memory containing data
            indexes: Indexes for data
            message_share_memory: Shared memory for message

        Returns:
            Deserialized data or None if no data available
        """
        data_index = -1
        while data_index == -1 and not stop_event.is_set():
            flag = struct.unpack_from("I", message_share_memory.buf[MESSAGE_INFO.DATA_SYNC_FLAG[0]:MESSAGE_INFO.DATA_SYNC_FLAG[1]], 0)[0]
            if flag != 1:
                continue
            data_index = struct.unpack_from("i", message_share_memory.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]], 0)[0]
        message_share_memory.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", -1)
        if data_index <0 or data_index >= len(indexes):
            self.logger.debug(f"Data index out of range: {data_index}, return None")
            return None
        index_data = indexes[data_index]
        if not index_data:
            self.logger.debug(f"Index data is None, return None")
            return None
        return self._read_and_unpickle(share_memory.buf, index_data)

    def _fill_janus_queue(
        self,
        dataset_share_memory: shared_memory.SharedMemory,
        message_share_memory: shared_memory.SharedMemory,
        indexes: Dict,
        janus_queue: janus.Queue,
        stop_event: threading.Event,
    ):
        """Pre-fill janus queue with initial data.

        Args:
            dataset_share_memory: Shared memory containing dataset
            indexes: Indexes for data
            data_index_value: Value for data index
            janus_queue: Janus queue for thread-async communication
            stop_event: Event to signal termination
        """
        # Pre-fill up to batch_size items first (mirrors your original behavior)
        for _ in range(self.batch_size):
            if stop_event.is_set():
                break
            data = self._get_single_data(
                dataset_share_memory, indexes, message_share_memory, stop_event
            )
            # Block if queue is full -> natural backpressure
            janus_queue.sync_q.put(data)
            if data is None:
                break
        self.logger.debug(f"Fill first batch of data to janus queue")

    def _producer_thread_target(
        self,
        dataset_share_memory: shared_memory.SharedMemory,
        message_share_memory: shared_memory.SharedMemory,
        indexes: Dict,
        janus_queue: janus.Queue,
        stop_event: threading.Event,
    ) -> None:
        """Thread target: read from shared memory/index queue and push into janus.sync_q.

        Args:
            dataset_share_memory: Shared memory containing dataset
            indexes: Indexes for data
            message_share_memory: Shared memory for message
            janus_queue: Janus queue for thread-async communication
            stop_event: Event to signal termination
        """
        # Continuous fill until stop_event or sentinel
        while not stop_event.is_set():
            data = self._get_single_data(
                dataset_share_memory, indexes, message_share_memory, stop_event
            )
            while True:
                try:
                    janus_queue.sync_q.put(data, timeout=1)
                except Exception as e:
                    if stop_event.is_set():
                        self.logger.debug(f"Producer thread stopped by stop_event with {e}")
                        return
                    continue
                break
            if data is None:
                self.logger.debug(f"Producer thread get sentinel, inference data producer exit")
                break

    def _sync_main_process_with_message(
        self, message_share_memory: shared_memory.SharedMemory,
    ):
        """Synchronize with main process using shared memory message.

        Args:
            message_share_memory: Shared memory for communication
        """
        message_buf = message_share_memory.buf
        struct.pack_into("I", message_buf, 0, WAIT_FLAG)
        self.logger.debug(f"Sync main process, wait for main process to sync flag to 0")
        while struct.unpack_from("I", message_buf, 0)[0] != 0:
            time.sleep(SYNC_MAIN_PROCESS_INTERVAL)
        self.logger.debug(f"Main process sync flag to 0")

    async def wait_get_data(self, async_queue: janus.Queue.async_q, stop_event: asyncio.Event):
        """Wait for data from async queue.
        """
        while not stop_event.is_set():
            try:
                data = await asyncio.wait_for(async_queue.get(), timeout=1)
                return data
            except asyncio.exceptions.TimeoutError:
                continue

    async def _worker_loop(
        self,
        token_bucket: BoundedSemaphore,
        async_queue: janus.Queue.async_q,
    ) -> None:
        """Worker task: repeatedly fetch data and call the async do_request.

        Consumes from janus.async_q (async_queue).

        Args:
            token_bucket: Semaphore for rate limiting
            async_queue: Async queue for data consumption
        """
        num_workers = self.batch_size if self.batch_size and self.batch_size > 0 else 1

        # Limit maximum concurrency
        semaphore = asyncio.Semaphore(num_workers) if num_workers else None
        # Reuse session to improve concurrency
        connector = aiohttp.TCPConnector(limit=num_workers + 1)
        timeout = aiohttp.ClientTimeout(total=get_request_time_out())
        session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, max_line_size=get_max_chunk_size()
        )
        self.logger.debug(f"Create aiohttp session with "
                     f"connector: {connector}, "
                     f"timeout: {timeout}, "
                     f"max_line_size: {get_max_chunk_size()}, "
                     f"max_concurrency: {num_workers}")
        start_time = time.perf_counter()

        stop_event = asyncio.Event()

        async def limited_request_func(data):
            if semaphore is None:
                await self.do_request(data, token_bucket, session)
                if self.pressure_mode:
                    raise ParameterValueError(
                        ICLI_CODES.CONCURRENCY_NOT_SET_IN_PRESSEURE_MODE,
                        f"Concurrency not set in pressure mode, please set `batch_size` in model config",
                    )
            async with semaphore:
                # Pressure mode: continuously send requests until pressure_time
                if self.pressure_mode:
                    # Prefetch next data immediately after first request
                    next_data_task = asyncio.create_task(self.wait_get_data(async_queue, stop_event))
                    await self.do_request(data, token_bucket, session)
                    try:
                        while time.perf_counter() - start_time < self.pressure_time:
                            if stop_event.is_set():
                                break
                            # Wait for next data
                            data = await next_data_task
                            # If sentinel (None) is received, exit the inner loop
                            if data is None:
                                stop_event.set()
                                break
                            # Start prefetching the next data immediately (before sending current request)
                            next_data_task = asyncio.create_task(self.wait_get_data(async_queue, stop_event))
                            # Send request (next data is being prefetched in parallel)
                            await self.do_request(data, token_bucket, session)
                    finally:
                        # Cancel the prefetch task if it's still running
                        if not next_data_task.done():
                            next_data_task.cancel()
                            try:
                                await next_data_task
                            except asyncio.CancelledError:
                                pass
                else:
                    await self.do_request(data, token_bucket, session)
        tasks = []
        try:
            while not stop_event.is_set():
                if token_bucket:
                    acquired = await asyncio.to_thread(token_bucket.acquire, timeout=1)
                    if not acquired:
                        continue
                else:
                    # Slightly limit RR when no token to avoid high CPU usage causing TTFT accumulation
                    await asyncio.sleep(BLOCK_INTERVAL)

                data = await self.wait_get_data(async_queue, stop_event)

                # data == None -> sentinel
                if data is None:
                    await asyncio.wait_for(async_queue.put(None), timeout=1)
                    break
                # Call user-provided async request
                tasks.append(asyncio.create_task(limited_request_func(data)))
                # Pressure mode: exit when stable state is reached
                if self.pressure_mode and len(tasks) >= num_workers:
                    self.logger.debug(f"Pressure mode, exit when stable state is reached")
                    break
            await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            self.logger.debug(f"Keyboard interrupt, set stop event")
            stop_event.set()
            # keyboard interrupt wait for all tasks to finish
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.logger.debug(f"Close aiohttp session")
            await session.close()

    def inference_with_shm(
        self,
        dataset_shm_name: str,
        message_shm_name: str,
        indexes: Dict,
        token_bucket: BoundedSemaphore,
        output_json_filepath: Optional[str] = None,
    ) -> Dict[str, int]:
        """Top-level runner using janus for thread<->async bridging.

        This function is synchronous: it creates a dedicated asyncio loop, starts
        producer threads (which put into janus.sync_q) and then runs consumer
        tasks on the new loop which consume from janus.async_q.

        Args:
            dataset_shm_name: Name of dataset shared memory
            message_shm_name: Name of message shared memory
            indexes: Indexes for data
            token_bucket: Semaphore for rate limiting
            output_json_filepath: Optional output file path

        Returns:
            Dictionary with status information
        """
        dataset_share_memory = shared_memory.SharedMemory(dataset_shm_name)
        message_share_memory = shared_memory.SharedMemory(message_shm_name)

        # status control
        stop_event = threading.Event()
        self.status_counter = StatusCounter(self.batch_size)
        self.status_counter.start()

        # create janus queue bound to that loop
        janus_queue = janus.Queue(maxsize=self.batch_size + 1)
        # start report thread
        report_thread = threading.Thread(
            target=self._monitor_status_thread,
            args=(stop_event, message_share_memory),
            daemon=True,
        )
        report_thread.start()

        self._fill_janus_queue(
            dataset_share_memory,
            message_share_memory,
            indexes,
            janus_queue,
            stop_event,
        )

        # start producer thread (fills janus_queue.sync_q)
        producer_thread = threading.Thread(
            target=self._producer_thread_target,
            args=(
                dataset_share_memory,
                message_share_memory,
                indexes,
                janus_queue,
                stop_event,
            ),
            daemon=True,
        )
        producer_thread.start()

        # Start cache consumer thread (preserve original behaviour)
        out_path = self.get_output_dir(output_json_filepath)
        tmp_json_filepath = os.path.join(out_path, "tmp")
        os.makedirs(tmp_json_filepath, exist_ok=True)
        tmp_file_name = f"tmp_{uuid.uuid4().hex[:8]}.jsonl"
        cache_consumer_thread = threading.Thread(
            target=self.output_handler.run_cache_consumer,
            args=(
                tmp_json_filepath,
                tmp_file_name,
                self.perf_mode,
                self.save_every,
            ),
        )
        cache_consumer_thread.start()
        # Notify main process to start generating tokens
        self._sync_main_process_with_message(message_share_memory)
        # Create a fresh event loop dedicated for running the async consumers
        loop = asyncio.new_event_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size)
        )
        asyncio.set_event_loop(loop)
        worker_task = loop.create_task(
            self._worker_loop(token_bucket, janus_queue.async_q)
        )
        # Run consumers on the created loop
        try:
            loop.run_until_complete(worker_task)
        except KeyboardInterrupt:
            self.logger.warning(
                "Keyboard interrupt. Please wait for tasks exit gracefully..."
            )
            stop_event.set()
            message_share_memory.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 1)
            message_share_memory.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", -1)
            worker_task.cancel()
            try:
                loop.run_until_complete(asyncio.wait_for(worker_task, timeout=10))
                self.logger.debug(f"Worker task completed")
            except Exception as e:
                self.logger.warning(f"Error waiting for worker task: {str(e)}")
        finally:
            # Orderly shutdown
            stop_event.set()
            producer_thread.join()
            message_share_memory.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 1)
            message_share_memory.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", -1)
            self.logger.debug(f"Stop event set")
             # Join threads
            report_thread.join()

            self.status_counter.stop()
            self.status_counter.join()
            self.output_handler.stop_cache_consumer()
            cache_consumer_thread.join()

            # Close janus queue properly
            janus_queue.close()
            loop.run_until_complete(janus_queue.wait_closed())

            loop.close()
            self.logger.debug(f"Asyncio event loop closed")

            # Write data with same abbr to same jsonl file
            self.output_handler.write_to_json(out_path, self.perf_mode)

            dataset_share_memory.close()
            message_share_memory.close()

        return {"status": 0}


class StatusCounter(threading.Thread):
    """Thread-safe status counter for tracking request statistics."""

    def __init__(self, batch_size: int = 0):
        """Initialize status counter.

        Args:
            batch_size: Size of batch for queue capacity calculation
        """
        super().__init__(daemon=True)
        self.logger = AISLogger()
        self.post_req = 0
        self.get_req = 0
        self.failed_req = 0
        self.finish_req = 0
        self.case_finish_req = 0
        # Use thread-safe standard library queue with capacity equal to batch_size * 5
        self.status_queue = None
        if batch_size > 0:
            self.status_queue: std_queue.Queue = std_queue.Queue(maxsize=batch_size * 5)
        self._stop_event = threading.Event()
        self._print_interval = 1.0  # Print status once per second

    # These maintain coroutine interface (caller uses await) but internally use synchronous put_nowait
    async def post(self):
        """Record a post request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("post_req")

    async def rev(self):
        """Record a get request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("get_req")

    async def failed(self):
        """Record a failed request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("failed_req")

    async def finish(self):
        """Record a finished request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("finish_req")

    async def case_finish(self):
        """Record a finished request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("case_finish_req")

    def stop(self):
        """Request thread to stop (called by main thread/coroutine)."""
        self._stop_event.set()

    def run(self):
        """
        Run in independent thread: print status at least once per second; continuously pull from queue and update counts.
        Use short polling with timeout=0.2 for better responsiveness, while using time accumulation to achieve once-per-second printing.
        """
        if not self.status_queue:
            return
        while not self._stop_event.is_set():
            try:
                # Small timeout to respond to stop requests promptly
                status = self.status_queue.get(timeout=0.2)
            except std_queue.Empty:
                status = None

            if status is not None:
                if status == "post_req":
                    self.post_req += 1
                elif status == "get_req":
                    self.get_req += 1
                elif status == "failed_req":
                    self.failed_req += 1
                elif status == "finish_req":
                    self.finish_req += 1
                elif status == "case_finish_req":
                    self.case_finish_req += 1

        # After stop request, try to consume remaining items in queue and update statistics (optional)
        while True:
            try:
                status = self.status_queue.get_nowait()
            except std_queue.Empty:
                break
            if status == "post_req":
                self.post_req += 1
            elif status == "get_req":
                self.get_req += 1
            elif status == "failed_req":
                self.failed_req += 1
            elif status == "finish_req":
                self.finish_req += 1
            elif status == "case_finish_req":
                self.case_finish_req += 1
        self.logger.debug("Status counter stopped. "
                          f"Process {os.getpid()} finished with status: "
                          f"{self.post_req} post requests "
                          f"{self.get_req} get requests "
                          f"{self.failed_req} failed requests "
                          f"{self.finish_req} finish requests "
                          f"{self.case_finish_req} case finish requests")