import json
import os
import queue
import shutil
import uuid
import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Union

import sqlite3
import numpy as np
import janus

from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.results import safe_write
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.db_utils import init_db, save_numpy_to_db
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError, ParameterValueError, FileOperationError, AISBenchRuntimeError

DB_REF_KEY = "__db_ref__"
DB_DATA_DIR = "db_data"


class BaseInferencerOutputHandler:
    """
    Base class for handling inferencer output results.

    This class provides the foundation for managing inference results, including
    caching, serialization, and file operations. It supports both performance
    and accuracy modes with different data handling strategies.

    Attributes:
        results_dict (defaultdict): Dictionary to store results by data abbreviation
        cache_queue (queue.Queue): Queue for caching results before writing
        all_success (bool): Flag indicating if all operations were successful
    """

    def __init__(self, perf_mode: bool = False, save_every: int = 100) -> None:
        """
        Initialize the base inferencer output handler.

        Args:
            perf_mode (bool): Whether to run in performance measurement mode
                            (default: False for accuracy mode)
            save_every (int): Number of items to batch before writing (default: 100)
        """
        self.logger = AISLogger()
        self.results_dict = defaultdict(dict)
        self.failed_results_dict = defaultdict(dict)
        self.cache_queue = janus.Queue()
        self.perf_mode = perf_mode
        self.all_success = True
        self.save_every = save_every

    @abstractmethod
    def get_prediction_result(self, output: Union[str, Output], gold: Optional[str] = None, input: Optional[Union[str, List[str]]] = None) -> dict:
        """
        Get the prediction result.

        Args:
            output (Union[str, Output]): Output result from inference
            gold (Optional[str]): Ground truth data for comparison
            input (Optional[Union[str, List[str]]]): Input data for the inference

        Returns:
            dict: Prediction result
        """
        raise AISBenchImplementationError(ICLI_CODES.UNKNOWN_ERROR,
                                       f"Method {self.__class__.__name__} hasn't been implemented yet")

    def get_result(
        self,
        conn: sqlite3.Connection,
        input: Union[str, List[str]],
        output: Union[str, Output],
        gold: Optional[str] = None,
    ) -> dict:
        """
        Save inference results to the results dictionary.

        Handles both performance and accuracy modes with different data storage
        strategies. In performance mode, only metrics are stored. In accuracy mode,
        full input/output data is preserved for evaluation.

        Args:
            conn (sqlite3.Connection): Database connection to write results to
            input (Union[str, List[str]]): Input data for the inference
            output (Union[str, Output]): Output result from inference
            gold (Optional[str]): Ground truth data for comparison

        Raises:
            AISBenchImplementationError: If not implemented by subclass
        """
        # Performance mode: only store metrics
        if self.perf_mode and isinstance(output, Output):
            result_data = output.get_metrics()
            result_data = self._extract_and_write_arrays(
                result_data, conn
            )

        elif isinstance(output, str):
            # Accuracy mode: store full input/output data
            result_data = {
                "success": True,
                "uuid": uuid.uuid4().hex[:8],
                "origin_prompt": input,
                "prediction": output,
            }

            if gold:
                result_data["gold"] = gold
        else:
            result_data = self.get_prediction_result(output, gold=gold, input=input)
        if not result_data.get("success", True):
            self.all_success = False
            if isinstance(output, Output) and hasattr(output, "error_info"):
                result_data["error_info"] = output.error_info
                self.logger.debug(f"Failed operation at data id {output.uuid}, error info: {result_data['error_info']}")
            else:
                self.logger.warning(
                    f"No error info available for failed operation at data id {output.uuid}"
                )
        return result_data

    def write_to_json(self, save_dir: str, perf_mode: bool) -> None:
        """
        Write results to JSON files.

        Saves results to JSONL files based on data_abbr to avoid process conflicts.
        Uses fcntl for file locking to ensure thread safety.

        Args:
            save_dir (str): Directory path to save the JSON files
            perf_mode (bool): If True, saves detailed performance data;
                            if False, saves basic result data

        Raises:
            OSError: If unable to create directory or write files
            ValueError: If save_dir is invalid
        """
        if not isinstance(save_dir, str) or not save_dir.strip():
            raise ParameterValueError(ICLI_CODES.UNKNOWN_ERROR,
                                      f"'save_dir' must be a non-empty string representing a directory path, but got {save_dir}")

        file_path = Path(save_dir)
        try:
            # Ensure directory exists
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            for data_abbr, results_dict in self.results_dict.items():
                if not results_dict:
                    continue

                if not perf_mode:
                    raw_data_name = data_abbr + ".jsonl"
                else:
                    raw_data_name = data_abbr + "_details.jsonl"

                file_path = Path(save_dir) / raw_data_name
                safe_write(results_dict, file_path)
                self.logger.debug(f"Process {os.getpid()} write results to {file_path}")

            for data_abbr, results_dict in self.failed_results_dict.items():
                if not results_dict:
                    continue
                failed_data_name = data_abbr + "_failed.jsonl"
                file_path = Path(save_dir) / failed_data_name
                safe_write(results_dict, file_path)
                self.logger.debug(f"Process {os.getpid()} write failed results to {file_path}")

        except Exception as e:
            raise FileOperationError(
                ICLI_CODES.INFER_RESULT_WRITE_ERROR,
                f"Failed to write results to {file_path}: {str(e)}",
            )

    async def report_cache_info(
        self,
        id: int,
        input: Union[List[str], str],
        output: Union[Output, str],
        data_abbr: str,
        gold: Optional[str] = None,
    ) -> bool:
        """
        Synchronously and non-blockingly add a record to the queue.

        Supports asyncio.Queue, queue.Queue, multiprocessing.Queue (if they implement
        put_nowait / put(block=False)). Returns True if successfully queued (or
        scheduled for queuing), False if failed (e.g., full queue or unable to schedule).

        Args:
            id (int): The index of the current result
            input (Union[List[str], str]): Input data for the inference
            output (Union[Output, str]): Output result from inference
            data_abbr (str): Abbreviation for the dataset
            gold (Optional[str]): Ground truth data for comparison

        Returns:
            bool: True if successfully queued, False otherwise

        Raises:
            queue.Full: If queue is full and cannot accept new items
        """
        try:
            item = (id, data_abbr, input, output, gold)
            self.cache_queue.async_q.put_nowait(item)
            return True
        except Exception as e:
            self.logger.debug(f"Failed to report cache info to async_q: {str(e)}")
            return False

    def report_cache_info_sync(
        self,
        id: int,
        input: Union[List[str], str],
        output: Union[Output, str],
        data_abbr: str,
        gold: Optional[str] = None,
    ) -> bool:
        """
        Synchronously and non-blockingly add a record to the queue.

        Supports queue.Queue, multiprocessing.Queue (if they implement
        put_nowait / put(block=False)). Returns True if successfully queued (or
        scheduled for queuing), False if failed (e.g., full queue or unable to schedule).

        Args:
            id (int): The index of the current result
            input (Union[List[str], str]): Input data for the inference
            output (Union[Output, str]): Output result from inference
            data_abbr (str): Abbreviation for the dataset
            gold (Optional[str]): Ground truth data for comparison

        Returns:
            bool: True if successfully queued, False otherwise

        Raises:
            queue.Full: If queue is full and cannot accept new items
        """
        try:
            item = (id, data_abbr, input, output, gold)
            self.cache_queue.sync_q.put_nowait(item)
            return True
        except Exception as e:
            self.logger.debug(f"Failed to report cache info to sync_q: {str(e)}")
            return False

    def _extract_and_write_arrays(
        self,
        obj: Any,
        conn: sqlite3.Connection,
    ) -> Any:
        """
        Recursively scan obj and immediately write numpy.ndarray objects to database.

        Returns a JSON-serializable replacement object where array positions are
        replaced with {"__db_ref__": "<id>"} placeholders.

        Args:
            obj: Object to scan for arrays
            conn (sqlite3.Connection): Database connection to write arrays to

        Returns:
            Any: JSON-serializable object with array references replaced

        Raises:
            ValueError: If obj cannot be processed
            RuntimeError: If database operations fail
        """

        # Atomic types return directly
        if obj is None or isinstance(obj, (bool, int, float, str, list, tuple)):
            return obj

        # If numpy array -> write to database and return reference
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj)
            try:
                id = save_numpy_to_db(conn, arr, self.save_every)
            except Exception as e:
                self.logger.warning(f"Failed to save numpy array to database: {str(e)}, will not save this array to database")
                return None

            # Return serializable placeholder
            return {DB_REF_KEY: id}


        # dict -> recursively process values
        if isinstance(obj, dict):
            # Use dict comprehension for better performance
            out = {
                str(k): self._extract_and_write_arrays(
                    v, conn
                )
                for k, v in obj.items()
            }
            return out

        # Other types: try JSON serialization, otherwise convert to string
        try:
            json.dumps(obj, ensure_ascii=False)
            return obj
        except Exception as json_error:
            try:
                str_obj = str(obj)
                return str_obj
            except Exception as str_error:
                self.logger.warning(
                    f"Failed to convert object to string: {str(str_error)}"
                )
                return None

    def run_cache_consumer(
        self,
        save_dir: str,
        file_name: str,
        perf_mode: bool,
        save_every: int = 1,
    ) -> None:
        """
        Run the cache consumer to process queued results.

        Processes items from the cache queue, saves results, and handles
        database file operations for array data. Manages file cleanup based on
        performance mode and success status.

        Args:
            save_dir (str): Directory to save output files
            file_name (str): Name of the output file
            perf_mode (bool): Whether in performance mode
            save_every (int): Number of items to batch before writing (default: 1)

        Raises:
            FileOperationError: If file operations fail
        """
        self.logger.debug("Running cache consumer to process queued results,"
                     f"save_dir: {save_dir}, "
                     f"file_name: {file_name}, "
                     f"perf_mode: {perf_mode}, "
                     f"save_every: {save_every}")
        db_path = Path(save_dir) / (Path(file_name).stem + ".db")
        db_name = db_path.name.replace("tmp_", "")
        conn = init_db(db_path)
        json_path = Path(save_dir) / file_name

        # Ensure directories exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        with open(json_path, "a", encoding="utf-8") as f:
                cache_data = []

                while True:
                    try:
                        item = self.cache_queue.sync_q.get(timeout=1)
                    except queue.Empty:
                        time.sleep(0.1)
                        continue

                    if item is None:
                        break
                    try:
                        uid = str(uuid.uuid4())[:8]

                        result_data = self.get_result(conn, *item[2:])
                        id, data_abbr = item[0], item[1]
                        json_data = {
                            "data_abbr": data_abbr,
                            "id": id,
                        }

                        json_data.update(result_data)
                        if perf_mode:
                            json_data["db_name"] = db_name
                            self.results_dict[data_abbr][uid] = json_data
                        else:
                            # accuracy mode: only save successful results in data_abbr.jsonl. otherwise, save to tmp file.
                            if result_data["success"]:
                                self.results_dict[data_abbr][uid] = json_data
                        if not result_data["success"]:
                            fail_data = {
                                "data_abbr": data_abbr,
                                "id": id,
                                "input": item[2],
                                "error_info": result_data["error_info"],
                            }
                            self.failed_results_dict[data_abbr][uid] = fail_data

                        # Pre-compute JSON string to avoid repeated serialization
                        json_str = json.dumps(json_data, ensure_ascii=False) + '\n'
                        # self.logger.debug(f"Saving result to cache_data: {json_str}")
                        cache_data.append(json_str)

                        # Write batch if reached save_every threshold
                        if len(cache_data) == save_every:
                            f.writelines(cache_data)
                            f.flush()  # Ensure data is written
                            cache_data = []

                    except Exception as e:
                        # Continue processing other items
                        self.logger.debug(f"Failed to process item {item}: {str(e)}")
                        continue

                # Write remaining cache data
                if cache_data:
                    f.writelines(cache_data)
                    f.flush()

        # Handle database file based on performance mode
        conn.commit()
        conn.close()
        if not perf_mode:
            if db_path.exists():
                os.remove(db_path)
        else:
            dest = db_path.parent.parent / DB_DATA_DIR
            dest.mkdir(exist_ok=True)
            if db_path.exists():
                shutil.move(str(db_path), str(dest / db_name))

        # Clean up JSON file if all operations were successful
        if self.all_success:
            if json_path.exists():
                os.remove(json_path)
        else:
            self.logger.warning(
                f"Not all items were successful, keeping JSON file for debugging: {json_path}"
            )

        # Clean up empty directories
        if json_path.parent.exists():
            try:
                empty = next(Path(json_path.parent).iterdir(), None) is None
                if empty:
                    self.logger.debug(f"Cleaning up empty directory: {json_path.parent}")
                    shutil.rmtree(json_path.parent)
            except Exception as e:
                self.logger.warning(f"Could not clean up directory {json_path.parent}: {str(e)}")
        self.logger.debug(f"Process {os.getpid()} cache consumer finished")


    def stop_cache_consumer(self) -> None:
        """
        Stop the cache consumer by sending a stop signal.

        This method signals the cache consumer to stop processing by
        adding a None item to the queue, which serves as a stop signal.
        """
        try:
            self.cache_queue.sync_q.put(None)
        except Exception as e:
            raise AISBenchRuntimeError(ICLI_CODES.UNKNOWN_ERROR, f"Failed to send stop signal to cache consumer: {str(e)}")
        self.logger.debug("Stop signal sent to cache consumer")

