"""Basic Inferencer."""

import os
import os.path as osp
from abc import abstractmethod
from typing import List, Optional, Dict
from collections import defaultdict

import json
from mmengine.dist import is_main_process


from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.utils.config import build_model_from_cfg
from ais_bench.benchmark.utils.core.abbr import model_abbr_from_cfg
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError, ParameterValueError


MAX_BATCH_SIZE = 100000


class BaseInferencer:
    """Base Inferencer class for all evaluation Inferencer.

    Attributes:
        model_cfg (Config): model config.
        model: built model instance (returned by build_model_from_cfg).
        batch_size (int): batch size for processing.
        output_json_filepath (str): output json path or directory.
        output_handler: output handler instance (provided by BaseInferencerOutputHandler).
        is_main_process (bool): whether the current process is the main process (distributed scenario).
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        output_json_filepath: Optional[str] = "./icl_inference_output",
    ) -> None:
        # basic parameters normalization
        self.logger = AISLogger()
        self.model_cfg = model_cfg
        self.batch_size = int(batch_size) if batch_size else 1

        if self.batch_size < 1 or self.batch_size > MAX_BATCH_SIZE:
            raise ParameterValueError(ICLI_CODES.BATCH_SIZE_OUT_OF_RANGE, 
                f"The range of batch_size is [1, {MAX_BATCH_SIZE}], but got {self.batch_size}. "
                "Please set it in datasets config"
            )

        # save output path (subclass does not need to repeat assignment)
        self.output_json_filepath = output_json_filepath
        self.logger.debug(f"Output JSON file path: {self.output_json_filepath}")

        # construct model and output handler (if needed, can be changed to lazy build)
        self.model = build_model_from_cfg(model_cfg)
        self.output_handler = BaseInferencerOutputHandler()

        # identify whether the current process is the main process (avoid covering the method with boolean)
        self.is_main_process = self._is_main_process()
        self.perf_mode = False
        self.task_state_manager = None

    @abstractmethod
    def get_data_list(
        self,
        retriever: BaseRetriever,
    ) -> List:
        """Get the data list for inference."""

        raise AISBenchImplementationError(ICLI_CODES.UNKNOWN_ERROR, 
                                           f"Method {self.__class__.__name__} hasn't been implemented yet")

    def set_task_state_manager(self, task_state_manager):
        self.logger.debug(f"Set task state manager: {task_state_manager}")
        self.task_state_manager = task_state_manager

    def get_finish_data_list(self) -> Dict[str, Dict[str, Dict]]:
        """Get the finish data list, which will not infer again in reuse mode.

        Returns:
            Dict[str, Dict[str, Dict]]: The finish data list, which is a dictionary of data_abbr and data_id to data_dict.
        """
        if self.perf_mode:
            self.logger.debug(f"Performance mode, return empty finish data dict")
            return {}
        output_dir = self.get_output_dir()
        if not os.path.exists(output_dir):
            self.logger.debug(f"Output directory {output_dir} does not exist, return empty finish data dict")
            return {}
        tmp_dir = os.path.join(output_dir, "tmp")
        abbr_finish_data_cache = defaultdict(dict) #
        tmp_finish_data_cache = defaultdict(dict) #
        finish_data_cache = defaultdict(dict)

        for finish_data in os.listdir(output_dir):
            if not finish_data.endswith(".jsonl"):
                continue
            data_abbr = finish_data.split(".")[0]
            
            with open(os.path.join(output_dir, finish_data), "r") as f:
                for line in f:
                    data = json.loads(line)
                    if not data.get("success"):
                        continue
                    abbr_finish_data_cache[data.get("uuid")] = data 

        if not os.path.exists(tmp_dir):
            return finish_data_cache
        for file in os.listdir(tmp_dir):
            if not file.endswith(".jsonl"):
                continue
            with open(os.path.join(tmp_dir, file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    if not data.get("success"):
                        continue
                    tmp_finish_data_cache[data.get("uuid")] = data
        to_write = defaultdict(dict)
        for key, value in tmp_finish_data_cache.items():
            data_abbr = value.get("data_abbr")
            finish_data_cache.setdefault(data_abbr, []).append(value)
            #abbr.json to write
            if key not in abbr_finish_data_cache.keys():
                to_write.setdefault(data_abbr, []).append(json.dumps(value) + "\n")
        for data_abbr, lines in to_write.items():
            with open(os.path.join(output_dir, f"{data_abbr}.jsonl"), "a") as f:
                f.writelines(lines)

        return finish_data_cache

    def _is_main_process(self):
        if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
            return int(os.getenv("RANK", "0")) == 0
        return is_main_process()

    def get_output_dir(self, output_json_filepath: Optional[str] = None):
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        output_json_filepath = osp.join(
            output_json_filepath,
            "performances" if self.perf_mode else "predictions",
            model_abbr_from_cfg(self.model_cfg),
        )
        self.logger.debug(f"Output directory: {output_json_filepath}")
        return output_json_filepath
