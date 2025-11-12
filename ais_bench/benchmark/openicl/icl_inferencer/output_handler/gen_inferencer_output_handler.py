from typing import List, Optional, Union

import sqlite3
import uuid

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError

class GenInferencerOutputHandler(BaseInferencerOutputHandler):
    """
    Output handler for generation-based inference tasks.

    This handler specializes in processing generation model outputs,
    supporting both performance measurement and accuracy evaluation modes.
    It handles different data formats and provides appropriate result storage.

    Attributes:
        all_success (bool): Flag indicating if all operations were successful
        perf_mode (bool): Whether in performance measurement mode
        cache_queue (queue.Queue): Queue for caching results before writing
    """

    def __init__(self, perf_mode: bool = False, save_every: int = 100) -> None:
        """
        Initialize the generation inferencer output handler.

        Args:
            perf_mode (bool): Whether to run in performance measurement mode
                            (default: False for accuracy mode)
        """
        super().__init__(save_every)
        self.all_success = True
        self.perf_mode = perf_mode

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
        """
        # Performance mode: only store metrics
        if self.perf_mode and isinstance(output, Output):
            result_data = output.get_metrics()
            result_data = self._extract_and_write_arrays(
                result_data, conn
            )
        elif isinstance(output, str):
            result_data = {
                "success": True,
                "uuid": uuid.uuid4().hex[:8],
                "origin_prompt": input,
                "prediction": output,
            }
            if gold:
                result_data["gold"] = gold
        else:
            # Accuracy mode: store full input/output data
            result_data = {
                "success": (
                    output.success if isinstance(output, Output) else True
                ),
                "uuid": output.uuid,
                "origin_prompt": input,
                "prediction": (
                    output.get_prediction()
                    if isinstance(output, Output)
                    else output
                ),
            }

            if gold:
                result_data["gold"] = gold

        # Check for failures and update success status
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