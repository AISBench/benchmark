from typing import List, Optional, Union

import sqlite3
import uuid

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler, BASE64_MAX_DISPLAY_LEN
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

    def get_prediction_result(
        self,
        output: Union[str, Output],
        gold: Optional[str] = None,
        input: Optional[Union[str, List[str]]] = None,
        data_abbr: Optional[str] = "",
    ) -> dict:
        """
        Get the prediction result for accuracy mode.

        Args:
            output (Union[str, Output]): Output result from inference
            gold (Optional[str]): Ground truth data for comparison
            input (Optional[Union[str, List[str]]]): Input data for the inference
            data_abbr (Optional[str]): Abbreviation of the dataset

        Returns:
            dict: Prediction result
        """
        if (
            isinstance(input, list)
            and len(input) > 0
            and isinstance(input[0], dict)
            and isinstance(input[0].get("prompt"), list)
        ):
            for item in input[0]["prompt"]:
                if not isinstance(item, dict):
                    continue
                image_url = item.get("image_url")
                if not isinstance(image_url, dict):
                    continue
                url = image_url.get("url")
                if isinstance(url, str) and len(url) > BASE64_MAX_DISPLAY_LEN:
                    image_url["url"] = url[:BASE64_MAX_DISPLAY_LEN] + " ..."
        result_data = {
            "success": (
                output.success if isinstance(output, Output) else True
            ),
            "uuid": output.uuid if isinstance(output, Output) else str(uuid.uuid4()).replace("-", ""),
            "origin_prompt": input if input is not None else "",
            "prediction": (
                output.get_prediction()
                if isinstance(output, Output)
                else output
            ),
        }

        if gold:
            result_data["gold"] = gold
        return result_data