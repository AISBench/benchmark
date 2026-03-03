from typing import List, Optional, Union
import sqlite3
import uuid
from pathlib import Path

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import LMMOutput
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError

class LMMGenInferencerOutputHandler(BaseInferencerOutputHandler):
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
    def set_output_path(self, output_path: str) -> None:
        self.output_path = output_path

    def get_prediction_result(
        self,
        output: Union[str, LMMOutput],
        gold: Optional[str] = None,
        input: Optional[Union[str, List[str]]] = None,
        data_abbr: Optional[str] = "",
    ) -> dict:
        """
        Get the prediction result for accuracy mode.

        Args:
            output (Union[str, LMMOutput]): Output result from inference
            gold (Optional[str]): Ground truth data for comparison
            input (Optional[Union[str, List[str]]]): Input data for the inference
            data_abbr (Optional[str]): Abbreviation of the dataset

        Returns:
            dict: Prediction result
        """
        try:
            save_dir = Path(self.output_path) / f"{data_abbr}_out_file"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            for item in input[0]['prompt']:
                if item.get('image_url'):
                    item['image_url']['url'] = item['image_url']['url'][:256]
            result_data = {
                "success": (
                    output.success if isinstance(output, LMMOutput) else True
                ),
                "uuid": output.uuid if isinstance(output, LMMOutput) else str(uuid.uuid4()).replace("-", ""),
                "origin_prompt": input if input is not None else "",
                "prediction": (
                    output.get_prediction(save_dir)
                    if isinstance(output, LMMOutput)
                    else output
                ),
            }
            if gold:
                result_data["gold"] = gold
        except Exception as e:
            import traceback
            print(f"[ERROR] LMMGenInferencerOutputHandler.get_prediction_result failed: {type(e).__name__}: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to get prediction result: {e}")
        return result_data