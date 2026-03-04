from typing import Optional, Union, List


from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import FunctionCallOutput
from ais_bench.benchmark.utils.logging.exceptions import AISBenchTypeError
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES


class BFCLV3OutputHandler(BaseInferencerOutputHandler):
    """
    Output handler for BFCLV3 inference tasks.
    """
    def get_prediction_result(
        self,
        output: FunctionCallOutput,
        gold: Optional[str] = None,
        input: Optional[Union[str, List[str]]] = None,
        data_abbr: Optional[str] = ""
    ) -> dict:
        """
        Get the prediction result for BFCLV3 inference tasks.

        Args:
            output (FunctionCallOutput): Output result from inference
            gold (Optional[str]): Ground truth data for comparison
            input (Optional[Union[str, List[str]]]): Input data for the inference (not used in this implementation)
            data_abbr (Optional[str]): Abbreviation of the dataset (not used in this implementation)

        Returns:
            dict: Prediction result containing success, uuid, prediction (tool_calls), and inference_log
        Raises:
            AISBenchTypeError: If output is not a FunctionCallOutput instance
        """
        if not isinstance(output, FunctionCallOutput):
            raise AISBenchTypeError(ICLI_CODES.UNKNOWN_ERROR, f"Expected FunctionCallOutput, but got {type(output)}")
        result_data = {
            "uuid": output.uuid,
            "success": output.success,
            "prediction": output.tool_calls,
            "inference_log": output.inference_log,
        }
        return result_data