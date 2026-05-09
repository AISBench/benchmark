from typing import List, Optional, Union, Dict

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError

logger = AISLogger()

class PPLRequestOutput(Output):
    def __init__(self, perf_mode: bool = False) -> None:
        super().__init__(perf_mode)
        self.ppl: float = 0
        self.origin_prompt_logprobs: Dict[str, Dict[str, Union[str, int, float]]] = None

class PPLResponseOutput(Output):
    def __init__(self, perf_mode: bool = False) -> None:
        super().__init__(perf_mode)
        self.input = []
        self.label_ppl_list: List[Dict[str, float]] = []
        self.origin_prompt_logprobs: List[Dict[str, Dict[str, Union[str, int, float]]]] = []

class PPLInferencerOutputHandler(BaseInferencerOutputHandler):
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
        self.perf_mode = perf_mode

    def get_prediction_result(
        self,
        output: Union[str, PPLResponseOutput],
        gold: Optional[str] = None,
        input: Optional[Union[str, List[str]]] = None,
        data_abbr: Optional[str] = ""
    ) -> dict:
        """
        Get the prediction result for performance mode.

        Args:
            output (Union[str, PPLResponseOutput]): Model output
            gold (Optional[str]): Ground truth data for comparison
            input (Optional[Union[str, List[str]]]): Input data for the inference
            data_abbr (Optional[str]): Abbreviation of the dataset

        Returns:
            dict: Prediction result
        """
        if not isinstance(output, PPLResponseOutput):
            raise AISBenchImplementationError(ICLI_CODES.UNKNOWN_ERROR, f"Output is not a PPLResponseOutput")
        result_data = {
            "success": (
                output.success
            ),
            "uuid": output.uuid,
            "origin_prompt": output.input,
            "ppl_list": output.label_ppl_list,
            "origin_prompt_logprobs": output.origin_prompt_logprobs,
            "prediction": (
                output.get_prediction()
                if isinstance(output, Output)
                else output
            ),
        }
        if gold:
            result_data["gold"] = gold
        return result_data