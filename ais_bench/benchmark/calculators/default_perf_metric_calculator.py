from ais_bench.benchmark.calculators.base_perf_metric_calculator import (
    BasePerfMetricCalculator,
)
from ais_bench.benchmark.registry import PERF_METRIC_CALCULATORS
from ais_bench.benchmark.calculators.base_perf_metric_calculator import DEFAULT_STATS
from ais_bench.benchmark.utils.logging.error_codes import CALC_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError


@PERF_METRIC_CALCULATORS.register_module()
class DefaultPerfMetricCalculator(BasePerfMetricCalculator):
    """
    Default performance metric calculator for comprehensive analysis.

    This calculator provides a comprehensive analysis of all benchmark results
    without focusing on specific stages. It processes all requests and provides
    overall performance metrics.

    Args:
        stats_list (list, optional): List of statistics to calculate
    """

    def _init_datas(self, perf_details: dict, max_concurrency: int):
        """
        Initialize data structures for comprehensive analysis.

        Args:
            perf_details (dict): Performance details dictionary
            max_concurrency (int): Maximum concurrency value

        Raises:
            ValueError: If all requests failed
        """
        if sum(perf_details["success"]) == 0:
            raise AISBenchDataContentError(
                CALC_CODES.ALL_REQUEST_DATAS_INVALID,
                "All requests failed, cannot calculate performance results. Please check the error logs from responses!",
            )
        self.stage_dict = {"total": self._get_requests_id(perf_details)}
        self.result = {}
        self.max_concurrency = max_concurrency
        self.data_count = {}
        self.decode_latencies = {}
        self.success_count = {}
        self.infer_time = {}
        self.metrics = {}
        self.common_metrics = {}

        for stage_name, _ in self.stage_dict.items():
            self._process_result(perf_details, stage_name)

    def _get_requests_id(self, perf_details: dict) -> list:
        """
        Get all request IDs for comprehensive analysis.

        Args:
            perf_details (dict): Performance details dictionary

        Returns:
            list: List of all request IDs
        """
        return list(range(len(perf_details["id"])))

    def _process_result(self, full_result: dict, stage_name: str):
        """
        Process performance results for a specific stage.

        Args:
            full_result (dict): Complete performance results
            stage_name (str): Name of the stage to process
        """
        id_list = self.stage_dict.get(stage_name)
        result = {}
        for k, v in full_result.items():
            if v is not None:
                result[k] = [v[i] for i in id_list]
        self.data_count[stage_name] = len(full_result["success"])
        self.decode_latencies[stage_name] = result["itl"]
        self.success_count[stage_name] = sum(full_result["success"])
        self.infer_time[stage_name] = max(result["end_time"]) - min(
            result["start_time"]
        )
        # Compute the average decode latency per request
        self.logger.info("Converting performance results for stage...")
        self.result[stage_name] = self.convert_result(result)
        self.logger.info("Performance results conversion completed!")
