from tqdm import tqdm

from ais_bench.benchmark.calculators.base_perf_metric_calculator import (
    BasePerfMetricCalculator,
)
from ais_bench.benchmark.registry import PERF_METRIC_CALCULATORS
from ais_bench.benchmark.calculators.base_perf_metric_calculator import (
    DEFAULT_STATS,
)
from ais_bench.benchmark.utils.logging.error_codes import CALC_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError

WAVE_OFFSET = 0.02
INTERVAL_OFFSET = 0.001

@PERF_METRIC_CALCULATORS.register_module()
class StablePerfMetricCalculator(BasePerfMetricCalculator):
    """
    Performance metric calculator for stable stage analysis.

    This calculator focuses on analyzing the stable phase of benchmark execution,
    where the system operates at maximum concurrency with minimal fluctuations.
    It identifies and analyzes the stable period to provide more accurate
    performance metrics.

    Args:
        stats_list (list, optional): List of statistics to calculate
    """

    def _init_datas(self, perf_details: dict, max_concurrency: int):
        """
        Initialize data structures for stable stage analysis.

        Args:
            perf_details (dict): Performance details dictionary
            max_concurrency (int): Maximum concurrency value

        Raises:
            ValueError: If all requests failed
        """
        self.max_concurrency = max_concurrency
        self.stage_section = [0, 0]
        if sum(perf_details["success"]) == 0:
            raise AISBenchDataContentError(
                CALC_CODES.ALL_REQUEST_DATAS_INVALID,
                "All requests failed, cannot calculate performance results. Please check the error logs from responses!",
            )
        self.stage_dict = {"stable": self._get_requests_id(perf_details)}

        self.result = {}
        self.data_count = {}
        self.decode_latencies = {}
        self.success_count = {}
        self.empty_count = {}
        self.infer_time = {}
        self.metrics = {}
        self.common_metrics = {}

        for stage_name, _ in self.stage_dict.items():
            self._process_result(perf_details, stage_name)

    def _get_requests_id(self, perf_details: dict) -> list:
        """
        Identify requests that belong to the stable stage.

        Args:
            perf_details (dict): Performance details dictionary

        Returns:
            list: List of request IDs in the stable stage

        Raises:
            RuntimeError: If no stable stage can be identified
        """
        # Calculate the minimum start time as the baseline
        min_start_time = min(perf_details["start_time"])
        time_point_concurrency = [0] * 2 * len(perf_details["id"])
        request_time_sections = []
        for id in range(len(perf_details["id"])):
            request_time_sections.append(
                {
                    "id": id,
                    "attr": "start",
                    "time": perf_details["start_time"][id],
                }
            )
            request_time_sections.append(
                {
                    "id": id,
                    "attr": "end",
                    "time": perf_details["end_time"][id],
                }
            )
        sorted_time_sections = sorted(request_time_sections, key=lambda x: x["time"])
        id_lists = []
        self.logger.info("Starting stable stage calculation...")
        requested = 0
        progress_bar = tqdm(
            total=len(sorted_time_sections),
            desc="Calculating stable stage",
            unit=" req",
        )
        for i, section in enumerate(sorted_time_sections):
            if section["attr"] == "start":
                time_point_concurrency[i] = time_point_concurrency[i - 1] + 1
                requested += 1
            else:
                time_point_concurrency[i] = time_point_concurrency[i - 1] - 1
            if (
                section["attr"] == "start"
                and time_point_concurrency[i] == self.max_concurrency
            ):
                id_lists.append(section["id"])
                if len(id_lists) == 2:
                    self.stage_section[0] = section["time"]  # total start time
            elif (
                section["attr"] == "start"
                and time_point_concurrency[i]
                >= int(self.max_concurrency * (1 - WAVE_OFFSET))
                and len(id_lists) > 2
            ):
                id_lists.append(section["id"])
            elif requested == len(perf_details["id"]) and section["attr"] == "end":
                self.stage_section[1] = section["time"]
                progress_bar.update(len(sorted_time_sections) - progress_bar.n)
                break
            elif (
                len(id_lists) > 1
                and section["attr"] == "end"
                and time_point_concurrency[i]
                < int(self.max_concurrency * (1 - WAVE_OFFSET))
            ):
                # Check if there's a start event within INTERVAL_OFFSET that recovers concurrency
                # Skip consecutive end events and look ahead for start events
                should_exit_stable = True
                current_time = section["time"]
                current_concurrency = time_point_concurrency[i]

                # Look ahead to find the next start event within INTERVAL_OFFSET
                for j in range(i + 1, len(sorted_time_sections)):
                    next_section = sorted_time_sections[j]
                    time_interval = next_section["time"] - current_time

                    # If time interval exceeds INTERVAL_OFFSET, exit stable stage
                    if time_interval > INTERVAL_OFFSET:
                        break

                    # Update concurrency based on the event type
                    if next_section["attr"] == "end":
                        current_concurrency -= 1
                    else:  # start event
                        current_concurrency += 1
                        # If concurrency recovers to threshold, don't exit stable stage
                        if current_concurrency >= int(
                            self.max_concurrency * (1 - WAVE_OFFSET)
                        ):
                            should_exit_stable = False
                            break

                if should_exit_stable:
                    self.stage_section[1] = section["time"]
                    progress_bar.update(len(sorted_time_sections) - progress_bar.n)
                    break
            progress_bar.update(1)
        progress_bar.close()
        if len(id_lists) > 0:
            id_lists.pop(0)  # ignore first request that reached max concurrency
        if len(id_lists) == 0:
            raise AISBenchDataContentError(
                CALC_CODES.CAN_NOT_FIND_STABLE_STAGE,
                "Can not find a stable stage from performance results! Please check the conccurency plot.",
            )
        # Convert to relative time based on minimum start time
        relative_start_time = self.stage_section[0] - min_start_time
        relative_end_time = self.stage_section[1] - min_start_time
        self.logger.info(
            f"Stable stage calculation completed. "
            f"Start time: {relative_start_time:.6f}, "
            f"End time: {relative_end_time:.6f}, "
            f"Stable Stage Duration: {relative_end_time - relative_start_time:.6f}"
        )
        return id_lists

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
        self.data_count[stage_name] = len(result["success"])
        self.decode_latencies[stage_name] = result["itl"]
        self.success_count[stage_name] = sum(result["success"])
        self.infer_time[stage_name] = self.stage_section[1] - self.stage_section[0]
        self.logger.info("Converting performance results for stage...")
        self.result[stage_name] = self.convert_result(result)
        self.logger.info("Performance results conversion completed!")

    def _calculate_concurrency(self, stage_name: str) -> float:
        """
        Calculate concurrency for stable stage with maximum concurrency limit.

        Overrides the base class method to ensure concurrency does not exceed
        the maximum concurrency value for stable stage analysis.

        Args:
            stage_name (str): Name of the stage

        Returns:
            float: Calculated concurrency value, capped at max_concurrency
        """
        return min(
            round(
                sum(self.result[stage_name]["E2EL"]) / self.infer_time[stage_name], 4
            ),
            self.max_concurrency,
        )
