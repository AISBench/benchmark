# flake8: noqa
# yapf: disable
import os.path as osp
from typing import Any, Dict, List

import mmengine
from mmengine import ConfigDict

from ais_bench.benchmark.summarizers.default import (
    DefaultSummarizer,
    model_abbr_from_cfg_used_in_summarizer,
)
from ais_bench.benchmark.utils.core.abbr import dataset_abbr_from_cfg, get_infer_output_path


class SWEBenchSummarizer(DefaultSummarizer):
    """Summarizer for SWE-bench evaluation results.
    """

    def _pick_up_results(self):
        # raw_results: {model_abbr: {dataset_abbr: result}}
        raw_results: Dict[str, Dict[str, Any]] = {}
        # parsed_results: {model_abbr: {dataset_abbr: {metric: score}}}
        parsed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        # dataset_metrics: {dataset_abbr: [metric]}
        dataset_metrics: Dict[str, List[str]] = {}
        # dataset_eval_mode: {dataset_abbr: eval_mode}
        dataset_eval_mode: Dict[str, str] = {}

        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            parsed_results.setdefault(model_abbr, {})
            raw_results.setdefault(model_abbr, {})

            for dataset in self.dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                base_dir = osp.join(
                    self.work_dir, "results", model_abbr, dataset_abbr
                )
                if not osp.isdir(base_dir):
                    continue

                aggregate_path = get_infer_output_path(
                    model,
                    dataset,
                    osp.join(self.work_dir, "results"),
                    file_extension="json",
                )
                aggregate_exists = osp.isfile(aggregate_path)
                if aggregate_exists:
                    try:
                        aggregate_data = mmengine.load(aggregate_path)
                        if isinstance(aggregate_data, dict):
                            total_instances = aggregate_data.get("total_instances")
                            resolved_instances = aggregate_data.get("resolved_instances")
                            if isinstance(total_instances, int) and isinstance(resolved_instances, int) and total_instances >= 0:
                                accuracy = (
                                    resolved_instances / total_instances * 100.0
                                    if total_instances > 0
                                    else 0.0
                                )
                                _rst = {
                                    "accuracy": accuracy,
                                    "correct_count": resolved_instances,
                                    "total_count": total_instances,
                                }
                                raw_results[model_abbr][dataset_abbr] = {
                                    "accuracy": accuracy,
                                    "correct_count": resolved_instances,
                                    "total_count": total_instances,
                                }
                                dataset_metrics[dataset_abbr] = ["accuracy"]
                                parsed_results[model_abbr][dataset_abbr] = _rst
                                continue
                    except Exception:
                        self.logger.warning(
                            "Failed to parse swebench aggregate result file: %s",
                            aggregate_path,
                        )
                continue

        for dataset in self.dataset_cfgs:
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            dataset_eval_mode[dataset_abbr] = "agent"

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
