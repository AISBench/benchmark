import os.path as osp
from typing import Any, Dict, List

import mmengine

from ais_bench.benchmark.summarizers.default import (
    DefaultSummarizer,
    model_abbr_from_cfg_used_in_summarizer,
)
from ais_bench.benchmark.utils.core.abbr import dataset_abbr_from_cfg


class SWEBenchProSummarizer(DefaultSummarizer):
    def _pick_up_results(self):
        raw_results: Dict[str, Dict[str, Any]] = {}
        parsed_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        dataset_metrics: Dict[str, List[str]] = {}
        dataset_eval_mode: Dict[str, str] = {}

        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            parsed_results.setdefault(model_abbr, {})
            raw_results.setdefault(model_abbr, {})

            for dataset in self.dataset_cfgs:
                dataset_abbr = dataset_abbr_from_cfg(dataset)

                # 读取 report 文件：model_abbr_dataset_abbr_report.json
                report_path = osp.join(
                    self.work_dir, "results", f"{model_abbr}_{dataset_abbr}_report.json"
                )
                self.logger.debug(f"[summary]report_path = {report_path}")

                report_exists = osp.isfile(report_path)
                if report_exists:
                    try:
                        report_data = mmengine.load(report_path)
                        if isinstance(report_data, dict):
                            total_instances = report_data.get("total_instances_num", 0)
                            resolved_instances = report_data.get("eval_resolved_instances_num", 0)
                            overall_accuracy = report_data.get("accuracy", 0.0)

                            _rst = {
                                "accuracy": round(overall_accuracy, 2),
                                "correct_count": resolved_instances,
                                "total_count": total_instances,
                            }
                            self.logger.debug(f"[summary]_rst = {_rst}")

                            raw_results[model_abbr][dataset_abbr] = {
                                "accuracy": round(overall_accuracy, 2),
                                "correct_count": resolved_instances,
                                "total_count": total_instances,
                            }
                            self.logger.debug(f"[summary]raw_results = {raw_results}")

                            dataset_metrics[dataset_abbr] = ["accuracy"]
                            parsed_results[model_abbr][dataset_abbr] = _rst
                            continue
                    except Exception:
                        self.logger.warning(
                            "Failed to parse swebench_pro report file: %s",
                            report_path,
                        )
                continue

        for dataset in self.dataset_cfgs:
            dataset_abbr = dataset_abbr_from_cfg(dataset)
            dataset_eval_mode[dataset_abbr] = "agent"

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode