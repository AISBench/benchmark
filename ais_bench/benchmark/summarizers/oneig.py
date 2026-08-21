"""OneIG summarizer for aggregating results across 5 sub-tasks.

Official reference: OneIG-Benchmark/fine_grained_analysis.py
"""
from typing import Dict, List

from ais_bench.benchmark.summarizers.default import DefaultSummarizer


class OneIGSummarizer(DefaultSummarizer):
    """OneIG summarizer that aggregates results from 5 sub-tasks.

    Computes:
    - oneig_total: average of all 5 sub-tasks (computed by summary_groups)
    - Diversity fine-grained: oneig_diversity_anime, oneig_diversity_human, etc.
    """

    def _calculate_group_metrics(
        self,
        raw_results: Dict,
        parsed_results: Dict,
        dataset_metrics: Dict,
        dataset_eval_mode: Dict,
    ):
        # First, run standard summary_groups aggregation
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            super()._calculate_group_metrics(
                raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # Then, add OneIG-specific aggregation (only diversity fine-grained)
        for model_abbr in self.model_abbrs:
            model_raw_results = raw_results.get(model_abbr, {})

            # Diversity: by class_item (anime, human, object, text, reasoning)
            self._aggregate_diversity_finegrain(
                model_abbr, raw_results, parsed_results,
                dataset_metrics, dataset_eval_mode, model_raw_results)

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode

    @staticmethod
    def _register_metric(
        metric_name: str,
        value: float,
        model_abbr: str,
        raw_results: Dict,
        parsed_results: Dict,
        dataset_metrics: Dict,
        dataset_eval_mode: Dict,
    ):
        """Register a computed metric into result dicts."""
        raw_results.setdefault(model_abbr, {}).setdefault(metric_name, {})['accuracy'] = value
        parsed_results.setdefault(model_abbr, {}).setdefault(metric_name, {})['accuracy'] = value
        if metric_name not in dataset_metrics:
            dataset_metrics[metric_name] = ['accuracy']
        dataset_eval_mode[metric_name] = 'gen'

    def _aggregate_diversity_finegrain(
        self,
        model_abbr: str,
        raw_results: Dict,
        parsed_results: Dict,
        dataset_metrics: Dict,
        dataset_eval_mode: Dict,
        model_results: Dict,
    ):
        """Aggregate diversity scores by class_item.

        Reference: diversity_score.py L61-86
        """
        abbr = 'oneig_diversity'
        details = model_results.get(abbr, {}).get('details')
        class_scores_raw = model_results.get(abbr, {}).get('class_scores')

        if class_scores_raw and isinstance(class_scores_raw, dict):
            # Use pre-computed class_scores from evaluator
            for class_item, scores in class_scores_raw.items():
                valid = [s for s in scores if s is not None]
                if valid:
                    avg = sum(valid) / len(valid)
                    # 命名风格统一：oneig_diversity_xxx
                    metric_name = f'oneig_diversity_{class_item}'
                    self._register_metric(
                        metric_name, avg * 100, model_abbr,
                        raw_results, parsed_results, dataset_metrics, dataset_eval_mode)
        elif details and isinstance(details, list):
            # Fallback: group from details
            class_scores: Dict[str, List[float]] = {}
            for item in details:
                class_item = item.get('class_item', '')
                score = item.get('score')
                if score is not None and class_item:
                    class_scores.setdefault(class_item, []).append(score)

            for class_item, scores in class_scores.items():
                avg = sum(scores) / len(scores)
                metric_name = f'oneig_diversity_{class_item}'
                self._register_metric(
                    metric_name, avg * 100, model_abbr,
                    raw_results, parsed_results, dataset_metrics, dataset_eval_mode)
