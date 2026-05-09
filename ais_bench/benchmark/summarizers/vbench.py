# flake8: noqa
# yapf: disable
"""VBench summarizer with official normalization and aggregation logic."""
import re
from typing import Dict

from ais_bench.benchmark.summarizers.default import DefaultSummarizer

# VBench official constants from scripts/constant.py
NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}
DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}
QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]
SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]
QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1

# Known dimension names (underscore form) for regex extraction from abbr
_DIM_PATTERN = re.compile(
    r'(subject_consistency|background_consistency|temporal_flickering|'
    r'motion_smoothness|dynamic_degree|aesthetic_quality|imaging_quality|'
    r'object_class|multiple_objects|human_action|color|spatial_relationship|'
    r'scene|appearance_style|temporal_style|overall_consistency)$'
)


def _abbr_to_const_key(abbr: str) -> str:
    """Extract dimension from abbr, e.g. vbench_custom_subject_consistency -> subject consistency."""
    m = _DIM_PATTERN.search(abbr)
    if m:
        return m.group(1).replace('_', ' ')
    if abbr.startswith('vbench_'):
        return abbr[7:].replace('_', ' ')
    return abbr.replace('_', ' ')


def _get_normalized_score(raw_score: float, const_key: str) -> float:
    """Normalize and apply DIM_WEIGHT per cal_final_score.py."""
    if const_key not in NORMALIZE_DIC or const_key not in DIM_WEIGHT:
        return 0.0
    raw = raw_score / 100.0 if raw_score > 1 else raw_score
    min_val = NORMALIZE_DIC[const_key]['Min']
    max_val = NORMALIZE_DIC[const_key]['Max']
    span = max_val - min_val
    if span <= 0:
        norm = 1.0 if raw >= max_val else 0.0
    else:
        norm = (raw - min_val) / span
    return norm * DIM_WEIGHT[const_key]


class VBenchSummarizer(DefaultSummarizer):
    """VBench summarizer using official cal_final_score.py logic.

    Computes Quality Score, Semantic Score, Total Score with:
    - Per-dimension normalization: (score - Min) / (Max - Min) * DIM_WEIGHT
    - Quality = weighted avg of QUALITY_LIST dims
    - Semantic = weighted avg of SEMANTIC_LIST dims
    - Total = (Quality * 4 + Semantic * 1) / 5
    """

    def _calculate_group_metrics(
        self,
        raw_results: Dict,
        parsed_results: Dict,
        dataset_metrics: Dict,
        dataset_eval_mode: Dict,
    ):
        """Compute vbench Quality, Semantic, Total using official formula."""
        for model_abbr in self.model_abbrs:
            model_results = parsed_results.get(model_abbr, {})
            vbench_scores = {}
            for abbr, data in model_results.items():
                if not abbr.startswith('vbench_'):
                    continue
                acc = data.get('accuracy')
                if acc is None or not isinstance(acc, (int, float)):
                    continue
                const_key = _abbr_to_const_key(abbr)
                vbench_scores[const_key] = acc

            if not vbench_scores:
                continue

            normalized = {
                k: _get_normalized_score(v, k)
                for k, v in vbench_scores.items()
            }

            quality_num = sum(normalized.get(k, 0) for k in QUALITY_LIST)
            quality_denom = sum(DIM_WEIGHT.get(k, 0) for k in QUALITY_LIST)
            quality_score = (
                quality_num / quality_denom if quality_denom else 0.0
            )

            semantic_num = sum(normalized.get(k, 0) for k in SEMANTIC_LIST)
            semantic_denom = sum(DIM_WEIGHT.get(k, 0) for k in SEMANTIC_LIST)
            semantic_score = (
                semantic_num / semantic_denom if semantic_denom else 0.0
            )

            total_score = (
                quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT
            ) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

            for name, score in [
                ('vbench_quality', quality_score * 100),
                ('vbench_semantic', semantic_score * 100),
                ('vbench_total', total_score * 100),
            ]:
                raw_results[model_abbr].setdefault(name, {})['accuracy'] = score
                parsed_results[model_abbr].setdefault(name, {})['accuracy'] = score
                if name not in dataset_metrics:
                    dataset_metrics[name] = ['accuracy']
                dataset_eval_mode[name] = 'gen'

        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
