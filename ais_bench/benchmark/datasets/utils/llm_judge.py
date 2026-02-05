import re
import os

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.datasets.base import BaseJDGDataset
from ais_bench.benchmark.utils.file.file import load_jsonl
logger = AISLogger()


@TEXT_POSTPROCESSORS.register_module("get_a_or_b")
def get_a_or_b(pred: str) -> str:
    """从模型回复中提取A或B"""
    match = re.search(r'[AB]', pred[-1:])
    return match.group(0) if match else 'B'


class LLMJudgeDataset(BaseJDGDataset):
    def _load_from_predictions(self, prediction_path: str):
        """Load predictions from a directory and merge them with the dataset.

        Args:
            prediction_path (str): The path to the prediction file.

        Returns:
            Dataset: The merged dataset with predictions.
        """
        if os.path.exists(prediction_path):
            preds = load_jsonl(prediction_path)
        preds.sort(key=lambda x: x.get('id',0))
        return preds


@ICL_EVALUATORS.register_module()
class LLMJudgeCorrectEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if i == "A":
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result