"""Google SimpleQA Verified: local data, configurable LLM judge, official metrics."""

import csv
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.utils.llm_judge import LLMJudgeDataset
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS, LOAD_DATASET

__all__ = [
    "SIMPLEQA_VERIFIED_JUDGE_PROMPT",
    "SimpleQAVerifiedDataset",
    "SimpleQAVerifiedEvaluator",
    "SimpleQAVerifiedJudgeDataset",
]

# Official starter notebook v9, linked by the Google dataset card.
SIMPLEQA_VERIFIED_JUDGE_PROMPT = (
    Path(__file__)
    .with_name("simpleqa_verified_prompt.txt")
    .read_text(encoding="utf-8")
    .replace("{target}", "{answer}")
    .replace("{predicted_answer}", "{model_answer}")
)


@LOAD_DATASET.register_module()
class SimpleQAVerifiedDataset(BaseDataset):
    @staticmethod
    def load(path, **kwargs):
        source = Path(get_data_path(path, local_mode=True))
        if source.is_dir():
            source = source / "simpleqa_verified.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle:
            if source.suffix.lower() == ".csv":
                rows = list(csv.DictReader(handle))
            elif source.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in handle if line.strip()]
            else:
                raise ValueError("Expected SimpleQA Verified CSV or JSONL")
        if not rows:
            raise ValueError("Empty SimpleQA Verified dataset")
        seen = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at row {index}")  # noqa: TRY004
            for field in ("problem", "answer"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Invalid {field} at row {index}")
            original_index = row.get("original_index")
            if (
                type(original_index) not in (str, int)
                or not str(original_index).strip()
            ):
                raise ValueError(f"Invalid original_index at row {index}")
            identity = str(original_index).strip()
            if identity in seen:
                raise ValueError(f"Duplicate original_index: {identity}")
            seen.add(identity)
            row["original_index"] = identity
        test = Dataset.from_list(rows)
        return DatasetDict(train=test.select([]), test=test)


@LOAD_DATASET.register_module()
class SimpleQAVerifiedJudgeDataset(LLMJudgeDataset):
    def _get_dataset_class(self):
        return SimpleQAVerifiedDataset

    def _modify_dataset_item(self, dataset_item, pred_item):
        super()._modify_dataset_item(dataset_item, pred_item)
        dataset_item["judge_prompt"] = SIMPLEQA_VERIFIED_JUDGE_PROMPT.format(
            question=dataset_item["problem"],
            answer=dataset_item["answer"],
            model_answer=dataset_item["model_answer"],
        )


@ICL_EVALUATORS.register_module()
class SimpleQAVerifiedEvaluator(BaseEvaluator):
    """Three-way grading; scores use [0, 1], matching the starter notebook."""

    @staticmethod
    def parse_grade(text):
        if not isinstance(text, str):
            return None
        return {
            "A": "A",
            "B": "B",
            "C": "C",
            "CORRECT": "A",
            "INCORRECT": "B",
            "NOT_ATTEMPTED": "C",
            "NOT ATTEMPTED": "C",
        }.get(text.strip().upper())

    def score(self, predictions, references):
        if not references or len(predictions) != len(references):
            return {"error": "Empty references or mismatched prediction count"}
        grades = [self.parse_grade(text) for text in predictions]
        invalid = grades.count(None)
        details = [
            {"judge_output": text, "model_pred_uuid": ref, "grade": grade}
            for text, ref, grade in zip(predictions, references, grades)
        ]
        if invalid:
            return {
                "error": f"{invalid} invalid judge outputs; metrics withheld",
                "invalid_judge_count": invalid,
                "details": details,
            }
        total = len(grades)
        correct = grades.count("A")
        incorrect = grades.count("B")
        attempted = correct + incorrect
        accuracy = correct / total
        accuracy_given_attempted = correct / attempted if attempted else 0.0
        f1 = 2 * correct / (total + attempted)
        return {
            "accuracy": accuracy,
            "accuracy_given_attempted": accuracy_given_attempted,
            "f1": f1,
            "correct": accuracy,
            "incorrect": incorrect / total,
            "not_attempted": grades.count("C") / total,
            "invalid_judge_count": 0,
            "details": details,
        }
