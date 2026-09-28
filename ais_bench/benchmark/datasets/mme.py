import base64
import os
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from datasets import Dataset

from ais_bench.benchmark.datasets.utils.datasets import (get_content_str,
                                                         get_data_path)
from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger

from .base import BaseDataset

PERCEPTION_TASKS = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
COGNITION_TASKS = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
MME_TASKS = PERCEPTION_TASKS + COGNITION_TASKS
MME_REQUIRED_COLUMNS = {
    "question_id",
    "image",
    "question",
    "answer",
    "category",
}

logger = AISLogger()


def _resolve_parquet_files(path):
    resolved_path = Path(get_data_path(path, local_mode=True))
    if resolved_path.is_file():
        if resolved_path.suffix.lower() != ".parquet":
            raise ValueError(f"MME data file must be parquet: {resolved_path}")
        return [resolved_path]

    if not resolved_path.is_dir():
        raise FileNotFoundError(f"MME data path does not exist: {resolved_path}")

    parquet_files = sorted(resolved_path.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {resolved_path}")
    return parquet_files


def _mime_type_from_name(image_name):
    suffix = Path(image_name).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


def _encode_image(image_data, row_index):
    if not isinstance(image_data, dict):
        raise ValueError(
            f"Invalid MME image at row {row_index}: expected a dict, "
            f"got {type(image_data).__name__}"
        )

    image_bytes = image_data.get("bytes")
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    elif isinstance(image_bytes, bytearray):
        image_bytes = bytes(image_bytes)
    if not isinstance(image_bytes, bytes):
        raise ValueError(f"Invalid MME image bytes at row {row_index}")

    image_name = str(image_data.get("path") or f"{row_index}.jpg")
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    mime_type = _mime_type_from_name(image_name)
    image_url = f"data:{mime_type};base64,{encoded_image}"
    return encoded_image, image_url, os.path.basename(image_name)


@LOAD_DATASET.register_module()
class MMEDataset(BaseDataset):

    @staticmethod
    def load(path):
        """Load all local MME parquet shards with inline base64 images."""
        parquet_files = _resolve_parquet_files(path)
        frames = [pd.read_parquet(parquet_file) for parquet_file in parquet_files]
        data = pd.concat(frames, ignore_index=True)

        missing_columns = MME_REQUIRED_COLUMNS.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"MME parquet is missing required columns: {missing}")

        records = []
        for row_index, row in data.iterrows():
            question_id = str(row["question_id"])
            question = str(row["question"])
            answer = str(row["answer"])
            category = str(row["category"])

            if category not in MME_TASKS:
                raise ValueError(
                    f"Invalid MME category at row {row_index}: {category}"
                )
            if answer.lower() not in ("yes", "no"):
                raise ValueError(
                    f"Invalid MME answer at row {row_index}: {answer}"
                )

            image, image_url, image_name = _encode_image(
                row["image"], row_index
            )
            # Follow InfoVQA's image-first multimodal message construction.
            # Unlike InfoVQA, MME must use the parquet question verbatim.
            messages = [
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": question},
            ]
            content = get_content_str(messages)
            reference = {
                "question_id": question_id,
                "image_name": image_name,
                "question": question,
                "answer": answer,
                "category": category,
            }
            records.append(
                {
                    "content": content,
                    "question_id": question_id,
                    "image": image,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "image_name": image_name,
                    "reference": reference,
                }
            )

        logger.info(
            f"Loaded {len(records)} samples from {len(parquet_files)} "
            "MME parquet shard(s)"
        )
        return Dataset.from_list(records)


class MMEEvaluator(BaseEvaluator):

    def __init__(self, results_subdir="mme_results", write_results=True):
        super().__init__()
        self.results_subdir = results_subdir
        self.write_results = write_results

    @staticmethod
    def parse_pred_answer(prediction):
        """Parse a prediction exactly like the official calculation.py."""
        prediction = str(prediction).lower()
        if prediction in ("yes", "no"):
            return prediction

        prefix = prediction[:4]
        if "yes" in prefix:
            return "yes"
        if "no" in prefix:
            return "no"
        return "other"

    @staticmethod
    def _single_line(value):
        return (
            str(value)
            .replace("\r", " ")
            .replace("\n", " ")
            .replace("\t", " ")
        )

    @staticmethod
    def _pair_records(records, category):
        pairs = OrderedDict()
        for record in records:
            pairs.setdefault(record["question_id"], []).append(record)

        for question_id, pair in pairs.items():
            if len(pair) != 2:
                raise ValueError(
                    f"MME category {category!r}, question_id "
                    f"{question_id!r} has {len(pair)} questions; expected 2"
                )
        return pairs

    def _write_result_files(self, category_pairs):
        if not self.write_results:
            return
        if not hasattr(self, "_out_dir"):
            self.logger.warning(
                "MME result txt files were not written because evaluator "
                "output directory is unavailable"
            )
            return

        results_dir = Path(self._out_dir) / self.results_subdir
        results_dir.mkdir(parents=True, exist_ok=True)
        for category, pairs in category_pairs.items():
            result_path = results_dir / f"{category}.txt"
            with result_path.open("w", encoding="utf-8", newline="\n") as file:
                for pair in pairs.values():
                    for record in pair:
                        fields = (
                            record["image_name"],
                            record["question"],
                            record["answer"],
                            record["prediction"],
                        )
                        file.write(
                            "\t".join(self._single_line(field) for field in fields)
                            + "\n"
                        )
        self.logger.info(f"MME result txt files saved to {results_dir}")

    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) have different length"
            )
        if not predictions:
            return {
                "ACC": 0.0,
                "ACC+": 0.0,
                "Perception": 0.0,
                "Cognition": 0.0,
                "MME Score": 0.0,
                "details": [],
            }

        category_records = OrderedDict()
        details = []
        for prediction, reference in zip(predictions, references):
            if not isinstance(reference, dict):
                raise ValueError("MME references must be dictionaries")

            category = str(reference.get("category", ""))
            if category not in MME_TASKS:
                raise ValueError(f"Invalid MME category: {category}")

            answer = str(reference.get("answer", ""))
            ground_truth = answer.lower()
            if ground_truth not in ("yes", "no"):
                raise ValueError(f"Invalid MME ground-truth answer: {answer}")

            prediction = self._single_line(prediction)
            parsed_prediction = self.parse_pred_answer(prediction)
            correct = parsed_prediction == ground_truth
            record = {
                "question_id": str(reference.get("question_id", "")),
                "image_name": str(reference.get("image_name", "")),
                "question": str(reference.get("question", "")),
                "answer": answer,
                "category": category,
                "prediction": prediction,
                "parsed_prediction": parsed_prediction,
                "correct": correct,
            }
            category_records.setdefault(category, []).append(record)
            details.append(
                {
                    "pred": prediction,
                    "answer": answer,
                    "parsed_prediction": parsed_prediction,
                    "question_id": record["question_id"],
                    "category": category,
                    "correct": correct,
                }
            )

        category_pairs = OrderedDict(
            (
                category,
                self._pair_records(records, category),
            )
            for category, records in category_records.items()
        )
        self._write_result_files(category_pairs)

        result = {}
        total_correct = 0
        total_questions = 0
        total_pair_correct = 0
        total_pairs = 0
        task_scores = {}
        for category, pairs in category_pairs.items():
            records = category_records[category]
            correct = sum(record["correct"] for record in records)
            pair_correct = sum(
                all(record["correct"] for record in pair)
                for pair in pairs.values()
            )
            acc = 100.0 * correct / len(records)
            acc_plus = 100.0 * pair_correct / len(pairs)
            task_score = acc + acc_plus

            result[f"{category}/ACC"] = acc
            result[f"{category}/ACC+"] = acc_plus
            result[f"{category}/score"] = task_score
            task_scores[category] = task_score
            total_correct += correct
            total_questions += len(records)
            total_pair_correct += pair_correct
            total_pairs += len(pairs)

        result["ACC"] = 100.0 * total_correct / total_questions
        result["ACC+"] = 100.0 * total_pair_correct / total_pairs
        result["Perception"] = sum(
            task_scores.get(category, 0.0) for category in PERCEPTION_TASKS
        )
        result["Cognition"] = sum(
            task_scores.get(category, 0.0) for category in COGNITION_TASKS
        )
        result["MME Score"] = result["Perception"] + result["Cognition"]
        result["details"] = details
        return result
