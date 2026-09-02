import json
import math
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import (get_content_str,
                                                         get_data_path)
from ais_bench.benchmark.datasets.utils.llm_judge import (
    LLMJudgeCorrectEvaluator, LLMJudgeDataset)
from ais_bench.benchmark.registry import ICL_EVALUATORS, LOAD_DATASET
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()


def parse_hle_item(item: Dict) -> Dict:
    """Parse HLE dataset item and generate AISBench standard content.

    Args:
        item: Raw HLE dataset item containing 'question' and optionally 'image'.

    Returns:
        Dictionary with 'content' (formatted prompt) and 'answer' fields.
    """
    question_text = item.get("question", "")

    msgs = [{"type": "text", "text": question_text}]

    image_url = item.get("image", "")
    if image_url and isinstance(image_url, str):
        msgs.append({"type": "image_url", "image_url": image_url})

    return {"content": get_content_str(msgs), "answer": item.get("answer", "")}


@LOAD_DATASET.register_module()
class HLEDataset(BaseDataset):
    """HLE (Humanity's Last Exam) Dataset class for loading exam questions."""

    @staticmethod
    def load(path, **kwargs):
        """Load HLE dataset from parquet file.

        Args:
            path: Path to the parquet file containing HLE dataset.
            **kwargs: Additional arguments (unused).

        Returns:
            HuggingFace Dataset with 'content' and 'answer' fields.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
        """
        resolved_path = get_data_path(path)
        logger.debug(f"Loading HLE dataset from: {resolved_path}")

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"HLE parquet file not found: {resolved_path}")

        data = pd.read_parquet(resolved_path)
        logger.debug(f"Loaded {len(data)} samples from parquet file")

        dataset = []
        for i in range(len(data)):
            line = data.iloc[i]
            parsed_item = parse_hle_item(line.to_dict())
            dataset.append(parsed_item)

        logger.debug(f"Processed {len(dataset)} samples")
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class HLEJGDataset(LLMJudgeDataset):
    """HLE Judge Dataset class for LLM-based evaluation.

    Wrapper class that provides LLM Judge evaluation capabilities for HLE dataset.
    """

    def _get_dataset_class(self):
        """Return the base dataset class for LLM Judge evaluation."""
        return HLEDataset


def parse_predictions(predictions: list) -> List[Dict[str, Any]]:
    """Parse prediction strings into structured format.

    Extracts model answer, reasoning, correctness, and confidence from
    JSON-formatted prediction strings.

    Args:
        predictions: List of JSON string predictions from the judge model.

    Returns:
        List of dictionaries with 'model_answer', 'reasoning', 'correct',
        and 'confidence' fields. Invalid predictions are skipped with a log.
    """
    results = []
    for pred_str in predictions:
        logger.debug(f"\n original_pred_input: {pred_str}")
        cleaned = re.sub(r"[\n\t\r]+", " ", pred_str)
        cleaned = re.sub(r"\s+", " ", cleaned)
        logger.debug(f"\n cleaned_pred_input: {cleaned}")
        try:
            data = json.loads(cleaned)
            logger.debug(f"\n after_proc_data: {data}")
            results.append(
                {
                    "model_answer": data.get("extracted_final_answer"),
                    "reasoning": data.get("reasoning"),
                    "correct": data.get("correct"),
                    "confidence": data.get("confidence"),
                }
            )
        except json.JSONDecodeError:
            logger.error(
                UTILS_CODES.UNKNOWN_ERROR, f"wrong format prediction: {cleaned}"
            )
            continue
    return results


def calib_err(confidence, correct, p="2", beta=100):
    """Calculate Expected Calibration Error (ECE) for model predictions.

    Computes the calibration error by binning predictions based on confidence
    scores and comparing predicted confidence with actual accuracy per bin.

    Args:
        confidence: Array of confidence scores (0-1 range).
        correct: Array of boolean correctness values.
        p: Norm type for error calculation. Options: '1' (L1), '2' (L2),
           'infty'/'infinity'/'max' (L-inf).
        beta: Target bin size for grouping predictions.

    Returns:
        Calibration error value (float).
    """
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]

    num_bins = len(confidence) // beta

    if num_bins == 0:
        # Edge case: too few samples, return overall error
        avg_conf = np.nanmean(confidence)
        avg_correct = np.nanmean(correct)
        return np.abs(avg_conf - avg_correct)

    bins = [[i * beta, (i + 1) * beta] for i in range(num_bins)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p == "infty" or p == "infinity" or p == "max":
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def dump_metrics(judge_results, n):
    """Calculate evaluation metrics from judge results.

    Computes accuracy, calibration error, and confidence interval for
    the evaluated predictions.

    Args:
        judge_results: List of judge response dicts with 'correct' and 'confidence'.
        n: Total number of expected predictions.

    Returns:
        Dictionary containing:
            - accuracy: Percentage string with confidence interval
            - calibration_error: Calibration error value
            - sample_num: Total number of samples
    """
    if not judge_results:
        logger.error(UTILS_CODES.UNKNOWN_ERROR, "No available judge_results")
        return {
            "accuracy": "0%",
            "confidence_half_width": "+/- 0%",
            "calibration_error": 0,
            "sample_num": n,
        }

    correct = []
    confidence = []
    for judge in judge_results:
        correct.append("yes" in judge.get("correct", ""))
        confidence.append(judge.get("confidence", 0))

    correct = np.array(correct)
    confidence = np.array(confidence) / 100

    logger.debug(
        f"after_process_correct: {correct}, after_process_confidence: {confidence}"
    )

    # Handle case where prediction count differs from expected
    if len(correct) != n:
        logger.error(
            UTILS_CODES.UNKNOWN_ERROR,
            f"Available predictions: {len(correct)} | Total questions: {n}",
        )

    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
    calibration_error = 100 * round(calib_err(confidence, correct, p="2", beta=100), 2)

    return {
        "accuracy": f"{accuracy}% +/- {confidence_half_width}%",
        "calibration_error": calibration_error,
        "sample_num": n,
    }


@ICL_EVALUATORS.register_module()
class HLEJudgeEvaluator(LLMJudgeCorrectEvaluator):
    """HLE Judge evaluator for assessing model responses using LLM-based judgment.

    Evaluates model predictions by parsing judge model outputs and computing
    accuracy and calibration metrics.
    """

    def score(self, predictions: List[Dict], references: List[str]) -> Dict[str, Any]:
        """Score predictions against references using LLM judge.

        Args:
            predictions: List of prediction strings from the model.
            references: List of reference answers (unused, included for API compatibility).

        Returns:
            Dictionary with evaluation metrics (accuracy, calibration_error, sample_num)
            or error message if validation fails.
        """
        logger.debug(
            f"predictions len: {len(predictions)}, references len: {len(references)}"
        )

        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        parsed_judge_results = parse_predictions(predictions)

        return dump_metrics(parsed_judge_results, len(references))
