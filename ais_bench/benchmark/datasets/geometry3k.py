import ast
import json
import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import (
    decode_base64_to_image_file,
    get_content_str,
    get_data_path,
)
from ais_bench.benchmark.utils.logging import AISLogger

from .base import BaseDataset

logger = AISLogger()

# ── Prompt template (same as veRL geo3k.py preprocessing) ──────────────
GEOMETRY3K_INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


# ── Answer extraction ───────────────────────────────────────────────────
def last_boxed_only_string(string):
    """Find the last \\boxed{...} occurrence in the string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    """Strip the leading '\\boxed{' and trailing '}' from a boxed string."""
    left = "\\boxed{"
    try:
        if not (s.startswith(left) and s.endswith("}")):
            return None
        return s[len(left) : -1]
    except (IndexError, TypeError):
        return None


def extract_boxed_content(pred_str):
    """Extract the content inside the last \\boxed{...} in prediction string.

    Replicates the behaviour of ``mathruler.grader.extract_boxed_content`` used
    in veRL's ``verl/utils/reward_score/geo3k.py``.
    """
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return pred_str  # if no \boxed{}, return whole prediction
    answer = remove_boxed(boxed_str)
    if answer is None:
        return pred_str
    return answer


# ── Math answer normalisation ────────────────────────────────────────────
def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison.

    This handles the most common differences between model outputs and
    ground-truth answers in Geometry3K: extra whitespace, LaTeX spacing
    variants, degree symbols, and common unit suffixes.
    """
    ans = str(ans).strip()
    # remove leading/trailing LaTeX math markers
    ans = re.sub(r"^\$+|\$+$", "", ans)
    # normalise LaTeX whitespace commands
    ans = ans.replace("\\ ", " ")
    # normalise multiple spaces
    ans = re.sub(r"\s+", " ", ans)
    # strip common spurious characters
    ans = ans.strip(". ,;:")
    # degree symbol normalisation
    ans = ans.replace("°", "")
    ans = ans.replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "")
    # LaTeX text wrapper removal (units often wrapped)
    ans = re.sub(r"\\text\{(.*?)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{(.*?)\}", r"\1", ans)
    # strip trailing/multiple whitespace again after replacements
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Compare a model answer against the ground truth.

    Replicates the behaviour of ``mathruler.grader.grade_answer`` used
    in veRL's ``verl/utils/reward_score/geo3k.py``.

    The comparison uses:
    1. Exact match after normalisation.
    2. Numeric comparison with tolerance for decimal values.
    3. LaTeX-aware string comparison.
    """
    given = normalize_answer(given_answer)
    gt = normalize_answer(ground_truth)

    # 1. Exact match after normalisation
    if given == gt:
        return True

    # 2. Strip common LaTeX markers and try again
    given_stripped = given.replace("\\", "").replace("{", "").replace("}", "")
    gt_stripped = gt.replace("\\", "").replace("{", "").replace("}", "")
    if given_stripped.strip() == gt_stripped.strip():
        return True

    # 3. Numeric comparison with tolerance
    try:
        given_num = float(given_stripped)
        gt_num = float(gt_stripped)
        return abs(given_num - gt_num) < 1e-4
    except (ValueError, TypeError):
        pass

    # 4. Case-insensitive comparison
    if given.lower() == gt.lower():
        return True

    return False


# ── Format reward (same as veRL) ─────────────────────────────────────────
def format_reward(predict_str: str) -> float:
    """Check whether the output follows the required format.

    Must contain both ``<think>...</think>`` and ``\\boxed{...}``.
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


# ── Resolve dataset path ─────────────────────────────────────────────────
def _resolve_dataset_path(path):
    if not path:
        return path, False
    if os.path.isabs(path):
        return path, True
    try:
        return get_data_path(path, local_mode=True), True
    except Exception:
        return path, False


# ── Image helpers ────────────────────────────────────────────────────────
def _save_image(image_obj, image_dir, index):
    """Save a PIL Image to disk and return the file path."""
    from PIL import Image as PILImage

    os.makedirs(image_dir, exist_ok=True)
    if isinstance(image_obj, PILImage.Image):
        img_path = os.path.join(image_dir, f"{index}.png")
        image_obj.convert("RGB").save(img_path)
        return img_path
    elif isinstance(image_obj, dict) and "bytes" in image_obj:
        from io import BytesIO

        img_path = os.path.join(image_dir, f"{index}.png")
        PILImage.open(BytesIO(image_obj["bytes"])).convert("RGB").save(img_path)
        return img_path
    elif isinstance(image_obj, str):
        return image_obj
    return ""


# ── Dataset ──────────────────────────────────────────────────────────────
@LOAD_DATASET.register_module()
class Geometry3KDataset(BaseDataset):

    @staticmethod
    def load(path="hiyouga/geometry3k", split="test", instruction=None):
        """Load the geometry3k dataset from HuggingFace.

        Args:
            path: HuggingFace dataset name or local path.
            split: Which split to load (``'test'`` gives 601 examples).
            instruction: Optional override for the instruction suffix.

        Returns:
            A HuggingFace ``Dataset`` object with fields:
            ``content``, ``question``, ``image``, ``answer``.
        """
        # Allow local parquet data as well as the remote HF name
        resolved_path, is_local = _resolve_dataset_path(path)
        if is_local:
            dataset = load_dataset(
                "parquet", data_files={split: resolved_path}, split=split
            )
        else:
            dataset = load_dataset(path, split=split)

        # Build the instruction string
        inst = instruction if instruction is not None else GEOMETRY3K_INSTRUCTION

        # Determine where to store extracted images
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..", "datasets")
        )
        image_root_path = os.path.join(base_dir, "geometry3k_images")
        os.makedirs(image_root_path, exist_ok=True)

        records = []
        for i, example in enumerate(dataset):
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            images = example.get("images", [])

            # Save the first image (geometry3k has exactly one image per example)
            image_path = ""
            if images:
                image_path = _save_image(images[0], image_root_path, i)

            # Construct the full prompt (exactly as veRL does)
            full_prompt = f"{problem} {inst}"

            # Build a message list so get_content_str renders it correctly
            msgs = [
                {"type": "image_url", "image_url": image_path},
                {"type": "text", "text": full_prompt},
            ]

            records.append(
                {
                    "content": get_content_str(msgs),
                    "question": full_prompt,
                    "image": image_path,
                    "answer": answer,
                    "index": i,
                }
            )

        return Dataset.from_list(records)


# ── Evaluator ────────────────────────────────────────────────────────────
class Geometry3KEvaluator(BaseEvaluator):
    """Evaluator for geometry3k that replicates veRL's scoring logic.

    For each prediction, it:
    1. Extracts the content inside ``\\boxed{...}``.
    2. Compares the extracted answer with ground truth via ``grade_answer``.
    3. Checks format compliance (``<think>...</think>`` + ``\\boxed{...}``).
    4. Computes weighted score: ``0.9 * accuracy + 0.1 * format``.
    """

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        total = len(predictions)
        accuracy_correct = 0
        format_correct = 0
        combined_scores = []
        details = []

        for pred, ref in zip(predictions, references):
            # Clean special tokens from prediction
            for char in ["<|im_end|>", "<|endoftext|>"]:
                pred = pred.replace(char, "")

            gt = ref if isinstance(ref, str) else ref.get("answer", str(ref))

            # Extract boxed answer and grade
            extracted = extract_boxed_content(pred)
            acc = 1.0 if grade_answer(extracted, gt) else 0.0
            fmt = format_reward(pred)
            combined = 0.9 * acc + 0.1 * fmt

            if acc == 1.0:
                accuracy_correct += 1
            if fmt == 1.0:
                format_correct += 1
            combined_scores.append(combined)

            details.append(
                {
                    "pred": pred,
                    "answer": gt,
                    "extracted_answer": extracted,
                    "accuracy": acc,
                    "format_score": fmt,
                    "combined_score": combined,
                }
            )

        result = {
            "accuracy": 100.0 * accuracy_correct / total,
            "format_score": 100.0 * format_correct / total,
            "combined_score": 100.0 * sum(combined_scores) / total,
            "details": details,
        }
        return result
