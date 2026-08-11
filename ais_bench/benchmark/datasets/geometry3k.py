import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from mathruler.grader import extract_boxed_content, grade_answer

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import (
    get_content_str,
    get_data_path,
)
from ais_bench.benchmark.utils.logging import AISLogger

from .base import BaseDataset

logger = AISLogger()

# ── Prompt template ────────────────────────────────────────────────────
GEOMETRY3K_INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


# ── Scoring functions (wrapping mathruler.grader, same as verl) ──────
def _extract_boxed_content(pred_str: str) -> str:
    """Wrapper around mathruler.grader.extract_boxed_content with logging.

    RETURNS "None" (string) when no \\boxed{} is found, consistent with verl.
    """
    result = extract_boxed_content(pred_str)
    if result == "None" or result == "":
        logger.debug(f"[extract_boxed_content] no \\boxed{{}} content, returning \"None\"")
    else:
        logger.debug(f"[extract_boxed_content] extracted: {result!r}")
    return result if result not in ("None", "") else "None"


def _grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Wrapper around mathruler.grader.grade_answer with logging.

    Uses sympy-based mathematical equivalence checking, strict integer matching,
    fraction comparison, and tuple/interval handling.
    """
    logger.debug(
        f"[grade_answer]\n"
        f"  given (raw)      : {given_answer!r}\n"
        f"  ground_truth (raw): {ground_truth!r}"
    )
    result = grade_answer(given_answer, ground_truth)
    logger.debug(f"[grade_answer] result={result}")
    return result


# ── Format reward ───────────────────────────────────────────────────────
def format_reward(predict_str: str) -> float:
    """Check whether the output has <think>...</think> and \\boxed{...}."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}", re.DOTALL)
    result = 1.0 if re.fullmatch(pattern, predict_str) else 0.0
    boxed_marker = '\\boxed{'
    logger.debug(
        f"[format_reward]\n"
        f"  has_think_tags={'<think>' in predict_str and '</think>' in predict_str}\n"
        f"  has_boxed={boxed_marker in predict_str}\n"
        f"  format_score={result}"
    )
    return result


# ── Image helpers ───────────────────────────────────────────────────────
def _save_image(image_obj, image_dir, index):
    """Save an image object to disk and return the file path.

    Handles:
        - PIL Image objects (datasets auto-decodes parquet image bytes)
        - dicts with 'bytes' key (raw PNG/JPEG bytes, fallback)
        - strings (already file paths)
    """
    from PIL import Image as PILImage

    os.makedirs(image_dir, exist_ok=True)
    logger.debug(f"[_save_image] index={index}, image_obj type={type(image_obj)}")

    if isinstance(image_obj, PILImage.Image):
        img_path = os.path.join(image_dir, f"{index}.png")
        logger.debug(
            f"[_save_image]\n"
            f"  type=PIL Image  size={image_obj.size}  mode={image_obj.mode}\n"
            f"  saved -> {img_path}"
        )
        image_obj.convert("RGB").save(img_path)
        return img_path

    elif isinstance(image_obj, dict) and "bytes" in image_obj:
        from io import BytesIO

        img_path = os.path.join(image_dir, f"{index}.png")
        img_bytes = image_obj["bytes"]
        logger.debug(
            f"[_save_image]\n"
            f"  type=dict with 'bytes'  bytes_len={len(img_bytes)}  path={image_obj.get('path', 'N/A')}\n"
            f"  saved -> {img_path}"
        )
        PILImage.open(BytesIO(img_bytes)).convert("RGB").save(img_path)
        return img_path

    elif isinstance(image_obj, str):
        logger.debug(f"[_save_image] already a path string: {image_obj}")
        return image_obj

    logger.warning(f"[_save_image] unknown image type={type(image_obj)}, returning ''")
    return ""


# ── Resolve dataset path ────────────────────────────────────────────────
def _resolve_parquet_path(path, split):
    """Resolve the parquet file path for a given split.

    Resolution order:
        1. If ``path`` is an absolute path to a file → use directly.
        2. If ``path`` is an absolute path to a directory → look for ``{split}-*.parquet`` inside.
        3. If ``path`` is relative → try ``get_data_path``, then fall back to the
           source-relative ``../../datasets/geometry3k`` directory.
    """
    # Absolute file path
    if path and os.path.isabs(path) and os.path.isfile(path):
        logger.debug(f"[_resolve_parquet_path] absolute file path: {path}")
        return path

    # Absolute directory path
    if path and os.path.isabs(path) and os.path.isdir(path):
        data_dir = Path(path)
    else:
        # Try get_data_path first
        resolved = None
        if path:
            try:
                resolved = get_data_path(path, local_mode=True)
            except Exception:
                logger.debug(f"[_resolve_parquet_path] get_data_path failed for {path!r}")

        if resolved and os.path.exists(resolved):
            if os.path.isfile(resolved):
                logger.debug(f"[_resolve_parquet_path] resolved via get_data_path (file): {resolved}")
                return resolved
            data_dir = Path(resolved)
        else:
            # Fallback: resolve relative to this source file
            source_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            data_dir = source_dir / ".." / ".." / "datasets" / "geometry3k"
            data_dir = data_dir.resolve()

    logger.debug(f"[_resolve_parquet_path] data_dir: {data_dir}")

    # Look for parquet files in data_dir/data/
    data_subdir = data_dir / "data"
    if data_subdir.is_dir():
        parquet_files = sorted(data_subdir.glob(f"{split}-*.parquet"))
        if not parquet_files:
            # Try any parquet files
            parquet_files = sorted(data_subdir.glob("*.parquet"))
        logger.debug(f"[_resolve_parquet_path] parquet files in data/: {[p.name for p in parquet_files]}")
    else:
        # Look directly in data_dir
        parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not parquet_files:
            parquet_files = sorted(data_dir.glob("*.parquet"))
        logger.debug(f"[_resolve_parquet_path] parquet files in data_dir: {[p.name for p in parquet_files]}")

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {data_dir}. "
            f"Expected pattern: {split}-*.parquet"
        )

    chosen = str(parquet_files[0])
    logger.debug(f"[_resolve_parquet_path] chosen file: {chosen}")
    return chosen


# ── Dataset ─────────────────────────────────────────────────────────────
@LOAD_DATASET.register_module()
class Geometry3KDataset(BaseDataset):

    @staticmethod
    def load(path=None, split="test", instruction=None):
        """Load the geometry3k dataset from local parquet files.

        Args:
            path: Path to the dataset directory or parquet file.
                  Defaults to the bundled ``datasets/geometry3k/`` directory.
            split: Which split to load (``'test'`` for 601 examples).
            instruction: Optional override for the instruction suffix.

        Returns:
            A HuggingFace ``Dataset`` with fields:
            ``content``, ``question``, ``image``, ``answer``, ``index``.
        """
        logger.debug(
            f"[Geometry3KDataset.load] ===== START =====\n"
            f"  input path={path!r}\n"
            f"  input split={split!r}\n"
            f"  input instruction={instruction!r}"
        )

        # Resolve the parquet file
        parquet_file = _resolve_parquet_path(path, split)
        logger.debug(f"[Geometry3KDataset.load] resolved parquet_file: {parquet_file}")

        # Load from local parquet
        dataset = load_dataset("parquet", data_files={split: parquet_file}, split=split)
        logger.debug(f"[Geometry3KDataset.load] dataset loaded: num_rows={len(dataset)}, columns={dataset.column_names}")

        # Build instruction string
        inst = instruction if instruction is not None else GEOMETRY3K_INSTRUCTION
        logger.debug(f"[Geometry3KDataset.load] instruction_text: {inst!r}")

        # Determine image output directory
        parquet_dir = Path(parquet_file).parent.parent  # geometry3k/
        image_root_path = str(parquet_dir / "geometry3k_images")
        os.makedirs(image_root_path, exist_ok=True)
        logger.debug(f"[Geometry3KDataset.load] image_root_path: {image_root_path}")

        records = []
        for i, example in enumerate(dataset):
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            images = example.get("images", [])

            if isinstance(problem, list) and len(problem) > 0:
                problem = problem[0]
            if isinstance(problem, str):
                problem = problem.replace("<image>", "", 1).lstrip()

            logger.debug(
                f"[Geometry3KDataset.load] --- record[{i}] ---\n"
                f"  problem: {problem!r}\n"
                f"  answer: {answer!r}\n"
                f"  images type: {type(images)}\n"
                f"  images len: {len(images) if hasattr(images, '__len__') else 'N/A'}"
            )

            # Save the first image
            image_path = ""
            if images is not None and hasattr(images, '__len__') and len(images) > 0:
                img_obj = images[0]
                if isinstance(img_obj, dict):
                    logger.debug(
                        f"[Geometry3KDataset.load] record[{i}] image[0]:\n"
                        f"  type=dict  keys={list(img_obj.keys())}\n"
                        + (f"  'bytes' len: {len(img_obj['bytes'])}\n" if "bytes" in img_obj else "")
                        + (f"  'path': {img_obj['path']}" if "path" in img_obj else "")
                    )
                elif hasattr(img_obj, 'size'):
                    # PIL Image (datasets auto-decodes parquet image bytes)
                    logger.debug(
                        f"[Geometry3KDataset.load] record[{i}] image[0]: PIL Image  size={img_obj.size}  mode={img_obj.mode}"
                    )
                else:
                    logger.debug(f"[Geometry3KDataset.load] record[{i}] image[0] type={type(img_obj)}")
                image_path = _save_image(img_obj, image_root_path, i)
            else:
                logger.debug(f"[Geometry3KDataset.load] record[{i}] no images found")

            # Construct the full prompt
            full_prompt = f"{problem} {inst}"
            logger.debug(
                f"[Geometry3KDataset.load] record[{i}] prompt:\n"
                f"  problem: {problem!r}\n"
                f"  full_prompt: {full_prompt!r}\n"
                f"  image_path: {image_path!r}"
            )

            # Build message list for get_content_str
            msgs = [
                {"type": "image_url", "image_url": image_path},
                {"type": "text", "text": full_prompt},
            ]
            content = get_content_str(msgs)
            logger.debug(f"[Geometry3KDataset.load] record[{i}] content: {content!r}")

            records.append(
                {
                    "content": content,
                    "question": full_prompt,
                    "image": image_path,
                    "answer": answer,
                    "index": i,
                }
            )

        logger.debug(f"[Geometry3KDataset.load] ===== END: {len(records)} records built =====")
        return Dataset.from_list(records)


# ── Evaluator ────────────────────────────────────────────────────────────
class Geometry3KEvaluator(BaseEvaluator):
    """Evaluator for geometry3k, matching verl's scoring logic.

    For each prediction:
    1. Extracts the content inside ``\\boxed{...}`` via mathruler.
    2. Compares with ground truth via mathruler ``grade_answer`` (sympy-based).
    3. Checks format compliance (``<think>...</think>`` + ``\\boxed{...}``).
    4. Computes weighted score: ``(1-w) * accuracy + w * format`` (same as verl).

    Args:
        format_weight: Weight of format reward in combined score.  Default 0.1.
    """

    def __init__(self, format_weight: float = 0.0):
        super().__init__()
        self.format_weight = format_weight
        logger.debug(f"[Geometry3KEvaluator] format_weight={format_weight}")

    def _compute_score(self, pred_str: str, ground_truth: str) -> dict:
        """Compute per-sample scores using verl's formula:
            combined = (1-w) * acc + w * fmt
        """
        # Clean special tokens
        for char in ["<|im_end|>", "<|endoftext|>"]:
            pred_str = pred_str.replace(char, "")

        # Extract boxed answer via mathruler
        extracted = _extract_boxed_content(pred_str)

        # Accuracy: compare extracted answer with ground truth
        acc = 1.0 if _grade_answer(extracted, ground_truth) else 0.0

        # Format: check for <think> + \boxed{}
        fmt = format_reward(pred_str)

        # Combined score: same formula as verl's compute_score
        combined = (1.0 - self.format_weight) * acc + self.format_weight * fmt

        return {
            "extracted_answer": extracted,
            "accuracy": acc,
            "format_score": fmt,
            "combined_score": combined,
        }

    def score(self, predictions, references):
        logger.debug(
            f"[Geometry3KEvaluator.score] ===== START =====\n"
            f"  num_predictions: {len(predictions)}\n"
            f"  num_references: {len(references)}\n"
            f"  format_weight: {self.format_weight}"
        )

        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        total = len(predictions)
        accuracy_correct = 0
        format_correct = 0
        combined_scores = []
        details = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            gt = ref if isinstance(ref, str) else ref.get("answer", str(ref))
            logger.debug(
                f"[Geometry3KEvaluator.score] --- sample {i}/{total} ---\n"
                f"  raw_pred (len={len(pred)}): {pred[:500]!r}\n"
                f"  ground_truth: {gt!r}"
            )

            sample_result = self._compute_score(pred, gt)
            acc = sample_result["accuracy"]
            fmt = sample_result["format_score"]
            combined = sample_result["combined_score"]
            extracted = sample_result["extracted_answer"]

            logger.debug(
                f"[Geometry3KEvaluator.score] sample[{i}]\n"
                f"  extracted_answer: {extracted!r}\n"
                f"  accuracy={acc}, format_score={fmt}, combined_score={combined}"
            )

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

        final_accuracy = 100.0 * accuracy_correct / total
        final_format = 100.0 * format_correct / total
        final_combined = 100.0 * sum(combined_scores) / total

        logger.debug(
            f"[Geometry3KEvaluator.score] ===== FINAL RESULTS =====\n"
            f"  total_samples: {total}\n"
            f"  accuracy_correct: {accuracy_correct}/{total}\n"
            f"  format_correct: {format_correct}/{total}\n"
            f"  final_accuracy: {final_accuracy:.2f}%\n"
            f"  final_format_score: {final_format:.2f}%\n"
            f"  final_combined_score: {final_combined:.2f}%"
        )

        result = {
            "accuracy": final_combined,
            "details": details,
        }
        return result
