import os
import re
from pathlib import Path

from datasets import Dataset, load_dataset

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


# ── Answer extraction ───────────────────────────────────────────────────
def last_boxed_only_string(string):
    """Find the last \\boxed{...} in the string."""
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
    """Strip '\\boxed{' and trailing '}'."""
    left = "\\boxed{"
    try:
        if not (s.startswith(left) and s.endswith("}")):
            return None
        return s[len(left) : -1]
    except (IndexError, TypeError):
        return None


def extract_boxed_content(pred_str):
    """Extract the content inside the last \\boxed{...}."""
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        logger.info(f"[extract_boxed_content] no \\boxed{{}} found, returning full pred_str")
        return pred_str
    answer = remove_boxed(boxed_str)
    if answer is None:
        logger.info(f"[extract_boxed_content] failed to remove boxed wrapper, returning full pred_str")
        return pred_str
    logger.info(f"[extract_boxed_content] extracted: {answer!r}")
    return answer


# ── Answer normalisation ────────────────────────────────────────────────
def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    ans = str(ans).strip()
    ans = re.sub(r"^\$+|\$+$", "", ans)
    ans = ans.replace("\\ ", " ")
    ans = re.sub(r"\s+", " ", ans)
    ans = ans.strip(". ,;:")
    ans = ans.replace("°", "")
    ans = ans.replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "")
    ans = re.sub(r"\\text\{(.*?)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{(.*?)\}", r"\1", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Compare a model answer against the ground truth.

    1. Exact match after normalisation.
    2. LaTeX-marker-stripped comparison.
    3. Numeric comparison with 1e-4 tolerance.
    4. Case-insensitive comparison.
    """
    given = normalize_answer(given_answer)
    gt = normalize_answer(ground_truth)

    logger.info(f"[grade_answer] given (raw)      : {given_answer!r}")
    logger.info(f"[grade_answer] ground_truth (raw): {ground_truth!r}")
    logger.info(f"[grade_answer] given (normalized) : {given!r}")
    logger.info(f"[grade_answer] ground_truth (norm): {gt!r}")

    # 1. Exact match after normalisation
    if given == gt:
        logger.info(f"[grade_answer] result=True (exact match after normalization)")
        return True

    # 2. Strip LaTeX markers and try again
    given_stripped = given.replace("\\", "").replace("{", "").replace("}", "")
    gt_stripped = gt.replace("\\", "").replace("{", "").replace("}", "")
    logger.info(f"[grade_answer] given_stripped: {given_stripped!r}")
    logger.info(f"[grade_answer] gt_stripped: {gt_stripped!r}")
    if given_stripped.strip() == gt_stripped.strip():
        logger.info(f"[grade_answer] result=True (stripped LaTeX match)")
        return True

    # 3. Numeric comparison with tolerance
    try:
        given_num = float(given_stripped)
        gt_num = float(gt_stripped)
        result = abs(given_num - gt_num) < 1e-4
        logger.info(f"[grade_answer] numeric: given_num={given_num}, gt_num={gt_num}, diff={abs(given_num - gt_num):.6f}, result={result}")
        return result
    except (ValueError, TypeError):
        logger.info(f"[grade_answer] numeric conversion failed, continuing...")

    # 4. Case-insensitive comparison
    if given.lower() == gt.lower():
        logger.info(f"[grade_answer] result=True (case-insensitive match)")
        return True

    logger.info(f"[grade_answer] result=False (all methods failed)")
    return False


# ── Format reward ───────────────────────────────────────────────────────
def format_reward(predict_str: str) -> float:
    """Check whether the output has <think>...</think> and \\boxed{...}."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    result = 1.0 if re.fullmatch(pattern, predict_str) else 0.0
    logger.info(f"[format_reward] has_think_tags={'<think>' in predict_str and '</think>' in predict_str}")
    logger.info(f"[format_reward] has_boxed={'\\boxed{' in predict_str}")
    logger.info(f"[format_reward] format_score={result}")
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
    logger.info(f"[_save_image] index={index}, image_obj type={type(image_obj)}")

    if isinstance(image_obj, PILImage.Image):
        img_path = os.path.join(image_dir, f"{index}.png")
        logger.info(f"[_save_image] PIL Image: size={image_obj.size}, mode={image_obj.mode}")
        image_obj.convert("RGB").save(img_path)
        logger.info(f"[_save_image] saved PIL Image -> {img_path}")
        return img_path

    elif isinstance(image_obj, dict) and "bytes" in image_obj:
        from io import BytesIO

        img_path = os.path.join(image_dir, f"{index}.png")
        img_bytes = image_obj["bytes"]
        logger.info(f"[_save_image] dict with 'bytes' key, bytes_len={len(img_bytes)}, path={image_obj.get('path', 'N/A')}")
        PILImage.open(BytesIO(img_bytes)).convert("RGB").save(img_path)
        logger.info(f"[_save_image] saved bytes-dict image -> {img_path}")
        return img_path

    elif isinstance(image_obj, str):
        logger.info(f"[_save_image] already a path string: {image_obj}")
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
        logger.info(f"[_resolve_parquet_path] absolute file path: {path}")
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
                logger.info(f"[_resolve_parquet_path] get_data_path failed for {path!r}")

        if resolved and os.path.exists(resolved):
            if os.path.isfile(resolved):
                logger.info(f"[_resolve_parquet_path] resolved via get_data_path (file): {resolved}")
                return resolved
            data_dir = Path(resolved)
        else:
            # Fallback: resolve relative to this source file
            source_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            data_dir = source_dir / ".." / ".." / "datasets" / "geometry3k"
            data_dir = data_dir.resolve()

    logger.info(f"[_resolve_parquet_path] data_dir: {data_dir}")

    # Look for parquet files in data_dir/data/
    data_subdir = data_dir / "data"
    if data_subdir.is_dir():
        parquet_files = sorted(data_subdir.glob(f"{split}-*.parquet"))
        if not parquet_files:
            # Try any parquet files
            parquet_files = sorted(data_subdir.glob("*.parquet"))
        logger.info(f"[_resolve_parquet_path] parquet files in data/: {[p.name for p in parquet_files]}")
    else:
        # Look directly in data_dir
        parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not parquet_files:
            parquet_files = sorted(data_dir.glob("*.parquet"))
        logger.info(f"[_resolve_parquet_path] parquet files in data_dir: {[p.name for p in parquet_files]}")

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {data_dir}. "
            f"Expected pattern: {split}-*.parquet"
        )

    chosen = str(parquet_files[0])
    logger.info(f"[_resolve_parquet_path] chosen file: {chosen}")
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
        logger.info(f"[Geometry3KDataset.load] ===== START =====")
        logger.info(f"[Geometry3KDataset.load] input path={path!r}")
        logger.info(f"[Geometry3KDataset.load] input split={split!r}")
        logger.info(f"[Geometry3KDataset.load] input instruction={instruction!r}")

        # Resolve the parquet file
        parquet_file = _resolve_parquet_path(path, split)
        logger.info(f"[Geometry3KDataset.load] resolved parquet_file: {parquet_file}")

        # Load from local parquet
        dataset = load_dataset("parquet", data_files={split: parquet_file}, split=split)
        logger.info(f"[Geometry3KDataset.load] dataset loaded: num_rows={len(dataset)}, columns={dataset.column_names}")

        # Build instruction string
        inst = instruction if instruction is not None else GEOMETRY3K_INSTRUCTION
        logger.info(f"[Geometry3KDataset.load] instruction_text: {inst!r}")

        # Determine image output directory
        parquet_dir = Path(parquet_file).parent.parent  # geometry3k/
        image_root_path = str(parquet_dir / "geometry3k_images")
        os.makedirs(image_root_path, exist_ok=True)
        logger.info(f"[Geometry3KDataset.load] image_root_path: {image_root_path}")

        records = []
        for i, example in enumerate(dataset):
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            images = example.get("images", [])

            logger.info(f"[Geometry3KDataset.load] --- record[{i}] ---")
            logger.info(f"[Geometry3KDataset.load] record[{i}] problem: {problem!r}")
            logger.info(f"[Geometry3KDataset.load] record[{i}] answer: {answer!r}")
            logger.info(f"[Geometry3KDataset.load] record[{i}] images type: {type(images)}")
            logger.info(f"[Geometry3KDataset.load] record[{i}] images len: {len(images) if hasattr(images, '__len__') else 'N/A'}")

            # Save the first image
            image_path = ""
            if images is not None and hasattr(images, '__len__') and len(images) > 0:
                img_obj = images[0]
                logger.info(f"[Geometry3KDataset.load] record[{i}] image[0] type: {type(img_obj)}")
                if isinstance(img_obj, dict):
                    logger.info(f"[Geometry3KDataset.load] record[{i}] image[0] keys: {list(img_obj.keys())}")
                    if "bytes" in img_obj:
                        logger.info(f"[Geometry3KDataset.load] record[{i}] image[0]['bytes'] len: {len(img_obj['bytes'])}")
                    if "path" in img_obj:
                        logger.info(f"[Geometry3KDataset.load] record[{i}] image[0]['path']: {img_obj['path']}")
                elif hasattr(img_obj, 'size'):
                    # PIL Image (datasets auto-decodes parquet image bytes)
                    logger.info(f"[Geometry3KDataset.load] record[{i}] image[0] PIL Image: size={img_obj.size}, mode={img_obj.mode}")
                image_path = _save_image(img_obj, image_root_path, i)
            else:
                logger.info(f"[Geometry3KDataset.load] record[{i}] no images found")

            logger.info(f"[Geometry3KDataset.load] record[{i}] final image_path: {image_path!r}")

            # Construct the full prompt
            full_prompt = f"{problem} {inst}"
            logger.info(f"[Geometry3KDataset.load] record[{i}] problem: {problem!r}")
            logger.info(f"[Geometry3KDataset.load] record[{i}] full_prompt: {full_prompt!r}")

            # Build message list for get_content_str
            msgs = [
                {"type": "image_url", "image_url": image_path},
                {"type": "text", "text": full_prompt},
            ]
            content = get_content_str(msgs)
            logger.info(f"[Geometry3KDataset.load] record[{i}] content: {content!r}")

            records.append(
                {
                    "content": content,
                    "question": full_prompt,
                    "image": image_path,
                    "answer": answer,
                    "index": i,
                }
            )

        logger.info(f"[Geometry3KDataset.load] ===== END: {len(records)} records built =====")
        return Dataset.from_list(records)


# ── Evaluator ────────────────────────────────────────────────────────────
class Geometry3KEvaluator(BaseEvaluator):
    """Evaluator for geometry3k.

    For each prediction:
    1. Extracts the content inside ``\\boxed{...}``.
    2. Compares with ground truth via ``grade_answer``.
    3. Checks format compliance (``<think>...</think>`` + ``\\boxed{...}``).
    4. Computes weighted score: ``0.9 * accuracy + 0.1 * format``.
    """

    def score(self, predictions, references):
        logger.info(f"[Geometry3KEvaluator.score] ===== START =====")
        logger.info(f"[Geometry3KEvaluator.score] num_predictions: {len(predictions)}")
        logger.info(f"[Geometry3KEvaluator.score] num_references: {len(references)}")

        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        total = len(predictions)
        accuracy_correct = 0
        format_correct = 0
        combined_scores = []
        details = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            logger.info(f"[Geometry3KEvaluator.score] --- sample {i}/{total} ---")
            logger.info(f"[Geometry3KEvaluator.score] raw_pred (len={len(pred)}): {pred[:500]!r}")

            # Clean special tokens
            for char in ["<|im_end|>", "<|endoftext|>"]:
                pred = pred.replace(char, "")
            logger.info(f"[Geometry3KEvaluator.score] cleaned_pred (len={len(pred)}): {pred[:500]!r}")

            gt = ref if isinstance(ref, str) else ref.get("answer", str(ref))
            logger.info(f"[Geometry3KEvaluator.score] ground_truth: {gt!r}")

            # Extract boxed answer and grade
            extracted = extract_boxed_content(pred)
            logger.info(f"[Geometry3KEvaluator.score] extracted_answer: {extracted!r}")

            acc = 1.0 if grade_answer(extracted, gt) else 0.0
            fmt = format_reward(pred)
            combined = 0.9 * acc + 0.1 * fmt

            logger.info(f"[Geometry3KEvaluator.score] sample[{i}] accuracy={acc}, format_score={fmt}, combined_score={combined}")

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

        logger.info(f"[Geometry3KEvaluator.score] ===== FINAL RESULTS =====")
        logger.info(f"[Geometry3KEvaluator.score] total_samples: {total}")
        logger.info(f"[Geometry3KEvaluator.score] accuracy_correct: {accuracy_correct}/{total}")
        logger.info(f"[Geometry3KEvaluator.score] format_correct: {format_correct}/{total}")
        logger.info(f"[Geometry3KEvaluator.score] final_accuracy: {final_accuracy:.2f}%")
        logger.info(f"[Geometry3KEvaluator.score] final_format_score: {final_format:.2f}%")
        logger.info(f"[Geometry3KEvaluator.score] final_combined_score: {final_combined:.2f}%")

        result = {
            "accuracy": final_accuracy,
            "format_score": final_format,
            "combined_score": final_combined,
            "details": details,
        }
        return result
