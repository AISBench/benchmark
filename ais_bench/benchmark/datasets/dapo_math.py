import os

import re
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, ICL_EVALUATORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError
from typing import Optional
from .base import BaseDataset
from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS

logger = AISLogger()

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.
    
    Args:
        string: Input string containing LaTeX code
        
    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
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

    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.
    
    Args:
        s: String with format "\\boxed{content}"
        
    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    if s[:len(left)] != left:
        raise AISBenchDataContentError(
            DSET_CODES.DATA_INVALID_STRUCTURE,
            f"box error: {s}"
        )
    if s[-1] != "}":
        raise AISBenchDataContentError(
            DSET_CODES.DATA_INVALID_STRUCTURE,
            f"box error: {s}"
        )
    return s[len(left):-1]

# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.
    
    Args:
        final_answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def extract_pred_by_minerva(solution_str: str, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> str:
    """Extract and normalize the answer from a solution string based on Minerva criteria.
    
    Args:
        solution_str: The solution string to check.
        answer_pattern: Regex pattern to extract the answer.
        
    Returns:
        The normalized prediction string.
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    return pred


def extract_pred_by_strict_box(pred: str) -> str:
    """Extract the answer from a prediction string using strict boxed answer criteria.
    
    Args:
        pred: The prediction string.
        
    Returns:
        The extracted prediction string from the last boxed expression, or None if not found.
    """
    # Extract the relevant part of the prediction
    pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else ""

    return extracted_pred

@TEXT_POSTPROCESSORS.register_module('dapo_math_postprocess')
def dapo_math_postprocess(solution_str: str) -> str:
    return extract_pred_by_minerva(solution_str)

@TEXT_POSTPROCESSORS.register_module('dapo_math_postprocess_v2')
def dapo_math_postprocess_v2(solution_str: str) -> Optional[str]:
    return extract_pred_by_strict_box(solution_str)

@LOAD_DATASET.register_module()
class DAPOMathDataset(BaseDataset):
    """DAPO-math-17k dataset for RL reasoning evaluation.
    
    Data format:
    {
        "data_source": "math_dapo",
        "prompt": [{"content": "...", "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": "...", "style": "..."},
        "extra_info": {"index": "..."}
    }
    """

    @staticmethod
    def load(path, file_name=None, **kwargs):
        """Load DAPO-math-17k dataset from Parquet file.
        
        Args:
            path (str): Path to the dataset directory or file.
            file_name (str, optional): Name of the Parquet file. 
                If None, will look for 'dapo-math-17k.parquet' or all .parquet files.
            **kwargs: Additional arguments.
            
        Returns:
            DatasetDict: Dataset with 'test' split.
        """
        path = get_data_path(path, local_mode=True)
        logger.debug(f"Loading DAPO-math-17k dataset from: {path}")
        
        # Determine file path
        if file_name:
            file_path = os.path.join(path, file_name) if not os.path.isabs(file_name) else file_name
        elif os.path.isfile(path) and path.endswith('.parquet'):
            # If path is already a Parquet file, use it directly
            file_path = path
        else:
            # Try default name first
            default_path = os.path.join(path, 'dapo-math-17k.parquet')
            if os.path.exists(default_path):
                file_path = default_path
            else:
                # Look for any .parquet file in the directory
                if not os.path.isdir(path):
                    raise AISBenchDataContentError(
                        DSET_CODES.FILE_NOT_FOUND,
                        f"Path is not a directory or Parquet file: {path}"
                    )
                parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
                if not parquet_files:
                    raise AISBenchDataContentError(
                        DSET_CODES.FILE_NOT_FOUND,
                        f"No Parquet file found in {path}"
                    )
                if len(parquet_files) > 1:
                    logger.debug(f"Multiple Parquet files found, using first one: {parquet_files[0]}")
                file_path = os.path.join(path, parquet_files[0])
        
        if not os.path.exists(file_path):
            raise AISBenchDataContentError(
                DSET_CODES.FILE_NOT_FOUND,
                f"Dataset file not found: {file_path}"
            )
        
        # Load data from Parquet file using datasets library
        try:
            raw_dataset = Dataset.from_parquet(file_path)
            logger.debug(f"Loaded Parquet file with {len(raw_dataset)} rows")
        except Exception as e:
            raise AISBenchDataContentError(
                DSET_CODES.FILE_READ_ERROR,
                f"Failed to read Parquet file {file_path}: {e}"
            )
        
        # Process and transform data
        dataset = []
        for idx, row in enumerate(raw_dataset):
            try:
                # Extract prompt content
                if 'prompt' not in row:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Missing 'prompt' field at row {idx}"
                    )
                
                prompt_list = row['prompt']
                if not isinstance(prompt_list, list) or len(prompt_list) == 0:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Invalid 'prompt' format at row {idx}: expected non-empty list"
                    )
                
                # Extract content from prompt (usually the first item with role='user')
                prompt_content = None
                for item in prompt_list:
                    if isinstance(item, dict) and item.get('role') == 'user':
                        prompt_content = item.get('content', '')
                        break
                
                if prompt_content is None:
                    # Fallback: use first item's content
                    if isinstance(prompt_list[0], dict):
                        prompt_content = prompt_list[0].get('content', '')
                    else:
                        prompt_content = str(prompt_list[0])
                
                # Extract ground truth from reward_model
                if 'reward_model' not in row:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Missing 'reward_model' field at row {idx}"
                    )
                
                reward_model = row['reward_model']
                if not isinstance(reward_model, dict) or 'ground_truth' not in reward_model:
                    raise AISBenchDataContentError(
                        DSET_CODES.DATA_INVALID_STRUCTURE,
                        f"Invalid 'reward_model' format at row {idx}: missing 'ground_truth'"
                    )
                
                ground_truth = str(reward_model['ground_truth'])
                
                # Build dataset entry
                entry = {
                    'prompt': prompt_content,
                    'answer': ground_truth,
                    'data_source': row.get('data_source', 'math_dapo'),
                    'ability': row.get('ability', 'MATH'),
                    'extra_info': row.get('extra_info', {}),
                }
                
                dataset.append(entry)
                
            except AISBenchDataContentError:
                raise
            except Exception as e:
                logger.debug(f"Unexpected error processing row {idx}: {e}")
                continue
        
        if not dataset:
            raise AISBenchDataContentError(
                DSET_CODES.DATA_INVALID_STRUCTURE,
                f"No valid data entries found in {file_path}"
            )
        
        logger.debug(f"DAPO-math-17k dataset loaded: {len(dataset)} samples")
        
        # Create DatasetDict with test split
        dataset_dict = DatasetDict({
            'test': Dataset.from_list(dataset),
            'train': Dataset.from_list(dataset)  # Use same data for train (for few-shot examples)
        })
        
        return dataset_dict

@ICL_EVALUATORS.register_module()
class DAPOMathEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            j = normalize_final_answer(j)
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result

@ICL_EVALUATORS.register_module()
class DAPOMathEvaluatorV2(BaseEvaluator):

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
            if i == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result