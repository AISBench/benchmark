import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, ICL_EVALUATORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, get_cache_dir
from ais_bench.benchmark.datasets.utils.llm_judge import LLMJudgeDataset
from ais_bench.benchmark.utils.logging.logger import AISLogger

from ..base import BaseDataset

logger = AISLogger()

# Prompt template for AA-LCR (matching evalscope format)
PROMPT_TEMPLATE = """\
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION"""

# Judge prompt template for LLM-based evaluation (adapted from evalscope)
JUDGE_PROMPT = """\
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT. \
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {answers}
CANDIDATE ANSWER TO ASSESS: {model_answer}

Reply only with CORRECT or INCORRECT."""

# Local zip and cache configuration
DEFAULT_CACHE_SUBDIR = 'aa_lcr'
DEFAULT_EXTRACTED_DIR_NAME = 'lcr'

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../../../'
)

# Local zip path (relative to this file)
_LOCAL_ZIP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../datasets/aa_lcr/AA-LCR_extracted-text.zip'
)


def _ensure_text_dir_downloaded() -> Path:
    """Ensure AA-LCR extracted texts are available locally.

    Expects the zip file to already exist at the local relative path:
        ais_bench/datasets/aa_lcr/AA-LCR_extracted-text.zip

    If the zip is missing, raises an error instructing the user to
    place the dataset file manually.
    """
    cache_root = Path(get_cache_dir(DEFAULT_CACHE_DIR)) / DEFAULT_CACHE_SUBDIR
    extracted_dir = cache_root / DEFAULT_EXTRACTED_DIR_NAME

    if extracted_dir.exists():
        logger.info(f'AA-LCR documents found: {extracted_dir}')
        return extracted_dir

    local_zip = Path(_LOCAL_ZIP_PATH).resolve()
    if not local_zip.exists():
        raise FileNotFoundError(
            f'AA-LCR dataset zip not found at: {local_zip}\n'
            'Please download AA-LCR_extracted-text.zip'
        )

    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f'Extracting {local_zip} to {cache_root}...')
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(cache_root)

        if not extracted_dir.exists():
            raise ValueError(
                f'Extraction succeeded but target directory not found: {extracted_dir}'
            )

        logger.info(f'AA-LCR documents ready at {extracted_dir}')
        return extracted_dir
    except Exception as e:
        raise ValueError(
            f'Failed to extract AA-LCR documents from {local_zip}: {e}. '
            'Please check that the zip file is valid and not corrupted.'
        ) from e


def _get_context(text_dir: Path, record: dict) -> str:
    """Read and format the document context for a given record."""
    doc_folder = text_dir / record['document_category'] / record['document_set_id']

    if not doc_folder.exists() or not doc_folder.is_dir():
        logger.warning(
            f'Document folder not found: {doc_folder}. Returning empty context.'
        )
        return ''

    doc_blocks = []
    try:
        for file_path in sorted(doc_folder.iterdir()):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8').strip()
                    if content:
                        doc_blocks.append(content)
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(
                        f'Could not read file {file_path}, skipping: {e}'
                    )
    except OSError as e:
        logger.warning(
            f'Could not access document folder {doc_folder}: {e}'
        )
        return (
            f"ERROR: Could not read documents for "
            f"{record['document_category']}/{record['document_set_id']}"
        )

    documents_text = '\n\n'.join(
        f'BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}'
        for i, doc in enumerate(doc_blocks)
    )
    return documents_text


@LOAD_DATASET.register_module()
class AALCRDataset(BaseDataset):
    """AA-LCR (Artificial Analysis Long Context Retrieval) dataset.

    A benchmark for evaluating long-context retrieval and reasoning
    capabilities of language models. It requires models to find and
    synthesize information across multiple documents.

    The dataset auto-downloads the document corpus on first use.
    """

    def __init__(self, reader_cfg=None, **kwargs):
        # Ensure documents are downloaded before loading
        self.text_dir = _ensure_text_dir_downloaded()
        super().__init__(reader_cfg=reader_cfg or {}, **kwargs)

    @staticmethod
    def load(path: str, name: str = 'default'):
        path = get_data_path(path, local_mode=True)
        logger.debug(f"Loading AA-LCR dataset from: {path}")
        from datasets import load_dataset

        text_dir = _ensure_text_dir_downloaded()

        dataset = load_dataset(path=path, name=name, trust_remote_code=True, split='test')
        raw_data = []
        for i in range(len(dataset)):
            item = dataset[i]
            # Build the long-context prompt using documents
            context = _get_context(text_dir, item)
            prompt = PROMPT_TEMPLATE.format(
                documents_text=context,
                question=item['question'],
            )
            raw_data.append({
                'input': prompt,
                'answers': item['answer'],
                'question': item['question'],
                'document_category': item.get('document_category', ''),
                'document_set_id': item.get('document_set_id', ''),
                'data_source_urls': item.get('data_source_urls', ''),
            })
        logger.debug(f"AA-LCR dataset loaded: {len(raw_data)} samples")
        return Dataset.from_list(raw_data)


@LOAD_DATASET.register_module()
class AALCRJGDataset(LLMJudgeDataset):
    """AA-LCR Judge Dataset class for LLM-based evaluation.

    Wrapper class that provides LLM Judge evaluation capabilities for
    AA-LCR dataset. Follows the same pattern as HLEJGDataset.

    The judge dataset merges original dataset items (question, answers)
    with model predictions (model_answer) so the judge model can compare
    the candidate answer against the reference answer.
    """

    def _get_dataset_class(self):
        """Return the base dataset class for LLM Judge evaluation."""
        return AALCRDataset


@ICL_EVALUATORS.register_module()
class AALCRJudgeEvaluator(BaseEvaluator):
    """AA-LCR Judge evaluator for assessing model responses using LLM-based judgment.

    Evaluates model predictions by parsing judge model outputs (CORRECT/INCORRECT)
    and computing accuracy metrics. Follows the same pattern as HLEJudgeEvaluator,
    adapted for the simpler binary CORRECT/INCORRECT judge output format.

    The judge model is expected to respond with "CORRECT" or "INCORRECT" for
    each candidate answer, as instructed by the JUDGE_PROMPT template.
    """

    def score(self, predictions: List, references: List) -> Dict[str, Any]:
        """Score predictions against references using LLM judge outputs.

        Args:
            predictions: List of judge model output strings (should contain
                CORRECT or INCORRECT).
            references: List of reference answers.

        Returns:
            Dictionary with accuracy and per-sample details.
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        details = {}
        correct = 0
        total = 0

        for index, (judge_output, ref) in enumerate(zip(predictions, references)):
            total += 1
            # Parse judge output: look for CORRECT using word boundary
            # to avoid matching "INCORRECT" as "CORRECT"
            is_correct = bool(
                re.search(r'\bCORRECT\b', judge_output, re.IGNORECASE)
            )

            if is_correct:
                correct += 1

            details[str(index)] = {
                'judge_output': judge_output,
                'answer': ref,
                'correct': is_correct,
            }

        accuracy = correct / total * 100 if total > 0 else 0
        return {
            'accuracy': accuracy,
            'details': details,
        }
