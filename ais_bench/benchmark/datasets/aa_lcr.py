import csv
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_cache_dir
from ais_bench.benchmark.datasets.utils.llm_judge import LLMJudgeDataset
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS, LOAD_DATASET
from ais_bench.benchmark.utils.logging.logger import AISLogger

logger = AISLogger()

# ---------------------------------------------------------------------------
# Local paths – document corpus ZIP and metadata CSV
# ---------------------------------------------------------------------------

# Directory containing this file (benchmark/datasets/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Data root: <repo>/ais_bench/datasets/aa_lcr/
#   _THIS_DIR  = ais_bench/benchmark/datasets/
#   _DATA_DIR  = ais_bench/datasets/aa_lcr/   (../../datasets/aa_lcr)
_DATA_DIR = os.path.abspath(os.path.join(
    _THIS_DIR, '..', '..', 'datasets', 'aa_lcr'
))

# Document corpus ZIP (benchmark/ais_bench/datasets/aa_lcr/extracted_text/).
_DOC_ZIP_PATH = os.path.join(
    _DATA_DIR, 'extracted_text', 'AA-LCR_extracted-text.zip'
)

# CSV metadata file (benchmark/ais_bench/datasets/aa_lcr/AA-LCR_Dataset.csv).
_CSV_PATH = os.path.join(_DATA_DIR, 'AA-LCR_Dataset.csv')

# Cache subdirectory where the ZIP is extracted.
DEFAULT_CACHE_SUBDIR: str = 'aa_lcr'
DEFAULT_EXTRACTED_DIR_NAME: str = 'lcr'

# Default cache root – user-level so the corpus survives package updates.
DEFAULT_CACHE_ROOT = os.path.abspath(os.path.join(
    _THIS_DIR, '..', '..', 'datasets'
))

# ---------------------------------------------------------------------------
# Prompt templates (matching evalscope format exactly)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION"""

JUDGE_PROMPT = """\
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT. \
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {answers}
CANDIDATE ANSWER TO ASSESS: {model_answer}

Reply only with CORRECT or INCORRECT."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_text_dir_downloaded() -> Path:
    """Ensure AA-LCR extracted texts are available locally.

    Looks for the document corpus ZIP at the relative path
    ``benchmark/ais_bench/datasets/aa_lcr/extracted_text/AA-LCR_extracted-text.zip``
    (sibling to this file in the repository), extracts it into the cache
    directory on first use, and returns the path to the ``lcr/`` directory
    containing the ``.txt`` files.  Subsequent calls return the cached path
    immediately.

    This mirrors evalscope's ``_ensure_text_dir_downloaded()`` but reads
    from a local ZIP instead of downloading from ModelScope.

    Returns:
        Path to the ``lcr/`` directory containing extracted ``.txt`` files.

    Raises:
        FileNotFoundError: If the local ZIP file does not exist.
        ValueError: If extraction fails.
    """
    cache_root = Path(get_cache_dir(DEFAULT_CACHE_ROOT)) / DEFAULT_CACHE_SUBDIR
    extracted_dir = cache_root / DEFAULT_EXTRACTED_DIR_NAME

    if extracted_dir.exists():
        logger.info(f'AA-LCR documents found in cache: {extracted_dir}')
        return extracted_dir

    local_zip = Path(_DOC_ZIP_PATH)
    if not local_zip.exists():
        raise FileNotFoundError(
            f'AA-LCR document corpus ZIP not found at: {local_zip}\n'
            'Please ensure the file is placed at '
            'benchmark/ais_bench/datasets/aa_lcr/extracted_text/'
            'AA-LCR_extracted-text.zip relative to the repository root.'
        )

    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f'Extracting {local_zip} to {cache_root} ...')
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(cache_root)

        if not extracted_dir.exists():
            raise ValueError(
                f'Extraction succeeded but target directory not found: '
                f'{extracted_dir}'
            )

        logger.info(f'AA-LCR documents ready at {extracted_dir}')
        return extracted_dir
    except Exception as exc:
        raise ValueError(
            f'Failed to extract AA-LCR documents from {local_zip}: {exc}. '
            'Please check that the zip file is valid and not corrupted.'
        ) from exc


def _get_context(text_dir: Path, record: dict) -> str:
    """Read and format the document context for a given record.

    Documents are loaded in the order specified by ``data_source_filenames``,
    matching the original AA-LCR reference implementation.  Each document is
    wrapped in ``BEGIN DOCUMENT … / END DOCUMENT …`` markers.

    This mirrors evalscope's ``AALCRAdapter._get_context()``.

    Args:
        text_dir: Root directory of the extracted document corpus
            (the ``lcr/`` directory).
        record: A single dataset record dict with ``document_category``,
            ``document_set_id`` and ``data_source_filenames``.

    Returns:
        Formatted string with all documents for this record, or an empty
        string if the document folder cannot be found / read.
    """
    doc_folder = text_dir / record['document_category'] / record['document_set_id']

    if not doc_folder.exists() or not doc_folder.is_dir():
        logger.warning(
            f'Document folder not found: {doc_folder}. '
            'Returning empty context.'
        )
        return ''

    # Resolve data_source_filenames – may be a semicolon-separated string
    # (from CSV) or already a list (from HuggingFace dataset).
    filenames_raw = record.get('data_source_filenames', '')
    if isinstance(filenames_raw, str) and filenames_raw.strip():
        ordered_filenames = [
            fn.strip() for fn in filenames_raw.split(';') if fn.strip()
        ]
    elif isinstance(filenames_raw, list):
        ordered_filenames = filenames_raw
    else:
        ordered_filenames = []

    doc_blocks: List[str] = []

    if ordered_filenames:
        # Load documents in the order specified by data_source_filenames
        # (matching the AA-LCR reference implementation).
        for filename in ordered_filenames:
            file_path = doc_folder / filename
            if not file_path.is_file():
                logger.warning(
                    f'Document file not found: {file_path}, skipping.'
                )
                continue
            try:
                content = file_path.read_text(encoding='utf-8').strip()
                if content:
                    doc_blocks.append(content)
            except (IOError, UnicodeDecodeError) as exc:
                logger.warning(
                    f'Could not read file {file_path}, skipping: {exc}'
                )
    else:
        # Fallback: iterate directory (sorted for determinism).
        try:
            for file_path in sorted(doc_folder.iterdir()):
                if not file_path.is_file():
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8').strip()
                    if content:
                        doc_blocks.append(content)
                except (IOError, UnicodeDecodeError) as exc:
                    logger.warning(
                        f'Could not read file {file_path}, skipping: {exc}'
                    )
        except OSError as exc:
            logger.warning(
                f'Could not access document folder {doc_folder}: {exc}'
            )
            return (
                f"ERROR: Could not read documents for "
                f"{record['document_category']}/{record['document_set_id']}"
            )

    if not doc_blocks:
        logger.warning(
            f'No valid documents found in {doc_folder}. '
            'Returning empty context.'
        )
        return ''

    documents_text = '\n\n'.join(
        f'BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}'
        for i, doc in enumerate(doc_blocks)
    )
    return documents_text


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

@LOAD_DATASET.register_module()
class AALCRDataset(BaseDataset):
    """AA-LCR (Artificial Analysis Long Context Retrieval) dataset.

    A benchmark for evaluating long-context retrieval and reasoning
    capabilities of language models. Models must find and synthesise
    information across multiple documents to answer each question.

    The document corpus is read from a local ZIP and cached after first
    extraction.  Question metadata is loaded from a local CSV file.
    The two are linked via ``document_category`` and
    ``document_set_id`` fields — the same data-separation design used
    by the evalscope adapter.

    .. note::

        Evaluation uses an LLM judge rather than exact-match or F1,
        because reference answers can be paraphrased.
    """

    def __init__(self, reader_cfg=None, k=1, n=1, task_state_manager=None, **kwargs):
        # Ensure the document corpus is available *before* BaseDataset
        # calls self.load(), which needs the text_dir.
        self.text_dir = _ensure_text_dir_downloaded()
        super().__init__(reader_cfg=reader_cfg or {}, k=k, n=n, task_state_manager=task_state_manager, **kwargs)

    @staticmethod
    def load(path: str, name: str = 'default', **kwargs):
        """Load AA-LCR dataset metadata and build long-context prompts.

        Loads question metadata from the local CSV file, then for each
        record reads the associated documents from the extracted corpus
        and builds the full prompt.  This mirrors evalscope's
        ``AALCRAdapter.record_to_sample()`` flow.

        Args:
            path: Local path to the dataset metadata directory
                (e.g. ``benchmark/ais_bench/datasets/aa_lcr``).
            name: Dataset configuration / subset name.

        Returns:
            A HuggingFace :class:`Dataset` with columns: ``input``,
            ``answers``, ``question``, ``document_category``,
            ``document_set_id``, ``data_source_urls``.
        """
        # Resolve CSV path: prefer the local CSV file over HuggingFace
        # load_dataset to avoid requiring network access.
        csv_path = _CSV_PATH
        if path and os.path.isabs(path):
            candidate = os.path.join(path, 'AA-LCR_Dataset.csv')
            if os.path.exists(candidate):
                csv_path = candidate

        logger.debug(f'Loading AA-LCR dataset metadata from: {csv_path}')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f'AA-LCR CSV metadata file not found at: {csv_path}\n'
                'Please ensure the file is placed at '
                'benchmark/ais_bench/datasets/aa_lcr/AA-LCR_Dataset.csv '
                'relative to the repository root.'
            )

        text_dir = _ensure_text_dir_downloaded()

        # Load records directly from local CSV (adapted from evalscope's
        # load_questions() in the AA-LCR README).
        records: List[Dict[str, Any]] = []
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        logger.info(f'Loaded {len(records)} records from AA-LCR CSV')

        raw_data: List[Dict[str, Any]] = []
        for record in records:
            context = _get_context(text_dir, record)
            prompt = PROMPT_TEMPLATE.format(
                documents_text=context,
                question=record['question'],
            )

            # ====== log constructed prompt ======
            logger.info(
                '========== CONSTRUCTED PROMPT '
                f'(question_id={record.get("question_id", "?")}) =========='
            )
            logger.info(f'QUESTION: {record["question"]}')
            logger.info(f'ANSWER:   {record["answer"]}')
            logger.info(
                f'DOC_CATEGORY: {record.get("document_category", "?")}  '
                f'DOC_SET: {record.get("document_set_id", "?")}'
            )
            logger.info(f'PROMPT ({len(prompt)} chars):\n{prompt}')
            logger.info('========== CONSTRUCTED PROMPT END ==========')
            # =======================================

            raw_data.append({
                'input': prompt,
                'answers': record['answer'],
                'question': record['question'],
                'document_category': record.get('document_category', ''),
                'document_set_id': record.get('document_set_id', ''),
                'data_source_urls': record.get('data_source_urls', ''),
            })

        logger.debug(f'AA-LCR dataset loaded: {len(raw_data)} samples')
        return Dataset.from_list(raw_data)


@LOAD_DATASET.register_module()
class AALCRJGDataset(LLMJudgeDataset):
    """AA-LCR Judge Dataset – merges model predictions with dataset items.

    Follows the same pattern as :class:`HLEJGDataset`: subclasses
    :class:`LLMJudgeDataset` and overrides ``_get_dataset_class`` to
    point back to :class:`AALCRDataset`.
    """

    def _get_dataset_class(self):
        return AALCRDataset

    def _modify_dataset_item(self, dataset_item, pred_item):
        """Merge prediction into the dataset item and log the judge prompt."""
        super()._modify_dataset_item(dataset_item, pred_item)

        # ====== log LLM Judge prompt ======
        logger.info(
            '========== JUDGE ITEM '
            f'(pred_uuid={pred_item.get("uuid", "?")}) =========='
        )
        logger.info(f'QUESTION:      {dataset_item.get("question", "N/A")}')
        logger.info(f'ANSWERS:       {dataset_item.get("answers", "N/A")}')
        logger.info(
            f'MODEL_ANSWER:  {dataset_item.get("model_answer", "N/A")}'
        )
        judge_prompt = JUDGE_PROMPT.format(
            question=dataset_item.get('question', ''),
            answers=dataset_item.get('answers', ''),
            model_answer=dataset_item.get('model_answer', ''),
        )
        logger.info(f'JUDGE_PROMPT ({len(judge_prompt)} chars):\n{judge_prompt}')
        logger.info('========== JUDGE ITEM END ==========')
        # ===================================


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@ICL_EVALUATORS.register_module()
class AALCRJudgeEvaluator(BaseEvaluator):
    """AA-LCR Judge evaluator for LLM-based answer assessment.

    Parses judge model outputs looking for ``CORRECT`` / ``INCORRECT``
    and computes accuracy.  Uses word-boundary matching so that
    "INCORRECT" is not falsely matched as "CORRECT".

    This mirrors evalscope's ``AALCRAdapter.llm_match_score()`` scoring
    logic: a single regex pass with ``\\bCORRECT\\b`` to determine
    correctness.
    """

    def score(self, predictions: List, references: List) -> Dict[str, Any]:
        """Score judge predictions against reference answers.

        Args:
            predictions: Raw judge model outputs (strings expected to
                contain ``CORRECT`` or ``INCORRECT``).
            references: Reference answers (used for per-sample detail).

        Returns:
            Dict with ``accuracy`` (float percentage) and ``details``
            (per-sample judge output, reference, and correctness flag).
        """
        if len(predictions) != len(references):
            return {
                'error': (
                    'predictions and references have different length. '
                    f'len(predictions): {len(predictions)}, '
                    f'len(references): {len(references)}'
                )
            }

        details: Dict[str, Dict[str, Any]] = {}
        correct = 0
        total = 0

        for index, (judge_output, ref) in enumerate(
            zip(predictions, references)
        ):
            total += 1
            # Use word-boundary matching to avoid matching "CORRECT"
            # inside "INCORRECT".  Same approach as evalscope's
            # AALCRAdapter.llm_match_score().
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

            # ====== log eval result ======
            logger.info(
                f'========== EVAL RESULT (index={index}) =========='
            )
            logger.info(f'JUDGE_OUTPUT: {judge_output}')
            logger.info(f'REFERENCE:    {ref}')
            logger.info(
                f'IS_CORRECT:   {is_correct}'
            )
            logger.info('========== EVAL RESULT END ==========')
            # ==============================

        accuracy = correct / total * 100 if total > 0 else 0

        # ====== log final accuracy summary ======
        logger.info(
            '========== EVAL SUMMARY '
            f'(correct={correct}/{total}, accuracy={accuracy:.2f}%) =========='
        )
        # =========================================
        return {
            'accuracy': accuracy,
            'details': details,
        }
