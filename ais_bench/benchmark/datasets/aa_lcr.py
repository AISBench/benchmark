import csv
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_cache_dir, get_data_path
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

# Data directory relative to repository root (resolved via get_data_path).
_DATA_REL_PATH: str = 'ais_bench/datasets/aa_lcr'

# Document corpus ZIP filename within the data directory.
_DOC_ZIP_REL_PATH: str = os.path.join(
    _DATA_REL_PATH, 'extracted_text', 'AA-LCR_extracted-text.zip'
)

# Cache subdirectory where the ZIP is extracted.
DEFAULT_CACHE_SUBDIR: str = 'aa_lcr'
DEFAULT_EXTRACTED_DIR_NAME: str = 'lcr'

# Default cache root – user-level so the corpus survives package updates.
DEFAULT_CACHE_ROOT = os.path.abspath(os.path.join(
    _THIS_DIR, '..', '..', 'datasets'
))

# ---------------------------------------------------------------------------
# Prompt templates
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
    ``ais_bench/datasets/aa_lcr/extracted_text/AA-LCR_extracted-text.zip``
    (resolved via :func:`get_data_path`), extracts it into the cache
    directory on first use, and returns the path to the ``lcr/`` directory
    containing the ``.txt`` files.  Subsequent calls return the cached path
    immediately.

    Returns:
        Path to the ``lcr/`` directory containing extracted ``.txt`` files.

    Raises:
        FileNotFoundError: If the local ZIP file does not exist.
        ValueError: If extraction fails.
    """
    cache_root = Path(get_cache_dir(DEFAULT_CACHE_ROOT)) / DEFAULT_CACHE_SUBDIR
    extracted_dir = cache_root / DEFAULT_EXTRACTED_DIR_NAME

    if extracted_dir.exists():
        return extracted_dir

    local_zip = Path(get_data_path(_DOC_ZIP_REL_PATH))
    if not local_zip.exists():
        raise FileNotFoundError(
            f'AA-LCR document corpus ZIP not found at: {local_zip}\n'
            'Please ensure the file is placed at '
            'ais_bench/datasets/aa_lcr/extracted_text/'
            'AA-LCR_extracted-text.zip relative to the repository root.'
        )

    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(cache_root)

        if not extracted_dir.exists():
            raise ValueError(
                f'Extraction succeeded but target directory not found: '
                f'{extracted_dir}'
            )

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

    Args:
        text_dir: Root directory of the extracted document corpus
            (the ``lcr/`` directory).
        record: A single dataset record dict with ``document_category``,
            ``document_set_id`` and ``data_source_filenames``.

    Returns:
        Formatted string with all documents for this record, or an empty
        string if the document folder cannot be found / read.
    """
    doc_category = record.get('document_category')
    doc_set_id = record.get('document_set_id')
    if not doc_category or not doc_set_id:
        logger.warning(
            f"Missing document_category or document_set_id in record: {record}. "
            "Returning empty context."
        )
        return ''
    doc_folder = text_dir / doc_category / doc_set_id

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
    ``document_set_id`` fields.

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
        and builds the full prompt.

        Args:
            path: Local path to the dataset metadata directory
                (e.g. ``ais_bench/datasets/aa_lcr``).
            name: Dataset configuration / subset name.

        Returns:
            A HuggingFace :class:`Dataset` with columns: ``input``,
            ``answers``, ``question``, ``document_category``,
            ``document_set_id``, ``data_source_urls``.
        """
        # Resolve CSV path via get_data_path: uses the config-supplied
        # ``path`` as the dataset directory when provided, otherwise falls
        # back to the default location relative to the repository root.
        if path and os.path.isabs(path):
            data_dir = path
        else:
            rel_path = path if path else _DATA_REL_PATH
            data_dir = get_data_path(rel_path)

        csv_path = os.path.join(data_dir, 'AA-LCR_Dataset.csv')
        logger.debug(f'Loading AA-LCR dataset metadata from: {csv_path}')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f'AA-LCR CSV metadata file not found at: {csv_path}\n'
                'Please ensure the file is placed at '
                'ais_bench/datasets/aa_lcr/AA-LCR_Dataset.csv '
                'relative to the repository root.'
            )

        text_dir = _ensure_text_dir_downloaded()

        # Load records directly from local CSV.
        records: List[Dict[str, Any]] = []
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

        raw_data: List[Dict[str, Any]] = []
        for record in records:
            context = _get_context(text_dir, record)
            prompt = PROMPT_TEMPLATE.format(
                documents_text=context,
                question=record['question'],
            )

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


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@ICL_EVALUATORS.register_module()
class AALCRJudgeEvaluator(BaseEvaluator):
    """AA-LCR Judge evaluator for LLM-based answer assessment.

    Parses judge model outputs looking for ``CORRECT`` / ``INCORRECT``
    and computes accuracy.  Uses word-boundary matching so that
    "INCORRECT" is not falsely matched as "CORRECT".
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
                'accuracy': 0.0,
                'details': {},
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
            # inside "INCORRECT".
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

        logger.info(
            '========== EVAL SUMMARY '
            f'(correct={correct}/{total}, accuracy={accuracy:.2f}%) =========='
        )
        return {
            'accuracy': accuracy,
            'details': details,
        }
