import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, ICL_EVALUATORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path, get_cache_dir
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

# Download URL and cache configuration
DOWNLOAD_URL = (
    'https://modelscope.cn/datasets/evalscope/AA-LCR/resolve/master/'
    'extracted_text/AA-LCR_extracted-text.zip'
)
DEFAULT_CACHE_SUBDIR = 'aa_lcr'
DEFAULT_ZIP_NAME = 'AA-LCR_extracted-text.zip'
DEFAULT_EXTRACTED_DIR_NAME = 'lcr'

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../../../'
)


def _ensure_text_dir_downloaded() -> Path:
    """Ensure AA-LCR extracted texts are available locally; download and extract if missing."""
    cache_root = Path(get_cache_dir(DEFAULT_CACHE_DIR)) / DEFAULT_CACHE_SUBDIR
    extracted_dir = cache_root / DEFAULT_EXTRACTED_DIR_NAME

    if extracted_dir.exists():
        logger.info(f'AA-LCR documents found: {extracted_dir}')
        return extracted_dir

    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = cache_root / DEFAULT_ZIP_NAME

    try:
        logger.info(f'Downloading AA-LCR documents from {DOWNLOAD_URL} to {zip_path}...')
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

        logger.info(f'Extracting {zip_path} to {cache_root}...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_root)

        if not extracted_dir.exists():
            raise ValueError(
                f'Extraction succeeded but target directory not found: {extracted_dir}'
            )

        logger.info(f'AA-LCR documents ready at {extracted_dir}')
        return extracted_dir
    except Exception as e:
        raise ValueError(
            f'Failed to download or extract AA-LCR documents: {e}. '
            'Please manually download and place documents in the cache directory.'
        ) from e
    finally:
        # Best-effort cleanup of the zip file
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass


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


@ICL_EVALUATORS.register_module()
class AALCREvaluator(BaseEvaluator):
    """AA-LCR evaluator using relaxed accuracy.

    Evaluates whether the model's answer contains the key information
    from the reference answer. Uses case-insensitive substring matching
    with normalization to provide a reasonable correctness check.

    For the most accurate evaluation, LLM-based judging is recommended
    (as used in the original evalscope implementation), but this evaluator
    provides a fast rule-based alternative.
    """

    def score(self, predictions: List, references: List) -> dict:
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        details = {}
        correct = 0
        total = 0

        for index, (pred, ref) in enumerate(zip(predictions, references)):
            total += 1
            is_correct = _check_answer_correctness(pred, ref)

            if is_correct:
                correct += 1

            details[str(index)] = {
                'pred': pred,
                'answer': ref,
                'correct': is_correct,
            }

        accuracy = correct / total * 100 if total > 0 else 0
        return {
            'accuracy': accuracy,
            'details': details,
        }


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Convert to lowercase
    text = text.lower().strip()
    return text


def _check_answer_correctness(prediction: str, reference: str) -> bool:
    """Check if the prediction matches the reference answer.

    Uses multiple strategies:
    1. Normalized exact match
    2. Check if all key phrases from reference appear in prediction
    3. Check if cleaned prediction contains cleaned reference
    """
    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)

    # Strategy 1: Normalized exact match (after stripping common prefixes)
    pred_clean = re.sub(
        r'^(answer:?\s*|the\s+answer\s+is:?\s*|response:?\s*)',
        '', pred_norm, flags=re.IGNORECASE
    ).strip()
    ref_clean = ref_norm.strip()

    if pred_clean == ref_clean:
        return True

    # Strategy 2: Check if reference is contained in prediction
    if ref_clean in pred_clean:
        return True

    # Strategy 3: Check keyword overlap for multi-line references
    ref_lines = [l.strip() for l in ref_clean.split('\n') if l.strip()]
    if len(ref_lines) > 1:
        # For structured answers, check that key information is present
        match_count = 0
        for line in ref_lines:
            # Extract key content from each reference line
            key_content = re.sub(r'^\d+\.\s*', '', line).strip()
            if key_content and key_content in pred_clean:
                match_count += 1
        # Require at least half of the reference lines to be present
        if match_count >= len(ref_lines) / 2:
            return True

    return False
