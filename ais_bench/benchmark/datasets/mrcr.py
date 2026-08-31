"""MRCR dataset integration for aisbench.

MRCR (Multi-Round Co-reference Resolution, https://huggingface.co/datasets/openai/mrcr)
is a long-context benchmark introduced by OpenAI (inspired by the Gemini
"Michelangelo" MRCR eval).  The model receives a long multi-turn synthetic
conversation in which the *same* writing request (e.g. "write a poem about
tapirs") appears 2/4/8 times among same-distribution distractor requests, and
must return the Nth instance, prepending a random string:

    "Prepend <hash> to the 2nd (1 indexed) poem about tapirs.
     Do not include any other text in your response."

Each parquet row contains:

- ``prompt``: JSON-encoded list of ``{role, content}`` chat messages
- ``answer``: the gold needle text (already prepended with the random string)
- ``random_string_to_prepend``: the hash the model must prepend
- ``date_added``: present in the 12/2025 bugfix revision

Official scoring (mirrored byte-for-byte in :class:`MRCREvaluator._grade`):

    def grade(response, answer, random_string_to_prepend) -> float:
        if not response.startswith(random_string_to_prepend):
            return 0
        response = response.removeprefix(random_string_to_prepend)
        answer = answer.removeprefix(random_string_to_prepend)
        return float(SequenceMatcher(None, response, answer).ratio())

Thinking-model note: aisbench's saved ``prediction`` concatenates
``reasoning_content + "\n\n" + content`` (see ``Output.get_prediction``),
which would *always* fail the ``startswith(prefix)`` check.  The evaluator
therefore prefers the reasoning-free ``content`` field persisted by
``GenInferencerOutputHandler`` when the model emits reasoning, and falls
back to ``prediction`` for non-thinking models / legacy result files.
"""

import json
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow.parquet as pq
from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.openicl.icl_prompt_template import BasePromptTemplate
from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import (
    ICL_EVALUATORS,
    ICL_PROMPT_TEMPLATES,
    LOAD_DATASET,
    TEXT_POSTPROCESSORS,
)
from ais_bench.benchmark.utils.logging import AISLogger

logger = AISLogger()


# Official bin boundaries (tokens of prompt + answer, tiktoken o200k_base).
# See the dataset card: "Bins are determined by number of tokens used by
# prompt + answer in the sample."
MRCR_BIN_BOUNDARIES: Dict[str, Tuple[int, int]] = {
    '4k': (4096, 8192),
    '8k': (8192, 16384),
    '16k': (16384, 32768),
    '32k': (32768, 65536),
    '64k': (65536, 131072),
    '128k': (131072, 262144),
    '256k': (262144, 524288),
    '1m': (524288, 1048576),
}

# Column names probed for a precomputed prompt+answer token count; the HF
# release computes bins at build time, so some revisions ship a ready-made
# column.  If none is present we count with tiktoken ourselves.
_TOKEN_COUNT_COLUMNS = ('num_tokens', 'n_tokens', 'tokens', 'token_count')


@ICL_PROMPT_TEMPLATES.register_module('mrcr_prompt')
class MRCRPromptTemplate(BasePromptTemplate):
    """Pass-through template for MRCR raw multi-turn prompts.

    MRCR stores the conversation as a JSON-encoded list of ``{role,
    content}`` messages that must reach the chat API unchanged (the
    official runner posts ``json.loads(row['prompt'])`` verbatim).  This
    template converts that list into a framework :class:`PromptList`:
    system messages live in the ``begin`` section, user/assistant turns in
    the ``round`` section, following the standard section emission used by
    the built-in chat templates of other gen datasets.
    """

    def __init__(
        self,
        template: Union[Dict, str] = '',
        ice_token: Optional[str] = None,
        sep_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            template=template or '', ice_token=ice_token, sep_token=sep_token)

    def generate_item(
        self,
        entry: Dict,
        output_field=None,
        output_field_replace_token: str = '',
        ice_field_replace_token: str = '',
    ):
        messages = entry.get('prompt', [])
        if isinstance(messages, str):
            # Plain-text prompt: forward as-is.
            return messages
        if isinstance(messages, dict):
            messages = [messages]

        # Map the released OpenAI-style roles onto the internal roles the
        # API chat model understands (see ROLE_MAP in vllm_custom_api_chat):
        #   system -> SYSTEM (begin section)
        #   user / assistant -> HUMAN / BOT (round section)
        template: Dict[str, list] = {}
        system_items = []
        round_items = []
        for msg in messages:
            if not isinstance(msg, dict):
                msg = {'role': 'user', 'content': str(msg)}
            raw_role = msg.get('role', 'user')
            if raw_role == 'system':
                role = 'SYSTEM'
            elif raw_role == 'assistant':
                role = 'BOT'
            else:
                role = 'HUMAN'
            item = {'role': role, 'prompt': msg.get('content', '')}
            for key, value in msg.items():
                if key not in ('role', 'content'):
                    item[key] = value
            if role == 'SYSTEM':
                system_items.append(item)
            else:
                round_items.append(item)
        if system_items:
            template['begin'] = system_items
        template['round'] = round_items
        return self._encode_template(template, ice=False)


@LOAD_DATASET.register_module()
class MRCRDataset(BaseDataset):
    """MRCR dataset.

    Streams the per-needle-count parquet shards row-group by row-group so
    that 1M-token samples never need to be materialised as a whole shard,
    then (optionally) filters rows to one token bin following the official
    bucket definition.
    """

    @staticmethod
    def load(
        path: str,
        subset: str = '2needle',
        length_bin: Optional[str] = '1m',
        tokenizer_model: str = 'o200k_base',
        **kwargs,
    ) -> Dataset:
        """Load one needle-count subset of MRCR.

        Args:
            path: dataset root, e.g. ``ais_bench/datasets/MRCR`` containing
                ``2needle/`` ``4needle/`` ``8needle/`` sub-directories.
            subset: needle-count sub-directory name.
            length_bin: bin key of :data:`MRCR_BIN_BOUNDARIES` (e.g. ``'1m'``);
                ``None`` disables filtering (fast smoke-test path).
            tokenizer_model: tiktoken encoding used for bin filtering when no
                precomputed token-count column exists (official: o200k_base).

        Returns:
            HuggingFace Dataset with ``id / prompt / answer /
        random_string_to_prepend`` columns.
        """
        # Fail fast on config typos even before the dataset is deployed.
        if length_bin is not None and length_bin not in MRCR_BIN_BOUNDARIES:
            raise ValueError(
                f'Unknown length_bin {length_bin!r}, expected one of '
                f'{sorted(MRCR_BIN_BOUNDARIES)} or None.')
        bin_lo, bin_hi = (
            MRCR_BIN_BOUNDARIES[length_bin] if length_bin else (None, None)
        )

        root = get_data_path(path)
        subset_dir = os.path.join(root, subset) if subset else root
        if not os.path.isdir(subset_dir):
            raise FileNotFoundError(
                f'MRCR subset directory not found: {subset_dir}. '
                f'Deploy openai/mrcr under {root} first (see the mrcr README).'
            )
        shard_files = sorted(
            os.path.join(subset_dir, f) for f in os.listdir(subset_dir)
            if f.endswith('.parquet')
        )
        if not shard_files:
            raise FileNotFoundError(
                f'No .parquet shards under {subset_dir}.')

        tokenizer = None
        if bin_lo is not None:
            # Prefer a precomputed column; fall back to counting ourselves.
            schema_names = pq.ParquetFile(shard_files[0]).schema.names
            token_column = next(
                (c for c in _TOKEN_COUNT_COLUMNS if c in schema_names), None)
            if token_column is None:
                import tiktoken
                tokenizer = tiktoken.get_encoding(tokenizer_model)
                logger.info(
                    'No precomputed token-count column found; counting '
                    f'tokens with tiktoken {tokenizer_model!r} for bin '
                    f'({bin_lo}, {bin_hi}].'
                )

        dataset: List[Dict[str, Any]] = []
        for shard in shard_files:
            parquet_file = pq.ParquetFile(shard)
            for batch in parquet_file.iter_batches(batch_size=8):
                frame = batch.to_pandas()
                for row_i in range(len(frame)):
                    row = frame.iloc[row_i].to_dict()
                    item = MRCRDataset._parse_row(
                        row, row_id=len(dataset), shard=shard)
                    if item is None:
                        continue
                    if bin_lo is not None:
                        n_tokens = MRCRDataset._count_tokens(
                            row, item, token_column, tokenizer)
                        if n_tokens is None or not (bin_lo < n_tokens <= bin_hi):
                            continue
                    dataset.append(item)
            logger.debug(f'Scanned {shard}')
        if not dataset:
            raise ValueError(
                f'No MRCR samples matched subset={subset!r} '
                f'length_bin={length_bin!r} under {root}.')
        logger.info(
            f'Loaded {len(dataset)} MRCR samples '
            f'(subset={subset}, length_bin={length_bin}).')
        return Dataset.from_list(dataset)

    @staticmethod
    def _parse_row(row: Dict[str, Any], row_id: int, shard: str) -> Optional[Dict]:
        """Parse one parquet row; ``None`` (with a warning) on bad rows."""
        raw_prompt = row.get('prompt')
        answer = row.get('answer')
        prefix = row.get('random_string_to_prepend')
        if raw_prompt is None or answer is None or prefix is None:
            logger.warning(
                f'Row misses prompt/answer/random_string_to_prepend, '
                f'skipped (shard={shard}).')
            return None
        messages = raw_prompt
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError as exc:
                logger.warning(f'Failed to parse prompt JSON, skipped: {exc}')
                return None
        if isinstance(messages, dict):
            messages = [messages]
        if not isinstance(messages, list) or not messages:
            logger.warning('Prompt is not a non-empty message list, skipped.')
            return None
        return {
            # Sequential id guarantees prediction/test_set alignment after the
            # eval task sorts predictions by ``id``.
            'id': row_id,
            'prompt': messages,
            'answer': str(answer),
            'random_string_to_prepend': str(prefix),
        }

    @staticmethod
    def _count_tokens(row, item, token_column, tokenizer) -> Optional[int]:
        """Token count following the official prompt+answer bucket rule."""
        if token_column is not None:
            value = row.get(token_column)
            return int(value) if value is not None else None
        if tokenizer is None:
            return None
        n = sum(
            len(tokenizer.encode(str(m.get('content', ''))))
            for m in item['prompt']
            if isinstance(m, dict)
        )
        return n + len(tokenizer.encode(item['answer']))


def _remove_prefix(s: str, prefix: str) -> str:
    """``str.removeprefix`` equivalent (keeps Python 3.8 compatibility)."""
    return s[len(prefix):] if s.startswith(prefix) else s


def _select_response(prediction: Any, content: Any) -> str:
    """Pick the response text the official grader would see.

    Prefers the reasoning-free ``content`` field (thinking models); falls
    back to ``prediction`` (non-thinking models, legacy result files).
    Handles pass@k list payloads by taking the first element.
    """
    for candidate in (content, prediction):
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        if candidate:
            return str(candidate)
    return ''


@ICL_EVALUATORS.register_module()
class MRCREvaluator(BaseEvaluator):
    """MRCR evaluator: official prefix gate + SequenceMatcher ratio.

    Metrics:
        - ``score``: 100 * mean(ratio) -- the headline number, equivalent to
          the official per-sample ``grade()`` mean.
        - ``prefix_hit_rate``: share of samples that passed the
          ``startswith(random_string)`` gate (0-ratio diagnosis).
        - ``strict_acc``: share of samples with ratio == 1.0 (exact
          reproduction, Context Arena style diagnostic).
        - ``details``: per-sample records for offline re-grading.
    """

    def __init__(self):
        super().__init__()

    def score(
        self,
        predictions: List[Any],
        references: List[Any],
        content: Optional[List[Any]] = None,
        test_set: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        if len(predictions) != len(references):
            return {
                'error': (
                    'predictions and references have different length: '
                    f'len(predictions)={len(predictions)}, '
                    f'len(references)={len(references)}'
                )
            }
        # Per-sample prefixes come from the dataset column (the prediction
        # file never stores random_string_to_prepend).
        if test_set is not None:
            prefixes = list(test_set['random_string_to_prepend'])
        else:
            prefixes = ['' for _ in predictions]
            logger.warning(
                'test_set unavailable; scoring with empty prepend strings.')

        details: List[Dict[str, Any]] = []
        total_ratio = 0.0
        prefix_hits = 0
        strict_hits = 0
        total = 0
        for index, (pred, ref) in enumerate(zip(predictions, references)):
            response = _select_response(pred, content[index] if content else None)
            prefix = prefixes[index]
            ratio = self._grade(response, str(ref), prefix)
            total_ratio += ratio
            prefix_hits += 1 if response.startswith(prefix) and prefix else 0
            strict_hits += 1 if ratio == 1.0 else 0
            total += 1
            details.append(
                {
                    'id': index,
                    'pred': response,
                    'answer': ref,
                    'random_string_to_prepend': prefix,
                    'prefix_hit': bool(response.startswith(prefix) and prefix),
                    'ratio': ratio,
                    'correct': ratio == 1.0,
                }
            )

        return {
            'score': 100.0 * total_ratio / total if total else 0.0,
            'prefix_hit_rate': 100.0 * prefix_hits / total if total else 0.0,
            'strict_acc': 100.0 * strict_hits / total if total else 0.0,
            'num_total': total,
            'details': details,
        }

    @staticmethod
    def _grade(response: str, answer: str, prefix: str) -> float:
        """Mirror the official ``grade`` helper line by line.

        The official snippet relies on ``str.removeprefix`` (Python 3.9+);
        ``_remove_prefix`` reproduces it exactly for older interpreters.
        """
        if not isinstance(response, str):
            response = str(response)
        if not response.startswith(prefix):
            return 0.0
        response = _remove_prefix(response, prefix)
        answer = _remove_prefix(answer, prefix)
        return float(SequenceMatcher(None, response, answer).ratio())


@TEXT_POSTPROCESSORS.register_module('mrcr_postprocess')
def mrcr_postprocess(text: str) -> str:
    """Identity postprocessor (predictions are graded raw, as official)."""
    return text
