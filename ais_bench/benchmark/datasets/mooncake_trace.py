import json
import os
import random
import hashlib
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils.file.load_tokenizer import load_tokenizer
from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchDataContentError,
    AISBenchRuntimeError,
    ParameterValueError,
)

logger = AISLogger()

# ============================================================================
# 1. RNG Management System
# ============================================================================


class RNGManager:
    """Random number generator manager"""

    def __init__(self, root_seed: int | None):
        self._root_seed = root_seed
        if root_seed is not None:
            # Set global random seed (defensive measure)
            random.seed(root_seed)
            np_seed = (root_seed ^ (root_seed >> 32)) & 0xFFFFFFFF
            np.random.seed(np_seed)

    def derive(self, identifier: str) -> random.Random:
        """Derive a child RNG from an identifier"""
        if self._root_seed is not None:
            # Deterministic derivation: use SHA-256 hash
            seed_string = f"{self._root_seed}:{identifier}"
            hash_bytes = hashlib.sha256(seed_string.encode("utf-8")).digest()
            child_seed = int.from_bytes(hash_bytes[:8], byteorder="big")
            return random.Random(child_seed)
        else:
            # Non-deterministic: use system random
            return random.Random()


_rng_manager: RNGManager | None = None


def init_rng(seed: int | None):
    """Initialize global RNG manager"""
    global _rng_manager
    _rng_manager = RNGManager(seed)


def derive_rng(identifier: str) -> random.Random:
    """Derive a child RNG"""
    if _rng_manager is None:
        raise AISBenchRuntimeError(
            DSET_CODES.UNKNOWN_ERROR,
            "RNG manager not initialized. Call init_rng() first.",
        )
    return _rng_manager.derive(identifier)


# ============================================================================
# 2. Corpus Loading
# ============================================================================

DEFAULT_CORPUS_FILE = "assets/shakespeare.txt"
MAX_CHARS_PER_CHUNK = 10_000


def initialize_corpus(tokenizer, corpus_path: Path) -> list[int]:
    """
    Load and tokenize corpus

    Uses a character-based chunking strategy to ensure identical chunk boundaries
    across different machines.
    """
    with open(corpus_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Preprocessing: filter empty lines
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    def tokenize_chunk(chunk: list[str]) -> list[int]:
        """Tokenize a text chunk"""
        text = " ".join(chunk)
        tokens = tokenizer.encode(
            text, add_special_tokens=False
        )  # Returns token ID list
        return tokens

    # Character-based chunking (deterministic chunking)
    chunks = []
    buffer = []
    char_count = 0

    for line in non_empty_lines:
        buffer.append(line)
        char_count += len(line)

        if char_count >= MAX_CHARS_PER_CHUNK:
            chunks.append(buffer)
            buffer = []
            char_count = 0

    # Add remaining lines as the last chunk
    if buffer:
        chunks.append(buffer)

    # Multi-threaded tokenization (thread count doesn't affect reproducibility
    # because chunking is deterministic)
    num_threads = min(os.cpu_count() or 4, 8)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        tokenized_chunks = list(executor.map(tokenize_chunk, chunks))

    # Flatten all tokens
    tokenized_corpus = [token for chunk in tokenized_chunks for token in chunk]

    return tokenized_corpus


# ============================================================================
# 3. PromptGenerator
# ============================================================================


class PromptGenerator:
    """Prompt generator"""

    def __init__(
        self,
        tokenizer,
        tokenized_corpus: list[int],
        root_seed: int | None = None,
        block_size: int = 512,
    ):
        self.tokenizer = tokenizer
        self._tokenized_corpus = tokenized_corpus
        self._corpus_size = len(tokenized_corpus)

        # Initialize RNG (for corpus sampling)
        self._corpus_rng = derive_rng("dataset.prompt.corpus")

        # Hash ID cache: hash_id -> token list
        self._cache: dict[int, list[int]] = {}

        # Block size (default 512 tokens)
        self.block_size = block_size

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
    ) -> str:
        """
        Main entry point for generating prompts

        Args:
            mean: Target number of tokens (if using hash_ids, this is the total token count)
            stddev: Standard deviation (usually 0 for hash_ids mode)
            hash_ids: Hash ID list for cache reuse

        Returns:
            Generated prompt text
        """
        if hash_ids:
            return self._generate_cached_prompt(mean, hash_ids, self.block_size)

        # No hash_ids: sample token count using normal distribution
        num_tokens = self._sample_num_tokens(mean, stddev)
        return self.generate_prompt(num_tokens)

    def generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt with specified number of tokens"""
        tokens = self._sample_tokens(num_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _sample_tokens(self, num_tokens: int, hash_id: int | None = None) -> list[int]:
        """
        Sample specified number of tokens from corpus

        Uses circular sampling: if beyond corpus end, continue from the beginning.

        Args:
            num_tokens: Number of tokens to sample
            hash_id: Optional hash ID. If provided, uses an independent RNG derived from hash_id
                     to ensure different hash_ids generate different token sequences.
        """
        if num_tokens > self._corpus_size:
            # If requested token count exceeds corpus size, return entire corpus
            return self._tokenized_corpus.copy()

        # Select RNG: use independent RNG for hash_id, otherwise use shared RNG
        if hash_id is not None:
            # Derive independent RNG based on hash_id to ensure uniqueness
            block_rng = derive_rng(f"dataset.prompt.block.{hash_id}")
            start_idx = block_rng.randrange(self._corpus_size)
        else:
            # Use shared RNG for backward compatibility
            start_idx = self._corpus_rng.randrange(self._corpus_size)

        end_idx = start_idx + num_tokens
        prompt_tokens = self._tokenized_corpus[start_idx:end_idx]

        # If beyond corpus end, continue from the beginning
        if end_idx > self._corpus_size:
            prompt_tokens += self._tokenized_corpus[: end_idx - self._corpus_size]

        return prompt_tokens

    def _sample_num_tokens(self, mean: int | None, stddev: int | None) -> int:
        """Sample token count from normal distribution"""
        if mean is None:
            raise ParameterValueError(
                DSET_CODES.MISSING_REQUIRED_PARAM, "mean must be provided"
            )

        if stddev is None or stddev == 0:
            return mean

        # Sample using normal distribution (ensure positive integer)
        length_rng = derive_rng("dataset.prompt.length")
        while True:
            value = int(length_rng.gauss(mean, stddev))
            if value > 0:
                return value

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        """
        Generate prompt based on hash_ids (using cache mechanism)

        Each hash_id corresponds to a token block. If hash_id is in cache,
        reuse cached tokens; otherwise generate new tokens and cache them.

        Args:
            num_tokens: Total number of tokens
            hash_ids: Hash ID list
            block_size: Number of tokens per hash block (default 512)

        Returns:
            Generated prompt text
        """
        final_prompt: list[int] = []
        current_block_size = block_size

        # Calculate size of the last block
        final_block_size = num_tokens - ((len(hash_ids) - 1) * block_size)

        # Validate parameters
        if final_block_size <= 0 or block_size < final_block_size:
            raise ParameterValueError(
                DSET_CODES.INVALID_PARAM_VALUE,
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. Final block size: {final_block_size} must be > 0 and <= {block_size}.",
            )

        # Process each hash_id
        for index, hash_id in enumerate(hash_ids):
            # Last hash_id uses remaining tokens
            if index == len(hash_ids) - 1:
                current_block_size = final_block_size

            # If hash_id not in cache, generate and cache
            if hash_id not in self._cache:
                prompt_tokens: list[int] = []

                # If tokenizer supports block separator, insert BOS/EOS token
                # This ensures different blocks won't merge
                block_separation_token_id = getattr(
                    self.tokenizer, 'block_separation_token_id', None
                )

                if block_separation_token_id is not None:
                    prompt_tokens.append(block_separation_token_id)
                    prompt_tokens += self._sample_tokens(current_block_size - 1, hash_id=hash_id)
                else:
                    prompt_tokens += self._sample_tokens(current_block_size, hash_id=hash_id)

                # Cache token list
                self._cache[hash_id] = prompt_tokens

            # Reuse cached tokens
            final_prompt.extend(self._cache[hash_id])

        # Decode to text (don't skip special tokens, preserve block separator)
        return self.tokenizer.decode(final_prompt, skip_special_tokens=False)


# ============================================================================
# 4. Mooncake Trace Data Model
# ============================================================================


class MooncakeTrace:
    """Mooncake trace data model"""

    def __init__(self, data: dict[str, Any]):
        # Support input_text field: if input_text exists, input_length and hash_ids become optional
        self.input_text = data.get("input_text")

        if self.input_text is None:
            # If no input_text, input_length must exist
            if "input_length" not in data or data["input_length"] is None:
                raise ParameterValueError(
                    DSET_CODES.MISSING_REQUIRED_PARAM,
                    "Either 'input_text' or 'input_length' must be provided",
                )
            self.input_length = data["input_length"]
            self.hash_ids = data.get("hash_ids")
        else:
            # If input_text exists, input_length and hash_ids become optional (will be ignored)
            self.input_length = data.get("input_length")
            self.hash_ids = data.get("hash_ids")

        self.output_length = data.get("output_length")
        self.timestamp = data.get("timestamp")
        if self.timestamp:
            if not isinstance(self.timestamp, (float, int)):
                raise ParameterValueError(
                    DSET_CODES.INVALID_PARAM_VALUE,
                    f"timestamp must be a float or int, but got {type(self.timestamp)}",
                )
            if self.timestamp < 0:
                raise ParameterValueError(
                    DSET_CODES.INVALID_PARAM_VALUE,
                    f"timestamp must be >= 0, but got {self.timestamp}",
                )


def load_mooncake_trace(filename: str) -> list[MooncakeTrace]:
    """
    Load Mooncake trace data from JSONL file

    Returns:
        List of trace data
    """
    traces = []

    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            traces.append(MooncakeTrace(json.loads(line)))

    return traces


def _process_timestamps(
    traces: list[MooncakeTrace],
    auto_offset: bool = False,
    start_offset: int = 0,
    end_offset: int = -1,
) -> list[MooncakeTrace]:
    """
    Process timestamps: apply auto offset, start offset, and end offset

    Args:
        traces: Original trace data list
        auto_offset: Whether to auto-offset timestamps (make first timestamp 0)
        start_offset: Start offset in milliseconds, filter out timestamps less than this offset
        end_offset: End offset in milliseconds, -1 means no limit, >=0 filters out timestamps greater than this offset

    Returns:
        Processed trace data list
    """
    if not traces:
        return traces

    # Filter traces without timestamps (keep them but they won't participate in timestamp processing)
    traces_with_timestamp = [(i, trace) for i, trace in enumerate(traces) if trace.timestamp is not None]
    traces_without_timestamp = [(i, trace) for i, trace in enumerate(traces) if trace.timestamp is None]

    if not traces_with_timestamp:
        # If no timestamps, return original list directly
        return traces

    # Extract timestamp list
    timestamps = [trace.timestamp for _, trace in traces_with_timestamp]

    # 1. Apply auto_offset: make first timestamp 0
    if auto_offset:
        min_timestamp = min(timestamps)
        if min_timestamp > 0:
            # Subtract minimum from all timestamps
            for _, trace in traces_with_timestamp:
                trace.timestamp = trace.timestamp - min_timestamp
            # Update timestamps list
            timestamps = [t - min_timestamp for t in timestamps]

    # 2. Apply start_offset and end_offset filtering
    filtered_indices = []
    for idx, (original_idx, trace) in enumerate(traces_with_timestamp):
        timestamp = trace.timestamp
        # Check if within range
        if timestamp < start_offset:
            continue
        if end_offset >= 0 and timestamp > end_offset:
            continue
        filtered_indices.append(original_idx)

    # 3. Build result list: keep filtered traces and traces without timestamps
    result_traces = []
    filtered_set = set(filtered_indices)
    without_timestamp_set = {i for i, _ in traces_without_timestamp}
    for i, trace in enumerate(traces):
        if i in filtered_set or i in without_timestamp_set:
            result_traces.append(trace)

    return result_traces


# ============================================================================
# 5. MooncakeTraceDataset
# ============================================================================


@LOAD_DATASET.register_module()
class MooncakeTraceDataset(BaseDataset):
    def load(
        self,
        path,
        model_path,
        random_seed=None,
        generated_prompts_path="",
        fixed_schedule_auto_offset=False,
        fixed_schedule_start_offset=0,
        fixed_schedule_end_offset=-1,
    ):
        """
        Load Mooncake trace dataset

        Args:
            path: Path to JSONL file containing hashid and trace data
            model_path: Model path for loading tokenizer
            random_seed: Random seed
            generated_prompts_path: Path to generated prompt cache file, will be reused if exists
            fixed_schedule_auto_offset: Whether to auto-offset timestamps (make first timestamp 0), default False
            fixed_schedule_start_offset: Start offset in milliseconds, default 0
            fixed_schedule_end_offset: End offset in milliseconds, -1 means no limit, default -1

        Returns:
            Dataset: Dataset containing prompt, timestamp, and max_out_len fields
        """
        # Parameter validation
        if (
            fixed_schedule_end_offset >= 0
            and fixed_schedule_start_offset > fixed_schedule_end_offset
        ):
            raise ParameterValueError(
                DSET_CODES.INVALID_PARAM_VALUE,
                f"fixed_schedule_start_offset ({fixed_schedule_start_offset}) must be <= "
                f"fixed_schedule_end_offset ({fixed_schedule_end_offset})",
            )

        path = get_data_path(path)
        self.logger.info(f"Loading mooncake trace dataset from: {path}")

        # Process generated_prompts_path: consider fixed_schedule parameters to generate unique cache filename
        if not generated_prompts_path:
            # Generate default cache file path: append _generated_prompts to original filename
            dir_name = os.path.dirname(path)
            base_name = os.path.basename(path)
            name_without_ext, ext = os.path.splitext(base_name)

            # If fixed_schedule parameters are used, include them in cache filename
            trace_data_file_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()

            cache_suffix = f"_{trace_data_file_md5}_generated_prompts"
            if (
                fixed_schedule_auto_offset
                or fixed_schedule_start_offset != 0
                or fixed_schedule_end_offset != -1
            ):
                schedule_params = []
                if fixed_schedule_auto_offset:
                    schedule_params.append("auto")
                if fixed_schedule_start_offset != 0:
                    schedule_params.append(f"start{fixed_schedule_start_offset}")
                if fixed_schedule_end_offset != -1:
                    schedule_params.append(f"end{fixed_schedule_end_offset}")
                cache_suffix += "_" + "_".join(schedule_params)

            generated_prompts_path = os.path.join(
                dir_name, f"{name_without_ext}{cache_suffix}{ext}"
            )
        else:
            generated_prompts_path = get_data_path(generated_prompts_path)

        self.logger.info(f"Generated prompts cache path: {generated_prompts_path}")

        # Check if cache file exists, if exists load directly
        if os.path.exists(generated_prompts_path):
            self.logger.warning(f"Found existing generated prompts file, loading from: {generated_prompts_path},"
                                f"if you want to regenerate the prompts, please delete the file and run again.")
            dataset_list = []
            with open(generated_prompts_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        dataset_list.append(json.loads(line.strip()))
            self.logger.info(f"Successfully loaded {len(dataset_list)} items from cache file")
            return Dataset.from_list(dataset_list)

        # If file doesn't exist, need to generate dataset
        self.logger.info(f"Cache file not found, generating prompts from source file")

        # 1. Initialize RNG system
        init_rng(random_seed)

        # 2. Load tokenizer
        self.logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = load_tokenizer(model_path)
        self.logger.info(f"Tokenizer loaded successfully, vocab_size: {tokenizer.vocab_size}")

        # 3. Load and tokenize corpus
        # Try to find corpus file from multiple possible locations
        corpus_path = None
        possible_paths = [
            Path(__file__).parent.parent.parent
            / "third_party/aiperf/assets/shakespeare.txt",
        ]

        for p in possible_paths:
            if p.exists():
                corpus_path = p
                break

        if corpus_path is None:
            # If not found, try to copy from aiperf or use absolute path
            # Here we use a fallback: if file not found, create a simple message
            raise AISBenchDataContentError(
                DSET_CODES.FILE_NOT_FOUND,
                f"Corpus file not found. Please ensure {DEFAULT_CORPUS_FILE} exists in "
                f"{[str(p) for p in possible_paths]}",
            )

        self.logger.info(f"Loading corpus from: {corpus_path}")
        tokenized_corpus = initialize_corpus(tokenizer, corpus_path)
        self.logger.info(f"Corpus loaded successfully, {len(tokenized_corpus)} tokens")

        # 4. Create PromptGenerator
        prompt_generator = PromptGenerator(tokenizer, tokenized_corpus, root_seed=random_seed)

        # 5. Load Mooncake trace data
        trace_data = load_mooncake_trace(path)
        self.logger.info(f"Loaded {len(trace_data)} traces from source file")

        # 6. Process timestamps (apply fixed_schedule parameters)
        if (
            fixed_schedule_auto_offset
            or fixed_schedule_start_offset != 0
            or fixed_schedule_end_offset != -1
        ):
            original_count = len(trace_data)
            trace_data = _process_timestamps(
                trace_data,
                auto_offset=fixed_schedule_auto_offset,
                start_offset=fixed_schedule_start_offset,
                end_offset=fixed_schedule_end_offset,
            )
            self.logger.info(
                f"Applied timestamp processing: {original_count} -> {len(trace_data)} traces "
                f"(auto_offset={fixed_schedule_auto_offset}, "
                f"start_offset={fixed_schedule_start_offset}, "
                f"end_offset={fixed_schedule_end_offset})"
            )

        # 7. Convert to prompts
        prompts = []
        input_text_count = 0
        generated_count = 0

        for trace in trace_data:
            # Check if input_text field exists
            if trace.input_text is not None:
                # Directly use input_text as prompt
                prompt = trace.input_text
                input_text_count += 1
            else:
                # Use PromptGenerator to generate prompt
                prompt = prompt_generator.generate(
                    mean=trace.input_length,
                    stddev=0,  # Mooncake trace usually doesn't use standard deviation
                    hash_ids=trace.hash_ids or [],
                )
                generated_count += 1

            item = {
                "prompt": prompt,
                "max_out_len": trace.output_length or 1,
                "answer": "mock_answer",
            }
            # If timestamp exists, add to result
            if trace.timestamp is not None:
                item["timestamp"] = trace.timestamp

            prompts.append(item)

        if input_text_count > 0:
            self.logger.info(f"Used input_text for {input_text_count} traces, generated prompts for {generated_count} traces")
        self.logger.info(f"Generated {len(prompts)} prompts, saving to cache file: {generated_prompts_path}")
        # Save to cache file
        prompts.sort(key=lambda x: x.get("timestamp", float("inf")))
        with open(generated_prompts_path, "w", encoding="utf-8") as f:
            for item in prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.logger.info(f"Successfully saved generated prompts to: {generated_prompts_path}")

        return Dataset.from_list(prompts)


class MooncakeTraceEvaluator(BaseEvaluator):
    pass
