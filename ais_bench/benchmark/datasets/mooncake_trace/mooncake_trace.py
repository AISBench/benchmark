import json
import os
import random

from tqdm import tqdm
from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils.file.load_tokenizer import load_tokenizer
from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class MooncakeTraceDataset(BaseDataset):
    # Global shared hash_id cache: maps hash_id -> list of 512 token IDs
    _global_hash_id_cache = {}

    def prompts_generator(
        self,
        tokenizer,
        token_length,
        hash_ids,
        block_size=512,
        hash_id_cache=None,
        prefix_ratio=0,
        prefix_pool=None,
    ):
        """Generate a prompt based on hash_ids and token_length.

        The generated prompt will have exactly token_length tokens when encoded by the tokenizer.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding
            token_length: Total number of tokens required
            hash_ids: List of hash IDs, each representing a block of tokens
            block_size: Number of tokens per hash block (default 512)
            hash_id_cache: Dictionary to cache prompts by hash_id
            prefix_ratio: Ratio of prefix tokens when hash_ids is empty (default 0)
            prefix_pool: List of token IDs to use as prefix pool (default None)

        Returns:
            str: Generated prompt text that encodes to exactly token_length tokens
        """
        # Use global shared cache if no cache is provided
        if hash_id_cache is None:
            hash_id_cache = self._global_hash_id_cache

        if not hash_ids:
            # If no hash_ids, generate a prompt with prefix_ratio prefix from pool
            return self._generate_prompt_with_prefix(
                tokenizer, token_length, prefix_ratio, prefix_pool
            )

        # Calculate the final block size
        final_block_size = token_length - ((len(hash_ids) - 1) * block_size)
        if final_block_size <= 0 or block_size < final_block_size:
            raise ValueError(
                f"Input length: {token_length}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. The final hash block size: {final_block_size} must be "
                f"greater than 0 and less than or equal to {block_size}."
            )

        # First, collect all token IDs for each hash_id block
        # Each hash_id represents a fixed sequence of 512 token IDs
        # The same hash_id always generates the same 512 token sequence
        final_prompt_tokens = []
        current_block_size = block_size
        for index, hash_id in enumerate(hash_ids):
            # Check if this hash_id is already in cache
            if hash_id not in hash_id_cache:
                if index == len(hash_ids) - 1:
                    current_block_size = final_block_size
                # Generate 512 tokens for this hash_id (standard block size)
                # Use a deterministic seed based on hash_id for reproducibility
                random.seed(hash_id)
                token_ids = [
                    random.randint(0, tokenizer.vocab_size - 1)
                    for _ in range(current_block_size)  # Always generate 512 tokens
                ]
                # Immediately write token IDs to cache
                hash_id_cache[hash_id] = token_ids
                random.seed()  # Reset seed

            # Get the cached token IDs for this hash_id (always 512 tokens)
            cached_token_ids = hash_id_cache[hash_id]
            final_prompt_tokens.extend(cached_token_ids)

        # Decode all token IDs to prompt text at once
        combined_prompt = tokenizer.decode(final_prompt_tokens, skip_special_tokens=False)

        # Final adjustment to ensure exact token length (accounting for potential token merging at boundaries)
        # return self._adjust_prompt_length(tokenizer, combined_prompt, token_length)
        return combined_prompt

    def _generate_prompt_with_prefix(
        self, tokenizer, token_length, prefix_ratio=0, prefix_pool=None
    ):
        """Generate a prompt with prefix from token ID pool and remaining tokens.

        Args:
            tokenizer: The tokenizer to use
            token_length: Total number of tokens required
            prefix_ratio: Ratio of prefix tokens (0-1)
            prefix_pool: List of token IDs to use as prefix pool (None if not available)

        Returns:
            str: Generated prompt text that encodes to exactly token_length tokens
        """
        # Calculate prefix token length
        prefix_token_length = (
            int(token_length * prefix_ratio) if prefix_ratio > 0 else 0
        )
        remaining_token_length = token_length - prefix_token_length

        # Collect all token IDs first
        all_token_ids = []

        # Select prefix token IDs from pool
        if (
            prefix_token_length > 0
            and prefix_pool is not None
            and len(prefix_pool) >= prefix_token_length
        ):
            # Select prefix_token_length tokens from prefix_pool
            # Use deterministic selection based on token_length for reproducibility
            selected_prefix_token_ids = prefix_pool[:prefix_token_length]
            all_token_ids.extend(selected_prefix_token_ids)
        elif prefix_token_length > 0:
            # Fallback: generate prefix token IDs if pool is not available or too small
            # Use a deterministic seed based on token_length for reproducibility
            random.seed(token_length)
            prefix_token_ids = [
                random.randint(0, tokenizer.vocab_size - 1)
                for _ in range(prefix_token_length)
            ]
            all_token_ids.extend(prefix_token_ids)
            random.seed()  # Reset seed
        # Generate remaining token IDs
        if remaining_token_length > 0:
            # Use a deterministic seed based on token_length and prefix_ratio for reproducibility
            random.seed(token_length * 1000 + int(prefix_ratio * 1000))
            remaining_token_ids = [
                random.randint(0, tokenizer.vocab_size - 1)
                for _ in range(remaining_token_length)
            ]
            all_token_ids.extend(remaining_token_ids)
            random.seed()  # Reset seed

        # Decode all token IDs to prompt text at once
        combined_prompt = tokenizer.decode(all_token_ids, skip_special_tokens=False)

        # Final adjustment to ensure exact token length
        return self._adjust_prompt_length(tokenizer, combined_prompt, token_length)

    def _generate_remaining_tokens(self, tokenizer, token_length):
        """Generate text that encodes to approximately token_length tokens."""
        # Start with a simple pattern
        base_text = "A " * max(1, token_length // 2)
        return self._adjust_prompt_length(tokenizer, base_text, token_length)

    def _adjust_prompt_length(self, tokenizer, prompt, target_token_length):
        """Adjust prompt text to ensure it encodes to exactly target_token_length tokens.

        Args:
            tokenizer: The tokenizer to use
            prompt: Initial prompt text
            target_token_length: Desired number of tokens

        Returns:
            str: Adjusted prompt text that encodes to exactly target_token_length tokens
        """
        # Encode to check current token count
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        current_length = len(encoded)

        if current_length == target_token_length:
            return prompt

        # If we need more tokens, append characters
        if current_length < target_token_length:
            # Find a character that adds tokens when appended
            padding_chars = [" ", "A", "B", "C", "D", "E"]
            padding_char = " "

            for char in padding_chars:
                test_text = prompt + char
                test_encoded = tokenizer.encode(test_text, add_special_tokens=False)
                if len(test_encoded) > current_length:
                    padding_char = char
                    break

            # Add padding until we reach target length
            max_iterations = target_token_length * 2  # Safety limit
            iteration = 0
            while current_length < target_token_length and iteration < max_iterations:
                prompt += padding_char
                encoded = tokenizer.encode(prompt, add_special_tokens=False)
                new_length = len(encoded)

                # If adding a character doesn't increase token count, try a different approach
                if new_length == current_length:
                    # Try adding a word instead
                    prompt += " word"
                    encoded = tokenizer.encode(prompt, add_special_tokens=False)
                    new_length = len(encoded)

                current_length = new_length
                iteration += 1

                if current_length >= target_token_length:
                    break

        # If we have too many tokens, use binary search to find the right length
        if current_length > target_token_length:
            # Binary search for the right text length
            left, right = 0, len(prompt)
            best_prompt = prompt
            best_length_diff = abs(current_length - target_token_length)

            # Limit binary search iterations
            max_binary_iterations = 50
            binary_iteration = 0

            while left < right and binary_iteration < max_binary_iterations:
                mid = (left + right) // 2
                if mid == 0:
                    break

                test_prompt = prompt[:mid]
                test_encoded = tokenizer.encode(test_prompt, add_special_tokens=False)
                test_length = len(test_encoded)

                if test_length == target_token_length:
                    return test_prompt
                elif test_length < target_token_length:
                    length_diff = target_token_length - test_length
                    if length_diff < best_length_diff:
                        best_prompt = test_prompt
                        best_length_diff = length_diff
                    left = mid + 1
                else:
                    right = mid

                binary_iteration += 1

            # Fine-tune from the best result
            prompt = best_prompt
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            current_length = len(encoded)

            # Fine-tune by adding/removing characters one by one
            max_fine_tune = 100  # Safety limit
            fine_tune_iter = 0

            while (
                current_length != target_token_length and fine_tune_iter < max_fine_tune
            ):
                if current_length < target_token_length:
                    # Try adding different characters
                    for char in [" ", "A", "B", "C"]:
                        test_prompt = prompt + char
                        test_encoded = tokenizer.encode(
                            test_prompt, add_special_tokens=False
                        )
                        if len(test_encoded) == target_token_length:
                            return test_prompt
                        elif len(test_encoded) > target_token_length:
                            break
                    # If no single char works, add a space and continue
                    prompt += " "
                else:
                    # Remove one character from end
                    if len(prompt) > 0:
                        prompt = prompt[:-1]
                    else:
                        break

                encoded = tokenizer.encode(prompt, add_special_tokens=False)
                current_length = len(encoded)
                fine_tune_iter += 1

        return prompt

    def load(self, path, model_path, prefix_ratio=0, generated_prompts_path=""):
        path = get_data_path(path)
        self.logger.info(f"Loading mooncake trace dataset from: {path}")

        # 生成落盘文件路径：在原文件名后加上_generated_prompts
        # 例如：abc.jsonl -> abc_generated_prompts.jsonl
        # 如果用户提供了generated_prompts_path，则使用用户提供的路径
        if not generated_prompts_path:
            dir_name = os.path.dirname(path)
            base_name = os.path.basename(path)
            name_without_ext, ext = os.path.splitext(base_name)
            generated_prompts_path = os.path.join(
                dir_name, f"{name_without_ext}_generated_prompts{ext}"
            )
        else:
            generated_prompts_path = get_data_path(generated_prompts_path)

        self.logger.info(f"Generated prompts cache path: {generated_prompts_path}")

        # 检查落盘文件是否存在，如果存在则直接加载
        if os.path.exists(generated_prompts_path):
            self.logger.info(f"Found existing generated prompts file, loading from: {generated_prompts_path}")
            dataset_list = []
            with open(generated_prompts_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        dataset_list.append(json.loads(line.strip()))
            self.logger.info(f"Successfully loaded {len(dataset_list)} items from cache file")
            return Dataset.from_list(dataset_list)

        # 如果文件不存在，需要生成dataset
        self.logger.info(f"Cache file not found, generating prompts from source file")
        self.logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = load_tokenizer(model_path)
        self.logger.info(f"Tokenizer loaded successfully, vocab_size: {tokenizer.vocab_size}")

        json_data = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                json_data.append(data)
        json_data.sort(key=lambda x: x.get("timestamp", 0))
        self.logger.info(f"Loaded {len(json_data)} items from source file")

        # First pass: find maximum input_length
        max_input_length = 0
        for data in json_data:
            input_length = data.get("input_length", 0)
            if input_length > max_input_length:
                max_input_length = input_length
        self.logger.info(f"Maximum input length: {max_input_length}")

        # Generate prefix pool: a list of token IDs with length = max_input_length
        prefix_pool = None
        if max_input_length > 0 and prefix_ratio > 0:
            self.logger.info(f"Generating prefix pool with ratio: {prefix_ratio}, size: {max_input_length}")
            # Use a fixed seed for reproducibility
            random.seed(42)
            prefix_pool = [
                random.randint(0, tokenizer.vocab_size - 1)
                for _ in range(max_input_length)
            ]
            random.seed()  # Reset seed
            self.logger.info(f"Prefix pool generated successfully")

        self.logger.info(f"Starting to generate prompts for {len(json_data)} items")
        dataset = []
        for data in tqdm(json_data, desc="Loading mooncake trace dataset ..."):
            input_length = data.get("input_length", 0)
            output_length = data.get("output_length", 0)
            hash_ids = data.get("hash_ids", [])

            # Generate prompt using hash_ids
            # hash_id_cache will use global shared cache automatically
            prompt = self.prompts_generator(
                tokenizer=tokenizer,
                token_length=input_length,
                hash_ids=hash_ids,
                block_size=512,
                hash_id_cache=None,  # None means use global shared cache
                prefix_ratio=prefix_ratio,
                prefix_pool=prefix_pool,
            )

            item = {
                "prompt": prompt,
                "answer": "",  # Empty answer as placeholder
                "max_out_len": output_length,
            }
            # 如果有timestamp，添加到结果中
            if "timestamp" in data:
                item["timestamp"] = data["timestamp"]
            dataset.append(item)

        self.logger.info(f"Generated {len(dataset)} prompts, saving to cache file: {generated_prompts_path}")
        dataset_obj = Dataset.from_list(dataset)
        with open(generated_prompts_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.logger.info(f"Successfully saved generated prompts to: {generated_prompts_path}")

        return dataset_obj


class MooncakeTraceEvaluator(BaseEvaluator):
    pass
