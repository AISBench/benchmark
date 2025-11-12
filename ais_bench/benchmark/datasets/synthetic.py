from typing import Dict, List, Any, Tuple, Set, Iterable
import json
import os
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer

from datasets import Dataset
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging import get_logger
from ais_bench.benchmark.datasets.base import BaseDataset


@dataclass
class NumberRange:
    lower: tuple[int, float] = None
    upper: tuple[int, float] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True


def _check_keys_equal(got_keys, true_keys, comment):
    for key in got_keys:
        if key not in true_keys:
            raise ValueError(f"{key} is not a valid key for {comment}.")
    if got_keys != true_keys:
        raise ValueError(f"Expect keys {true_keys} for {comment}, but got keys {set(got_keys)}.")

def _ensure_keys_present(check_keys: Iterable, required_keys:Set, comment:str):
    if not required_keys.issubset(set(check_keys)):
            raise ValueError(f"Missing required key(s): {{{required_keys - set(check_keys)}}} for {comment}.")

def check_type(name: str, value: Any, types: Tuple):
    if not isinstance(value, types):
        raise ValueError(f"Parameter {name} should have type {types} for SyntheticConfig,",
                         f"but got {type(value).__name__}.")

def check_range(name: str, value: Any, param: NumberRange):
    lower, upper = param.lower, param.upper
    lower_inclusive, upper_inclusive = param.lower_inclusive, param.upper_inclusive
    # 构造区间的字符串表示
    lower_bound = '[' if lower_inclusive else '('
    upper_bound = ']' if upper_inclusive else ')'
    lower_str = str(lower) if lower is not None else '-inf'
    upper_str = str(upper) if upper is not None else '+inf'

    # 构造完整区间表示字符串
    interval_str = f"{lower_bound}{lower_str}, {upper_str}{upper_bound}"

    # 检查下限
    if lower is not None:
        lt = (lower_inclusive and value < lower)
        le = (not lower_inclusive and value <= lower)
        if le or lt:
            raise ValueError(f"Parameter {name} is {value}, not within the required range {interval_str}")
    # 检查上限
    if upper is not None:
        gt = (upper_inclusive and value > upper)
        ge = (not upper_inclusive and value >= upper)
        if gt or ge:
            raise ValueError(f"Parameter {name} is {value}, not within the required range {interval_str}")

def normalize_file_path(file_path:str) -> str:
    return os.path.abspath(os.path.expanduser(file_path))

@LOAD_DATASET.register_module()
class SyntheticDataset(BaseDataset):

    @staticmethod
    def check_synthetic_string_config(synthetic_config: Dict):
        input_str = "Input"
        output_str = "Output"
        _ensure_keys_present(synthetic_config.keys(), {input_str, output_str}, "SyntheticConfig")

        for key in (input_str, output_str):
            conf = synthetic_config.get(key)
            _check_keys_equal(conf.keys(), {"Method", "Params"}, 'SyntheticConfig["{key}"]')
            method = conf.get("Method")
            params = conf.get("Params")
            uniform_str = "uniform"
            gaussian_str = "gaussian"
            zipf_str = "zipf"
            min_value_str = "MinValue"
            max_value_str = "MaxValue"
            mean_str = "Mean"
            var_str = "Var"
            alpha_str = "Alpha"

            if method == uniform_str:
                _check_keys_equal(params.keys(), {min_value_str, max_value_str}, uniform_str)
            elif method == gaussian_str:
                _check_keys_equal(params.keys(), {mean_str, var_str, min_value_str, max_value_str}, gaussian_str)
            elif method == zipf_str:
                _check_keys_equal(params.keys(), {alpha_str, min_value_str, max_value_str}, zipf_str)
            else:
                raise ValueError(f'Method should be one of {{{uniform_str, gaussian_str, zipf_str}}}, '
                                 f'but got {method}.')
            min_float32_value = -3.0e38
            max_float32_value = 3.0e38
            for param_name, param_value in params.items():
                desc_name = key + " " + param_name
                if param_name in (min_value_str, max_value_str):
                    check_type(desc_name, param_value, types=(int,))
                    if key == input_str:
                        check_range(desc_name, param_value, NumberRange(1, 2**20))    # 2**20 = 1M
                    elif key == output_str:
                        check_range(desc_name, param_value, NumberRange(1, 2**20))
                elif param_name == mean_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(min_float32_value, max_float32_value))
                elif param_name == var_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(0, max_float32_value))
                elif param_name == alpha_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(1.0, 10.0, lower_inclusive=False))
            min_value = params.get(min_value_str)
            max_value = params.get(max_value_str)
            if min_value > max_value:
                raise ValueError(f'MinValue should less than MaxValue, '
                                 f'but got MinValue is {min_value}, and MaxValue is {max_value}.')

    @staticmethod
    def check_synthetic_tokenid_config(synthetic_config: Dict):
        request_size_key = "RequestSize"
        _ensure_keys_present(synthetic_config.keys(), {request_size_key}, "SyntheticConfig")

        request_size_value = synthetic_config.get(request_size_key)
        check_type(request_size_key, request_size_value, types=(int, ))
        check_range(request_size_key, request_size_value, NumberRange(1, 2**20))

    @staticmethod
    def _check_synthetic_config(synthetic_config: Dict):
        config_type_key = "Type"
        request_count_key = "RequestCount"

        required_keys = {config_type_key, request_count_key}
        _ensure_keys_present(synthetic_config.keys(), required_keys, "SyntheticConfig")

        config_type_value = synthetic_config.get(config_type_key)
        check_type(config_type_key, config_type_value, types=(str, ))
        config_type_value = config_type_value.lower()

        request_count_value = synthetic_config.get(request_count_key)
        check_type(request_count_key, request_count_value, types=(int, ))
        check_range(request_count_key, request_count_value, NumberRange(1, 2**20))

        check_map = {
            "string": SyntheticDataset.check_synthetic_string_config,
            "tokenid": SyntheticDataset.check_synthetic_tokenid_config
        }

        check_func = check_map.get(config_type_value)
        if not check_func:
            raise ValueError(f"Expect type should from {check_map.keys()}(case-insensitive) for SyntheticConfig,",
                             f"but got {config_type_value}.")

        type_config = {}
        type_config_key = "StringConfig" if config_type_value == "string" else "TokenIdConfig"
        _ensure_keys_present(synthetic_config.keys(), {type_config_key}, "SyntheticConfig")
        type_config = synthetic_config.get(type_config_key)

        check_func(type_config)

    @staticmethod
    def sample_one_value(method: str, params: dict) -> int:
        # Sample one value, the args have been checked before
        min_value = params["MinValue"]
        max_value = params["MaxValue"]
        if method == "uniform":
            value = np.random.uniform(min_value, max_value)
        elif method == "gaussian":
            mean = params["Mean"]
            stddev = np.sqrt(params["Var"])
            value = np.random.normal(mean, stddev)
            value = np.clip(value, min_value, max_value)
        elif method == "zipf":
            alpha = params["Alpha"]
            value = np.random.zipf(alpha)
            value = np.clip(value, min_value, max_value)
        else:
            raise ValueError(f"Unknown method: {method}, method should be one of {{uniform, gaussian, zipf}}.")
        return int(value)

    @staticmethod
    def read_line(self, line: List[int]) -> Dict:
        """Get a data dict according to line.

        Args:
            line (List[int]): Input line should be a list with 2 integral elements, it represents the number of input
                token and output token respectively.

        Returns:
            data (str): Constructed input with 'A'.
            num_expect_generated_tokens (int): max_tokens.
        """
        if not hasattr(line, '__len__') or len(line) != 2:
            raise ValueError("Input line should be a list with 2 integral elements.")
        default_str = "A"
        num_input_token, num_expect_generated_tokens = line
        data = " ".join([default_str] * num_input_token)
        return data, num_expect_generated_tokens

    @staticmethod
    def find_first_file_path(search_path: str, search_file: str) -> str:
        """
        Recursively find the first search_file file path in a directory tree (Linux compatible).

        Args:
            search_path (str): Path to the root directory for searching
            search_file (str): Full name of the expected file for searching

        Returns:
            str: Full path to the first found search_file file

        Raises:
            ValueError: If input path is invalid or not a directory
            FileNotFoundError: If no search_file file is found in the directory tree

        Note:
            Uses breadth-first search strategy for finding files
            Follows Linux filesystem case sensitivity rules
        """
        # Normalize path (expand ~ and resolve relative paths)
        normalized_path = normalize_file_path(search_path)

        # Validate input path
        if not os.path.exists(normalized_path):
            raise ValueError(f"Path does not exist: {normalized_path}")
        if not os.path.isdir(normalized_path):
            raise ValueError(f"Not a directory: {normalized_path}")

        # Implement BFS search using os.walk
        for root, dirs, files in os.walk(normalized_path):
            # Check current directory first before descending into subdirectories
            if search_file in files:
                # Return immediately when first match is found
                return os.path.join(root, search_file)

        # Handle case where no config file is found
        raise FileNotFoundError(f"No {search_file} found in directory tree: {normalized_path}")

    @staticmethod
    def generate_valid_random_ids(valid_indices, request_size: int) -> torch.Tensor:
        """
        Generates random integers in [0, vocab_size) excluding special IDs.

        Args:
            valid_indices: valid indices in tokenizer file
            request_size: Number of random values to generate

        Returns:
            Tensor of shape (request_size,) with dtype torch.int64
        """

        # Randomly select from valid indices
        rand_indices = torch.randint(0, len(valid_indices), (request_size,))

        return valid_indices[rand_indices].to(torch.int64)

    def load(self, config, **kwargs):
        self.logger = get_logger()
        dataset = []
        model_path_key = "ModelPath"
        config[model_path_key] = kwargs.get("model_path", None)
        self._check_synthetic_config(config)
        request_count = config.get("RequestCount")
        config_type = config.get("Type").lower()
        trust_remote_code = config.get("TrustRemoteCode")
        if config_type == "string":
            string_config = config.get("StringConfig")
            input_method = string_config["Input"]["Method"]
            input_params = string_config["Input"]["Params"]
            output_method = string_config["Output"]["Method"]
            output_params = string_config["Output"]["Params"]
            num_input_output_tokens = [[self.sample_one_value(input_method, input_params),
                                            self.sample_one_value(output_method, output_params)]
                                        for _ in range(request_count)]

            for num_input_output_token in tqdm(num_input_output_tokens, desc="Constructing synthetic string datasets ..."):
                data, max_tokens = self.read_line(self, num_input_output_token)
                dataset.append({"question": data, "answer": "aaa", "max_out_len": max_tokens})

        elif config_type == "tokenid":
            tokenid_config = config.get("TokenIdConfig")
            request_size = tokenid_config.get("RequestSize", None)
            model_path_value = config.get("ModelPath", None)

            check_type(model_path_key, model_path_value, types=(str, ))
            if not os.path.exists(normalize_file_path(model_path_value)):
                raise ValueError(f"ModelPath does not exist: {str(model_path_value)}")

            model_path_value = normalize_file_path(model_path_value)
            tokenizer_file_path = self.find_first_file_path(model_path_value, "tokenizer_config.json")

            tokenizer_model = AutoTokenizer.from_pretrained(
                os.path.dirname(tokenizer_file_path), 
                trust_remote_code=trust_remote_code
            )

            vocab_size = tokenizer_model.vocab_size
            vocab_size = tokenid_config.get("VocabSize", None) if not vocab_size else vocab_size # The vocab_size defined in the model has higher priority
            if not vocab_size:
                raise ValueError(f"The configuration vocab_size was not found in the dataset config file {model_path_value}",
                                 f"or tokenizer config file {tokenizer_file_path}")

            all_special_ids = tokenizer_model.all_special_ids
            self.logger.info(f"Current tokenizer model: {tokenizer_model.__class__.__name__}")
            self.logger.debug(f"Token id range: (0, {vocab_size}) excluding the values {all_special_ids}")

            # Create mask of valid IDs
            valid_ids = torch.ones(vocab_size, dtype=torch.bool)
            original_array = np.array(all_special_ids)
            filtered_array = original_array[original_array < vocab_size]
            valid_ids[filtered_array.tolist()] = False

            # Generate random indices for valid IDs
            valid_indices = torch.where(valid_ids)[0]

            for _ in tqdm(range(request_count), desc="Constructing synthetic tokenid datasets ..."):
                input_ids = self.generate_valid_random_ids(valid_indices, request_size)
                decode_str = tokenizer_model.decode(input_ids)
                dataset.append({"question":decode_str,"answer":"aaa"})

        else:
            raise ValueError(f"Invalid type:{config_type}. Should choose one from {{string, tokenid}}")

        return Dataset.from_list(dataset)