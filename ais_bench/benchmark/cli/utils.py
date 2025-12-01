import sys
import os
from datetime import datetime

from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

DATASETS_NEED_MODELS = ["ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
                      "ais_bench.benchmark.datasets.sharegpt.ShareGPTDataset"]
MAX_NUM_WORKERS = int(os.cpu_count() * 0.8)
DEFAULT_PRESSURE_TIME = 15
MAX_PRESSURE_TIME = 60 * 60 * 24 # 24 hours

logger = AISLogger()

def get_config_type(obj) -> str:
    if isinstance(obj, str):
        return obj
    return f"{obj.__module__}.{obj.__name__}"


def get_current_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fill_model_path_if_datasets_need(model_cfg, dataset_cfg):
    data_type = get_config_type(dataset_cfg.get("type"))
    if data_type in DATASETS_NEED_MODELS:
        model_path = model_cfg.get("path")
        if not model_path:
            raise AISBenchConfigError(
                UTILS_CODES.SYNTHETIC_DS_MISS_REQUIRED_PARAM,
                "[path] in model config is required for synthetic(tokenid) and sharegpt dataset."
            )
        dataset_cfg.update({"model_path": model_path})

def fill_test_range_use_num_prompts(num_prompts: int, dataset_cfg: dict):
    if not num_prompts:
        return
    reader_cfg = dataset_cfg["reader_cfg"]
    if "test_range" in reader_cfg:
        if isinstance(num_prompts, int):
            logger.warning("`test_range` has been set, `--num-prompts` will be ignored")
        return
    reader_cfg["test_range"] = f"[:{str(num_prompts)}]"
    logger.info(f"Keeping the first {num_prompts} prompts for dataset [{dataset_cfg.get('abbr')}]")

def create_int_validator(
    param_name: str,
    min_value: int = None,
    max_value: int = None,
    allow_none: bool = False,
    error_message_suffix: str = None,
):
    """Create a validator function for integer command-line arguments.

    Args:
        param_name: Name of the parameter (used in error messages)
        min_value: Minimum allowed value (inclusive). If None, no minimum check.
        max_value: Maximum allowed value (inclusive). If None, no maximum check.
        allow_none: If True, allows None values. If False, raises error on None.
        error_message_suffix: Optional suffix to append to error messages for additional context.

    Returns:
        A validator function that can be used as argparse type parameter.

    Example:
        >>> validator = create_int_validator("max_workers", min_value=1, max_value=10)
        >>> validator("5")  # Returns 5
        >>> validator("0")  # Raises ArgumentTypeError
        >>> validator("abc")  # Raises ArgumentTypeError
    """
    def validator(value):
        # Handle None values
        if allow_none and (value is None or str(value).lower() == 'none'):
            return None

        # Convert to integer
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            error_msg = f"`{param_name}` must be an integer, but got {value!r}"
            if error_message_suffix:
                error_msg += f" {error_message_suffix}"
            raise AISBenchConfigError(UTILS_CODES.INVALID_INTEGER_TYPE, error_msg)
        # Check minimum value
        if min_value is not None and int_value < min_value:
            error_msg = f"`{param_name}` must be >= {min_value}, but got {int_value}"
            if error_message_suffix:
                error_msg += f" {error_message_suffix}"
            raise AISBenchConfigError(UTILS_CODES.ARGUMENT_TOO_SMALL, error_msg)

        # Check maximum value
        if max_value is not None and int_value > max_value:
            error_msg = f"`{param_name}` must be <= {max_value}, but got {int_value}"
            if error_message_suffix:
                error_msg += f" {error_message_suffix}"
            raise AISBenchConfigError(UTILS_CODES.ARGUMENT_TOO_LARGE, error_msg)

        return int_value

    return validator


# Create specific validators using the factory function
validate_max_workers = create_int_validator(
    param_name="--max-num-workers",
    min_value=1,
    max_value=MAX_NUM_WORKERS,
    error_message_suffix=f"(maximum recommended: {MAX_NUM_WORKERS}, which is 0.8 * total_cpu_count)"
)

validate_max_workers_per_gpu = create_int_validator(
    param_name="--max-workers-per-gpu",
    min_value=1,
)

validate_num_prompts = create_int_validator(
    param_name="--num-prompts",
    min_value=1,
    allow_none=True,
)

validate_num_warmups = create_int_validator(
    param_name="--num-warmups",
    min_value=0,
)

validate_pressure_time = create_int_validator(
    param_name="--pressure-time",
    min_value=1,
    max_value=MAX_PRESSURE_TIME,
    error_message_suffix=f"(maximum: {MAX_PRESSURE_TIME} seconds, which is 24 hours)"
)