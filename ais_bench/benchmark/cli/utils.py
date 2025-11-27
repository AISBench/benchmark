import sys
import os
import argparse
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


def is_running_in_background():
    # check whether stdin and stdout are connected to TTY
    stdin_is_tty = sys.stdin.isatty()
    stdout_is_tty = sys.stdout.isatty()

    # if stdin and stdout are not connected to TTY, the script is running in background
    return not (stdin_is_tty and stdout_is_tty)


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

def validate_max_workers(value):
    """Validate and normalize the max_num_workers parameter (used for argparse type parameter)"""
    try:
        max_num_workers = int(value)
    except (ValueError, TypeError):
        logger.warning(f"`max_num_workers` must be an integer, but got {value}, setting to default value 1")
        return 1
    
    # Check if it is less than 1
    if max_num_workers < 1:
        logger.warning(f"`max_num_workers` must be greater than or equal to 1, but got {max_num_workers}, setting to default value 1")
        return 1
    
    # Check if it is greater than the maximum value
    if max_num_workers > MAX_NUM_WORKERS:
        logger.warning(f"`max_num_workers` is more than 0.8 * total_cpu_count ({MAX_NUM_WORKERS}), setting to {MAX_NUM_WORKERS}")
        return MAX_NUM_WORKERS
    
    return max_num_workers


def validate_max_workers_per_gpu(value):
    """Validate and normalize the max_workers_per_gpu parameter (used for argparse type parameter)"""
    try:
        max_workers_per_gpu = int(value)
    except (ValueError, TypeError):
        logger.warning(f"`max_workers_per_gpu` must be an integer, but got {value}, setting to default value 1")
        return 1
    
    # Check if it is less than 1
    if max_workers_per_gpu < 1:
        logger.warning(f"`max_workers_per_gpu` must be greater than or equal to 1, but got {max_workers_per_gpu}, setting to default value 1")
        return 1
    
    return max_workers_per_gpu


def validate_num_prompts(value):
    """Validate and normalize the num_prompts parameter (used for argparse type parameter)"""
    # argparse type function only receives string values when user provides the argument
    # None is handled by default parameter, so we don't need to handle None here
    try:
        num_prompts = int(value)
    except (ValueError, TypeError):
        logger.warning(f"`num_prompts` must be an integer, but got {value}, setting to default value None")
        return None
    # Check if it is less than 1
    if num_prompts < 1:
        logger.warning(f"`num_prompts` must be greater than or equal to 1, but got {num_prompts}, setting to default value None")
        return None
    return num_prompts


def validate_num_warmups(value):
    """Validate and normalize the num_warmups parameter (used for argparse type parameter)"""
    try:
        num_warmups = int(value)
    except (ValueError, TypeError):
        logger.warning(f"`num_warmups` must be an integer, but got {value}, setting to default value 1")
        return 1
    
    # Check if it is less than 0
    if num_warmups < 0:
        logger.warning(f"`num_warmups` must be greater than or equal to 0, but got {num_warmups}, setting to default value 1")
        return 1
    
    return num_warmups


def validate_pressure_time(value):
    """Validate and normalize the pressure_time parameter (used for argparse type parameter)"""
    try:
        pressure_time = int(value)
    except (ValueError, TypeError):
        logger.warning(f"`pressure_time` must be an integer, but got {value}, setting to default value 15s")
        return DEFAULT_PRESSURE_TIME
    
    # Check if it is less than 1
    if pressure_time < 1:
        logger.warning(f"`pressure_time` must be greater than or equal to 1, but got {pressure_time}, setting to default value 15s")
        return DEFAULT_PRESSURE_TIME
    if pressure_time > MAX_PRESSURE_TIME:
        logger.warning(f"`pressure_time` is more than 24 hours, setting to {MAX_PRESSURE_TIME}")
        return MAX_PRESSURE_TIME
    return pressure_time