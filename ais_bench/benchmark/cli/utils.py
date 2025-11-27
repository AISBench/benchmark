import sys
from datetime import datetime

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import ConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

DATASETS_NEED_MODELS = ["ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
                      "ais_bench.benchmark.datasets.sharegpt.ShareGPTDataset"]

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
            raise ConfigError(
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