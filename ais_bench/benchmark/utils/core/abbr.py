import os.path as osp
from typing import Dict, List, Union

from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError


logger = AISLogger()


def model_abbr_from_cfg(cfg: Union[ConfigDict, List[ConfigDict]]) -> str:
    """Generate model abbreviation from the model's confg."""
    if isinstance(cfg, (list, tuple)):
        return "_".join(model_abbr_from_cfg(c) for c in cfg)
    if "abbr" in cfg:
        return cfg["abbr"]
    
    # Fallback: generate from type and path
    if "path" not in cfg:
        logger.warning(f"Model config missing both 'abbr' and 'path': {cfg.get('type', 'unknown')}")
    
    model_abbr = cfg["type"] + "_" + "_".join(osp.realpath(cfg["path"]).split("/")[-2:])
    model_abbr = model_abbr.replace("/", "_")
    logger.debug(f"Generated model abbr from path: {model_abbr}")
    return model_abbr


def dataset_abbr_from_cfg(cfg: ConfigDict) -> str:
    """Returns dataset abbreviation from the dataset's confg."""
    if "abbr" in cfg:
        return cfg["abbr"]
    dataset_abbr = cfg["path"]
    if "name" in cfg:
        dataset_abbr += "_" + cfg["name"]
    dataset_abbr = dataset_abbr.replace("/", "_")
    logger.debug(f"Generated dataset abbr from path: {dataset_abbr}")
    return dataset_abbr


def merged_dataset_abbr_from_class(cfg: ConfigDict) -> str:
    return cfg.get("type").split(".")[-1].lower()


def task_abbr_from_cfg(task: Dict) -> str:
    """Returns task abbreviation from the task's confg."""
    if len(task["datasets"][0]) > 1:
        name = f"{task['models'][0]['abbr']}/{task['datasets'][0][0].get('type').split('.')[-1].lower()}"
    else:
        name = f"{task['models'][0]['abbr']}/{task['datasets'][0][0]['abbr']}"
    logger.debug(f"Generated task abbr: {name}")
    return name

def merge_dataset_abbr_from_cfg(task: ConfigDict) -> str:
    """Returns task abbreviation from the task's confg."""
    if len(task["datasets"][0]) > 1:
        name = f"{task['datasets'][0][0].get('type').split('.')[-1].lower()}"
    else:
        name = f"{task['datasets'][0][0]['abbr']}"
    return name

def get_infer_output_path(
    model_cfg: ConfigDict,
    dataset_cfgs: List[ConfigDict] | ConfigDict,
    root_path: str = None,
    file_extension: str = "json",
) -> str:
    # change to raise exception
    if root_path is None:
        raise AISBenchRuntimeError(UTILS_CODES.ROOT_PATH_NOT_SET, "root_path is not set")
    model_abbr = model_abbr_from_cfg(model_cfg)
    if isinstance(dataset_cfgs, list):
        if len(dataset_cfgs) > 1:
            dataset_abbr = dataset_cfgs[0].get('type').split('.')[-1].lower()
        else:
            dataset_abbr = dataset_cfgs[0].get('abbr')
    else:
        dataset_abbr = dataset_abbr_from_cfg(dataset_cfgs)
    
    output_path = osp.join(root_path, model_abbr, f"{dataset_abbr}.{file_extension}")
    logger.debug(f"Generated output path: {output_path}")
    return output_path


def deal_with_judge_model_abbr(model_cfg, judge_model_cfg, meta=False):
    if isinstance(model_cfg, ConfigDict):
        model_cfg = (model_cfg,)
    if meta:
        for m_cfg in model_cfg:
            if "summarized-by--" in m_cfg["abbr"]:
                return model_cfg
        suffix_abbr = "summarized-by--" + model_abbr_from_cfg(judge_model_cfg)
        logger.debug(f"Adding meta judge suffix: {suffix_abbr}")
        model_cfg += ({"abbr": suffix_abbr},)
    else:
        for m_cfg in model_cfg:
            if "judged-by--" in m_cfg["abbr"]:
                return model_cfg
        suffix_abbr = "judged-by--" + model_abbr_from_cfg(judge_model_cfg)
        logger.debug(f"Adding judge suffix: {suffix_abbr}")
        model_cfg += ({"abbr": suffix_abbr},)
    return model_cfg
