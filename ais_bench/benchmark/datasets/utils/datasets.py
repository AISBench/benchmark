import os
import random
import json

from ais_bench.benchmark.utils.logging.logging import get_logger

logger = get_logger()
# These datasets can only be used to evaluate performance.
ONLY_PERF_DATASETS = [
    "ais_bench.benchmark.datasets.MTBenchDataset",
    "ais_bench.benchmark.datasets.ShareGPTDataset",
    "ais_bench.benchmark.datasets.SyntheticDataset",
]
# Multimodal datasets.
MM_DATASETS = [
    "ais_bench.benchmark.datasets.TEXTVQADataset",
    "ais_bench.benchmark.datasets.VideoBenchDataset",
    "ais_bench.benchmark.datasets.VocalSoundDataset",
]
# Multimodal APIs.
MM_APIS = ["ais_bench.benchmark.models.VLLMCustomAPIChat"]

def get_cache_dir(default_dir):
    # TODO Add any necessary supplementary information for here
    return os.environ.get('AIS_BENCH_DATASETS_CACHE', default_dir)


def get_data_path(dataset_path: str, local_mode: bool = True):
    """return dataset id when getting data from ModelScope/HuggingFace repo, otherwise just
    return local path as is.

    Args:
        dataset_path (str): data path
        local_mode (bool): whether to use local path or
            ModelScope/HuggignFace repo
    """
    # update the path with CACHE_DIR
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../") # site-package
    cache_dir = get_cache_dir(default_dir)

    # For absolute path customized by the users, will not auto download dataset
    if dataset_path.startswith('/'):
        return dataset_path

    # For relative path, with CACHE_DIR
    if local_mode:
        local_path = os.path.join(cache_dir, dataset_path)

        if not os.path.exists(local_path):
            readme_path = os.path.join(default_dir, "README.md")
            raise FileExistsError(f"Dataset path: {local_path} is not exist! " +
                                  "Please check section \"--datasets支持的数据集\" of " +
                                  f"{readme_path} to check how to prepare supported datasets.")
        else:
            return local_path
    else:
        raise TypeError('Customized dataset path type is not a absolute path!')


def get_sample_data(data_list: list, sample_mode: str = "default", request_count: int = 0):
    if not request_count:
        logger.info("If u do not provide 'request_count' when using custom-dataset sampling feature, "
                       "we will sample all available data by default.")
        sample_index = len(data_list)
    elif request_count > len(data_list):
        repeat_times = (request_count // len(data_list)) + (1 if request_count % len(data_list) != 0 else 0)
        data_list = (data_list * repeat_times)[:request_count]
        sample_index = request_count
    elif request_count < 0:
        raise ValueError("The 'request_count' is negative, we only support positive integer.")
    else:
        sample_index = request_count
    # sampling data
    if sample_mode == "default":
        return data_list[:sample_index]
    elif sample_mode == "random":
        return random.sample(data_list, sample_index)
    elif sample_mode == "shuffle":
        shuffle_data = data_list[:sample_index]
        random.shuffle(shuffle_data)
        return shuffle_data
    else:
        raise ValueError(f"Sample mode: {sample_mode} is not supported!")
    
def get_meta_json(dataset_path, meta_path):
    ori_meta_path = meta_path
    if not meta_path:
        meta_path = dataset_path + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_json_conf = json.load(f)
    else:
        if ori_meta_path:
            # user set meta_path does not exists
            raise ValueError(f'The file path specified by parameter "meta_path" does not exist: {ori_meta_path}')
        meta_json_conf = {}
    return meta_json_conf