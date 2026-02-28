import os
import math
import copy
import argparse
import json
import csv
import tabulate
import shutil
from tqdm import tqdm

from datasets import Dataset, load_from_disk

from ais_bench.benchmark.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_multi_reasoning_4k import language
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
from ais_bench.benchmark.cli.config_manager import CustomConfigChecker
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.logger import AISLogger
from mmengine.config import Config

logger = AISLogger(__name__)

def load_gedit_dataset(path):
    path = get_data_path(path)
    return load_from_disk(path)


def load_config(config_path: str) -> Config:
    """加载配置文件并进行校验"""
    if not os.path.exists(config_path):
        raise ParameterValueError(f"Config path: {config_path} is not exist!")
    try:
        config = Config.fromfile(config_path, format_python_code=False)
    except BaseException as e:
        raise RuntimeError(f"Fail to load config {config_path}, failed reason: {e}")
    CustomConfigChecker(config, config_path).check()
    return config

class GEditPredsParser:
    def __init__(self, args):
        self.config = load_config(args.config_path)
        self.output_dir = args.timestamp_path
        dataset = load_gedit_dataset(args.dataset_path)
        # 将Dataset转换为字典以提高访问速度
        self.dataset = {}
        for i in range(len(dataset)):
            item = dataset[i]
            self.dataset[item["id"]] = item
        self.paths_map = dict(
            org_pred_path = [],
        )
        for comb in self.config["model_dataset_combinations"]:
            model_abbr = comb["models"][0]["abbr"]
            dataset_org_abbr = comb["datasets"][0]["abbr"]
            self.paths_map["org_pred_path"].append(os.path.join(self.output_dir, "predictions", model_abbr, f"{dataset_org_abbr}.jsonl"))

    def parse_results(self):
        logger.info(f"Start parse infer result from: {self.output_dir}")
        org_pred_data_list = self._load_and_merge_jsonl("org_pred_path")
        org_pred_data_dict = {item["uuid"]: item for item in org_pred_data_list}

        self.all_data_results = {}

        for uuid in tqdm(org_pred_data_dict.keys(), desc="Parsing results"):
            id = org_pred_data_dict[uuid]["id"]
            output_img_path = org_pred_data_dict[uuid]["prediction"]
            self.all_data_results[id] = {
                "key": self.dataset[id]["key"],
                "task_type": self.dataset[id]["task_type"],
                "instruction_language": self.dataset[id]["instruction_language"],
                "output_img_path": output_img_path,
            }

        logger.info(f"Finish parsing results")

    def dump_gedit_format_result(self):
        save_path = os.path.join(self.output_dir, "results", "fullset")
        logger.info(f"Start dumping gedit format result ......")
        for id, item in tqdm(self.all_data_results.items(), desc="Dumping gedit format results"):
            dump_dir = os.path.join(save_path, item["task_type"], item["instruction_language"])
            os.makedirs(dump_dir, exist_ok=True)
            # 将output_img_path copy到dump_dir
            shutil.copy(item['output_img_path'], os.path.join(dump_dir, item['key'] + '.png'))
        logger.info(f"Finish dumping gedit format result ......")

    def _load_and_merge_jsonl(self, path_kind: "org_pred_path"):
        merged_data = []
        start_index = 0
        for path in self.paths_map[path_kind]:
            offset_index = copy.deepcopy(start_index)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    data["prediction"] = os.path.join(os.path.dirname(path), data["prediction"])
                    data["id"] = data["id"] + offset_index
                    start_index += 1
                    merged_data.append(data)
        return merged_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="显示gedit数据集的推理结果")
    parser.add_argument("--config_path", default="./multi_device_run_qwen_image_edit.py", help="配置文件路径")
    parser.add_argument("--timestamp_path", help="结果时间戳路径")
    parser.add_argument("--dataset_path", default="ais_bench/datasets/GEdit-Bench")

    args = parser.parse_args()
    eval_parser = GEditPredsParser(args)
    eval_parser.parse_results()
    eval_parser.dump_gedit_format_result()


if __name__ == "__main__":
    main()
