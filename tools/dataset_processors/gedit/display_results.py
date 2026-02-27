import os
import math
import copy
import argparse
import json
import csv
import tabulate

from ais_bench.benchmark.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_multi_reasoning_4k import language
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
from ais_bench.benchmark.cli.config_manager import CustomConfigChecker
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.datasets.utils.lmm_judge import get_lmm_point_list
from mmengine.config import Config

logger = AISLogger(__name__)


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

class GEditEvalResultParser:
    def __init__(self, args):
        self.config = load_config(args.config_path)
        self.output_dir = args.timestamp_path
        self.paths_map = dict(
            org_pred_path = [],
            sc_judge_pred_path = [],
            pq_judge_pred_path = [],
        )
        for comb in self.config["model_dataset_combinations"]:
            model_abbr = comb["models"][0]["abbr"]
            dataset_org_abbr = comb["datasets"][0]["abbr"]
            dataset_sc_abbr = f'{comb["datasets"][0]["abbr"]}-{comb["datasets"][0]["judge_infer_cfg"]["judge_model"]["abbr"]}'
            dataset_pq_abbr = f'{comb["datasets"][1]["abbr"]}-{comb["datasets"][1]["judge_infer_cfg"]["judge_model"]["abbr"]}'
            self.paths_map["org_pred_path"].append(os.path.join(self.output_dir, "predictions", model_abbr, f"{dataset_org_abbr}.jsonl"))
            self.paths_map["sc_judge_pred_path"].append(os.path.join(self.output_dir, "predictions", model_abbr, f"{dataset_sc_abbr}.jsonl"))
            self.paths_map["pq_judge_pred_path"].append(os.path.join(self.output_dir, "predictions", model_abbr, f"{dataset_pq_abbr}.jsonl"))

    def parse_results(self):
        logger.info(f"Start parse judge infer result from: {self.output_dir}")
        org_pred_data_list = self._load_and_merge_jsonl("org_pred_path")
        org_pred_data_dict = {item["uuid"]: item for item in org_pred_data_list}
        sc_judge_pred_data_list = self._load_and_merge_jsonl("sc_judge_pred_path")
        sc_judge_pred_data_dict = {item["gold"]: item for item in sc_judge_pred_data_list}
        pq_judge_pred_data_list = self._load_and_merge_jsonl("pq_judge_pred_path")
        pq_judge_pred_data_dict = {item["gold"]: item for item in pq_judge_pred_data_list}

        self.all_data_results = {}

        for uuid in org_pred_data_dict.keys():
            id = org_pred_data_dict[uuid]["id"]
            output_img_path = org_pred_data_dict[uuid]["prediction"]
            sc_point = self._calc_meta_point(sc_judge_pred_data_dict[uuid]["prediction"])
            pq_point = self._calc_meta_point(pq_judge_pred_data_dict[uuid]["prediction"])
            o_point = math.sqrt(sc_point * pq_point)
            question, case_language = self._get_question_and_language(org_pred_data_dict[uuid]["origin_prompt"][0]["prompt"])
            self.all_data_results[id] = {
                "uuid": uuid,
                "question": question,
                "language": case_language,
                "output_img_path": output_img_path,
                "SC_point": sc_point,
                "PQ_point": pq_point,
                "O_point": o_point,
            }
        logger.info(f"Finish parsing results")

    def dump_result_csv(self):
        save_path = os.path.join(self.output_dir, "results")
        logger.info(f"Start dumping details ......")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "gedit_gathered_result.csv"), "w", encoding="utf-8", newline="") as f:
            fieldnames = ["id", "uuid", "question", "language", "output_img_path", "SC_point", "PQ_point", "O_point"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for id in self.all_data_results.keys():
                writer.writerow(self.all_data_results[id])
        logger.info(f"Finish dumping csv to: {os.path.join(save_path, 'gedit_gathered_result.csv')}")

    def display_results(self):
        evaluate_result_list = []

        lang_count = {"zh": 0, "en": 0}
        for id in self.all_data_results.keys():
            lang_count[self.all_data_results[id]["language"]] += 1

        for lang in ["zh", "en"]:
            sc_point_sum = 0
            pq_point_sum = 0
            o_point_sum = 0
            count = 0
            for id in self.all_data_results.keys():
                if self.all_data_results[id]["language"] == lang:
                    sc_point_sum += self.all_data_results[id]["SC_point"]
                    pq_point_sum += self.all_data_results[id]["PQ_point"]
                    o_point_sum += self.all_data_results[id]["O_point"]
                    count += 1
            if count > 0:
                evaluate_result_list.append(copy.deepcopy([lang, sc_point_sum / count, pq_point_sum / count, o_point_sum / count]))

        sc_point_sum = 0
        pq_point_sum = 0
        o_point_sum = 0
        count = len(self.all_data_results)
        for id in self.all_data_results.keys():
            sc_point_sum += self.all_data_results[id]["SC_point"]
            pq_point_sum += self.all_data_results[id]["PQ_point"]
            o_point_sum += self.all_data_results[id]["O_point"]
        evaluate_result_list.append(copy.deepcopy(["all case", sc_point_sum / count, pq_point_sum / count, o_point_sum / count]))

        print(tabulate.tabulate(evaluate_result_list,
                                headers=["language", "SC_point", "PQ_point", "O_point"],
                                floatfmt=".4f"))

    def _calc_meta_point(self, pred):
        results_list = get_lmm_point_list(pred)
        if not results_list:
            return 0
        try:
            point_list = json.loads(results_list)
        except BaseException as e:
            raise RuntimeError(f"Illegal prediction: {pred}")
        return min(point_list)

    def _get_question_and_language(self, input_prompt):
        question = ""
        for item in input_prompt:
            if item.get("type") == "text":
                question = item.get("text")
        # 判断question是否包含中文
        if any("\u4e00" <= char <= "\u9fff" for char in question):
            return question, "zh"
        else:
            return question, "en"

    def _load_and_merge_jsonl(self, path_kind: "org_pred_path"):
        merged_data = []
        start_index = 0
        for path in self.paths_map[path_kind]:
            offset_index = copy.deepcopy(start_index)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    data["id"] = data["id"] + offset_index
                    start_index += 1
                    merged_data.append(data)
        return merged_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="显示gedit数据集的推理结果")
    parser.add_argument("--config_path", help="配置文件路径")
    parser.add_argument("--timestamp_path", help="结果时间戳路径")

    args = parser.parse_args()
    eval_parser = GEditEvalResultParser(args)
    eval_parser.parse_results()
    eval_parser.dump_result_csv()
    eval_parser.display_results()


if __name__ == "__main__":
    main()
