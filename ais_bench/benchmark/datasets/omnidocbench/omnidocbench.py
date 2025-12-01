import json
import os
import re
import base64
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START
from .omnidocbench_dependency import *

from ..base import BaseDataset

DEFAULT_QUESTION = "Parse the image"


@LOAD_DATASET.register_module()
class OmniDocBenchDataset(BaseDataset):

    @staticmethod
    def load(path, image_path):
        path = get_data_path(path, local_mode=True)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = []
        for item in data:
            item_bak = item.copy()
            item['answer'] = item_bak
            image_url = os.path.abspath(os.path.join(image_path, item['page_info']['image_path']))
            item["content"] = AIS_IMAGE_START + image_url + AIS_CONTENT_TAG \
                                + AIS_TEXT_START + DEFAULT_QUESTION + AIS_CONTENT_TAG
            dataset.append(item)
        return Dataset.from_list(dataset)


class OmniDocBenchEvaluator(BaseEvaluator):
    def __init__(self, metrics_list):
        super().__init__()
        self.metrics_list = metrics_list

    def score(self, predictions, references):
        save_name = 'end2end_quick_match'
        if not OMNIDOCBENCH_INSTALLED:
            raise ImportError(
                "Missing required packages to run OmniDocBench evaluation, "
                "install it via: pip3 install -r requirements/datasets/omnidocbench_dependencies.txt"
            )
        end2end_dataset = End2EndDataset(predictions, references)  
        res = end2end_eval(end2end_dataset, self.metrics_list, references, save_name, self._out_dir)
        overall = sum(res[key] for key in res) / len(res)
        res['overall'] = overall
        return res


def end2end_eval(dataset, metrics_list, pages, save_name, out_dir):
    result_all = {}
    results = {}
    page_info = {}
    md_flag = False
    if not md_flag:
        for page in pages:
            img_path = os.path.basename(page['page_info']['image_path'])
            page_info[img_path] = page['page_info']['page_attribute']

    for element in metrics_list.keys():
        result = {}
        group_info = metrics_list[element].get('group', [])
        samples = dataset.samples[element]
        for metric in metrics_list[element]['metric']:
            metric_val = METRIC_REGISTRY.get(metric)
            samples, result_s = metric_val(samples).evaluate(group_info, f"{save_name}_{element}", out_dir)
            if result_s:
                results[f'[{element}]: {metric}'] = 100 * float(result_s[metric]['ALL_page_avg'])
                result.update(result_s)
        result_all[element] = {}
        
        if md_flag:
            group_result =  {}
            page_result = {}
        else:
            group_result = get_full_labels_results(samples)
            page_result = get_page_split(samples, page_info)
        result_all[element] = {
            'all': result,
            'group':  group_result,
            'page': page_result}

        if isinstance(samples, list):
            saved_samples = samples
        else:
            saved_samples = samples.samples
        try:

            with open(f'{out_dir}/{save_name}_{element}_result.json', 'w', encoding='utf-8') as f:
                json.dump(saved_samples, f, indent=4, ensure_ascii=False)
        except TypeError as e:
            raise ValueError(f"write json error: {e}")
            
            # print out problematic data types
            def find_non_serializable(data):
                if isinstance(data, dict):
                    for k, v in data.items():
                        try:
                            json.dumps(v)
                        except TypeError:
                            print(f"key '{k}' contains a non-serializable value: {v} (type: {type(v)})")
                            find_non_serializable(v)
                elif isinstance(data, (list, tuple)):
                    for i, item in enumerate(data):
                        try:
                            json.dumps(item)
                        except TypeError:
                            print(f"key {i} contains a non-serializable value: {item} (type: {type(item)})")
                            find_non_serializable(item)
            
            find_non_serializable(saved_samples)


    with open(f'{out_dir}/{save_name}_metric_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_all, f, indent=4, ensure_ascii=False)
    return results
