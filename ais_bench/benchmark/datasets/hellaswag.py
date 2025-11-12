import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HellaswagDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'ctx': data['query'].split(': ', 2)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class HellaswagDataset_V2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'ctx': data['query'].split(': ', 1)[-1],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'label': 'ABCD'[data['gold']],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class HellaswagDataset_V3(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'query': data['query'],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'gold': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset


@LOAD_DATASET.register_module()
class HellaswagDatasetwithICE(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path)
        dataset_dict = DatasetDict()
        for split, filename in [
            ['train', 'hellaswag_train_sampled25.jsonl'],
            ['val', 'hellaswag.jsonl'],
        ]:
            dataset = []
            with open(osp.join(path, filename), 'r',
                        encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    dataset.append({
                        'ctx': data['query'].split(': ', 1)[-1],
                        'A': data['choices'][0],
                        'B': data['choices'][1],
                        'C': data['choices'][2],
                        'D': data['choices'][3],
                        'label': 'ABCD'[data['gold']],
                    })
            dataset_dict[split] = Dataset.from_list(dataset)
        return dataset_dict
