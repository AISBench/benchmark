import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class CEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, local_mode: bool = True):
        path = get_data_path(path, local_mode=local_mode)
        dataset = {}
        for split in ['dev', 'val', 'test']:
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    item = dict(zip(header, row))
                    item.setdefault('explanation', '')
                    item.setdefault('answer', '')
                    dataset.setdefault(split, []).append(item)
        dataset = DatasetDict(
            {i: Dataset.from_list(dataset[i])
                for i in dataset})
        return dataset