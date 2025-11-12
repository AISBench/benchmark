import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ARCDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = DatasetDict()
        for split in ["Dev", "Test"]:
            with open(osp.join(path, f"{name}-{split}.jsonl"), 'r', errors='ignore') as in_f:
                rows = []
                for line in in_f:
                    item = json.loads(line.strip())
                    question = item['question']
                    if len(question['choices']) != 4:
                        continue
                    labels = [c['label'] for c in question['choices']]
                    answerKey = 'ABCD'[labels.index(item['answerKey'])]
                    rows.append({
                        'question': question['stem'],
                        'answerKey': answerKey,
                        'textA': question['choices'][0]['text'],
                        'textB': question['choices'][1]['text'],
                        'textC': question['choices'][2]['text'],
                        'textD': question['choices'][3]['text'],
                    })
            dataset[split] = Dataset.from_list(rows)
        return dataset