
import json
from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.utils.llm_judge import LLMJudgeDataset

from ais_bench.benchmark.datasets.base import BaseDataset

@LOAD_DATASET.register_module()
class Aime2025Dataset(BaseDataset):
    @staticmethod
    def load(path, **kwargs):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                dataset.append(line)
        return Dataset.from_list(dataset)


@LOAD_DATASET.register_module()
class Aime2025JDGDataset(LLMJudgeDataset):
    def _get_dataset_class(self):
        return Aime2025Dataset
