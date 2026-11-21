from datasets import load_dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from ..base import BaseDataset
from ._qa import _project_qa_test_split


@LOAD_DATASET.register_module()
class LongBench2wikimqaDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = load_dataset(path=path,
                               name=name,
                               data_dir=path,
                               trust_remote_code=True)
        return _project_qa_test_split(dataset)
