
import json
from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

from .base import BaseDataset, BaseJDGDatasetMethod


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

class Aime2025JDGDataset(Aime2025Dataset):
    def load(self, path, predictions_path, **kwargs):

        dataset_content = Aime2025Dataset.load(path, **kwargs)

        # 加载被测模型的推理结果(排序后)
        predictions: list = BaseJDGDatasetMethod.load_from_predictions(predictions_path)

        # 为数据集添加 model_answer 列
        dataset_list = []

        for item in predictions:
            item_dict = dataset_content[int(item["id"])]
            item_dict["model_answer"] = item["prediction"]
            item_dict["model_pred_uuid"] = item["uuid"] # Be filled in gold
            dataset_list.append(item_dict)

        return Dataset.from_list(dataset_list)
