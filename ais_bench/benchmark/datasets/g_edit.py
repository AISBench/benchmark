import json
from datasets import Dataset, load_from_disk, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.utils.llm_judge import LLMJudgeDataset
from ais_bench.benchmark.utils.image_process import pil_to_base64
from PIL import Image
from tqdm import tqdm

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START

GEDIT_COUNT = 10

class GEditEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        details = []
        for i, pred in enumerate(predictions):
            details.append({
                'pred': pred,
                'ref': references[i],
            })
        result = {'accuracy': 100 * len(predictions) / len(references), 'details': details}
        return result

@LOAD_DATASET.register_module()
class GEditDataset(BaseDataset):
    @staticmethod
    def load(path, use_raw=False, split_count=1, split_index=0, **kwargs):
        path = get_data_path(path)
        dataset = load_from_disk(path)

        # 数据集切分：分成 split_count 份，取第 split_index 份
        if split_count > 1:
            total_len = len(dataset)
            base_size = total_len // split_count  # 每份基础大小
            remainder = total_len % split_count    # 余数

            # 计算当前 split_index 的起始和结束位置
            # 前 remainder 份每份多一个元素
            if split_index < remainder:
                start_idx = split_index * (base_size + 1)
                end_idx = start_idx + (base_size + 1)
            else:
                start_idx = remainder * (base_size + 1) + (split_index - remainder) * base_size
                end_idx = start_idx + base_size

            dataset = dataset.select(range(start_idx, end_idx))
        else:
            dataset = dataset.select(range(GEDIT_COUNT))

        if use_raw:
            image_column = 'input_image_raw'
        else:
            image_column = 'input_image'

        def process_example_to_dataset(example):
            """处理单条数据并转换为 Dataset"""
            image_url = pil_to_base64(example[image_column], "PNG")
            example['content'] = AIS_IMAGE_START + image_url + AIS_CONTENT_TAG \
                + AIS_TEXT_START + example['instruction'] + AIS_CONTENT_TAG
            # 使用 from_dict 替代 from_list 以提高性能
            data_dict = {key: [example[key]] for key in example.keys()}
            return Dataset.from_dict(data_dict)

        max_workers = 4  # Adjust based on system resources
        processed_datasets = [None] * len(dataset)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            with tqdm(total=len(dataset), desc=f"Submitting tasks split_count: {split_count}, split_index={split_index}", unit="example") as submit_pbar:
                futures = {}
                for i, example in enumerate(dataset):
                    future = executor.submit(process_example_to_dataset, example)
                    futures[future] = i
                    submit_pbar.update(1)

            # 收集处理完成的 Dataset
            with tqdm(total=len(dataset), desc="Processing GEdit dataset", unit="example") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    processed_datasets[idx] = future.result()
                    pbar.update(1)

        # 合并所有 Dataset
        return concatenate_datasets(processed_datasets)

@LOAD_DATASET.register_module()
class GEditJDGDataset(LLMJudgeDataset):
    def _get_dataset_class(self):
        return GEditDataset