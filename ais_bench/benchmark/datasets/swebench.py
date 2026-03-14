import re
import random
from datasets import load_dataset, Dataset, DatasetDict

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.datasets.base import BaseDataset

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
}


@LOAD_DATASET.register_module()
class SWEBenchDataset(BaseDataset):
    def filter_instances(
        self, instances: list[dict], *, filter_spec: str, shuffle: bool = False
    ) -> list[dict]:
        """Filter and slice a list of SWEBench instances."""
        if shuffle:
            instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
            random.seed(42)
            random.shuffle(instances)
        before_filter = len(instances)
        instances = [
            instance
            for instance in instances
            if re.match(filter_spec, instance["instance_id"])
        ]
        if (after_filter := len(instances)) != before_filter:
            self.logger.info(
                f"Instance filter: {before_filter} -> {after_filter} instances"
            )
        return instances

    def load(
        self,
        path: str,
        name: str,
        split: str = "test",
        filter_spec: str = "",
        shuffle: bool = False,
    ):
        if name not in DATASET_MAPPING:
            raise ParameterValueError(
                DSET_CODES.INVALID_PARAM_VALUE,
                f"Invalid swebench dataset name, expected one of {list(DATASET_MAPPING.keys())} but got {name}",
            )
        try:
            dataset = load_dataset("parquet", data_files={split: path})
        except Exception as e:
            self.logger.warning(
                f"Failed to load swebench dataset {name} from {path} with error: {e}, trying to load from Hugging Face"
            )
            try:
                dataset = load_dataset(DATASET_MAPPING[name], split=split)
            except Exception as e:
                raise ParameterValueError(
                    DSET_CODES.DATA_PREPROCESSING_ERROR,
                    f"Failed to load swebench dataset {name} from Hugging Face with error: {e}.",
                )
        dataset = self.filter_instances(list(dataset), filter_spec=filter_spec, shuffle=shuffle)
        return DatasetDict({"test": Dataset.from_list(dataset)})
