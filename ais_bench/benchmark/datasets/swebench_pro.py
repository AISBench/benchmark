import re
import random
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchDataContentError,
    FileOperationError,
    ParameterValueError,
)
from ais_bench.benchmark.utils.logging.error_codes import SWEBP_CODES
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

DATASET_MAPPING = {
    "full": "ScaleAI/SWE-bench_Pro",
    "mini": "",
}


def _parquet_shards_for_split(dataset_root: Path, split: str) -> list[str] | None:
    shards: list[Path] = []
    data_dir = dataset_root / "data"
    if data_dir.is_dir():
        shards = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not shards and dataset_root.is_dir():
        shards = sorted(dataset_root.glob(f"{split}-*.parquet"))
    if not shards:
        return None
    return [str(p) for p in shards]


def _parquet_data_files_for_root(root: Path, split: str) -> dict[str, str | list[str]] | None:
    if root.is_file():
        return {split: str(root)}
    shards = _parquet_shards_for_split(root, split)
    if not shards:
        return None
    return {split: shards if len(shards) > 1 else shards[0]}


@LOAD_DATASET.register_module()
class SWEBenchProDataset(BaseDataset):
    def filter_instances(
        self, instances: list[dict], *, filter_spec: str, shuffle: bool = False
    ) -> list[dict]:
        if shuffle:
            instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
            random.seed(42)
            random.shuffle(instances)
        before_filter = len(instances)
        if filter_spec:
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
        name: str = "full",
        path: str = "",
        split: str = "test",
        filter_spec: str = "",
        shuffle: bool = False,
        **kwargs,
    ):
        if name not in DATASET_MAPPING:
            raise ParameterValueError(
                SWEBP_CODES.INVALID_DATASET_NAME,
                f"Invalid swebench_pro dataset name, expected one of {list(DATASET_MAPPING.keys())} but got {name}",
            )
        hf_id = DATASET_MAPPING[name]
        path = (path or "").strip()

        if name == "mini" and not path:
            raise ParameterValueError(
                SWEBP_CODES.INVALID_DATASET_NAME,
                "mini dataset requires a local path, please configure `path` to a local parquet directory/file.",
            )

        if not path:
            try:
                dataset = load_dataset(hf_id, split=split)
                self.logger.info(
                    f"Loaded swebench_pro dataset {name} split={split} from Hugging Face (online)"
                )
            except Exception as e:
                raise AISBenchDataContentError(
                    SWEBP_CODES.HF_DATASET_LOAD_FAILED,
                    (
                        f"Failed to load swebench_pro dataset {name} split={split} from Hugging Face: {e}. "
                        "Please manually download the dataset and configure `path` to a local parquet directory/file."
                    ),
                )
        else:
            try:
                root = Path(get_data_path(path, local_mode=True))
            except Exception as e:
                raise FileOperationError(
                    SWEBP_CODES.LOCAL_PATH_RESOLVE_FAILED,
                    f"Failed to resolve local swebench_pro dataset path {path!r}: {e}",
                )

            data_files = _parquet_data_files_for_root(root, split)
            if data_files is None:
                raise FileOperationError(
                    SWEBP_CODES.LOCAL_PARQUET_NOT_FOUND,
                    (
                        f"No parquet found for split {split!r} under {root}. "
                        "Please verify `path` points to a local parquet file, "
                        "or a directory containing `data/<split>-*.parquet` "
                        "or `<split>-*.parquet` files."
                    ),
                )
            try:
                loaded = load_dataset("parquet", data_files=data_files)
                dataset = loaded[split] if isinstance(loaded, DatasetDict) else loaded
                self.logger.info(
                    f"Loaded swebench_pro dataset {name} split={split} from local path: {root}"
                )
            except Exception as e:
                raise AISBenchDataContentError(
                    SWEBP_CODES.LOCAL_PARQUET_LOAD_FAILED,
                    f"Failed to load local swebench_pro parquet from {root}: {e}",
                )

        dataset = self.filter_instances(list(dataset), filter_spec=filter_spec, shuffle=shuffle)
        return Dataset.from_list(dataset)
