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
from ais_bench.benchmark.utils.logging.error_codes import SWEB_CODES
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_data_path

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
}

def _parquet_shards_for_split(dataset_root: Path, split: str) -> list[str] | None:
    """Resolve parquet shards for a split (HF snapshot layout: <root>/data/<split>-*.parquet)."""
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
    return _parquet_data_files_from_dir(root, split)


def _parquet_data_files_from_dir(
    root: Path, split: str
) -> dict[str, str | list[str]] | None:
    shards = _parquet_shards_for_split(root, split)
    if not shards:
        return None
    return {split: shards if len(shards) > 1 else shards[0]}


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
        name: str,
        path: str = "",
        split: str = "test",
        filter_spec: str = "",
        shuffle: bool = False,
        **kwargs,
    ):
        """Load SWE-bench rows.
        Args:
            name (str): The name of the dataset to load.
            path: The path to the dataset.
            split (str): The split of the dataset to load.
            filter_spec (str): The filter specification to apply to the dataset.
            shuffle (bool): Whether to shuffle the dataset.
            **kwargs: Additional keyword arguments.

        Returns:
            A Dataset object.
        """
        if name not in DATASET_MAPPING:
            raise ParameterValueError(
                SWEB_CODES.INVALID_DATASET_NAME,
                f"Invalid swebench dataset name, expected one of {list(DATASET_MAPPING.keys())} but got {name}",
            )
        hf_id = DATASET_MAPPING[name]
        path = (path or "").strip()

        if not path:
            try:
                dataset = load_dataset(hf_id, split=split)
                self.logger.info(
                    f"Loaded swebench dataset {name} split={split} from Hugging Face (online)"
                )
            except Exception as e:
                raise AISBenchDataContentError(
                    SWEB_CODES.HF_DATASET_LOAD_FAILED,
                    (
                        f"Failed to load swebench dataset {name} split={split} from Hugging Face: {e}. "
                        "Please manually download the dataset and configure `path` to a local parquet directory/file."
                    ),
                )
        else:
            try:
                root = Path(get_data_path(path, local_mode=True))
            except Exception as e:
                raise FileOperationError(
                    SWEB_CODES.LOCAL_PATH_RESOLVE_FAILED,
                    f"Failed to resolve local swebench dataset path {path!r}: {e}",
                )

            data_files = _parquet_data_files_for_root(root, split)
            if data_files is None:
                raise FileOperationError(
                    SWEB_CODES.LOCAL_PARQUET_NOT_FOUND,
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
                    f"Loaded swebench dataset {name} split={split} from local path: {root}"
                )
            except Exception as e:
                raise AISBenchDataContentError(
                    SWEB_CODES.LOCAL_PARQUET_LOAD_FAILED,
                    f"Failed to load local swebench parquet from {root}: {e}",
                )
        dataset = self.filter_instances(list(dataset), filter_spec=filter_spec, shuffle=shuffle)
        return Dataset.from_list(dataset)
