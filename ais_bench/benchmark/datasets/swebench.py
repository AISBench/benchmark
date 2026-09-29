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
    "verified_mini": "MariusHobbhahn/swe-bench-verified-mini",
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
    def _load_instance_ids_file(self, instance_ids_file: str) -> set[str]:
        path = Path(instance_ids_file).expanduser()
        if not path.is_file():
            raise FileOperationError(
                SWEB_CODES.LOCAL_PATH_RESOLVE_FAILED,
                f"SWE-Bench instance ids file does not exist: {instance_ids_file!r}",
            )
        if path.suffix.lower() != ".txt":
            raise FileOperationError(
                SWEB_CODES.LOCAL_PATH_RESOLVE_FAILED,
                f"SWE-Bench instance ids file must be a .txt file: {instance_ids_file!r}",
            )

        try:
            instance_ids = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except OSError as e:
            raise FileOperationError(
                SWEB_CODES.LOCAL_PATH_RESOLVE_FAILED,
                f"Failed to read SWE-Bench instance ids file {instance_ids_file!r}: {e}",
            )
        return instance_ids

    def filter_instances(
        self,
        instances: list[dict],
        *,
        filter_spec: str,
        instance_ids: set[str] | None = None,
        shuffle: bool = False,
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

        if instance_ids is not None:
            available_ids = {instance["instance_id"] for instance in instances}
            missing_ids = instance_ids - available_ids
            before_ids_filter = len(instances)
            instances = [
                instance
                for instance in instances
                if instance["instance_id"] in instance_ids
            ]
            if (after_ids_filter := len(instances)) != before_ids_filter:
                self.logger.info(
                    f"Instance ids file filter: {before_ids_filter} -> {after_ids_filter} instances"
                )
            if missing_ids:
                self.logger.warning(
                    "Instance ids file contains %d ids not present after dataset/filter_spec selection: %s",
                    len(missing_ids),
                    ", ".join(sorted(missing_ids)[:10]),
                )
        return instances

    def load(
        self,
        name: str,
        path: str = "",
        split: str = "test",
        filter_spec: str = "",
        instance_ids_file: str = "",
        shuffle: bool = False,
        **kwargs,
    ):
        """Load SWE-bench rows.
        Args:
            name (str): The name of the dataset to load.
            path: The path to the dataset.
            split (str): The split of the dataset to load.
            filter_spec (str): The filter specification to apply to the dataset.
            instance_ids_file (str): Text file containing one instance_id per line.
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
        instance_ids = None
        if instance_ids_file:
            instance_ids = self._load_instance_ids_file(instance_ids_file)
            self.logger.info(
                "Loaded %d SWE-Bench instance ids from %s",
                len(instance_ids),
                instance_ids_file,
            )

        dataset = self.filter_instances(
            list(dataset),
            filter_spec=filter_spec,
            instance_ids=instance_ids,
            shuffle=shuffle,
        )
        return Dataset.from_list(dataset)
