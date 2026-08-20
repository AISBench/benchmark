import json
import os
from abc import ABC, abstractmethod


class EvalDatasetBase(ABC):
    """Base class for evaluation datasets."""

    def __init__(self, dataset_path, output_dir):
        self.dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))

    @abstractmethod
    def dataset_size(self) -> int:
        """Return total number of items in the dataset."""

    @abstractmethod
    def dataset_prompts(self):
        """Yield prompt strings one by one."""

    @abstractmethod
    def save_data_by_indices(self, indices, outpath):
        """Save selected items into a subdirectory under ``self.output_dir``."""

    def load_indices(self):
        return list(range(self.dataset_size()))

    @staticmethod
    def save_indices(indices, outpath):
        """Save selected source indices to ``indices.json``."""
        indices_path = os.path.join(outpath, "indices.json")
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump(indices, f)

    def load_indices_from_strategy(self, strategy_name):
        """Load indices saved under another strategy subdirectory, if present."""
        indices_path = os.path.join(self.output_dir, strategy_name, "indices.json")
        if os.path.exists(indices_path):
            with open(indices_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


EVAL_DATASETS = {}


def reg_eval_dataset(dataset_name):
    def wrapper(dataset_cls):
        EVAL_DATASETS[dataset_name] = dataset_cls
        return dataset_cls

    return wrapper


def get_eval_dataset(dataset_name, dataset_path, output_dir) -> EvalDatasetBase:
    """Create a registered evaluation dataset adapter."""
    dataset_cls = EVAL_DATASETS.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(
            f'Unknown dataset "{dataset_name}". '
            f"Registered: {list(EVAL_DATASETS.keys())}"
        )
    return dataset_cls(dataset_path=dataset_path, output_dir=output_dir)
