import os
import json
from abc import ABC, abstractmethod


# --- Required environment variables --------------------------------------
# Read eagerly with a clear error message instead of bare KeyError, so users
# know which env vars to set when they see the failure.
# This list is the union of all modules' requirements (dataset_base needs the
# first three, __main__.py needs CORESET_RATIO) — listed together so a single
# missing-var error tells the user everything they need to set.

_REQUIRED_ENV_VARS = ('EVAL_DATASET', 'CORESET_METRIC', 'LLM_MODEL', 'CORESET_RATIO')


def require_env(name: str) -> str:
    """Read a required env var; raise a clear error if missing."""
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            f"All required: {', '.join(_REQUIRED_ENV_VARS)}."
        )
    return val


EVAL_DATASET = require_env('EVAL_DATASET')
CORESET_METRIC = require_env('CORESET_METRIC')
LLM_MODEL = require_env('LLM_MODEL')


# --- Configurable paths --------------------------------------------------
# Override via env vars if the defaults don't match your environment.

CFG_BASE_DATASET_DIR = os.environ.get('CORESET_BASE_DIR', '/workspace/data')
CFG_CORESET_OUT_DIR = f'./datasets/{EVAL_DATASET}/{CORESET_METRIC}/{LLM_MODEL}'


# --- Dataset base class --------------------------------------------------

class EvalDatasetBase(ABC):
    """Base class for evaluation datasets.

    Subclasses must implement:
        - dataset_size
        - dataset_prompts
        - save_data_by_indices
    """

    @abstractmethod
    def dataset_size(self) -> int:
        """Return total number of items in the dataset."""

    @abstractmethod
    def dataset_prompts(self):
        """Yield prompt strings one by one."""

    @abstractmethod
    def save_data_by_indices(self, indices, outpath):
        """Save items at the given indices into outpath (a subdir of CFG_CORESET_OUT_DIR)."""

    def load_indices(self):
        return list(range(self.dataset_size()))

    def save_indices(self, indices, outpath):
        """
        Save indices to indices.json in the output directory.

        Args:
            indices: List of indices to save
            outpath: Output directory path
        """
        indices_path = os.path.join(outpath, 'indices.json')
        with open(indices_path, 'w', encoding='utf-8') as f:
            json.dump(indices, f)

    def load_indices_from_strategy(self, strategy_name):
        """
        Load indices from a previously saved strategy.

        Args:
            strategy_name: Name of the strategy to load indices from

        Returns:
            List[int]: Loaded indices, or None if not found
        """
        indices_path = os.path.join(CFG_CORESET_OUT_DIR, strategy_name, 'indices.json')
        if os.path.exists(indices_path):
            with open(indices_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None  # Graceful fallback


EVAL_DATASETS = dict()


def reg_eval_dataset(dataset_name):
    def wrapper(dataset_cls):
        EVAL_DATASETS[dataset_name] = dataset_cls
        return dataset_cls
    return wrapper


def get_eval_dataset() -> EvalDatasetBase:
    # !! This is NOT singleton
    dataset_cls = EVAL_DATASETS.get(EVAL_DATASET, None)
    if dataset_cls is None:
        raise ValueError(
            f'Unknown dataset "{EVAL_DATASET}". '
            f'Registered: {list(EVAL_DATASETS.keys())}'
        )
    return dataset_cls()
