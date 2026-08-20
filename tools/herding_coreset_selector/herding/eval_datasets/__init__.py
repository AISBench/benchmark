from herding.eval_datasets.dataset_base import (
    EVAL_DATASETS,
    EvalDatasetBase,
    get_eval_dataset,
    reg_eval_dataset,
)

# Import adapters so their registration decorators are executed.
from herding.eval_datasets import aime2025 as _aime2025  # noqa: F401,E402
from herding.eval_datasets import gpqa as _gpqa  # noqa: F401,E402

__all__ = [
    "EVAL_DATASETS",
    "EvalDatasetBase",
    "get_eval_dataset",
    "reg_eval_dataset",
]
