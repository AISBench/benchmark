import os

from herding.eval_datasets.dataset_base import get_eval_dataset

_name = os.environ.get('EVAL_DATASET')
if _name == 'gpqa':
    from herding.eval_datasets import gpqa
elif _name == 'aime2025':
    from herding.eval_datasets import aime2025
