"""VBench 1.0 dataset config type for video/image quality evaluation (eval-only, no loader)."""
from datasets import Dataset

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.base import BaseDataset


@LOAD_DATASET.register_module()
class VBenchDataset(BaseDataset):
    """Placeholder dataset for VBench evaluation.

    VBench evaluation uses only dataset config (path/videos_path, dimension_list,
    full_json_dir, eval_cfg). This class provides a minimal load() so that
    LOAD_DATASET.build(dataset_cfg) does not fail if ever called; the actual
    evaluation is done in VBenchEvalTask which reads the config directly.
    """

    @staticmethod
    def load(path: str, **kwargs):
        """Return a minimal placeholder dataset. VBench eval uses config only."""
        return Dataset.from_list([{"dummy": 0}])
