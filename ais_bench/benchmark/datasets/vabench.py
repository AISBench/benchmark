"""VABench prepared T2AV dataset, without inference dependencies."""
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.vabench.data import read_json


@LOAD_DATASET.register_module()
class VABenchDataset:
    """Eval-only dataset; avoids importing the text generation model stack.

    The eval task consumes this config directly. A standalone dataset build
    exposes the manifest rows, but is not a T2AV generation inferencer.
    """

    def __init__(self, path, **kwargs):
        self.dataset = self.load(path)

    @staticmethod
    def load(path, **kwargs):
        from pathlib import Path
        from datasets import Dataset
        manifest = read_json(Path(path) / 'manifest.json')
        return Dataset.from_list(manifest['records'])
