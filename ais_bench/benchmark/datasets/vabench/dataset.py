"""VABench prepared T2AV dataset, without inference dependencies."""
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils.vabench.data import read_json, select_indices


@LOAD_DATASET.register_module()
class VABenchDataset:
    """Eval-only dataset; avoids importing the text generation model stack.

    The eval task consumes this config directly. A standalone dataset build
    exposes the manifest rows, but is not a T2AV generation inferencer.
    """

    def __init__(self, path, reader_cfg=None, **kwargs):
        self.dataset = self.load(path, reader_cfg=reader_cfg, **kwargs)

    @staticmethod
    def load(path, reader_cfg=None, **kwargs):
        from pathlib import Path
        from datasets import Dataset
        manifest = read_json(Path(path) / 'manifest.json')
        records = manifest['records']
        test_range = (reader_cfg or {}).get('test_range')
        if test_range is not None:
            indices = select_indices(len(records), test_range)
            records = [records[index] for index in indices]
        return Dataset.from_list(records)
