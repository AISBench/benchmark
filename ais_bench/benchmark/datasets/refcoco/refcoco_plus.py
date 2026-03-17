from ais_bench.benchmark.registry import LOAD_DATASET

from ais_bench.benchmark.datasets.refcoco.refcoco import RefCOCODataset


@LOAD_DATASET.register_module()
class RefCOCOPlusDataset(RefCOCODataset):
    """
    RefCOCOplus is a variant of RefCOCO with more complex referring expressions. 
    Because the dataset field is same as the RefCOCO dataset, we can reuse the loading and evaluation code.
    The only difference is refcoco_plus only has three splits:
    - `val`: 3.81k rows
    - `testA`: 1.98k rows
    - `testB`: 1.8k rows
    """
    pass
