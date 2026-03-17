from ais_bench.benchmark.registry import LOAD_DATASET

from ais_bench.benchmark.datasets.refcoco.refcoco import RefCOCODataset


@LOAD_DATASET.register_module()
class RefCOCOgDataset(RefCOCODataset):
    """
    RefCOCOg is a variant of RefCOCO with more complex referring expressions. 
    Because the dataset field is same as the RefCOCO dataset, we can reuse the loading and evaluation code.
    The only difference is refcoco_g only has two splits:
    - `val`: 7.57k rows
    - `test`: 5.02k rows
    """
    pass
