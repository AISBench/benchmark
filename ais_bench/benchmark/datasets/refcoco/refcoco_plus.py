from ais_bench.benchmark.registry import LOAD_DATASET

from ais_bench.benchmark.datasets.refcoco.refcoco import RefCOCODataset


@LOAD_DATASET.register_module()
class RefCOCOPlusDataset(RefCOCODataset):
    TEMP_REFCOCO_IMAGE_STORE_DIR = 'RefCOCOPlus_images'
