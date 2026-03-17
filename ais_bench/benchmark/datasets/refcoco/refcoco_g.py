from ais_bench.benchmark.registry import LOAD_DATASET

from ais_bench.benchmark.datasets.refcoco.refcoco import RefCOCODataset


@LOAD_DATASET.register_module()
class RefCOCOgDataset(RefCOCODataset):
    TEMP_REFCOCO_IMAGE_STORE_DIR = 'RefCOCOg_images'
