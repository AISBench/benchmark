import unittest
from unittest.mock import patch

from ais_bench.benchmark.datasets.refcoco.refcoco import RefCOCODataset
from ais_bench.benchmark.datasets.refcoco.refcoco_g import RefCOCOgDataset
from ais_bench.benchmark.registry import LOAD_DATASET


class TestRefCOCOgDataset(unittest.TestCase):
    def test_should_delegate_to_base_loader_when_loading_refcocog_split(self):
        # given
        sentinel_dataset = object()

        with patch.object(RefCOCODataset, 'load', return_value=sentinel_dataset) as mock_load:
            # when
            result = RefCOCOgDataset.load('/dataset/root', 'test', local_mode=False, image_type='image_base64')

        # then
        self.assertIs(result, sentinel_dataset)
        mock_load.assert_called_once_with('/dataset/root', 'test', local_mode=False, image_type='image_base64')

    def test_should_register_refcocog_dataset_when_module_is_imported(self):
        # given / when
        registered_dataset = LOAD_DATASET.get('RefCOCOgDataset')

        # then
        self.assertIs(registered_dataset, RefCOCOgDataset)
        self.assertTrue(issubclass(RefCOCOgDataset, RefCOCODataset))


if __name__ == '__main__':
    unittest.main()
