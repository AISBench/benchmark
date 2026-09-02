import unittest
from unittest.mock import patch

from ais_bench.benchmark.datasets.refcoco.refcoco import RefCOCODataset
from ais_bench.benchmark.datasets.refcoco.refcoco_plus import RefCOCOPlusDataset
from ais_bench.benchmark.registry import LOAD_DATASET


class TestRefCOCOPlusDataset(unittest.TestCase):
    def test_should_delegate_to_base_loader_when_loading_refcoco_plus_split(self):
        # given
        sentinel_dataset = object()

        with patch.object(
            RefCOCODataset, "load", return_value=sentinel_dataset
        ) as mock_load:
            # when
            result = RefCOCOPlusDataset.load(
                "/dataset/root", "testB", local_mode=False, image_type="base64"
            )

        # then
        self.assertIs(result, sentinel_dataset)
        mock_load.assert_called_once_with(
            "/dataset/root", "testB", local_mode=False, image_type="base64"
        )

    def test_should_register_refcoco_plus_dataset_when_module_is_imported(self):
        # given / when
        registered_dataset = LOAD_DATASET.get("RefCOCOPlusDataset")

        # then
        self.assertIs(registered_dataset, RefCOCOPlusDataset)
        self.assertTrue(issubclass(RefCOCOPlusDataset, RefCOCODataset))


if __name__ == "__main__":
    unittest.main()
