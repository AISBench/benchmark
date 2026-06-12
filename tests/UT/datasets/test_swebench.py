import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ais_bench.benchmark.datasets.swebench import SWEBenchDataset
from ais_bench.benchmark.utils.logging.exceptions import FileOperationError


class TestSWEBenchDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = object.__new__(SWEBenchDataset)
        self.dataset.logger = mock.MagicMock()

    def test_load_instance_ids_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ids_file = Path(temp_dir) / "ids.txt"
            ids_file.write_text("django__django-1\n\nsympy__sympy-2\nsympy__sympy-2\n", encoding="utf-8")

            instance_ids = self.dataset._load_instance_ids_file(str(ids_file))

        self.assertEqual(instance_ids, {"django__django-1", "sympy__sympy-2"})

    def test_load_instance_ids_file_requires_txt_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ids_file = Path(temp_dir) / "ids.csv"
            ids_file.write_text("django__django-1\n", encoding="utf-8")

            with self.assertRaises(FileOperationError):
                self.dataset._load_instance_ids_file(str(ids_file))

    def test_filter_instances_by_filter_spec_and_instance_ids(self):
        instances = [
            {"instance_id": "django__django-1"},
            {"instance_id": "django__django-2"},
            {"instance_id": "sympy__sympy-1"},
        ]

        filtered = self.dataset.filter_instances(
            instances,
            filter_spec=r"^django__",
            instance_ids={"django__django-2", "sympy__sympy-1"},
        )

        self.assertEqual(filtered, [{"instance_id": "django__django-2"}])


if __name__ == "__main__":
    unittest.main()
