import unittest
import tempfile
import os
import json
import fcntl
from unittest.mock import patch, mock_open

from ais_bench.benchmark.utils.results.results import safe_write, dump_results_dict


class TestResults(unittest.TestCase):
    """Tests for results.py functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_results.jsonl")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_safe_write(self):
        """Test safe_write function with file locking."""
        results_dict = {
            "task1": {"result": "success", "score": 0.95},
            "task2": {"result": "success", "score": 0.87}
        }

        safe_write(results_dict, self.test_file)

        # Verify file was created and contains expected content
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            # Verify JSON content
            result1 = json.loads(lines[0].strip())
            result2 = json.loads(lines[1].strip())
            self.assertIn("result", result1)
            self.assertIn("result", result2)

    def test_safe_write_empty_dict(self):
        """Test safe_write with empty dictionary."""
        safe_write({}, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertEqual(content, "")

    def test_dump_results_dict_formatted(self):
        """Test dump_results_dict with formatted=True (default)."""
        results_dict = {
            "task1": {"result": "success", "score": 0.95},
            "task2": {"result": "success", "score": 0.87}
        }

        dump_results_dict(results_dict, self.test_file, formatted=True)

        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertEqual(content, results_dict)
            # Check that file has indentation (formatted)
            f.seek(0)
            raw_content = f.read()
            self.assertIn("\n", raw_content)  # Should have newlines for formatting

    def test_dump_results_dict_not_formatted(self):
        """Test dump_results_dict with formatted=False."""
        results_dict = {
            "task1": {"result": "success", "score": 0.95}
        }

        dump_results_dict(results_dict, self.test_file, formatted=False)

        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertEqual(content, results_dict)

    def test_dump_results_dict_empty(self):
        """Test dump_results_dict with empty dictionary."""
        dump_results_dict({}, self.test_file)

        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertEqual(content, {})

    def test_dump_results_dict_unicode(self):
        """Test dump_results_dict with unicode characters."""
        results_dict = {
            "task1": {"result": "成功", "message": "测试"}
        }

        dump_results_dict(results_dict, self.test_file, formatted=True)

        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertEqual(content["task1"]["result"], "成功")
            self.assertEqual(content["task1"]["message"], "测试")


if __name__ == "__main__":
    unittest.main()

