import os
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open

from ais_bench.benchmark.datasets.utils.datasets import (
    get_cache_dir,
    get_data_path,
    get_sample_data,
    get_meta_json,
)


class TestGetCacheDir(unittest.TestCase):
    def test_get_cache_dir_env(self):
        with patch.dict(os.environ, {"AIS_BENCH_DATASETS_CACHE": "/env/cache"}, clear=False):
            self.assertEqual(get_cache_dir("/default"), "/env/cache")

    def test_get_cache_dir_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_cache_dir("/default"), "/default")


class TestGetDataPath(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    def test_absolute_path_returns_directly(self, mock_exists):
        self.assertEqual(get_data_path("/abs/path", local_mode=True), "/abs/path")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.dirname")
    @patch("os.path.abspath")
    def test_relative_path_exists(self, mock_abspath, mock_dirname, mock_exists):
        # fabricate default_dir and cache_dir
        mock_abspath.return_value = "/pkg/utils/datasets.py"
        mock_dirname.return_value = "/pkg/utils"
        with patch.dict(os.environ, {"AIS_BENCH_DATASETS_CACHE": "/cache"}, clear=False):
            self.assertEqual(get_data_path("rel/path", local_mode=True), "/cache/rel/path")

    @patch("os.path.exists", return_value=False)
    @patch("os.path.dirname")
    @patch("os.path.abspath")
    def test_relative_path_not_exists(self, mock_abspath, mock_dirname, mock_exists):
        mock_abspath.return_value = "/pkg/utils/datasets.py"
        mock_dirname.return_value = "/pkg/utils"
        with patch.dict(os.environ, {"AIS_BENCH_DATASETS_CACHE": "/cache"}, clear=False):
            with self.assertRaises(FileExistsError):
                get_data_path("rel/path", local_mode=True)

    def test_relative_path_non_local_mode_raises(self):
        with self.assertRaises(TypeError):
            get_data_path("rel/path", local_mode=False)


class TestGetSampleData(unittest.TestCase):
    def test_no_request_count_defaults_all(self):
        data = [1, 2, 3]
        out = get_sample_data(data, sample_mode="default", request_count=0)
        self.assertEqual(out, data)

    def test_request_count_greater_than_len(self):
        data = [1, 2]
        out = get_sample_data(data, sample_mode="default", request_count=5)
        self.assertEqual(len(out), 5)
        # should be repeated from original list order
        self.assertEqual(out[:2], data)

    def test_request_count_negative_raises(self):
        with self.assertRaises(ValueError):
            get_sample_data([1], sample_mode="default", request_count=-1)

    def test_random_mode(self):
        data = list(range(10))
        with patch("random.sample", return_value=[9, 8, 7]):
            out = get_sample_data(data, sample_mode="random", request_count=3)
            self.assertEqual(out, [9, 8, 7])

    def test_shuffle_mode(self):
        data = [1, 2, 3]
        with patch("random.shuffle", side_effect=lambda x: x.reverse()):
            out = get_sample_data(data, sample_mode="shuffle", request_count=3)
            self.assertEqual(out, [3, 2, 1])

    def test_unsupported_mode_raises(self):
        with self.assertRaises(ValueError):
            get_sample_data([1, 2], sample_mode="bad", request_count=1)


class TestGetMetaJson(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    def test_meta_path_none_but_file_exists(self, mock_exists):
        m = mock_open(read_data='{"k": 1}')
        with patch("builtins.open", m):
            out = get_meta_json("/data/path", None)
            self.assertEqual(out, {"k": 1})

    @patch("os.path.exists", return_value=False)
    def test_meta_path_none_and_file_not_exists_returns_empty(self, mock_exists):
        out = get_meta_json("/data/path", None)
        self.assertEqual(out, {})

    @patch("os.path.exists", return_value=False)
    def test_user_meta_path_not_exists_raises(self, mock_exists):
        with self.assertRaises(ValueError):
            get_meta_json("/data/path", "/custom/meta.json")


if __name__ == "__main__":
    unittest.main()
