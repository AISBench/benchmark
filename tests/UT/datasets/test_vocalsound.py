import unittest
from unittest.mock import patch, MagicMock, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.vocalsound import VocalSoundDataset, VocalSoundEvaluator


class TestVocalSound(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.vocalsound.Path")
    @patch("ais_bench.benchmark.datasets.vocalsound.get_data_path", return_value="/fake/dir")
    def test_load_audio_path(self, mock_get_path, mock_path_cls):
        # 模拟目录下有两个wav文件
        mock_path = MagicMock()
        mock_path.glob.return_value = ["/fake/dir/a_cat.wav", "/fake/dir/b_42.wav"]
        mock_path_cls.return_value = mock_path
        out = VocalSoundDataset.load("/any", audio_type="audio_path")
        self.assertIsInstance(out, Dataset)
        self.assertEqual(len(out), 2)
        row = out[0]
        self.assertIn("audio_url", row)
        self.assertIn("answer", row)

    @patch("ais_bench.benchmark.datasets.vocalsound.Path")
    @patch("ais_bench.benchmark.datasets.vocalsound.get_data_path", return_value="/fake/dir")
    @patch("builtins.open")
    def test_load_audio_base64(self, mock_open_file, mock_get_path, mock_path_cls):
        mock_path = MagicMock()
        mock_path.glob.return_value = ["/fake/dir/a_cat.wav"]
        mock_path_cls.return_value = mock_path
        m = mock_open(read_data=b"bin")
        mock_open_file.return_value = m.return_value
        out = VocalSoundDataset.load("/any", audio_type="audio_base64")
        self.assertEqual(len(out), 1)

    def test_evaluator(self):
        eva = VocalSoundEvaluator()
        # find_choice 映射
        self.assertEqual(eva.find_choice("A"), "laughter")
        # score 正常
        res = eva.score([["A"]], ["laughter"])
        self.assertIn("accuracy", res)
        # 长度不匹配
        res2 = eva.score(["A"], ["laughter", "sniff"])
        self.assertIn("error", res2)


if __name__ == "__main__":
    unittest.main()
