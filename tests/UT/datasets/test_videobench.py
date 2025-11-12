import unittest
from unittest.mock import patch, MagicMock, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.videobench import VideoBenchDataset, VideoBenchEvaluator


class TestVideoBenchDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.videobench.get_data_path", return_value="/fake/root")
    @patch("ais_bench.benchmark.datasets.videobench.os.path.exists", return_value=True)
    @patch("ais_bench.benchmark.datasets.videobench.Path")
    @patch("builtins.open")
    def test_load_video_path(self, mock_open_file, mock_Path, mock_exists, mock_get_path):
        # ANSWER.json
        answers = {"ds": {"k1": "A"}}
        m_ans = mock_open(read_data=str(answers).replace("'", '"'))
        # new.json 内容
        new_data = {"k1": {"vid_path": "/x/ds/file.mp4", "video_id": 1, "question": "q?", "choices": {"A": "a", "B": "b"}}}
        m_new = mock_open(read_data=str(new_data).replace("'", '"'))
        # 依次打开 ANSWER.json 与 new.json
        mock_open_file.side_effect = [m_ans.return_value, m_new.return_value]
        mock_path = MagicMock()
        mock_path.glob.return_value = ["/fake/root/ds_new.json"]
        mock_Path.return_value = mock_path

        ds = VideoBenchDataset.load(path="/any", video_type="video_path", num_frames=3)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        row = ds[0]
        self.assertIn("video_url", row)
        self.assertIn("choices_prompt", row)

    @patch("ais_bench.benchmark.datasets.videobench.get_data_path", return_value="/fake/root")
    @patch("ais_bench.benchmark.datasets.videobench.os.path.exists", return_value=True)
    @patch("ais_bench.benchmark.datasets.videobench.Path")
    @patch("builtins.open")
    @patch("ais_bench.benchmark.datasets.videobench.VideoAsset")
    @patch("ais_bench.benchmark.datasets.videobench.image_to_base64")
    def test_load_video_base64(self, mock_b64, mock_asset, mock_open_file, mock_Path, mock_exists, mock_get_path):
        import json
        answers = {"ds": {"k1": "A"}}
        m_ans = mock_open(read_data=json.dumps(answers))
        new_data = {"k1": {"vid_path": "/x/ds/file.mp4", "video_id": 1, "question": "q?", "choices": {"A": "a", "B": "b", "C": None}}}
        m_new = mock_open(read_data=json.dumps(new_data))
        mock_open_file.side_effect = [m_ans.return_value, m_new.return_value]
        mock_path = MagicMock()
        mock_path.glob.return_value = ["/fake/root/ds_new.json"]
        mock_Path.return_value = mock_path
        # mock frames
        mock_asset.return_value.pil_images = [MagicMock(), MagicMock()]
        mock_b64.side_effect = ["f1", "f2"]

        ds = VideoBenchDataset.load(path="/any", video_type="video_base64", num_frames=2)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        self.assertIn(",", ds[0]["video_url"])  # 多帧拼接

    @patch("ais_bench.benchmark.datasets.videobench.get_data_path", return_value="/fake/root")
    @patch("ais_bench.benchmark.datasets.videobench.os.path.exists", return_value=False)
    def test_missing_answer_file(self, mock_exists, mock_get_path):
        with self.assertRaises(FileNotFoundError):
            VideoBenchDataset.load(path="/any", video_type="video_path")

    @patch("ais_bench.benchmark.datasets.videobench.get_data_path", return_value="/fake/root")
    @patch("ais_bench.benchmark.datasets.videobench.os.path.exists", return_value=True)
    @patch("ais_bench.benchmark.datasets.videobench.Path")
    @patch("builtins.open")
    def test_invalid_video_type(self, mock_open_file, mock_Path, mock_exists, mock_get_path):
        # 提供含有效条目的 new.json 与 ANSWER.json，以确保进入 video_type 分支
        answers = {"ds": {"k1": "A"}}
        m_ans = mock_open(read_data=str(answers).replace("'", '"'))
        new_data = {"k1": {"vid_path": "/x/ds/file.mp4", "video_id": 1, "question": "q?", "choices": {"A": "a", "B": "b", "C": "c"}}}
        m_new = mock_open(read_data=str(new_data).replace("'", '"'))
        mock_open_file.side_effect = [m_ans.return_value, m_new.return_value]
        mock_path = MagicMock()
        mock_path.glob.return_value = ["/fake/root/new.json"]
        mock_Path.return_value = mock_path
        with self.assertRaises(ValueError):
            VideoBenchDataset.load(path="/any", video_type="bad")


class TestVideoBenchEvaluator(unittest.TestCase):
    def test_score_and_find_choice(self):
        eva = VideoBenchEvaluator()
        out = eva.score(["pred A"], references=[{"answer": "A"}])
        self.assertIn("accuracy", out)
        out2 = eva.score(["A"], references=[{"answer": "A"}, {"answer": "B"}])
        self.assertIn("error", out2)


if __name__ == '__main__':
    unittest.main()
