import unittest
from unittest.mock import patch, mock_open, MagicMock

from datasets import Dataset

from ais_bench.benchmark.datasets.sharegpt import ShareGPTDataset, ShareGPTEvaluator


class TestShareGPT(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.sharegpt.get_data_path", return_value="/fake/data.json")
    @patch("ais_bench.benchmark.datasets.sharegpt.load_tokenizer")
    @patch("builtins.open")
    def test_load_disable_shuffle(self, mock_open_file, mock_load_tokenizer, mock_get_path):
        data = [
            {
                "id": "1",
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {"from": "assistant", "value": "hello"}
                ]
            }
        ]
        m = mock_open(read_data=str(data).replace("'", '"'))
        mock_open_file.return_value = m.return_value
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3]
        mock_load_tokenizer.return_value = tok
        ds = ShareGPTDataset.load("/any", disable_shuffle=True, model_path="/tok")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.sharegpt.get_data_path", return_value="/fake/data.json")
    @patch("ais_bench.benchmark.datasets.sharegpt.load_tokenizer")
    @patch("builtins.open")
    def test_load_with_shuffle_and_tokenizer_error(self, mock_open_file, mock_load_tokenizer, mock_get_path):
        # 两个对话，确保长度>=2且human在前
        data = [
            {
                "id": "1",
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {"from": "assistant", "value": "hello"},
                    {"from": "human", "value": "again"},
                    {"from": "assistant", "value": "ok"}
                ]
            }
        ]
        m = mock_open(read_data=str(data).replace("'", '"'))
        mock_open_file.return_value = m.return_value
        tok = MagicMock()
        # 触发 except 分支
        tok.encode.side_effect = Exception("encode error")
        mock_load_tokenizer.return_value = tok
        ds = ShareGPTDataset.load("/any", disable_shuffle=False, model_path="/tok")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.sharegpt.get_data_path", return_value="/fake/data.json")
    @patch("ais_bench.benchmark.datasets.sharegpt.load_tokenizer")
    @patch("builtins.open")
    def test_filtering_logic(self, mock_open_file, mock_load_tokenizer, mock_get_path):
        data = [
            {  # 奇数对话 -> 被过滤
                "id": "1",
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {"from": "assistant", "value": "hello"},
                    {"from": "assistant", "value": "extra"}
                ]
            },
            {  # 非human开头 -> 被过滤
                "id": "2",
                "conversations": [
                    {"from": "assistant", "value": "hi"},
                    {"from": "human", "value": "hello"}
                ]
            },
            {  # 仅合法的一个
                "id": "3",
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {"from": "assistant", "value": "hello"}
                ]
            }
        ]
        m = mock_open(read_data=str(data).replace("'", '"'))
        mock_open_file.return_value = m.return_value
        tok = MagicMock()
        tok.encode.return_value = [1]
        mock_load_tokenizer.return_value = tok
        ds = ShareGPTDataset.load("/any", disable_shuffle=True, model_path="/tok")
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]['id'], '3')

    def test_evaluator(self):
        eva = ShareGPTEvaluator()
        res = eva.score([["A"]], ["laughter"])
        self.assertIn("accuracy", res)
        res2 = eva.score(["A"], ["B", "C"])  # 长度不一致
        self.assertIn("error", res2)


if __name__ == '__main__':
    unittest.main()
