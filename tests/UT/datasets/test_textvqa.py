import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.textvqa import (
    TEXTVQADataset,
    TEXTEvaluator,
    VQAEvalMethod,
    MAX_TARGET_LENGTH,
)


class TestTEXTVQA(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.textvqa.get_data_path", return_value="/root/q.json")
    @patch("builtins.open")
    def test_load_image_path(self, mock_open_file, mock_get_path):
        # annotations 文件
        annot = {"annotations": [{"question_id": 1, "answers": [{"answer": "foo"}]}]}
        m_ann = mock_open(read_data=str(annot).replace("'", '"'))
        # questions 文件（每行 JSON）
        q_line = '{"question_id": 1, "image": "/img.png", "question": "xxx"}'
        m_q = mock_open(read_data=q_line + "\n")
        mock_open_file.side_effect = [m_ann.return_value, m_q.return_value]
        ds = TEXTVQADataset.load("/any", image_type="image_path")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["image_url"], "/img.png")

    @patch("ais_bench.benchmark.datasets.textvqa.get_data_path", return_value="/root/q.json")
    @patch("builtins.open")
    def test_load_image_base64(self, mock_open_file, mock_get_path):
        annot = {"annotations": [{"question_id": 1, "answers": [{"answer": "foo"}]}]}
        m_ann = mock_open(read_data=str(annot).replace("'", '"'))
        q_line = '{"question_id": 1, "image": "/img.png", "question": "xxx"}'
        m_q = mock_open(read_data=q_line + "\n")
        m_img = mock_open()
        m_img.return_value.read.return_value = b"bin"
        mock_open_file.side_effect = [m_ann.return_value, m_q.return_value, m_img.return_value]
        ds = TEXTVQADataset.load("/any", image_type="image_base64")
        self.assertIsInstance(ds, Dataset)
        self.assertTrue(isinstance(ds[0]["image_url"], str))


class TestTEXTEvaluator(unittest.TestCase):
    def test_score(self):
        eva = TEXTEvaluator()
        preds = ["The answer is Foo!"]
        refs = [[{"answer": "foo"}, {"answer": "foo"}, {"answer": "bar"}]]
        out = eva.score(preds, refs)
        self.assertIn("accuracy", out)
        out2 = eva.score(["a"], [[{"answer": "a"}], [{"answer": "a"}]])
        self.assertIn("error", out2)

    def test_eval_method_exceptions(self):
        method = VQAEvalMethod()
        long_text = 'x' * (MAX_TARGET_LENGTH + 1)
        with self.assertRaises(ValueError):
            method.process_punctuation(long_text)
        with self.assertRaises(ValueError):
            method.process_digit_article(long_text)
        self.assertEqual(method.remove_special_characters('foo<unk>bar'), 'foobar')


if __name__ == '__main__':
    unittest.main()
