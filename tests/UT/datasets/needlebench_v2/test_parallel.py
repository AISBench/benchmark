import unittest
from unittest.mock import patch, mock_open, MagicMock
import json

from datasets import Dataset

from ais_bench.benchmark.datasets.needlebench_v2.parallel import (
    get_unique_entries,
    NeedleBenchParallelDataset,
    NeedleBenchParallelEvaluator,
)


class TestGetUniqueEntries(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("random.shuffle")
    def test_get_unique_entries_basic(self, mock_shuffle, mock_open_file):
        """测试基本的 get_unique_entries 功能"""
        lines = [
            '{"language": "English", "arg1": "a1", "arg2": "a2", "needle": "n1"}\n',
            '{"language": "English", "arg1": "a3", "arg2": "a4", "needle": "n2"}\n',
        ]
        mock_open_file.return_value.readlines.return_value = lines
        mock_shuffle.return_value = None
        
        results = get_unique_entries("/fake/path", 2, "English", 
                                    unique_arg1=True, unique_arg2=True, unique_combination=True)
        self.assertEqual(len(results), 2)

    @patch("builtins.open", new_callable=mock_open)
    @patch("random.shuffle")
    def test_get_unique_entries_wrong_language(self, mock_shuffle, mock_open_file):
        """测试过滤错误语言"""
        lines = [
            '{"language": "Chinese", "arg1": "a1", "arg2": "a2"}\n',
        ]
        mock_open_file.return_value.readlines.return_value = lines
        mock_shuffle.return_value = None
        
        results = get_unique_entries("/fake/path", 2, "English")
        self.assertEqual(len(results), 0)

    @patch("builtins.open", new_callable=mock_open)
    @patch("random.shuffle")
    def test_get_unique_entries_invalid_json(self, mock_shuffle, mock_open_file):
        """测试处理无效 JSON"""
        lines = [
            '{"language": "English", "arg1": "a1"}\n',
            'invalid json\n',
            '{"language": "English", "arg1": "a2"}\n',
        ]
        mock_open_file.return_value.readlines.return_value = lines
        mock_shuffle.return_value = None
        
        results = get_unique_entries("/fake/path", 2, "English")
        self.assertEqual(len(results), 2)


class TestNeedleBenchParallelDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path", return_value="/fake/path")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.os.environ.get", return_value=None)
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.os.path.join")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_basic(self, mock_open_file, mock_tiktoken, mock_get_unique, 
                       mock_join, mock_environ_get, mock_get_path):
        """测试基本的 load 功能"""
        # Mock tiktoken
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "decoded text"
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        # Mock get_unique_entries
        mock_get_unique.return_value = [
            {"needle": "n1", "arg2": "k1", "retrieval_question": "Q1? 'A1'."},
            {"needle": "n2", "arg2": "k2", "retrieval_question": "Q2? 'A2'."},
        ]
        
        # Mock file reading
        file_content = '{"text": "context text"}'
        mock_open_file.return_value.__iter__ = lambda self: iter([file_content + "\n"])
        mock_join.side_effect = lambda *args: "/".join(args)
        
        ds = NeedleBenchParallelDataset.load(
            path="/any",
            needle_file_name="needles.jsonl",
            length=1000,
            depths=[10, 20],
            tokenizer_model="gpt-3.5-turbo",
            file_list=["PaulGrahamEssays.jsonl"],
            num_repeats_per_file=1,
            length_buffer=100,
            language="English",
            quesiton_position="End"
        )
        self.assertIsInstance(ds, Dataset)  

    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path", return_value="/fake/path")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.os.environ.get", return_value=None)
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.os.path.join")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_chinese(self, mock_open_file, mock_tiktoken, mock_get_unique,
                         mock_join, mock_environ_get, mock_get_path):
        """测试中文语言"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "解码文本"
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        mock_get_unique.return_value = [
            {"needle": "n1", "arg2": "k1", "retrieval_question": "问题1？'答案1'。"},
        ]
        
        file_content = '{"text": "上下文"}'
        mock_open_file.return_value.__iter__ = lambda self: iter([file_content + "\n"])
        mock_join.side_effect = lambda *args: "/".join(args)
        
        ds = NeedleBenchParallelDataset.load(
            path="/any",
            needle_file_name="needles.jsonl",
            length=1000,
            depths=[10],
            tokenizer_model="gpt-3.5-turbo",
            file_list=["zh_general.jsonl"],
            num_repeats_per_file=1,
            length_buffer=100,
            language="Chinese",
            quesiton_position="End"
        )
        self.assertIsInstance(ds, Dataset)

    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path", return_value="/fake/path")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.os.environ.get", return_value=None)
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.os.path.join")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries")
    @patch("ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_invalid_question_position(self, mock_open_file, mock_tiktoken, mock_get_unique,
                                           mock_join, mock_environ_get, mock_get_path):
        """测试无效的问题位置"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "decoded text"
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        mock_get_unique.return_value = [
            {"needle": "n1", "arg2": "k1", "retrieval_question": "Q1? 'A1'."},
        ]
        mock_join.side_effect = lambda *args: "/".join(args)
        file_content = '{"text": "context"}'
        mock_open_file.return_value.__iter__ = lambda self: iter([file_content + "\n"])
        
        with self.assertRaises(ValueError):
            NeedleBenchParallelDataset.load(
                path="/any",
                needle_file_name="needles.jsonl",
                length=1000,
                depths=[10],
                tokenizer_model="gpt-3.5-turbo",
                file_list=["PaulGrahamEssays.jsonl"],
                num_repeats_per_file=1,
                length_buffer=100,
                language="English",
                quesiton_position="Invalid"
            )


class TestNeedleBenchParallelEvaluator(unittest.TestCase):
    def test_score_success(self):
        """测试评分成功"""
        evaluator = NeedleBenchParallelEvaluator()
        predictions = ["keyword1 and keyword2"]
        gold = ["keyword1*keyword2#10*20"]
        
        with patch('builtins.print'):  # Suppress print output
            result = evaluator.score(predictions, gold)
        
        self.assertIn("average_score", result)
        self.assertIn("details", result)
        self.assertIn("Depth10", result)
        self.assertIn("Depth20", result)

    def test_score_length_mismatch(self):
        """测试长度不匹配"""
        evaluator = NeedleBenchParallelEvaluator()
        result = evaluator.score(["pred1"], ["ref1", "ref2"])
        self.assertIn("error", result)

    def test_score_partial_match(self):
        """测试部分匹配"""
        evaluator = NeedleBenchParallelEvaluator()
        predictions = ["keyword1 only"]
        gold = ["keyword1*keyword2#10*20"]
        
        with patch('builtins.print'):
            result = evaluator.score(predictions, gold)
        
        self.assertIn("average_score", result)
        self.assertGreater(result["average_score"], 0)
        self.assertLess(result["average_score"], 100)


if __name__ == "__main__":
    unittest.main()
