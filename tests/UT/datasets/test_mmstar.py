import json
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
from datasets import Dataset

from ais_bench.benchmark.datasets.mmstar import (
    MMStarDataset,
    MMStarEvaluator,
    IMAGE_MAP_LEN,
)


class TestMMStarEvaluatorScore(unittest.TestCase):
    def setUp(self):
        self.evaluator = MMStarEvaluator()

    def test_length_mismatch(self):
        predictions = ["A"]
        references = [
            {"choices": '{"A": "1", "B": "2"}', "answer": "A", "category": "math"},
            {"choices": '{"A": "3", "B": "4"}', "answer": "B", "category": "science"},
        ]
        result = self.evaluator.score(predictions, references)
        self.assertIn("error", result)
        self.assertIn("different length", result["error"])

    def test_correct_prediction(self):
        predictions = ["A"]
        references = [
            {"choices": '{"A": "1", "B": "2"}', "answer": "A", "category": "math"},
        ]
        with patch(
            "ais_bench.benchmark.datasets.mmstar.can_infer", return_value="A"
        ):
            result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertAlmostEqual(result["math"], 100.0)
        self.assertTrue(result["details"][0]["correct"])

    def test_incorrect_prediction(self):
        predictions = ["B"]
        references = [
            {"choices": '{"A": "1", "B": "2"}', "answer": "A", "category": "math"},
        ]
        with patch(
            "ais_bench.benchmark.datasets.mmstar.can_infer", return_value="B"
        ):
            result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["Overall"], 0.0)
        self.assertAlmostEqual(result["math"], 0.0)
        self.assertFalse(result["details"][0]["correct"])

    def test_multiple_categories(self):
        predictions = ["A", "B", "A"]
        references = [
            {"choices": '{"A": "1", "B": "2"}', "answer": "A", "category": "math"},
            {"choices": '{"A": "3", "B": "4"}', "answer": "B", "category": "science"},
            {"choices": '{"A": "5", "B": "6"}', "answer": "A", "category": "math"},
        ]
        with patch(
            "ais_bench.benchmark.datasets.mmstar.can_infer",
            side_effect=["A", "B", "B"],
        ):
            result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["Overall"], 100 * 2 / 3)
        self.assertAlmostEqual(result["math"], 50.0)
        self.assertAlmostEqual(result["science"], 100.0)
        self.assertEqual(len(result["details"]), 3)

    def test_details_structure(self):
        predictions = ["A"]
        references = [
            {
                "choices": '{"A": "x", "B": "y"}',
                "answer": "A",
                "category": "cat1",
            },
        ]
        with patch(
            "ais_bench.benchmark.datasets.mmstar.can_infer", return_value="A"
        ):
            result = self.evaluator.score(predictions, references)
        detail = result["details"][0]
        self.assertIn("pred", detail)
        self.assertIn("answer", detail)
        self.assertIn("correct", detail)
        self.assertEqual(detail["pred"], "A")
        self.assertEqual(detail["answer"], references[0])
        self.assertTrue(detail["correct"])

    def test_all_correct(self):
        predictions = ["A", "B"]
        references = [
            {"choices": '{"A": "1"}', "answer": "A", "category": "c1"},
            {"choices": '{"B": "2"}', "answer": "B", "category": "c2"},
        ]
        with patch(
            "ais_bench.benchmark.datasets.mmstar.can_infer",
            side_effect=["A", "B"],
        ):
            result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertAlmostEqual(result["c1"], 100.0)
        self.assertAlmostEqual(result["c2"], 100.0)
        for d in result["details"]:
            self.assertTrue(d["correct"])

    def test_all_wrong(self):
        predictions = ["B", "A"]
        references = [
            {"choices": '{"A": "1"}', "answer": "A", "category": "c1"},
            {"choices": '{"B": "2"}', "answer": "B", "category": "c2"},
        ]
        with patch(
            "ais_bench.benchmark.datasets.mmstar.can_infer",
            side_effect=["B", "A"],
        ):
            result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["Overall"], 0.0)
        self.assertAlmostEqual(result["c1"], 0.0)
        self.assertAlmostEqual(result["c2"], 0.0)
        for d in result["details"]:
            self.assertFalse(d["correct"])


class TestMMStarDatasetLoad(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "opt1", "B": "opt2"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    @patch("builtins.open", create=True)
    def test_load_basic(self, mock_open_file, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["a" * 100],
            "question": ["What is this?"],
            "A": ["opt1"],
            "B": ["opt2"],
            "answer": ["A"],
            "category": ["math"],
            "split": ["test"],
            "l2-category": ["algebra"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        row = ds[0]
        self.assertIn("content", row)
        self.assertIn("answer", row)
        self.assertIn("<AIS_IMAGE_START>", row["content"])
        self.assertIn("What is this?", row["content"])
        answer = row["answer"]
        self.assertEqual(answer["answer"], "A")
        self.assertEqual(answer["category"], "math")
        self.assertEqual(answer["split"], "test")
        self.assertEqual(answer["l2-category"], "algebra")

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "opt1"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_no_image_column(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "question": ["Q1?"],
            "A": ["opt1"],
            "answer": ["A"],
            "category": ["science"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertEqual(len(ds), 1)
        self.assertIn("Q1?", ds[0]["content"])

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_with_hint(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Q?"],
            "A": ["a"],
            "answer": ["A"],
            "category": ["cat"],
            "hint": ["some hint"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertIn("Hint: some hint", ds[0]["content"])

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_without_hint(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Q?"],
            "A": ["a"],
            "answer": ["A"],
            "category": ["cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertNotIn("Hint:", ds[0]["content"])

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_with_image_path(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Q?"],
            "A": ["a"],
            "answer": ["A"],
            "category": ["cat"],
            "image_path": ["/some/path.png"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertEqual(len(ds), 1)
        answer = ds[0]["answer"]
        self.assertEqual(answer["category"], "cat")

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_image_map_short_entry_redirect(self, mock_get_path, mock_build_choices, mock_dump_image):
        long_img = "x" * 100
        short_ref = "1"
        df = pd.DataFrame({
            "index": ["0", "1"],
            "image": [short_ref, long_img],
            "question": ["Q1?", "Q2?"],
            "A": ["a1", "a2"],
            "answer": ["A", "A"],
            "category": ["cat", "cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertEqual(len(ds), 2)

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_no_options(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Open-ended Q?"],
            "answer": ["some_answer"],
            "category": ["cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertNotIn("Options:", ds[0]["content"])
        self.assertIn("Open-ended Q?", ds[0]["content"])

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a", "B": "b"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_options_prompt(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Pick one?"],
            "A": ["a"],
            "B": ["b"],
            "answer": ["A"],
            "category": ["cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        content = ds[0]["content"]
        self.assertIn("Options:", content)
        self.assertIn("Please select the correct answer", content)

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_integer_index(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0, 1],
            "image": ["x" * 65, "x" * 65],
            "question": ["Q1?", "Q2?"],
            "A": ["a0", "a1"],
            "answer": ["A", "A"],
            "category": ["cat", "cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertEqual(len(ds), 2)

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_string_index(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": ["idx0", "idx1"],
            "image": ["x" * 65, "x" * 65],
            "question": ["Q1?", "Q2?"],
            "A": ["a0", "a1"],
            "answer": ["A", "A"],
            "category": ["cat", "cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertEqual(len(ds), 2)

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_filters_nan_image(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0, 1, 2],
            "image": ["x" * 65, float("nan"), "x" * 65],
            "question": ["Q1?", "Q2?", "Q3?"],
            "A": ["a0", "a1", "a2"],
            "answer": ["A", "A", "A"],
            "category": ["cat", "cat", "cat"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertEqual(len(ds), 2)

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_hint_nan(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Q?"],
            "A": ["a"],
            "answer": ["A"],
            "category": ["cat"],
            "hint": [float("nan")],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        self.assertNotIn("Hint:", ds[0]["content"])

    @patch("ais_bench.benchmark.datasets.mmstar.dump_image", return_value=["/fake/img.png"])
    @patch("ais_bench.benchmark.datasets.mmstar.build_choices", return_value={"A": "a"})
    @patch("ais_bench.benchmark.datasets.mmstar.get_data_path", return_value="/fake/data.tsv")
    def test_load_answer_dict_keys(self, mock_get_path, mock_build_choices, mock_dump_image):
        df = pd.DataFrame({
            "index": [0],
            "image": ["x" * 65],
            "question": ["Q?"],
            "A": ["a"],
            "answer": ["A"],
            "category": ["cat"],
            "split": ["val"],
            "l2-category": ["sub"],
        })
        with patch("pandas.read_csv", return_value=df):
            ds = MMStarDataset.load("/fake/data.tsv")
        answer = ds[0]["answer"]
        self.assertIn("choices", answer)
        self.assertIn("answer", answer)
        self.assertIn("split", answer)
        self.assertIn("l2-category", answer)
        self.assertIn("category", answer)
        parsed_choices = json.loads(answer["choices"])
        self.assertEqual(parsed_choices, {"A": "a"})


if __name__ == "__main__":
    unittest.main()
