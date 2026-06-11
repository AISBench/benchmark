import json
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
from datasets import Dataset

from ais_bench.benchmark.datasets.mmmu_pro import (
    cot_postproc,
    MMMUProEvaluator,
    MMMUProCotEvaluator,
    MMMUProOptions10Dataset,
    MMMUProVisionDataset,
    IMAGE_MAP_LEN,
    ANSWER_STR_LEN,
)


IM_END_TAG = "</im_end>"


class TestConstants(unittest.TestCase):
    def test_image_map_len(self):
        self.assertEqual(IMAGE_MAP_LEN, 64)

    def test_answer_str_len(self):
        self.assertEqual(ANSWER_STR_LEN, 7)


class TestCotPostproc(unittest.TestCase):
    def test_single_answer(self):
        response = "Let me think step by step.\nAnswer: B"
        result = cot_postproc(response)
        self.assertEqual(result, " B")

    def test_single_answer_with_extra_text(self):
        response = "Some reasoning here.\nMore reasoning.\nAnswer: C"
        result = cot_postproc(response)
        self.assertEqual(result, " C")

    def test_multiple_answers_returns_raw(self):
        response = "Answer: A\nAnother line\nAnswer: B"
        result = cot_postproc(response)
        self.assertEqual(result, response)

    def test_no_answer_line_returns_raw(self):
        response = "Just some text without answer"
        result = cot_postproc(response)
        self.assertEqual(result, response)

    def test_empty_string(self):
        response = ""
        result = cot_postproc(response)
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        response = "   \n  \n  "
        result = cot_postproc(response)
        self.assertEqual(result, "   \n  \n  ")

    def test_answer_with_leading_trailing_whitespace(self):
        response = "  A\n  \nAnswer: A"
        result = cot_postproc(response)
        self.assertEqual(result, "A")

    def test_answer_with_multiple_uppercase(self):
        response = "Some reasoning\nAnswer: AB"
        result = cot_postproc(response)
        self.assertEqual(result, " AB")

    def test_answer_with_no_uppercase(self):
        response = "Some reasoning\nAnswer: 42"
        result = cot_postproc(response)
        self.assertEqual(result, "A")

    def test_answer_line_at_end(self):
        response = "Step 1\nStep 2\nAnswer: D"
        result = cot_postproc(response)
        self.assertEqual(result, " D")

    def test_answer_with_extra_text_after(self):
        response = "Thinking...\nAnswer: E\nMore text"
        result = cot_postproc(response)
        self.assertEqual(result, " E")

    def test_lowercase_not_detected(self):
        response = "reasoning\nAnswer: a"
        result = cot_postproc(response)
        self.assertEqual(result, "A")

    def test_surrounding_text_with_same_letter(self):
        response = "The answer is clearly A because...\nAnswer: A"
        result = cot_postproc(response)
        self.assertEqual(result, "A")


class TestMMMUProEvaluator(unittest.TestCase):
    def _make_ref(self, choices, answer, category):
        return json.dumps([choices, answer, category])

    def test_length_mismatch(self):
        evaluator = MMMUProEvaluator()
        result = evaluator.score(["A"], [self._make_ref({}, "A", "c"), self._make_ref({}, "B", "c")])
        self.assertIn("error", result)
        self.assertIn("length", result["error"])

    def test_correct_prediction(self):
        evaluator = MMMUProEvaluator()
        choices = {"A": "apple", "B": "banana", "C": "cherry"}
        ref = self._make_ref(choices, "A", "fruit")
        result = evaluator.score(["A"], [ref])
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertAlmostEqual(result["fruit"], 100.0)
        self.assertTrue(result["details"][0]["correct"])

    def test_incorrect_prediction(self):
        evaluator = MMMUProEvaluator()
        choices = {"A": "apple", "B": "banana", "C": "cherry"}
        ref = self._make_ref(choices, "A", "fruit")
        result = evaluator.score(["B"], [ref])
        self.assertAlmostEqual(result["Overall"], 0.0)
        self.assertAlmostEqual(result["fruit"], 0.0)
        self.assertFalse(result["details"][0]["correct"])

    def test_multiple_categories(self):
        evaluator = MMMUProEvaluator()
        ref1 = self._make_ref({"A": "cat", "B": "dog"}, "A", "animals")
        ref2 = self._make_ref({"A": "red", "B": "blue"}, "B", "colors")
        ref3 = self._make_ref({"A": "x", "B": "y"}, "B", "animals")
        result = evaluator.score(["A", "B", "B"], [ref1, ref2, ref3])
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertAlmostEqual(result["animals"], 100.0)
        self.assertAlmostEqual(result["colors"], 100.0)

    def test_special_character_cleaning(self):
        evaluator = MMMUProEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        pred_with_special = "A<|im_end|>"
        result = evaluator.score([pred_with_special], [ref])
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertTrue(result["details"][0]["correct"])

    def test_details_structure(self):
        evaluator = MMMUProEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        result = evaluator.score(["A"], [ref])
        detail = result["details"][0]
        self.assertIn("pred", detail)
        self.assertIn("answer", detail)
        self.assertIn("correct", detail)
        self.assertEqual(detail["pred"], "A")
        self.assertEqual(detail["answer"], ref)
        self.assertTrue(detail["correct"])

    def test_overall_key_present(self):
        evaluator = MMMUProEvaluator()
        choices = {"A": "a", "B": "b"}
        ref = self._make_ref(choices, "A", "cat")
        result = evaluator.score(["A"], [ref])
        self.assertIn("Overall", result)
        self.assertIn("details", result)
        self.assertIn("cat", result)

    def test_result_keys_sorted(self):
        evaluator = MMMUProEvaluator()
        refs = [
            self._make_ref({"A": "z", "B": "y"}, "A", "zebra"),
            self._make_ref({"A": "a", "B": "b"}, "A", "apple"),
        ]
        result = evaluator.score(["A", "A"], refs)
        keys = [k for k in result.keys() if k not in ("Overall", "details")]
        self.assertEqual(keys, sorted(keys))

    def test_mixed_correct_incorrect(self):
        evaluator = MMMUProEvaluator()
        choices = {"A": "x", "B": "y"}
        ref1 = self._make_ref(choices, "A", "cat")
        ref2 = self._make_ref(choices, "B", "cat")
        ref3 = self._make_ref(choices, "A", "dog")
        result = evaluator.score(["A", "A", "B"], [ref1, ref2, ref3])
        self.assertAlmostEqual(result["Overall"], 100 * 1 / 3)
        self.assertAlmostEqual(result["cat"], 50.0)
        self.assertAlmostEqual(result["dog"], 0.0)


class TestMMMUProCotEvaluator(unittest.TestCase):
    def _make_ref(self, choices, answer, category):
        return json.dumps([choices, answer, category])

    def test_length_mismatch(self):
        evaluator = MMMUProCotEvaluator()
        result = evaluator.score(["A"], [self._make_ref({}, "A", "c"), self._make_ref({}, "B", "c")])
        self.assertIn("error", result)

    def test_cot_single_answer_correct(self):
        evaluator = MMMUProCotEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        pred = "Let me think...\nAnswer: A"
        result = evaluator.score([pred], [ref])
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertTrue(result["details"][0]["correct"])

    def test_cot_single_answer_incorrect(self):
        evaluator = MMMUProCotEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        pred = "Let me think...\nAnswer: B"
        result = evaluator.score([pred], [ref])
        self.assertAlmostEqual(result["Overall"], 0.0)
        self.assertFalse(result["details"][0]["correct"])

    def test_cot_no_answer_line_falls_back(self):
        evaluator = MMMUProCotEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        pred = "Just apple"
        result = evaluator.score([pred], [ref])
        self.assertIn("details", result)
        self.assertIn("Overall", result)

    def test_cot_special_character_cleaning(self):
        evaluator = MMMUProCotEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        pred = "Reasoning\nAnswer: A" + IM_END_TAG
        result = evaluator.score([pred], [ref])
        self.assertAlmostEqual(result["Overall"], 100.0)

    def test_cot_multiple_categories(self):
        evaluator = MMMUProCotEvaluator()
        ref1 = self._make_ref({"A": "x", "B": "y"}, "A", "cat")
        ref2 = self._make_ref({"A": "x", "B": "y"}, "B", "dog")
        result = evaluator.score(["Think\nAnswer: A", "Think\nAnswer: B"], [ref1, ref2])
        self.assertAlmostEqual(result["Overall"], 100.0)
        self.assertAlmostEqual(result["cat"], 100.0)
        self.assertAlmostEqual(result["dog"], 100.0)

    def test_cot_details_structure(self):
        evaluator = MMMUProCotEvaluator()
        choices = {"A": "apple", "B": "banana"}
        ref = self._make_ref(choices, "A", "fruit")
        result = evaluator.score(["Reasoning\nAnswer: A"], [ref])
        detail = result["details"][0]
        self.assertIn("pred", detail)
        self.assertIn("answer", detail)
        self.assertIn("correct", detail)

    def test_cot_result_keys_sorted(self):
        evaluator = MMMUProCotEvaluator()
        refs = [
            self._make_ref({"A": "z", "B": "y"}, "A", "zebra"),
            self._make_ref({"A": "a", "B": "b"}, "A", "apple"),
        ]
        result = evaluator.score(["Answer: A", "Answer: A"], refs)
        keys = [k for k in result.keys() if k not in ("Overall", "details")]
        self.assertEqual(keys, sorted(keys))


@patch("ais_bench.benchmark.datasets.mmmu_pro.split_MMMU", side_effect=lambda x: x)
@patch("ais_bench.benchmark.datasets.mmmu_pro.build_choices", return_value={"A": "apple", "B": "banana"})
@patch("ais_bench.benchmark.datasets.mmmu_pro.get_content_str", side_effect=lambda x: "mock_content")
@patch("ais_bench.benchmark.datasets.mmmu_pro.dump_image", return_value=["/fake/img.png"])
@patch("ais_bench.benchmark.datasets.mmmu_pro.pd.read_csv")
@patch("ais_bench.benchmark.datasets.mmmu_pro.get_data_path", return_value="/fake/path.tsv")
class TestMMMUProOptions10DatasetLoad(unittest.TestCase):

    def _make_df(self, extra_cols=None):
        data = {
            "index": [1, 2],
            "question": ["What is this?", "What is that?"],
            "A": ["apple", "cat"],
            "B": ["banana", "dog"],
            "answer": ["A", "B"],
            "category": ["fruit", "animal"],
            "image": ["base64data1", "base64data2"],
        }
        if extra_cols:
            data.update(extra_cols)
        return pd.DataFrame(data)

    def test_basic_load(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProOptions10Dataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)
        self.assertIn("content", ds.column_names)
        self.assertIn("answer", ds.column_names)

    def test_load_with_hint(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df(extra_cols={"hint": ["hint1", "hint2"]})
        ds = MMMUProOptions10Dataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_load_with_cot(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProOptions10Dataset.load("/fake/path.tsv", is_cot=True)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_load_with_image_path_column(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        df = self._make_df(extra_cols={"image_path": ["/path/a.png", "/path/b.png"]})
        mock_csv.return_value = df
        ds = MMMUProOptions10Dataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_answer_json_format(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProOptions10Dataset.load("/fake/path.tsv")
        for item in ds:
            parsed = json.loads(item["answer"])
            self.assertIsInstance(parsed, list)
            self.assertEqual(len(parsed), 3)
            self.assertIsInstance(parsed[0], dict)
            self.assertIsInstance(parsed[1], str)
            self.assertIsInstance(parsed[2], str)

    def test_dump_image_called(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        MMMUProOptions10Dataset.load("/fake/path.tsv")
        self.assertEqual(mock_dump.call_count, 2)

    def test_list_image_path(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_dump.return_value = ["/fake/img1.png", "/fake/img2.png"]
        mock_csv.return_value = self._make_df()
        ds = MMMUProOptions10Dataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_file_not_found(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.side_effect = Exception("file not found")
        with self.assertRaises(FileNotFoundError):
            MMMUProOptions10Dataset.load("/nonexistent/path.tsv")


@patch("ais_bench.benchmark.datasets.mmmu_pro.split_MMMU", side_effect=lambda x: x)
@patch("ais_bench.benchmark.datasets.mmmu_pro.build_choices", return_value={"A": "apple", "B": "banana"})
@patch("ais_bench.benchmark.datasets.mmmu_pro.get_content_str", side_effect=lambda x: "mock_content")
@patch("ais_bench.benchmark.datasets.mmmu_pro.dump_image", return_value=["/fake/img.png"])
@patch("ais_bench.benchmark.datasets.mmmu_pro.pd.read_csv")
@patch("ais_bench.benchmark.datasets.mmmu_pro.get_data_path", return_value="/fake/path.tsv")
class TestMMMUProVisionDatasetLoad(unittest.TestCase):

    def _make_df(self, extra_cols=None):
        data = {
            "index": [1, 2],
            "question": ["What is this?", "What is that?"],
            "A": ["apple", "cat"],
            "B": ["banana", "dog"],
            "answer": ["A", "B"],
            "category": ["fruit", "animal"],
            "image": ["x" * 65, "x" * 65],
        }
        if extra_cols:
            data.update(extra_cols)
        return pd.DataFrame(data)

    def test_basic_load(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProVisionDataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)
        self.assertIn("content", ds.column_names)
        self.assertIn("answer", ds.column_names)

    def test_load_with_cot(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProVisionDataset.load("/fake/path.tsv", is_cot=True)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_load_with_image_path_column(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        df = self._make_df(extra_cols={"image_path": ["/path/a.png", "/path/b.png"]})
        mock_csv.return_value = df
        ds = MMMUProVisionDataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_answer_json_format(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProVisionDataset.load("/fake/path.tsv")
        for item in ds:
            parsed = json.loads(item["answer"])
            self.assertIsInstance(parsed, list)
            self.assertEqual(len(parsed), 3)

    def test_dump_image_called(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        MMMUProVisionDataset.load("/fake/path.tsv")
        self.assertEqual(mock_dump.call_count, 2)

    def test_list_image_path(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_dump.return_value = ["/fake/img1.png", "/fake/img2.png"]
        mock_csv.return_value = self._make_df()
        ds = MMMUProVisionDataset.load("/fake/path.tsv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_vision_prompt_without_cot(self, mock_path, mock_csv, mock_dump, mock_content, mock_build, mock_split):
        mock_csv.return_value = self._make_df()
        ds = MMMUProVisionDataset.load("/fake/path.tsv", is_cot=False)
        self.assertIsInstance(ds, Dataset)


if __name__ == '__main__':
    unittest.main()
