import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.datasets.geometry3k import (
    GEOMETRY3K_INSTRUCTION,
    Geometry3KDataset,
    Geometry3KEvaluator,
    _extract_boxed_content,
    _grade_answer,
    _resolve_parquet_path,
    _save_image,
    format_reward,
)


class TestExtractBoxedContent(unittest.TestCase):
    def test_normal_boxed(self):
        result = _extract_boxed_content("Some text \\boxed{42}")
        self.assertEqual(result, "42")

    def test_no_boxed(self):
        result = _extract_boxed_content("No boxed content here")
        self.assertEqual(result, "None")

    def test_multiple_boxed_returns_last(self):
        result = _extract_boxed_content("\\boxed{1} and \\boxed{2}")
        self.assertEqual(result, "2")

    def test_empty_boxed(self):
        result = _extract_boxed_content("\\boxed{}")
        self.assertEqual(result, "None")

    def test_boxed_with_think_tags(self):
        result = _extract_boxed_content(
            "<think>reasoning</think> The answer is \\boxed{3.14}"
        )
        self.assertEqual(result, "3.14")


class TestGradeAnswer(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(_grade_answer("42", "42"))

    def test_exact_match_false(self):
        self.assertFalse(_grade_answer("42", "43"))

    def test_numeric_equivalence(self):
        self.assertTrue(_grade_answer("3.14000", "3.14"))

    def test_fraction_match(self):
        self.assertTrue(_grade_answer("\\frac{1}{2}", "0.5"))

    def test_latex_superscript(self):
        self.assertTrue(_grade_answer("90^{\\circ}", "90"))

    def test_unit_text(self):
        self.assertTrue(_grade_answer("\\text{cm}", "cm"))


class TestFormatReward(unittest.TestCase):
    def test_valid_format(self):
        self.assertEqual(
            format_reward("<think>step by step</think> answer is \\boxed{42}"),
            1.0,
        )

    def test_no_think(self):
        self.assertEqual(format_reward("answer is \\boxed{42}"), 0.0)

    def test_no_boxed(self):
        self.assertEqual(format_reward("<think>step by step</think> answer is 42"), 0.0)

    def test_neither(self):
        self.assertEqual(format_reward("answer is 42"), 0.0)

    def test_think_after_boxed(self):
        """format_reward uses re.fullmatch: <think> must appear before \\boxed{}."""
        self.assertEqual(
            format_reward("\\boxed{42} <think>too late</think>"),
            0.0,
        )

    def test_partial_match_trailing_text(self):
        """re.fullmatch rejects text after the \\boxed{}."""
        self.assertEqual(
            format_reward("<think>x</think> \\boxed{42} extra"),
            0.0,
        )


class TestSaveImage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_pil_image(self):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (10, 10), color="red")
        path = _save_image(img, self.tmpdir, 0)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("0.png"))

    def test_save_bytes_dict(self):
        from io import BytesIO

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (10, 10), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_dict = {"bytes": buf.getvalue()}
        path = _save_image(img_dict, self.tmpdir, 1)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("1.png"))

    def test_save_string_path(self):
        path = _save_image("/tmp/existing.png", self.tmpdir, 2)
        self.assertEqual(path, "/tmp/existing.png")


class TestResolveParquetPath(unittest.TestCase):
    def test_absolute_file_path(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(b"")
            f.flush()
            result = _resolve_parquet_path(f.name, "test")
            self.assertEqual(result, f.name)

    def test_absolute_directory_with_split_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "data", "test-0000.parquet")
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            Path(test_file).touch()
            result = _resolve_parquet_path(tmpdir, "test")
            self.assertTrue(result.endswith("test-0000.parquet"), f"Got: {result}")

    def test_no_parquet_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(FileNotFoundError):
                _resolve_parquet_path(tmpdir, "test")


class TestGeometry3KDataset(unittest.TestCase):
    def _make_parquet(self, directory, split, rows):
        """Helper: write a minimal parquet file for testing."""
        import pandas as pd

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{split}-0000.parquet")
        df = pd.DataFrame(rows)
        df.to_parquet(path)
        return path

    def test_load_minimal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_parquet(
                os.path.join(tmpdir, "data"),
                "test",
                [
                    {
                        "problem": "What is 1+1?",
                        "answer": "2",
                        "images": [],
                    }
                ],
            )
            ds = Geometry3KDataset.load(path=tmpdir, split="test")
            self.assertEqual(len(ds), 1)
            row = ds[0]
            self.assertIn("content", row)
            self.assertIn("question", row)
            self.assertIn("image", row)
            self.assertIn("answer", row)
            self.assertIn("index", row)
            self.assertEqual(row["answer"], "2")
            self.assertEqual(row["index"], 0)

    def test_load_with_custom_instruction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_parquet(
                os.path.join(tmpdir, "data"),
                "test",
                [
                    {
                        "problem": "What is 2+2?",
                        "answer": "4",
                        "images": [],
                    }
                ],
            )
            custom_inst = "Just give the answer."
            ds = Geometry3KDataset.load(
                path=tmpdir, split="test", instruction=custom_inst
            )
            row = ds[0]
            self.assertIn(custom_inst, row["question"])

    def test_load_default_split_is_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_parquet(
                os.path.join(tmpdir, "data"),
                "test",
                [
                    {
                        "problem": "Q",
                        "answer": "A",
                        "images": [],
                    }
                ],
            )
            ds = Geometry3KDataset.load(path=tmpdir)
            self.assertEqual(len(ds), 1)


class TestGeometry3KEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Geometry3KEvaluator(format_weight=0.1)

    def test_compute_score_perfect(self):
        pred = "<think>1+1=2</think> \\boxed{2}"
        result = self.evaluator._compute_score(pred, "2")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["format_score"], 1.0)
        self.assertEqual(result["combined_score"], 1.0)

    def test_compute_score_wrong_answer(self):
        pred = "<think>x</think> \\boxed{999}"
        result = self.evaluator._compute_score(pred, "42")
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["format_score"], 1.0)
        # combined = 0.9 * 0 + 0.1 * 1 = 0.1
        self.assertAlmostEqual(result["combined_score"], 0.1)

    def test_compute_score_no_format(self):
        pred = "answer is 2"
        result = self.evaluator._compute_score(pred, "2")
        self.assertEqual(result["format_score"], 0.0)
        # combined = 0.9 * acc + 0.1 * 0
        self.assertAlmostEqual(
            result["combined_score"], 0.9 * result["accuracy"]
        )

    def test_compute_score_format_weight_zero(self):
        eva = Geometry3KEvaluator(format_weight=0.0)
        pred = "answer is 2"
        result = eva._compute_score(pred, "2")
        # format_weight=0 → combined = 1.0*acc + 0*fmt = acc
        self.assertEqual(result["combined_score"], result["accuracy"])

    def test_compute_score_strips_special_tokens(self):
        pred = "<think>x</think> \\boxed{2}<|im_end|>"
        result = self.evaluator._compute_score(pred, "2")
        self.assertEqual(result["accuracy"], 1.0)

    def test_score_different_length(self):
        result = self.evaluator.score(["a"], ["1", "2"])
        self.assertIn("error", result)

    def test_score_all_correct(self):
        predictions = [
            "<think>reason</think> \\boxed{2}",
            "<think>math</think> \\boxed{4}",
        ]
        references = ["2", "4"]
        result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["combined_score"], 100.0)

    def test_score_all_wrong(self):
        predictions = [
            "not even trying",
            "<think>x</think> \\boxed{0}",
        ]
        references = ["2", "4"]
        result = self.evaluator.score(predictions, references)
        self.assertLess(result["combined_score"], 50.0)

    def test_score_mixed(self):
        predictions = [
            "<think>calc</think> \\boxed{2}",   # perfect
            "no box",                             # no format, maybe correct
        ]
        references = ["2", "no box"]
        result = self.evaluator.score(predictions, references)
        # combined scores: 1st = 1.0, 2nd = 0.9*1.0 + 0.1*0 = 0.9
        expected = 100.0 * (1.0 + 0.9) / 2  # = 95.0
        self.assertAlmostEqual(result["combined_score"], expected)

    def test_score_returns_details(self):
        predictions = ["<think>x</think> \\boxed{5}"]
        references = ["5"]
        result = self.evaluator.score(predictions, references)
        self.assertIn("details", result)
        self.assertIn("combined_score", result)
        self.assertNotIn("accuracy", result)
        self.assertNotIn("format_score", result)

    def test_score_reference_is_dict(self):
        predictions = ["<think>x</think> \\boxed{7}"]
        references = [{"answer": "7"}]
        result = self.evaluator.score(predictions, references)
        self.assertAlmostEqual(result["combined_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
