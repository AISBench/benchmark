import base64
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ais_bench.benchmark.configs.datasets.mme.mme_gen_base64 import (
    mme_datasets, mme_infer_cfg)
from ais_bench.benchmark.datasets.mme import MMEDataset, MMEEvaluator
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate


def _row(question_id, image_name, question, answer, category, image_bytes=b"img"):
    return {
        "question_id": question_id,
        "image": {"bytes": image_bytes, "path": image_name},
        "question": question,
        "answer": answer,
        "category": category,
    }


class TestMMEDataset(unittest.TestCase):

    def test_loads_sorted_shards_and_keeps_question_verbatim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    _row(
                        "landmark/1",
                        "1.png",
                        "Is this a landmark? Please answer yes or no.",
                        "Yes",
                        "landmark",
                        b"first",
                    )
                ]
            ).to_parquet(data_dir / "test-00000-of-00002.parquet")
            pd.DataFrame(
                [
                    _row(
                        "landmark/1",
                        "1.png",
                        "Is this a vehicle? Please answer yes or no.",
                        "No",
                        "landmark",
                        b"second",
                    )
                ]
            ).to_parquet(data_dir / "test-00001-of-00002.parquet")

            dataset = MMEDataset.load(str(data_dir))

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                dataset[0]["question"],
                "Is this a landmark? Please answer yes or no.",
            )
            self.assertEqual(
                dataset[0]["image"],
                base64.b64encode(b"first").decode("ascii"),
            )
            self.assertIn("data:image/png;base64,", dataset[0]["content"])
            self.assertEqual(dataset[0]["reference"]["image_name"], "1.png")
            self.assertEqual(
                [item["answer"] for item in dataset], ["Yes", "No"]
            )

            template = MMPromptTemplate(
                template=mme_infer_cfg["prompt_template"]["template"]
            )
            prompt = template.generate_item(dataset[0])
            prompt_mm = next(
                item["prompt_mm"] for item in prompt if "prompt_mm" in item
            )
            self.assertEqual(prompt_mm[0]["type"], "image_url")
            self.assertTrue(
                prompt_mm[0]["image_url"]["url"].startswith(
                    "data:image/png;base64,"
                )
            )
            self.assertEqual(prompt_mm[1]["type"], "text")
            self.assertEqual(prompt_mm[1]["text"], dataset[0]["question"])

    def test_config_uses_required_local_data_path(self):
        self.assertEqual(mme_datasets[0]["path"], r"C:\需求\MME\MME\data")

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            pd.DataFrame([{"question_id": "x"}]).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                MMEDataset.load(str(path))

    def test_invalid_image_bytes_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            row = _row(
                "existence/1", "1.jpg", "Question", "Yes", "existence"
            )
            row["image"] = {"bytes": None, "path": "1.jpg"}
            pd.DataFrame([row]).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "image bytes"):
                MMEDataset.load(str(path))


class TestMMEEvaluator(unittest.TestCase):

    @staticmethod
    def _references():
        return [
            {
                "question_id": "existence/1",
                "image_name": "1.jpg",
                "question": "Is A present?",
                "answer": "Yes",
                "category": "existence",
            },
            {
                "question_id": "existence/1",
                "image_name": "1.jpg",
                "question": "Is B present?",
                "answer": "No",
                "category": "existence",
            },
            {
                "question_id": "commonsense_reasoning/2",
                "image_name": "2.png",
                "question": "Is C true?",
                "answer": "Yes",
                "category": "commonsense_reasoning",
            },
            {
                "question_id": "commonsense_reasoning/2",
                "image_name": "2.png",
                "question": "Is D true?",
                "answer": "No",
                "category": "commonsense_reasoning",
            },
        ]

    def test_official_acc_acc_plus_and_txt_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = MMEEvaluator()
            evaluator._out_dir = tmpdir
            result = evaluator.score(
                ["yes\nreason", "no, it is not", "yes", "yes"],
                self._references(),
            )

            self.assertEqual(result["ACC"], 75.0)
            self.assertEqual(result["ACC+"], 50.0)
            self.assertEqual(result["existence/ACC"], 100.0)
            self.assertEqual(result["existence/ACC+"], 100.0)
            self.assertEqual(result["Perception"], 200.0)
            self.assertEqual(result["Cognition"], 50.0)
            self.assertEqual(result["MME Score"], 250.0)

            result_dir = Path(tmpdir) / "mme_results"
            existence_lines = (result_dir / "existence.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(existence_lines), 2)
            self.assertEqual(len(existence_lines[0].split("\t")), 4)
            self.assertIn("yes reason", existence_lines[0])

    def test_prediction_parser_matches_official_prefix_rule(self):
        self.assertEqual(MMEEvaluator.parse_pred_answer("yes"), "yes")
        self.assertEqual(MMEEvaluator.parse_pred_answer("No."), "no")
        self.assertEqual(MMEEvaluator.parse_pred_answer("yes, because"), "yes")
        self.assertEqual(MMEEvaluator.parse_pred_answer("The answer is yes"), "other")

    def test_incomplete_pair_raises(self):
        evaluator = MMEEvaluator(write_results=False)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            evaluator.score(["yes"], self._references()[:1])

    def test_length_mismatch_raises(self):
        evaluator = MMEEvaluator(write_results=False)
        with self.assertRaisesRegex(ValueError, "different length"):
            evaluator.score(["yes"], [])

    def test_empty_input(self):
        evaluator = MMEEvaluator(write_results=False)
        result = evaluator.score([], [])
        self.assertEqual(result["ACC"], 0.0)
        self.assertEqual(result["ACC+"], 0.0)


if __name__ == "__main__":
    unittest.main()
