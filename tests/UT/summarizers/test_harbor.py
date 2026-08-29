import csv
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from ais_bench.benchmark.summarizers.harbor import HarborSummarizer
from ais_bench.benchmark.utils.config import ConfigDict


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestHarborSummarizerBuildRow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = {
            "abbr": "m",
            "agent_name": "terminus-2",
            "model_names": ["openai/qwen3", "hosted_vllm/qwen3"],
        }
        self.dataset_abbr = "d"
        self.summarizer = HarborSummarizer(
            ConfigDict(
                {
                    "work_dir": self.temp_dir,
                    "models": [self.model_cfg],
                    "datasets": [{"abbr": self.dataset_abbr}],
                }
            )
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _write_result(self, data):
        _write_json(
            os.path.join(self.temp_dir, "results", "m", "d.json"), data
        )

    def test_missing_file_returns_none(self):
        self.assertIsNone(self.summarizer._build_row(self.model_cfg, self.dataset_abbr))

    def test_counts_and_agent(self):
        self._write_result(
            {
                "avg_score": 0.75,
                "n_errors": 1,
                "reward_distribution": [
                    {"score": 1.0, "count": 2},
                    {"score": 0.5, "count": 1},
                    {"score": 0.0, "count": 1},
                ],
            }
        )
        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertEqual(row["agent"], "terminus-2")
        self.assertEqual(row["model_name"], "openai/qwen3, hosted_vllm/qwen3")
        self.assertEqual(row["dataset"], "d")
        self.assertEqual(row["avg_score"], 0.75)
        self.assertEqual(row["correct"], 2)
        self.assertEqual(row["wrong"], 2)
        self.assertEqual(row["exception"], 1)

    def test_model_name_empty_dash(self):
        cfg = {"abbr": "m", "agent_name": "oracle", "model_names": []}
        self._write_result({"avg_score": 0.5, "reward_distribution": []})
        row = self.summarizer._build_row(cfg, self.dataset_abbr)
        self.assertEqual(row["model_name"], "-")


class TestHarborSummarizerSummarize(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_writes_csv_and_prints(self):
        model = {"abbr": "m", "agent_name": "oracle", "model_names": ["mm"]}
        cfg = ConfigDict(
            {
                "work_dir": self.temp_dir,
                "models": [model],
                "datasets": [{"abbr": "d"}],
            }
        )
        _write_json(
            os.path.join(self.temp_dir, "results", "m", "d.json"),
            {"avg_score": 0.8, "n_errors": 0, "reward_distribution": []},
        )
        summarizer = HarborSummarizer(cfg)
        with mock.patch("builtins.print"):
            summarizer.summarize(time_str="20260101")
        csv_path = os.path.join(self.temp_dir, "summary", "summary_20260101.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][0], "agent")
        self.assertEqual(rows[1][0], "oracle")

    def test_no_results_warns(self):
        summarizer = HarborSummarizer(
            ConfigDict(
                {
                    "work_dir": self.temp_dir,
                    "models": [{"abbr": "m", "agent_name": "oracle"}],
                    "datasets": [{"abbr": "d"}],
                }
            )
        )
        summarizer.logger = mock.MagicMock()
        with mock.patch("builtins.print"):
            summarizer.summarize()
        summarizer.logger.warning.assert_called_once()
        self.assertFalse(
            os.path.exists(os.path.join(self.temp_dir, "summary"))
        )


if __name__ == "__main__":
    unittest.main()