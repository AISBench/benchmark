import csv
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from ais_bench.benchmark.summarizers.deepswe import DeepSWESummarizer
from ais_bench.benchmark.utils.config import ConfigDict


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _resolved_trial(reward=1.0, f2p=1, p2p=1, partial=0,
                    tok_in=100, tok_out=200, started="2026-01-01T10:00:00Z",
                    finished="2026-01-01T10:00:30Z"):
    return {
        "exception_info": None,
        "verifier_result": {
            "rewards": {"reward": reward, "f2p": f2p, "p2p": p2p,
                        "partial": partial},
        },
        "agent_result": {
            "n_input_tokens": tok_in,
            "n_output_tokens": tok_out,
        },
        "started_at": started,
        "finished_at": finished,
    }


class TestDeepSWESummarizerBuildRow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = {
            "abbr": "m",
            "agent_name": "mini-swe-agent",
            "model_names": ["openai/deepseek-v4-pro"],
        }
        self.dataset_abbr = "d"
        self.summarizer = DeepSWESummarizer(
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

    def _write_trial(self, name, data):
        _write_json(
            os.path.join(self.temp_dir, "results", "m", "d", "details", name,
                         "result.json"),
            data,
        )

    def test_missing_details_dir_returns_none(self):
        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertIsNone(row)

    def test_empty_details_dir_returns_zeroed_row(self):
        # 目录存在但无 trial：不返回 None，而是全零/'-' 的行
        os.makedirs(
            os.path.join(self.temp_dir, "results", "m", "d", "details"))
        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertIsNotNone(row)
        self.assertEqual(row["n_trials"], 0)
        self.assertEqual(row["resolved"], 0)
        self.assertEqual(row["resolved_rate"], "-")
        self.assertEqual(row["avg_time_sec"], "-")

    def test_counts_and_metrics(self):
        # trial 1: resolved, 30s
        self._write_trial("t1", _resolved_trial())
        # trial 2: unresolved, f2p=0, partial=0.5, 60s
        self._write_trial(
            "t2",
            _resolved_trial(reward=0, f2p=0, p2p=1, partial=0.5,
                             tok_in=50, tok_out=150,
                             started="2026-01-01T11:00:00Z",
                             finished="2026-01-01T11:01:00Z"))
        # trial 3: exception
        self._write_trial(
            "t3", {"exception_info": {"name": "SomeError"}})

        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertEqual(row["agent"], "mini-swe-agent")
        self.assertEqual(row["model_name"], "openai/deepseek-v4-pro")
        self.assertEqual(row["dataset"], "d")
        self.assertEqual(row["n_trials"], 3)
        self.assertEqual(row["resolved"], 1)
        self.assertEqual(row["resolved_rate"], round(1 / 3, 4))
        self.assertEqual(row["avg_f2p"], 0.5)
        self.assertEqual(row["avg_p2p"], 1.0)
        self.assertEqual(row["avg_partial"], 0.25)
        self.assertEqual(row["exceptions"], 1)
        self.assertEqual(row["input_tokens"], 150)
        self.assertEqual(row["output_tokens"], 350)
        self.assertEqual(row["avg_time_sec"], 45.0)

    def test_exception_trial_excluded_from_rewards(self):
        self._write_trial(
            "t1",
            {
                "exception_info": {"name": "RewardFileEmptyError"},
                "verifier_result": {
                    "rewards": {"reward": 1.0, "f2p": 1, "p2p": 1,
                                "partial": 0},
                },
            },
        )
        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertEqual(row["n_trials"], 1)
        self.assertEqual(row["resolved"], 0)
        self.assertEqual(row["avg_f2p"], "-")
        self.assertEqual(row["exceptions"], 1)

    def test_missing_fields_default_to_dash(self):
        # reward 部分字段缺失 / 无时间戳 / 无 token 统计
        self._write_trial("t1", {"exception_info": None})
        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertEqual(row["n_trials"], 1)
        self.assertEqual(row["resolved"], 0)
        self.assertEqual(row["resolved_rate"], 0.0)
        self.assertEqual(row["avg_f2p"], "-")
        self.assertEqual(row["input_tokens"], 0)
        self.assertEqual(row["avg_time_sec"], "-")

    def test_non_dir_and_missing_result_json_skipped(self):
        details = os.path.join(self.temp_dir, "results", "m", "d", "details")
        os.makedirs(details)
        # 普通文件、空目录均不应计入 trial
        with open(os.path.join(details, "stray.txt"), "w") as f:
            f.write("x")
        os.makedirs(os.path.join(details, "empty_trial"))
        row = self.summarizer._build_row(self.model_cfg, self.dataset_abbr)
        self.assertEqual(row["n_trials"], 0)

    def test_agent_and_model_name_fallback(self):
        cfg = {"abbr": "m", "model_names": []}
        self._write_trial("t1", _resolved_trial())
        row = self.summarizer._build_row(cfg, self.dataset_abbr)
        self.assertEqual(row["agent"], "m")
        self.assertEqual(row["model_name"], "-")


class TestDeepSWESummarizerSummarize(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_writes_csv_and_prints(self):
        _write_json(
            os.path.join(self.temp_dir, "results", "m", "d", "details", "t1",
                         "result.json"),
            _resolved_trial(),
        )
        summarizer = DeepSWESummarizer(
            ConfigDict(
                {
                    "work_dir": self.temp_dir,
                    "models": [{"abbr": "m", "agent_name": "mini-swe-agent",
                                "model_names": ["mm"]}],
                    "datasets": [{"abbr": "d"}],
                }
            )
        )
        with mock.patch("builtins.print"):
            summarizer.summarize(time_str="20260101")
        csv_path = os.path.join(self.temp_dir, "summary",
                                "summary_deepswe_20260101.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][0], "agent")
        self.assertEqual(rows[1][0], "mini-swe-agent")
        self.assertIn("resolved_rate", rows[0])

    def test_no_results_warns(self):
        summarizer = DeepSWESummarizer(
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
