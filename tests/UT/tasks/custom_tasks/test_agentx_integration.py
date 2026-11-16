import http.client
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mmengine.config import ConfigDict

from ais_bench.benchmark.summarizers.agentx import AgentXSummarizer
from ais_bench.benchmark.tasks.custom_tasks.agentx_task import AgentXTask


FAKE_HARNESS = r'''
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output-artifact-dir", required=True)
parser.add_argument("--api-key")
args, _ = parser.parse_known_args()
mode = os.environ.get("FAKE_AGENTX_MODE", "success")
print("fake harness api-key=" + (args.api_key or ""), flush=True)
print("fake harness api-token=" + os.environ.get("API_TOKEN", ""), flush=True)
for index in range(200):
    print("progress line %d" % index, flush=True)
print("long-log:" + "x" * 10000, flush=True)
if mode == "missing":
    sys.exit(0)
raw = {
    "schema_version": "1.4",
    "aiperf_version": "agentx-1.0.13",
    "request_count": {"unit": "requests", "avg": 4},
    "request_throughput": {"unit": "requests/s", "avg": 2},
    "output_token_throughput": {"unit": "tokens/s", "avg": 20},
    "time_to_first_token": {"unit": "s", "avg": 0.1, "p90": 0.2},
    "inter_token_latency": {"unit": "ms", "avg": 5, "p90": 8},
    "request_latency": {"unit": "s", "avg": 1, "p90": 2},
    "warmup_metrics": {"request_throughput": {"unit": "requests/s", "avg": 1}},
    "metadata": {
        "submission_valid": mode != "submission_invalid",
        "submission_invalid_reasons": ["injected invalid"] if mode == "submission_invalid" else [],
        "metric_duration_coverage": [],
    },
    "branch_stats": {"sessions": 1},
    "error_summary": [],
    "was_cancelled": mode == "cancelled",
}
os.makedirs(args.output_artifact_dir, exist_ok=True)
summary = os.path.join(args.output_artifact_dir, "profile_export_aiperf.json")
if mode == "corrupt":
    with open(summary, "w", encoding="utf-8") as file:
        file.write("{bad")
else:
    with open(summary, "w", encoding="utf-8") as file:
        json.dump(raw, file)
    with open(os.path.join(args.output_artifact_dir, "profile_export_aiperf.csv"), "w", encoding="utf-8") as file:
        file.write("metric,avg\nrequest_throughput,2\n")
if mode == "nonzero":
    sys.exit(7)
sys.exit(0)
'''


class TestAgentXFakeHarnessIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.fake_harness = self.root / "fake_harness.py"
        self.fake_harness.write_text(FAKE_HARNESS, encoding="utf-8")

    def make_task(self, name, **arg_overrides):
        work_dir = self.root / name
        work_dir.mkdir()
        args = {
            "concurrency": 2,
            "cooldown": 0,
            "variant": "full",
            "api_key": "secret",
            "env": {"API_TOKEN": "token-secret", "PUBLIC": "visible"},
        }
        args.update(arg_overrides)
        cfg = ConfigDict({
            "work_dir": str(work_dir),
            "models": [{
                "abbr": "qwen",
                "model": "qwen3.8",
                "path": str(self.model_dir),
                "url": "http://127.0.0.1:8000",
            }],
            "datasets": [{"abbr": "weka", "args": args}],
            "cli_args": {"debug": False},
        })
        return AgentXTask(cfg), work_dir

    def run_task(self, task, mode, probe_engine=None):
        metadata = {
            "distribution": "agentx-harness",
            "version": "1.0.13",
            "agentx_version": "1.0.13",
            "python_version": "%d.%d.%d" % sys.version_info[:3],
            "python": sys.executable,
            "entry": [sys.executable, str(self.fake_harness)],
            "platform": "fake",
        }
        engine = {
            "type": "vllm", "version": "0.8.5", "served_model": "qwen3.8",
            "max_model_len": 262144, "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/chat/completions",
            "probe": {"version": {"status": "ok", "error": None},
                      "models": {"status": "ok", "error": None}},
            "collected_at": "2026-01-01T00:00:00+00:00",
        }
        # _probe_engine 必须替换成可调用对象；直接 patch 成 (engine, errors) 元组时
        # 调用会抛 TypeError 并被 _execute_plan 的容错分支吞掉，engine 恒为 None。
        probe = (
            probe_engine if probe_engine is not None else mock.Mock(return_value=(engine, []))
        )
        with mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.resolve_harness_python",
            return_value=sys.executable,
        ), mock.patch(
            "ais_bench.benchmark.tasks.custom_tasks.agentx_task.probe_harness_environment",
            return_value=metadata,
        ), mock.patch.object(
            AgentXTask, "_probe_engine", probe
        ), mock.patch.dict(os.environ, {"FAKE_AGENTX_MODE": mode}, clear=False):
            return task.run()

    def load_manifest(self, task, work_dir):
        path = (
            work_dir / "performances"
            / f"agentx_full_batchsize_{task.args['concurrency']}" / "manifest.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def run_dir(self, work_dir, concurrency):
        return work_dir / "performances" / f"agentx_full_batchsize_{concurrency}"

    def test_success_full_chain_and_secret_redaction(self):
        task, work_dir = self.make_task("success")
        result = self.run_task(task, "success")
        self.assertEqual(result["run"]["status"], "completed")
        self.assertEqual(result["run"]["parse_status"], "success")
        self.assertTrue(result["run"]["benchmark_valid"])
        self.assertEqual(result["identity"]["concurrency"], 2)

        run_dir = self.run_dir(work_dir, 2)
        raw_dir = run_dir / "agentx_raw"
        self.assertTrue((raw_dir / "profile_export_aiperf.json").is_file())
        self.assertTrue((raw_dir / "profile_export_aiperf.csv").is_file())
        self.assertEqual(
            (run_dir / "agentx_summary.json").read_bytes(),
            (raw_dir / "profile_export_aiperf.json").read_bytes(),
        )
        self.assertEqual(
            (run_dir / "agentx_summary.csv").read_bytes(),
            (raw_dir / "profile_export_aiperf.csv").read_bytes(),
        )

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "agentx_full_batchsize_2")
        self.assertNotIn("hardware", manifest)
        self.assertEqual(manifest["engine"]["version"], "0.8.5")
        self.assertEqual(manifest["test_params"]["batch_size"], 2)
        self.assertEqual(manifest["test_params"]["random_seed"], 42)
        self.assertEqual(manifest["result"]["summary_json"], "agentx_summary.json")
        self.assertEqual(manifest["result"]["summary_csv"], "agentx_summary.csv")
        self.assertEqual(manifest["result"]["raw_artifacts"], "agentx_raw")
        self.assertEqual(manifest["dataset"]["variant"], "full")
        self.assertEqual(manifest["agentx_command"]["parameters"]["--scenario"], "agentx")
        self.assertEqual(manifest["agentx_command"]["parameters"]["--concurrency"], "2")
        self.assertEqual(manifest["agentx_command"]["parameters"]["--model"], "qwen3.8")
        self.assertEqual(manifest["agentx_command"]["extra_args"], [])
        self.assertNotIn("argv", manifest["agentx_command"])
        self.assertIn("--concurrency", manifest["run_params"]["command"])
        for key in ("upstream", "run_params", "agentx_command", "dataset", "model",
                    "engine", "test_params", "result"):
            self.assertIn(key, manifest)

        with mock.patch("builtins.print"):
            AgentXSummarizer(ConfigDict({"work_dir": str(work_dir)})).summarize()
        logs_dir = work_dir / "logs" / "agentx_full_batchsize_2"
        summary_log = (logs_dir / "agentx_summary.log").read_text(encoding="utf-8")
        self.assertIn("agentx_full_batchsize_2", summary_log)
        self.assertIn("performances/agentx_full_batchsize_2/agentx_summary.csv", summary_log)
        # 根目录只保留框架级 configs/、logs/ 和 performances/；任务自身不创建 configs/。
        self.assertFalse((work_dir / "agentx_summary.csv").exists())
        self.assertFalse((work_dir / "agentx_summary.json").exists())
        self.assertFalse((work_dir / "summary").exists())
        self.assertFalse((work_dir / "configs").exists())

        evidence_paths = [
            run_dir / "execution_plan.redacted.json",
            logs_dir / "agentx_stdout.log",
            logs_dir / "agentx_adapter.log",
            logs_dir / "agentx_summary.log",
            run_dir / "manifest.json",
            run_dir / "agentx_summary.json",
            run_dir / "agentx_summary.csv",
        ]
        for path in evidence_paths:
            with self.subTest(path=path.name):
                self.assertNotIn("secret", path.read_text(encoding="utf-8"))
        for path in (
            run_dir / "execution_plan.redacted.json",
            logs_dir / "agentx_stdout.log",
            logs_dir / "agentx_adapter.log",
            logs_dir / "agentx_summary.log",
            run_dir / "manifest.json",
        ):
            with self.subTest(path=path.name):
                self.assertIn("agentx_full_batchsize_2", path.read_text(encoding="utf-8"))

    def test_manifest_records_actual_agentx_parameters(self):
        task, work_dir = self.make_task(
            "params", concurrency=[1, 2], extra_args="--concurrency 5 --model Qwen"
        )
        self.run_task(task, "success")
        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency):
                manifest = json.loads(
                    (self.run_dir(work_dir, concurrency) / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["test_params"]["batch_size"], concurrency)
                self.assertEqual(
                    manifest["agentx_command"]["parameters"]["--concurrency"], "5"
                )
                self.assertEqual(manifest["agentx_command"]["parameters"]["--model"], "Qwen")
                self.assertEqual(
                    manifest["agentx_command"]["extra_args"],
                    ["--concurrency", "5", "--model", "Qwen"],
                )

    def test_multiple_concurrency_levels_run_independently(self):
        task, work_dir = self.make_task("multi", concurrency=[1, 4, 8])
        result = self.run_task(task, "success")
        self.assertEqual([item["identity"]["concurrency"] for item in result["runs"]], [1, 4, 8])
        run_ids = [item["run"]["run_id"] for item in result["runs"]]
        self.assertEqual(run_ids, [
            "agentx_full_batchsize_1", "agentx_full_batchsize_4", "agentx_full_batchsize_8",
        ])
        for concurrency in (1, 4, 8):
            run_dir = self.run_dir(work_dir, concurrency)
            self.assertTrue((run_dir / "agentx_raw" / "profile_export_aiperf.json").is_file())
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "agentx_summary.json").is_file())
            self.assertTrue(
                (work_dir / "logs" / f"agentx_full_batchsize_{concurrency}" / "agentx_summary.log").is_file()
            )

        with mock.patch("builtins.print"):
            AgentXSummarizer(ConfigDict({"work_dir": str(work_dir)})).summarize()
        # 档位信息由各档位 manifest/summary 承载，根目录不再生成跨档位宽表。
        self.assertFalse((work_dir / "agentx_summary.csv").exists())

    def test_submission_invalid_keeps_metrics_but_is_not_benchmark_valid(self):
        task, work_dir = self.make_task("submission_invalid")
        result = self.run_task(task, "submission_invalid")
        self.assertEqual(result["run"]["status"], "completed")
        self.assertFalse(result["run"]["benchmark_valid"])
        self.assertFalse(result["validity"]["harness_submission_valid"])
        self.assertIn("injected invalid", result["validity"]["harness_invalid_reasons"])
        # 判定结果不复制指标本体，上游指标原样保留在 performances/ 拷贝中。
        self.assertNotIn("metrics", result)
        summary = json.loads(
            (self.run_dir(work_dir, 2) / "agentx_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["request_throughput"]["avg"], 2)

    def test_failure_cancelled_missing_and_corrupt_summary(self):
        cases = (
            ("nonzero", "failed", "success", 7),
            ("cancelled", "cancelled", "success", 0),
            ("missing", "failed", "missing", 0),
            ("corrupt", "failed", "invalid_json", 0),
        )
        for mode, status, parse_status, return_code in cases:
            with self.subTest(mode=mode):
                task, work_dir = self.make_task(mode)
                with self.assertRaises(RuntimeError):
                    self.run_task(task, mode)
                manifest = self.load_manifest(task, work_dir)
                self.assertEqual(manifest["result"]["status"], status)
                self.assertEqual(manifest["result"]["parse_status"], parse_status)
                self.assertEqual(manifest["result"]["return_code"], return_code)
                self.assertFalse(manifest["result"]["benchmark_valid"])
                if mode == "nonzero":
                    self.assertEqual(manifest["result"]["data_status"], "partial")
                    summary_path = self.run_dir(work_dir, 2) / "agentx_summary.json"
                    raw = json.loads(summary_path.read_text(encoding="utf-8"))
                    self.assertEqual(raw["request_throughput"]["avg"], 2)
                if mode == "cancelled":
                    self.assertTrue(manifest["result"]["was_cancelled"])
                    self.assertIn("AGENTX_CANCELLED", {item["code"] for item in manifest["result"]["errors"]})

    def test_engine_probe_crash_keeps_manifest_evidence(self):
        # 探测异常（如 IncompleteRead）必须降级为失败记录，不能吞掉整档 manifest。
        task, work_dir = self.make_task("probe_crash")
        crashing = mock.Mock(side_effect=http.client.IncompleteRead(b"partial"))
        result = self.run_task(task, "success", probe_engine=crashing)
        self.assertEqual(result["run"]["status"], "completed")
        self.assertTrue(result["run"]["benchmark_valid"])
        manifest = self.load_manifest(task, work_dir)
        self.assertIsNone(manifest["engine"])
        self.assertEqual(
            manifest["result"]["errors"][0]["code"], "AGENTX_ENGINE_PROBE_FAILED"
        )

    def test_extra_arg_api_key_is_added_to_secrets(self):
        # --agentx-extra-arg 传入的 --api-key 取值也要进脱敏名单；
        # 上游按 "api-key=" 回显时不匹配 --api-key 正则，只能靠名单兜底。
        task, work_dir = self.make_task("extra_key", extra_args="--api-key extra-secret")
        result = self.run_task(task, "success")
        self.assertTrue(result["run"]["benchmark_valid"])
        logs_dir = work_dir / "logs" / "agentx_full_batchsize_2"
        stdout_log = (logs_dir / "agentx_stdout.log").read_text(encoding="utf-8")
        self.assertNotIn("extra-secret", stdout_log)
        self.assertIn("fake harness api-key=<redacted>", stdout_log)
        manifest_text = (
            self.run_dir(work_dir, 2) / "manifest.json"
        ).read_text(encoding="utf-8")
        self.assertNotIn("extra-secret", manifest_text)

    def test_test_params_follow_effective_argv(self):
        # test_params 与最终生效 argv 口径一致：extra args 覆盖 duration/seed/warmup 后，
        # 不再出现 test_params 与 agentx_command.parameters 互相矛盾的记录。
        task, work_dir = self.make_task(
            "params_argv", duration=1000,
            extra_args="--benchmark-duration 1800 --random-seed 7 --num-warmup-requests 5",
        )
        self.run_task(task, "success")
        manifest = self.load_manifest(task, work_dir)
        self.assertEqual(manifest["test_params"]["benchmark_duration_s"], 1800.0)
        self.assertEqual(manifest["test_params"]["random_seed"], 7)
        self.assertEqual(manifest["test_params"]["warmup_requests_per_lane"], 5)
        parameters = manifest["agentx_command"]["parameters"]
        self.assertEqual(parameters["--benchmark-duration"], "1800")
        self.assertEqual(parameters["--random-seed"], "7")
        self.assertEqual(parameters["--num-warmup-requests"], "5")


if __name__ == "__main__":
    unittest.main()
