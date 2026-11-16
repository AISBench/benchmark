import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mmengine.config import ConfigDict

from ais_bench.benchmark.summarizers.agentx import (
    AgentXSummarizer,
    manifest_row,
)
from ais_bench.benchmark.tasks.custom_tasks.agentx_result import atomic_write_json


def make_raw():
    return {
        "schema_version": "1.4",
        "request_count": {"unit": "requests", "avg": 10},
        "error_request_count": {"unit": "requests", "avg": 0},
        "request_throughput": {"unit": "requests/s", "avg": 8},
        "output_token_throughput": {"unit": "tokens/s", "avg": 80},
        "time_to_first_token": {"unit": "s", "avg": 0.1, "p90": 0.2},
        "inter_token_latency": {"unit": "ms", "avg": 12},
        "request_latency": {"unit": "us", "avg": 1000, "p90": 2000},
        "metadata": {
            "submission_valid": True,
            "submission_invalid_reasons": [],
            "metric_duration_coverage": [{
                "required_ratio": 0.9,
                "ttft_ratio": 1.0,
                "inter_token_latency_ratio": 0.95,
            }],
        },
    }


def make_manifest(run_id="agentx_full_batchsize_8", *, model="qwen3.8",
                  dataset="weka"):
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "upstream": {"framework": "agentx-harness", "version": "1.0.13"},
        "run_params": {"run_mode": "benchmark", "benchmark_id": f"{model}/{dataset}"},
        "dataset": {"abbr": dataset, "variant": "full", "source": "public"},
        "model": {"name": "=1+1", "abbr": "qwen,\"x\"\nnext", "path": "/models/qwen"},
        "engine": {"type": "vllm", "version": "0.8.5"},
        "test_params": {"batch_size": 8, "random_seed": 42},
        "result": {
            "status": "completed", "parse_status": "success", "data_status": "complete",
            "benchmark_valid": True, "comparison_scope": "official_agentx",
            "harness_submission_valid": True, "invalid_reasons": [],
            "aisbench_invalid_reasons": [], "errors": [],
            "summary_json": "agentx_summary.json",
            "summary_csv": "agentx_summary.csv",
            "raw_artifacts": "agentx_raw",
        },
    }


def write_run(work_dir, manifest=None):
    manifest = manifest or make_manifest()
    run_dir = Path(work_dir) / "performances" / manifest["run_id"]
    (run_dir / "agentx_raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "agentx_summary.json").write_text(
        json.dumps(make_raw()), encoding="utf-8"
    )
    (run_dir / "agentx_summary.csv").write_text(
        "metric,avg\nrequest_throughput,8\n", encoding="utf-8"
    )
    atomic_write_json(str(run_dir / "manifest.json"), manifest)
    return run_dir


class TestAgentXSummaryMapping(unittest.TestCase):
    def test_manifest_row_passes_through_raw_metric_values(self):
        with tempfile.TemporaryDirectory() as work_dir:
            manifest = make_manifest()
            run_dir = str(Path(work_dir) / "performances" / manifest["run_id"])
            row = manifest_row(manifest, make_raw(), work_dir, run_dir)
            self.assertEqual(row["run_id"], "agentx_full_batchsize_8")
            self.assertEqual(row["request_throughput"], 8)
            self.assertEqual(row["error_request_count"], 0)
            # 时延列直接透传上游 p90 原始值和原始单位，不做换算。
            self.assertEqual(row["time_to_first_token_p90"], 0.2)
            self.assertEqual(row["request_latency_p90"], 2000)
            self.assertIsNone(row["inter_token_latency_p90"])
            self.assertTrue(row["metric_coverage_passed"])
            self.assertEqual(
                row["manifest_path"], "performances/agentx_full_batchsize_8/manifest.json"
            )
            self.assertEqual(
                row["raw_summary_path"],
                "performances/agentx_full_batchsize_8/agentx_summary.json",
            )
            self.assertEqual(
                row["summary_csv_path"],
                "performances/agentx_full_batchsize_8/agentx_summary.csv",
            )


class TestAgentXSummarizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = Path(self.temp_dir.name)
        self.summarizer = AgentXSummarizer(ConfigDict({"work_dir": str(self.work_dir)}))

    def test_finalize_run_writes_slot_artifacts_and_summary_log(self):
        run_dir = self.work_dir / "performances" / "agentx_full_batchsize_8"
        raw_dir = run_dir / "agentx_raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )
        (raw_dir / "profile_export_aiperf.csv").write_text(
            "metric,avg\nrequest_throughput,8\n", encoding="utf-8"
        )
        logs_dir = self.work_dir / "logs" / "agentx_full_batchsize_8"
        manifest = make_manifest()
        manifest["result"]["summary_json"] = None
        manifest["result"]["summary_csv"] = None
        manifest["result"]["raw_artifacts"] = None
        self.summarizer.finalize_run(str(run_dir), manifest, str(raw_dir), str(logs_dir))

        self.assertEqual(
            (run_dir / "agentx_summary.json").read_bytes(),
            (raw_dir / "profile_export_aiperf.json").read_bytes(),
        )
        self.assertEqual(
            (run_dir / "agentx_summary.csv").read_bytes(),
            (raw_dir / "profile_export_aiperf.csv").read_bytes(),
        )
        written = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(written["result"]["summary_json"], "agentx_summary.json")
        self.assertEqual(written["result"]["summary_csv"], "agentx_summary.csv")
        self.assertEqual(written["result"]["raw_artifacts"], "agentx_raw")

        # 档位结束即写本档位 summary 日志，且不产生额外目录。
        summary_log = (logs_dir / "agentx_summary.log").read_text(encoding="utf-8")
        self.assertIn("agentx_full_batchsize_8", summary_log)
        self.assertIn(
            "performances/agentx_full_batchsize_8/agentx_summary.csv", summary_log
        )
        self.assertFalse((run_dir / "performances").exists())
        self.assertFalse((run_dir / "configs").exists())

    def test_summarize_reads_manifests_and_refreshes_slot_logs(self):
        write_run(self.work_dir)
        with mock.patch("builtins.print") as output:
            payload = self.summarizer.summarize()
        self.assertEqual(payload["counts"]["runs"], 1)
        self.assertEqual(payload["counts"]["benchmark_valid"], 1)
        self.assertTrue(output.called)

        # 根目录不再生成跨档位汇总文件，档位信息由 manifest 和 summary 日志承载。
        self.assertFalse((self.work_dir / "agentx_summary.csv").exists())
        self.assertFalse((self.work_dir / "agentx_summary.json").exists())
        self.assertFalse((self.work_dir / "summary").exists())
        summary_log = (
            self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log"
        ).read_text(encoding="utf-8")
        self.assertIn("agentx_full_batchsize_8", summary_log)
        self.assertIn("performances/agentx_full_batchsize_8/agentx_summary.json", summary_log)

    def test_partial_corrupt_manifest_retains_other_reports(self):
        good_dir = write_run(self.work_dir)
        bad_dir = self.work_dir / "performances" / "agentx_full_batchsize_16"
        bad_dir.mkdir(parents=True)
        (bad_dir / "manifest.json").write_text("{bad", encoding="utf-8")
        with mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_INVALID"):
                self.summarizer.summarize()
        self.assertTrue((bad_dir / "manifest.json").exists())
        self.assertTrue((good_dir / "manifest.json").exists())
        self.assertTrue(
            (
                self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log"
            ).is_file()
        )

    def test_summarize_degrades_row_when_summary_unreadable(self):
        # 载入失败时该行必须降级：不再出现 benchmark_valid=True 但指标全 N/A 的矛盾。
        run_dir = write_run(self.work_dir)
        (run_dir / "agentx_summary.json").write_text("{bad", encoding="utf-8")
        with mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_INVALID"):
                self.summarizer.summarize()
        summary_log = (
            self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log"
        ).read_text(encoding="utf-8")
        self.assertIn("summary_load_failed", summary_log)
        self.assertIn("False", summary_log)
        self.assertNotIn("official_agentx", summary_log)

    def test_summarize_degrades_row_when_summary_not_archived(self):
        # summary_json 为 None 时同样降级并记录根因，不允许静默通过。
        manifest = make_manifest()
        manifest["result"]["summary_json"] = None
        write_run(self.work_dir, manifest)
        with mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_INVALID"):
                self.summarizer.summarize()
        summary_log = (
            self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log"
        ).read_text(encoding="utf-8")
        self.assertIn("summary_missing", summary_log)
        self.assertIn("False", summary_log)
        self.assertNotIn("official_agentx", summary_log)

    def test_finalize_run_reads_console_log_from_raw_artifact_dir(self):
        # console 日志路径从 raw_artifact_dir 派生（不再硬编码 agentx_raw），
        # 内容与 agentx_stdout.log 同规则脱敏并剥离品牌前缀；上游原始文件不改。
        run_dir = self.work_dir / "performances" / "agentx_full_batchsize_8"
        raw_dir = run_dir / "custom_raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )
        (raw_dir / "profile_export_console.txt").write_text(
            "NVIDIA AIPerf | ready --api-key sk-leak\n", encoding="utf-8"
        )
        logs_dir = self.work_dir / "logs" / "agentx_full_batchsize_8"
        self.summarizer.finalize_run(str(run_dir), make_manifest(), str(raw_dir), str(logs_dir))
        summary_log = (logs_dir / "agentx_summary.log").read_text(encoding="utf-8")
        self.assertIn("ready --api-key <redacted>", summary_log)
        self.assertNotIn("sk-leak", summary_log)
        self.assertNotIn("NVIDIA AIPerf", summary_log)
        written = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(written["result"]["raw_artifacts"], "custom_raw")

    def test_summarize_keeps_console_log_when_artifacts_lie_outside_run_dir(self):
        # agentx_raw 场景：extra args 可把 --output-artifact-dir 覆盖到档位目录之外，
        # viz 重建 summary 日志时仍必须能读到 console 指标，不能被覆盖成空。
        run_dir = self.work_dir / "performances" / "agentx_full_batchsize_8"
        raw_dir = self.work_dir / "custom_artifacts"
        raw_dir.mkdir(parents=True)
        (raw_dir / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )
        (raw_dir / "profile_export_console.txt").write_text(
            "| TTFT | 100 |\n", encoding="utf-8"
        )
        logs_dir = self.work_dir / "logs" / "agentx_full_batchsize_8"
        self.summarizer.finalize_run(
            str(run_dir), make_manifest(), str(raw_dir), str(logs_dir)
        )
        written = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            written["result"]["raw_artifacts"],
            os.path.relpath(str(raw_dir), str(run_dir)),
        )
        with mock.patch("builtins.print"):
            self.summarizer.summarize()
        summary_log = (logs_dir / "agentx_summary.log").read_text(encoding="utf-8")
        self.assertIn("| TTFT | 100 |", summary_log)

    def test_summarize_does_not_rewrite_unchanged_slot_log(self):
        """汇总阶段内容未变化时不重写运行期写好的日志。"""
        run_dir = self.work_dir / "performances" / "agentx_full_batchsize_8"
        raw_dir = run_dir / "agentx_raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )
        logs_dir = self.work_dir / "logs" / "agentx_full_batchsize_8"
        self.summarizer.finalize_run(
            str(run_dir), make_manifest(), str(raw_dir), str(logs_dir)
        )
        log_path = logs_dir / "agentx_summary.log"
        before = (log_path.read_text(encoding="utf-8"), log_path.stat().st_mtime_ns)

        with mock.patch(
            "ais_bench.benchmark.summarizers.agentx._atomic_text"
        ) as atomic, mock.patch("builtins.print"):
            self.summarizer.summarize(print_table=False)

        atomic.assert_not_called()
        self.assertEqual(
            (log_path.read_text(encoding="utf-8"), log_path.stat().st_mtime_ns),
            before,
        )

    def test_no_manifests_is_explicit_error(self):
        with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_MISSING"):
            self.summarizer.summarize()

    def test_slot_with_artifacts_but_no_manifest_is_degraded(self):
        """有产物但 manifest 未写成时显式降级，不能静默丢档。"""
        run_dir = self.work_dir / "performances" / "agentx_full_batchsize_8"
        raw_dir = run_dir / "agentx_raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )

        with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_INVALID"):
            self.summarizer.summarize(print_table=False)
        self.assertTrue(
            (self.work_dir / "logs" / "agentx_full_batchsize_8" / "agentx_summary.log").is_file()
        )

    def test_slot_with_overridden_artifacts_outside_run_dir_is_degraded(self):
        """raw 模式把产物目录覆盖到档位目录外、进程被 kill 时也不能静默丢档。"""
        run_dir = self.work_dir / "performances" / "agentx_raw_batchsize_8"
        run_dir.mkdir(parents=True)
        external = self.work_dir / "custom_artifacts"
        external.mkdir()
        (external / "profile_export_aiperf.json").write_text(
            json.dumps(make_raw()), encoding="utf-8"
        )
        (run_dir / "execution_plan.redacted.json").write_text(
            json.dumps(
                {"run_id": "agentx_raw_batchsize_8", "artifact_dir": str(external)}
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_INVALID"):
            self.summarizer.summarize(print_table=False)
        summary_log = (
            self.work_dir / "logs" / "agentx_raw_batchsize_8" / "agentx_summary.log"
        )
        self.assertIn("manifest_missing", summary_log.read_text(encoding="utf-8"))

    def test_slot_with_plan_but_no_artifacts_is_ignored(self):
        """dry-run 只写计划和空目录，不能当成缺 manifest 的失败档位。"""
        run_dir = self.work_dir / "performances" / "agentx_raw_batchsize_8"
        run_dir.mkdir(parents=True)
        external = self.work_dir / "custom_artifacts"
        external.mkdir()
        (run_dir / "execution_plan.redacted.json").write_text(
            json.dumps(
                {"run_id": "agentx_raw_batchsize_8", "artifact_dir": str(external)}
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_MISSING"):
            self.summarizer.summarize(print_table=False)

    def test_slot_without_artifacts_is_ignored(self):
        (self.work_dir / "performances" / "agentx_full_batchsize_8").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "AGENTX_RESULT_MISSING"):
            self.summarizer.summarize(print_table=False)


if __name__ == "__main__":
    unittest.main()
