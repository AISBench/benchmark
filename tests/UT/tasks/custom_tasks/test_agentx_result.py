import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ais_bench.benchmark.tasks.custom_tasks.agentx_result import (
    RESULT_SCHEMA_VERSION,
    atomic_write_json,
    json_safe,
    normalize_agentx_result,
)


def raw_summary():
    return {
        "schema_version": "1.4",
        "aiperf_version": "agentx-1.0.13",
        "request_count": {"unit": "requests", "avg": 10, "count": 1},
        "error_request_count": {"unit": "requests", "avg": 0, "count": 1},
        "request_error_rate": {"unit": "ratio", "avg": 0},
        "request_throughput": {"unit": "requests/s", "avg": 8, "min": 7, "max": 9},
        "output_token_throughput": {"unit": "tokens/s", "avg": 80, "p90": 100},
        "time_to_first_token": {"unit": "s", "avg": 0.1, "p90": 0.2},
        "inter_token_latency": {"unit": "ms", "avg": 12},
        "request_latency": {"unit": "us", "avg": 1000, "p90": 2000},
        "future_metric": {"unit": "widgets", "avg": 0, "p99": 12},
        "warmup_metrics": {
            "request_throughput": {"unit": "requests/s", "avg": 2},
            "not_a_metric": {"avg": 99},
        },
        "metadata": {
            "submission_valid": True,
            "submission_invalid_reasons": [],
            "metric_duration_coverage": [{
                "required_ratio": 0.9,
                "ttft_ratio": 0.95,
                "inter_token_latency_ratio": 1.0,
            }],
            "dataset": {"alias": "official", "revision": "abc"},
        },
        "branch_stats": {"sessions": 3},
        "error_summary": [],
        "was_cancelled": False,
    }


def normalized(raw=None, **overrides):
    run = {
        "run_id": "run-1",
        "mode": "benchmark",
        "status": "completed",
        "parse_status": "success",
        "return_code": 0,
        "was_cancelled": False,
    }
    run.update(overrides.pop("run", {}))
    identity = {
        "model": "qwen3.8",
        "model_abbr": "qwen",
        "dataset_abbr": "weka",
        "variant": "full",
        "concurrency": 8,
    }
    identity.update(overrides.pop("identity", {}))
    dataset = {
        "source": "public",
        "verification": ["not_verified"],
        "sha256_mismatch": False,
        "variant": "full",
    }
    dataset.update(overrides.pop("dataset", {}))
    kwargs = {
        "raw": raw if raw is not None else raw_summary(),
        "run": run,
        "identity": identity,
        "harness": {"version": "1.0.13"},
        "dataset": dataset,
        "command": ["agentx", "profile"],
        "artifacts": {"raw_summary": "/work/raw.json"},
        "errors": [],
    }
    kwargs.update(overrides)
    return normalize_agentx_result(**kwargs)


class TestAgentXResultAdapter(unittest.TestCase):
    def test_complete_validity_and_no_metric_copy(self):
        result = normalized()
        self.assertEqual(result["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertTrue(result["run"]["benchmark_valid"])
        self.assertEqual(result["run"]["comparison_scope"], "official_agentx")
        self.assertEqual(result["run"]["data_status"], "complete")
        self.assertEqual(result["dataset"]["harness_provenance"]["revision"], "abc")
        # 指标本体保留在 performances/，判定结果不复制指标。
        for key in ("metrics", "warmup_metrics", "coverage", "branch_stats", "error_summary", "derived_metrics"):
            self.assertNotIn(key, result)

    def test_submission_invalid_is_separate_from_process_status(self):
        raw = raw_summary()
        raw["metadata"]["submission_valid"] = False
        raw["metadata"]["submission_invalid_reasons"] = ["coverage too low"]
        result = normalized(raw)
        self.assertEqual(result["run"]["status"], "completed")
        self.assertEqual(result["run"]["data_status"], "complete")
        self.assertFalse(result["run"]["benchmark_valid"])
        self.assertFalse(result["validity"]["harness_submission_valid"])
        # 只保留上游具体原因，不再追加概括性派生标记。
        self.assertEqual(result["validity"]["harness_invalid_reasons"], ["coverage too low"])
        self.assertNotIn("harness_submission_not_valid", result["validity"]["aisbench_invalid_reasons"])
        self.assertNotIn("harness_submission_invalid_reasons", result["validity"]["aisbench_invalid_reasons"])
        self.assertEqual(result["validity"]["primary_invalid_reason"], "coverage too low")
        self.assertIn("coverage too low", result["validity"]["all_invalid_reasons"])

    def test_smoke_hash_mismatch_and_cancelled_are_invalid(self):
        smoke = normalized(
            run={"mode": "smoke"},
            dataset={"source": "local", "sha256_mismatch": True},
        )
        self.assertEqual(smoke["run"]["comparison_scope"], "smoke")
        # 对外只报一条根因，其余根因保留在 all_invalid_reasons 中。
        self.assertEqual(smoke["validity"]["primary_invalid_reason"], "run_mode_not_benchmark")
        self.assertIn("dataset_sha256_mismatch", smoke["validity"]["all_invalid_reasons"])

        cancelled = normalized(run={"status": "cancelled", "was_cancelled": True})
        self.assertEqual(cancelled["run"]["data_status"], "partial")
        self.assertIn("run_cancelled", cancelled["validity"]["aisbench_invalid_reasons"])

    def test_local_verified_result_is_internal_only(self):
        result = normalized(dataset={"source": "local", "verification": ["local_content_hash_verified"]})
        self.assertTrue(result["run"]["benchmark_valid"])
        self.assertEqual(result["run"]["comparison_scope"], "internal_only")
        empty = normalized(
            raw={}, dataset={"source": "local", "verification": ["local_content_hash_verified"]}
        )
        self.assertEqual(empty["run"]["comparison_scope"], "internal_only")

    def test_completed_without_metrics_is_not_benchmark_valid(self):
        # 退出码 0、解析成功但 summary 无任何指标：data_status=unavailable 时
        # benchmark_valid 不能仍为 True，comparison_scope 不能仍是 official_agentx。
        empty = normalized(raw={
            "schema_version": "1.4",
            "metadata": {"submission_valid": True, "submission_invalid_reasons": []},
        })
        self.assertEqual(empty["run"]["data_status"], "unavailable")
        self.assertFalse(empty["run"]["benchmark_valid"])
        self.assertEqual(empty["run"]["comparison_scope"], "invalid")
        self.assertEqual(empty["validity"]["primary_invalid_reason"], "data_status:unavailable")
        self.assertIn("data_status:unavailable", empty["validity"]["all_invalid_reasons"])

        # 数据不完整只作为归档根因，run_status 仍是对外主根因。
        partial = normalized(run={"status": "failed"}, raw=raw_summary())
        self.assertEqual(partial["run"]["data_status"], "partial")
        self.assertEqual(partial["validity"]["primary_invalid_reason"], "run_status:failed")
        self.assertIn("data_status:partial", partial["validity"]["all_invalid_reasons"])

    def test_dataset_verification_requires_recorded_state(self):
        # verification 未按 *_verified 状态标记命名（缺失/空/非字符串项）时判 provenance 未验证；
        # not_verified 与 local_content_hash_verified 一样是显式记录的验证状态。
        for verification in ([], ["pending"], [True], "local_content_hash_verified"):
            with self.subTest(verification=verification):
                result = normalized(dataset={"verification": verification})
                self.assertIn(
                    "dataset_provenance_unverified",
                    result["validity"]["aisbench_invalid_reasons"],
                )
        self.assertNotIn(
            "dataset_provenance_unverified",
            normalized()["validity"]["aisbench_invalid_reasons"],
        )

    def test_unhashable_harness_reasons_do_not_block_result(self):
        raw = raw_summary()
        raw["metadata"]["submission_valid"] = False
        raw["metadata"]["submission_invalid_reasons"] = [{"code": "coverage", "ratio": 0.5}]
        result = normalized(raw)
        # 原因项统一转字符串，dict.fromkeys 去重不再抛 TypeError，整档证据可正常落盘。
        self.assertEqual(
            result["validity"]["harness_invalid_reasons"],
            ["{'code': 'coverage', 'ratio': 0.5}"],
        )
        self.assertEqual(result["validity"]["primary_invalid_reason"],
                         "{'code': 'coverage', 'ratio': 0.5}")
        json.dumps(result, allow_nan=False)

    def test_missing_metrics_unknown_schema_and_non_finite_values(self):
        empty = normalized(raw={})
        self.assertEqual(empty["run"]["data_status"], "unavailable")
        unknown = normalized(raw={"schema_version": "1.3"})
        self.assertEqual(unknown["run"]["parse_status"], "unsupported_schema")
        raw = raw_summary()
        raw["request_throughput"]["avg"] = float("nan")
        raw["output_token_throughput"]["avg"] = float("inf")
        result = normalized(raw)
        # 指标原始值留在 performances/ 拷贝中；判定结果自身可序列化。
        json.dumps(result, allow_nan=False)

    def test_json_safe_and_atomic_write_failure_preserve_old_file(self):
        self.assertEqual(json_safe([float("nan"), float("inf"), 0]), [None, None, 0])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with mock.patch("os.replace", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    atomic_write_json(str(path), {"new": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(Path(temp_dir).glob(".agentx-*")), [])


if __name__ == "__main__":
    unittest.main()
