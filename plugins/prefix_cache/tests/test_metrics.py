import unittest
from pathlib import Path

from ais_bench_prefix_cache.errors import RuntimeCapabilityError
from ais_bench_prefix_cache.metrics import (
    diff_metrics,
    diff_spec_decode,
    parse_metrics,
    snapshot_to_dict,
    spec_decode_to_dict,
    summarize_kv_usage,
)
from ais_bench_prefix_cache.runtime import VLLMClient
from ais_bench_prefix_cache.scenario import Scenario


BASE = """
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="engine_0",model_name="m"} 100
vllm:prefix_cache_queries{engine="engine_1",model_name="m"} 200
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{engine="engine_0",model_name="m"} 50
vllm:prefix_cache_hits{engine="engine_1",model_name="m"} 100
vllm:kv_cache_usage_perc{engine="engine_0",model_name="m"} 0.2
vllm:kv_cache_usage_perc{engine="engine_1",model_name="m"} 0.3
"""

AFTER = """
vllm:prefix_cache_queries{engine="engine_0",model_name="m"} 140
vllm:prefix_cache_queries{engine="engine_1",model_name="m"} 280
vllm:prefix_cache_hits{engine="engine_0",model_name="m"} 80
vllm:prefix_cache_hits{engine="engine_1",model_name="m"} 160
vllm:kv_cache_usage_perc{engine="engine_0",model_name="m"} 0.4
vllm:kv_cache_usage_perc{engine="engine_1",model_name="m"} 0.5
"""


def _spec_metrics(
    engine: str,
    draft: int,
    accepted: int,
    emitted: int,
    acceptance_rate: float,
    efficiency: float,
) -> str:
    return "\n".join([
        f'vllm:spec_decode_num_draft_tokens_total{{engine="{engine}"}} {draft}',
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="{engine}"}} {accepted}',
        f'vllm:spec_decode_num_emitted_tokens_total{{engine="{engine}"}} {emitted}',
        f'vllm:spec_decode_draft_acceptance_rate{{engine="{engine}"}} {acceptance_rate}',
        f'vllm:spec_decode_efficiency{{engine="{engine}"}} {efficiency}',
    ])


SPEC_BASE_RANK0 = _spec_metrics("engine_0", 100, 60, 80, 0.6, 0.8)
SPEC_BASE_RANK1 = _spec_metrics("engine_1", 200, 100, 150, 0.5, 0.75)
SPEC_AFTER_RANK0 = _spec_metrics("engine_0", 300, 200, 260, 0.7, 0.9)
SPEC_AFTER_RANK1 = _spec_metrics("engine_1", 500, 260, 400, 0.52, 0.8)
SPEC_BASE = SPEC_BASE_RANK0 + "\n" + SPEC_BASE_RANK1
SPEC_AFTER = SPEC_AFTER_RANK0 + "\n" + SPEC_AFTER_RANK1



class MetricsTest(unittest.TestCase):
    def test_parse_and_token_weighted_diff(self):
        actual = diff_metrics(parse_metrics(BASE, 2), parse_metrics(AFTER, 2))
        self.assertEqual(actual.global_queries, 120)
        self.assertEqual(actual.global_hits, 90)
        self.assertEqual(actual.global_hit_rate, 0.75)

    def test_legacy_metric_aliases(self):
        legacy = "\n".join([
            'vllm:gpu_prefix_cache_queries_total{engine="worker-a"} 20',
            'vllm:gpu_prefix_cache_hits_total{engine="worker-a"} 10',
            'vllm:gpu_cache_usage_perc{engine="worker-a"} 0.25',
        ])
        snapshot = parse_metrics(legacy, 1, {"worker-a": 0})
        self.assertEqual(snapshot.by_rank[0].queries, 20)
        self.assertEqual(snapshot.by_rank[0].hits, 10)
        self.assertEqual(snapshot.by_rank[0].kv_cache_usage, 0.25)

    def test_parse_and_diff_speculative_decoding_metrics(self):
        before = parse_metrics(BASE + SPEC_BASE, 2)
        after = parse_metrics(AFTER + SPEC_AFTER, 2)
        self.assertIsNotNone(before.spec_decode)
        self.assertEqual(before.spec_decode.by_rank[0]["draft_tokens"], 100)
        self.assertEqual(
            before.spec_decode.metric_names["accepted_tokens"],
            "vllm:spec_decode_num_accepted_tokens_total",
        )
        actual = diff_spec_decode(before, after)
        self.assertEqual(actual.status, "ok")
        self.assertEqual(actual.covered_ranks, (0, 1))
        self.assertEqual(actual.missing_ranks, ())
        self.assertEqual(actual.global_draft_tokens, 500)
        self.assertEqual(actual.global_accepted_tokens, 300)
        self.assertEqual(actual.global_emitted_tokens, 430)
        self.assertAlmostEqual(actual.global_draft_acceptance_rate, 0.6)
        self.assertAlmostEqual(actual.global_efficiency, 0.86)
        self.assertEqual(actual.by_rank[0]["deltas"], {
            "draft_tokens": 200,
            "accepted_tokens": 140,
            "emitted_tokens": 180,
        })
        self.assertAlmostEqual(actual.by_rank[0]["draft_acceptance_rate"], 0.7)
        self.assertAlmostEqual(actual.by_rank[0]["efficiency"], 0.9)
        self.assertEqual(actual.by_rank[1]["before"]["draft_tokens"], 200)
        self.assertEqual(actual.by_rank[1]["after"]["draft_tokens"], 500)

    def test_speculative_decoding_legacy_aliases_without_total(self):
        legacy = "\n".join([
            'vllm:prefix_cache_queries{engine="worker-a"} 10',
            'vllm:prefix_cache_hits{engine="worker-a"} 5',
            'vllm:spec_decode_num_draft_tokens{engine="worker-a"} 10',
            'vllm:spec_decode_num_accepted_tokens{engine="worker-a"} 6',
            'vllm:spec_decode_num_emitted_tokens{engine="worker-a"} 9',
        ])
        snapshot = parse_metrics(legacy, 1, {"worker-a": 0})
        self.assertIsNotNone(snapshot.spec_decode)
        self.assertEqual(snapshot.spec_decode.by_rank[0]["draft_tokens"], 10)

    def test_speculative_decoding_absent_is_non_fatal(self):
        before = parse_metrics(BASE, 2)
        after = parse_metrics(AFTER, 2)
        self.assertIsNone(before.spec_decode)
        actual = diff_spec_decode(before, after)
        self.assertEqual(actual.status, "unavailable")
        self.assertIsNone(actual.global_draft_tokens)
        self.assertIn("no speculative-decoding metrics", actual.reason)
        # Prefix-cache accounting still works when spec metrics are absent.
        self.assertEqual(diff_metrics(before, after).global_hit_rate, 0.75)

    def test_speculative_decoding_partial_ranks(self):
        before = parse_metrics(BASE + SPEC_BASE_RANK0, 2)
        after = parse_metrics(AFTER + SPEC_AFTER_RANK0, 2)
        actual = diff_spec_decode(before, after)
        self.assertEqual(actual.status, "partial")
        self.assertEqual(actual.covered_ranks, (0,))
        self.assertEqual(actual.missing_ranks, (1,))
        self.assertEqual(actual.global_draft_tokens, 200)
        self.assertAlmostEqual(actual.global_draft_acceptance_rate, 0.7)

    def test_speculative_decoding_counter_regression_is_reported(self):
        regressed = _spec_metrics("engine_0", 50, 40, 60, 0.8, 1.2)
        before = parse_metrics(BASE + SPEC_BASE, 2)
        after = parse_metrics(AFTER + regressed + "\n" + SPEC_AFTER_RANK1, 2)
        actual = diff_spec_decode(before, after)
        self.assertEqual(actual.status, "counter_regressed")
        self.assertEqual(actual.by_rank[0]["status"], "counter_regressed")
        self.assertIsNone(actual.by_rank[0]["draft_acceptance_rate"])
        self.assertEqual(actual.covered_ranks, (1,))
        self.assertEqual(actual.global_draft_tokens, 300)

    def test_speculative_decoding_serialization(self):
        before = parse_metrics(BASE + SPEC_BASE, 2)
        after = parse_metrics(AFTER + SPEC_AFTER, 2)
        snapshot = snapshot_to_dict(before, include_raw=False)
        self.assertTrue(snapshot["spec_decode"]["available"])
        self.assertEqual(
            snapshot["spec_decode"]["by_dp"]["0"]["draft_tokens"], 100
        )
        self.assertEqual(
            snapshot["spec_decode"]["by_dp"]["0"]["draft_acceptance_rate"], 0.6
        )
        actual = spec_decode_to_dict(diff_spec_decode(before, after))
        self.assertEqual(actual["status"], "ok")
        self.assertEqual(actual["global_draft_tokens"], 500)
        self.assertAlmostEqual(actual["global_draft_acceptance_rate"], 0.6)
        self.assertAlmostEqual(actual["global_efficiency"], 0.86)
        self.assertEqual(actual["covered_ranks"], [0, 1])

    def test_summarize_kv_usage_skips_missing_values(self):
        samples = [
            {0: 0.5, 1: 0.1},
            {0: 0.9, 1: None},
            {0: None, 1: 0.3},
        ]
        summary = summarize_kv_usage(samples)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["by_dp"]["0"], {"peak": 0.9, "avg": 0.7, "sample_count": 2})
        self.assertEqual(summary["by_dp"]["1"], {"peak": 0.3, "avg": 0.2, "sample_count": 2})
        self.assertEqual(summary["global_peak"], 0.9)
        self.assertEqual(summary["global_avg"], (0.5 + 0.9 + 0.1 + 0.3) / 4)

    def test_summarize_kv_usage_empty(self):
        summary = summarize_kv_usage([])
        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["by_dp"], {})
        self.assertIsNone(summary["global_peak"])
        self.assertIsNone(summary["global_avg"])

    def test_missing_rank_fails(self):
        one = BASE.replace('vllm:prefix_cache_queries{engine="engine_1",model_name="m"} 200\n', "").replace('vllm:prefix_cache_hits{engine="engine_1",model_name="m"} 100\n', "").replace('vllm:kv_cache_usage_perc{engine="engine_1",model_name="m"} 0.3\n', "")
        with self.assertRaisesRegex(RuntimeCapabilityError, "missing DP ranks: 1"):
            parse_metrics(one, 2)

    def test_warmup_requires_and_sends_every_group_rank(self):
        data = {"service": {"dp_size": 2, "timeout_seconds": 1, "model": "m", "inference_url": "http://x", "metrics_url": "http://x/metrics", "reset_url": None, "assume_empty_cache": True, "engine_label_map": {}, "api_key": ""}}
        client = VLLMClient(Scenario(Path("scenario.json"), data))
        calls = []
        client.send_completion = lambda prompt, max_tokens, dp_rank=None: calls.append(dp_rank) or {}  # type: ignore[method-assign]
        plan = [{"group_id": group, "dp_rank": rank, "prompt": f"{group}-{rank}", "max_tokens": 1} for group in ("g0", "g1") for rank in range(2)]
        results = client.warm_every_group_rank(plan)
        self.assertEqual(calls, [0, 1, 0, 1])
        self.assertTrue(all(row["success"] for row in results))
        with self.assertRaisesRegex(RuntimeCapabilityError, "does not cover"):
            client.warm_every_group_rank(plan[:-1])

    def test_single_dp_warmup_does_not_send_routing_header_rank(self):
        data = {"service": {"dp_size": 1, "timeout_seconds": 1, "model": "m", "inference_url": "http://x", "metrics_url": "http://x/metrics", "reset_url": None, "assume_empty_cache": True, "engine_label_map": {}, "api_key": ""}}
        client = VLLMClient(Scenario(Path("scenario.json"), data))
        calls = []
        client.send_completion = lambda prompt, max_tokens, dp_rank=None: calls.append(dp_rank) or {}  # type: ignore[method-assign]
        client.warm_every_group_rank([{"group_id": "g0", "dp_rank": 0, "prompt": "g0", "max_tokens": 1}])
        self.assertEqual(calls, [None])

    def test_reset_requires_explicit_assume_empty(self):
        base = {"dp_size": 1, "timeout_seconds": 1, "model": "m", "inference_url": "http://x", "metrics_url": "http://x/metrics", "reset_url": None, "engine_label_map": {}, "api_key": ""}
        client = VLLMClient(Scenario(Path("scenario.json"), {"service": base | {"assume_empty_cache": True}}))
        self.assertEqual(client.reset()[0]["code"], "ASSUME_EMPTY_CACHE")
        client = VLLMClient(Scenario(Path("scenario.json"), {"service": base | {"assume_empty_cache": False}}))
        with self.assertRaisesRegex(RuntimeCapabilityError, "reset_url"):
            client.reset()


if __name__ == "__main__":
    unittest.main()

