import unittest
from pathlib import Path

from ais_bench_prefix_cache.errors import RuntimeCapabilityError
from ais_bench_prefix_cache.metrics import diff_metrics, parse_metrics
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
