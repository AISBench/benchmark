import json
import itertools
import tempfile
import unittest
from pathlib import Path

from ais_bench_prefix_cache.errors import ScenarioValidationError
from ais_bench_prefix_cache.generation import (
    RequestPlan,
    assign_cold_routes,
    assign_groups,
    build_canonical_prefixes,
    build_input_lengths,
    build_output_lengths,
    build_unique_seed,
    build_unique_seed_tokens,
    find_boundary_safe_token_ids,
    GSMRecord,
    load_gsm8k,
    order_indices,
    select_gsm8k,
    simulate_theory,
    solve_prefix_lengths,
)


class _FakeTokenizer:
    """Ids 0..25 map to single letters; the pair 'ab' re-encodes as id 52."""

    def __init__(self):
        self.all_special_ids = []

    def __len__(self):
        return 64

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(97 + (token_id % 26)) for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        ids = [ord(ch) - 97 for ch in text if "a" <= ch <= "z"]
        out, i = [], 0
        while i < len(ids):
            if ids[i] == 0 and i + 1 < len(ids) and ids[i + 1] == 1:
                out.append(52)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out
from ais_bench_prefix_cache.scenario import load_scenario


def scenario_dict(root: Path, mode: str = "cold", dp_size: int = 2) -> dict:
    return {
        "schema_version": "1.0",
        "run": {"run_id": "pc-test", "random_seed": 42, "output_dir": str(root / "out")},
        "tokenizer": {"path": "fake", "block_size": 4},
        "corpus": {"path": str(root / "gsm.jsonl"), "field": "question", "selection": {"mode": "random"}},
        "requests": {"count": 8, "input_length": {"mode": "fixed", "value": 32}, "output_length": {"mode": "fixed", "value": 2}},
        "prefix_cache": {"mode": mode, "target_hit_rate": 0.5, "seed_blocks": 1, "groups": {"count": 2, "assignment": {"mode": "uniform"}}, "order": {"strategy": "interleave"}},
        "service": {"inference_url": "http://127.0.0.1:8000/v1/completions", "metrics_url": "http://127.0.0.1:8000/metrics", "reset_url": "http://127.0.0.1:8000/reset_prefix_cache", "model": "m", "dp_size": dp_size, "assume_empty_cache": False},
        "validation": {"target_warning_pp": 1.0, "actual_warning_pp": 5.0},
    }


class CoreTest(unittest.TestCase):
    def test_omitted_values_use_current_example_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "scenario.json"
            path.write_text("{}", encoding="utf-8")
            scenario = load_scenario(path)
            effective = scenario.to_effective_dict()
            self.assertEqual(effective["schema_version"], "1.0")
            self.assertEqual(effective["run"]["run_id"], "gsm8k-prefix-cache-60")
            self.assertEqual(effective["run"]["random_seed"], 42)
            self.assertEqual(effective["run"]["output_dir"], str((root / "outputs/gsm8k-prefix-cache-60").resolve()))
            self.assertEqual(effective["tokenizer"]["path"], "/home/weights/Qwen3.6-27B")
            self.assertEqual(effective["tokenizer"]["block_size"], 16)
            self.assertEqual(effective["corpus"]["path"], str((root / "GSM8K.jsonl").resolve()))
            self.assertEqual(effective["corpus"]["selection"], {"mode": "random"})
            self.assertEqual(effective["requests"]["count"], 100)
            self.assertEqual(effective["requests"]["input_length"], {"mode": "fixed", "value": 1024})
            self.assertEqual(effective["requests"]["output_length"], {"mode": "fixed", "value": 32})
            self.assertEqual(effective["prefix_cache"]["mode"], "warmup")
            self.assertEqual(effective["prefix_cache"]["target_hit_rate"], 0.6)
            self.assertEqual(effective["prefix_cache"]["minimum_non_shared_length"], 16)
            self.assertEqual(effective["prefix_cache"]["groups"]["count"], 1)
            self.assertEqual(effective["prefix_cache"]["groups"]["assignment"], {"mode": "uniform"})
            self.assertEqual(effective["service"]["dp_size"], 2)
            self.assertEqual(effective["service"]["inference_url"], "http://127.0.0.1:8000/v1/completions")
            self.assertEqual(effective["validation"], {"target_warning_pp": 1.0, "actual_warning_pp": 5.0})

    def test_partially_empty_sections_receive_nested_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "scenario.json"
            path.write_text(json.dumps({
                "run": {},
                "tokenizer": {},
                "corpus": {"selection": {}},
                "requests": {"input_length": {}, "output_length": {}},
                "prefix_cache": {"groups": {"assignment": {}}, "order": {}},
                "service": {},
                "validation": {},
            }), encoding="utf-8")
            effective = load_scenario(path).to_effective_dict()
            self.assertEqual(effective["corpus"]["selection"]["mode"], "random")
            self.assertEqual(effective["requests"]["input_length"], {"mode": "fixed", "value": 1024})
            self.assertEqual(effective["requests"]["output_length"], {"mode": "fixed", "value": 32})
            self.assertEqual(effective["prefix_cache"]["groups"]["count"], 1)
            self.assertEqual(effective["prefix_cache"]["groups"]["assignment"]["mode"], "uniform")
            self.assertEqual(effective["prefix_cache"]["order"]["strategy"], "interleave")

    def test_scenario_rejects_unknown_multi_instance_field(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = scenario_dict(root)
            data["service"]["instances"] = ["forbidden"]
            path = root / "scenario.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "service.instances"):
                load_scenario(path)

    def test_lengths_are_deterministic(self):
        ranges = {"mode": "range", "ranges": [{"min": 10, "max": 12, "count": 5}]}
        self.assertEqual(build_input_lengths(ranges, 5, 7), build_input_lengths(ranges, 5, 7))
        normal = {"mode": "truncated_normal", "min": 2, "max": 8}
        self.assertEqual(build_output_lengths(normal, 6, 9), build_output_lengths(normal, 6, 9))

    def test_explicit_and_truncated_normal_input_lengths(self):
        self.assertEqual(
            build_input_lengths({"mode": "explicit", "values": [8, 12, 16]}, 3, 7),
            [8, 12, 16],
        )
        normal = {"mode": "truncated_normal", "min": 8, "max": 16, "mean": 12, "std": 2}
        first = build_input_lengths(normal, 20, 9)
        self.assertEqual(first, build_input_lengths(normal, 20, 9))
        self.assertTrue(all(8 <= value <= 16 for value in first))

    def test_csv_lengths_and_specified_gsm_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "lengths.csv").write_text(
                "input_tokens,output_tokens\n16,2\n20,3\n", encoding="utf-8"
            )
            self.assertEqual(
                build_input_lengths({"mode": "csv", "path": str(root / "lengths.csv")}, 2, 1),
                [16, 20],
            )
            self.assertEqual(
                build_output_lengths({"mode": "csv", "path": str(root / "lengths.csv")}, 2, 1),
                [2, 3],
            )
            corpus = root / "gsm.jsonl"
            corpus.write_text(
                "".join(json.dumps({"question": value}) + "\n" for value in ("alpha", "beta", "gamma")),
                encoding="utf-8",
            )
            records = load_gsm8k(corpus)
            by_index = select_gsm8k(records, {"mode": "indices", "values": [2, 0]}, 3, 1)
            self.assertEqual([row.line_index for row in by_index], [2, 0, 2])
            by_hash = select_gsm8k(
                records,
                {"mode": "question_sha256", "values": [records[1].question_sha256]},
                2,
                1,
            )
            self.assertEqual([row.line_index for row in by_hash], [1, 1])

    def test_mixed_selection_allows_hashes_without_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / "gsm.jsonl"
            corpus.write_text(
                "".join(json.dumps({"question": value}) + "\n" for value in ("alpha", "beta")),
                encoding="utf-8",
            )
            records = load_gsm8k(corpus)
            selected = select_gsm8k(
                records,
                {"mode": "mixed", "indices": [], "question_sha256": [records[1].question_sha256]},
                3,
                1,
            )
            self.assertEqual([row.line_index for row in selected], [1, 1, 1])

    def test_canonical_collision_uses_group_fallback(self):
        class CharTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

            def decode(self, token_ids, skip_special_tokens=False):
                return "".join(chr(token_id) for token_id in token_ids)

        record = GSMRecord(0, "same question", "hash")
        canonical = build_canonical_prefixes(
            CharTokenizer(),
            {"group-0": [record], "group-1": [record]},
            {"group-0": 8, "group-1": 8},
            4,
        )
        self.assertNotEqual(
            canonical["group-0"].token_ids[:4], canonical["group-1"].token_ids[:4]
        )

    def test_seed_capacity_is_rejected_during_scenario_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = scenario_dict(root)
            data["requests"]["input_length"] = {"mode": "fixed", "value": 2}
            path = root / "scenario.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioValidationError, "at least 4 tokens"):
                load_scenario(path)

    def test_seed_generation_round_trips(self):
        tokenizer = _FakeTokenizer()
        safe_ids = find_boundary_safe_token_ids(tokenizer, 8)
        self.assertNotIn(52, safe_ids)
        for index in range(20):
            seed = build_unique_seed(tokenizer, safe_ids, f"r{index}", 4, 42)
            self.assertEqual(tokenizer.encode(tokenizer.decode(seed), add_special_tokens=False), list(seed))

    def test_groups_and_orders(self):
        groups = assign_groups(10, {"count": 3, "assignment": {"mode": "weights", "weights": [0.5, 0.3, 0.2]}}, 42)
        self.assertEqual([groups.count(f"group-{i}") for i in range(3)], [5, 3, 2])
        for strategy in ("sequential", "within_group_shuffle", "interleave", "global_shuffle"):
            self.assertEqual(sorted(order_indices(groups, strategy, 42)), list(range(10)))
        lengths = [30, 20, 10, 40, 15, 25]
        grouped = ["g0", "g1", "g0", "g1", "g0", "g1"]
        ordered = order_indices(grouped, "input_len_asc", 42, lengths)
        for group in ("g0", "g1"):
            selected = [lengths[index] for index in ordered if grouped[index] == group]
            self.assertEqual(selected, sorted(selected))

    def test_routes_and_dp_watermarks(self):
        groups = ["g0", "g1", "g0", "g1", "g0", "g1", "g0", "g1"]
        ranks, lanes = assign_cold_routes(groups, 2)
        self.assertEqual(ranks, [0, 0, 1, 1, 0, 0, 1, 1])
        plans = [RequestPlan(f"r{i}", i, group, i // 2, rank, lane, 32, 32, 1, 16, 4, 12) for i, (group, rank, lane) in enumerate(zip(groups, ranks, lanes))]
        theory = simulate_theory(plans, "cold")
        self.assertEqual([row.theoretical_hit_tokens for row in theory.rows], [0, 0, 0, 0, 16, 16, 16, 16])

    def test_unique_seed_and_solver(self):
        seeds = build_unique_seed_tokens(list(range(32, 96)), [f"r{i}" for i in range(20)], 4, 42)
        self.assertEqual(len(set(seeds.values())), 20)
        groups = ["g0"] * 4
        ranks, lanes = assign_cold_routes(groups, 1)
        result = solve_prefix_lengths([32] * 4, [1] * 4, groups, ranks, lanes, 4, 4, "cold", 0.5)
        self.assertTrue(all(value % 4 == 0 for value in result.shared_prefix_tokens))
        self.assertLessEqual(result.effective_hit_rate, result.max_reachable_rate)
        self.assertIn("g0", result.group_reachability)

    def test_solver_reserves_minimum_non_shared_length(self):
        groups = ["g0", "g0"]
        ranks, lanes = assign_cold_routes(groups, 1)
        result = solve_prefix_lengths([20, 20], [1, 1], groups, ranks, lanes, 4, 8, "cold", 1.0)
        self.assertTrue(all(prefix <= 12 for prefix in result.shared_prefix_tokens))
        self.assertFalse(result.target_reachable)

    def test_unreachable_large_cold_target_uses_exact_upper_boundary(self):
        lengths = [
            2504, 2150, 3174, 3051, 2962, 2619, 2467, 2404, 3776, 2178,
            2170, 2431, 2943, 3000, 2156, 2862, 3766, 2950, 3887, 3187,
        ]
        groups = ["g0"] * len(lengths)
        ranks, lanes = assign_cold_routes(groups, 1)
        result = solve_prefix_lengths(
            lengths, [32] * len(lengths), groups, ranks, lanes,
            128, 128, "cold", 0.9,
        )
        self.assertEqual(result.effective_hit_tokens, 48_896)
        self.assertEqual(result.effective_hit_rate, result.max_reachable_rate)
        self.assertFalse(result.target_reachable)

    def test_solver_matches_exhaustive_small_oracle(self):
        lengths = [16, 20, 24]
        outputs = [1, 1, 1]
        groups = ["g0", "g0", "g0"]
        ranks, lanes = assign_cold_routes(groups, 1)
        for mode in ("cold", "warmup"):
            for target in (0.2, 0.4, 0.6):
                result = solve_prefix_lengths(lengths, outputs, groups, ranks if mode == "cold" else [None] * 3, lanes if mode == "cold" else [None] * 3, 4, 4, mode, target)
                target_tokens = int(sum(lengths) * target + 0.5)
                best_error = None
                for prefixes in itertools.product(*(range(0, ((length - 4) // 4) * 4 + 1, 4) for length in lengths)):
                    plans = [RequestPlan(f"r{i}", i, "g0", i, ranks[i] if mode == "cold" else None, lanes[i] if mode == "cold" else None, lengths[i], lengths[i], 1, prefixes[i], 4, lengths[i] - prefixes[i] - 4) for i in range(3)]
                    warm = {"g0": max(prefixes)} if mode == "warmup" else None
                    hit = simulate_theory(plans, mode, warm).total_hit_tokens
                    error = abs(hit - target_tokens)
                    best_error = error if best_error is None else min(best_error, error)
                self.assertEqual(abs(result.effective_hit_tokens - target_tokens), best_error, (mode, target, result))

    def test_exact_cold_solver_matches_multi_lane_exhaustive_oracle(self):
        lengths = [16, 20, 24, 28]
        outputs = [1] * len(lengths)
        groups = ["g0"] * len(lengths)
        ranks = [0, 0, 1, 1]
        lanes = [0, 1, 0, 1]
        candidates = [range(0, ((length - 4) // 4) * 4 + 1, 4) for length in lengths]
        for target in (0.0, 0.1, 0.35, 0.6, 0.95, 1.0):
            result = solve_prefix_lengths(lengths, outputs, groups, ranks, lanes, 4, 4, "cold", target)
            target_tokens = int(sum(lengths) * target + 0.5)
            best_error = min(
                abs(
                    simulate_theory([
                        RequestPlan(
                            f"r{i}", i, "g0", i, ranks[i], lanes[i], lengths[i], lengths[i],
                            1, prefixes[i], 4, lengths[i] - prefixes[i] - 4,
                        )
                        for i in range(len(lengths))
                    ], "cold").total_hit_tokens - target_tokens
                )
                for prefixes in itertools.product(*candidates)
            )
            self.assertEqual(abs(result.effective_hit_tokens - target_tokens), best_error)


if __name__ == "__main__":
    unittest.main()
