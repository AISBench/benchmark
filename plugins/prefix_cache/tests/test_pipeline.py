import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ais_bench_prefix_cache.artifacts import read_jsonl, sha256_file, validate_artifacts
from ais_bench_prefix_cache.pipeline import inspect_scenario, prepare_scenario
from ais_bench_prefix_cache.cli import main as cli_main
from ais_bench_prefix_cache.errors import PrefixCacheError
from tests.test_core import scenario_dict
from ais_bench_prefix_cache.runtime import run_scenario


class FakeTokenizer:
    all_special_ids = list(range(32))

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


def write_case(root: Path, mode: str = "cold") -> Path:
    questions = ["alpha arithmetic question", "beta arithmetic question", "gamma arithmetic question", "delta arithmetic question"]
    (root / "gsm.jsonl").write_text("".join(json.dumps({"question": value}) + "\n" for value in questions), encoding="utf-8")
    data = scenario_dict(root, mode=mode)
    path = root / "scenario.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class PipelineTest(unittest.TestCase):
    def test_four_artifacts_and_minimal_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder))
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            self.assertTrue(all(path.exists() for path in paths.__dict__.values()))
            requests = read_jsonl(paths.requests)
            self.assertEqual(set(requests[0]), {"question", "answer", "max_tokens"})
            first_line = paths.requests.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(list(json.loads(first_line)), ["question", "answer", "max_tokens"])
            self.assertTrue(validate_artifacts(paths.manifest)["ok"])

    def test_inspect_reports_reachability_without_persisting_run_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            summary = inspect_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            self.assertIn("reachable_min", summary)
            self.assertIn("reachable_max", summary)
            self.assertEqual(sum(summary["groups"].values()), 8)
            self.assertFalse(summary["sends_requests"])
            self.assertFalse((root / "out").exists())

    def test_deterministic_content_hashes(self):
        hashes = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as folder:
                scenario = write_case(Path(folder))
                paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
                hashes.append((sha256_file(paths.full), sha256_file(paths.requests)))
        self.assertEqual(hashes[0], hashes[1])

    def test_manifest_does_not_persist_api_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            data = json.loads(scenario.read_text(encoding="utf-8"))
            data["service"]["api_key"] = "do-not-write-this-secret"
            scenario.write_text(json.dumps(data), encoding="utf-8")
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            manifest_text = paths.manifest.read_text(encoding="utf-8")
            self.assertNotIn("do-not-write-this-secret", manifest_text)
            manifest = json.loads(manifest_text)
            self.assertTrue(manifest["effective_config"]["service"]["api_key_configured"])

    def test_warmup_manifest_has_every_group_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            scenario = write_case(Path(folder), mode="warmup")
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            pairs = {(row["group_id"], row["dp_rank"]) for row in manifest["warmup"]["plan"]}
            self.assertEqual(pairs, {(f"group-{group}", rank) for group in range(2) for rank in range(2)})
            self.assertTrue(all(not row["included_in_formal_statistics"] for row in manifest["warmup"]["plan"]))

    def test_group_override_and_multi_sample_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            data = json.loads(scenario.read_text(encoding="utf-8"))
            data["prefix_cache"]["groups"]["overrides"] = {
                "group-0": {
                    "input_length": {"mode": "fixed", "value": 80},
                    "corpus_selection": {"mode": "indices", "values": [0, 1]},
                }
            }
            data["prefix_cache"]["target_hit_rate"] = 0.0
            scenario.write_text(json.dumps(data), encoding="utf-8")
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            rows = [row for row in read_jsonl(paths.full) if row["group_id"] == "group-0"]
            self.assertTrue(all(row["actual_input_tokens"] == 80 for row in rows))
            self.assertTrue(any(len(set(row["gsm_indices"])) >= 2 for row in rows))

    def test_analysis_deviation_is_warning_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            paths = prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            baseline = root / "baseline.prom"
            after = root / "after.prom"
            baseline.write_text('\n'.join([
                'vllm:prefix_cache_queries{engine="0"} 100',
                'vllm:prefix_cache_queries{engine="1"} 100',
                'vllm:prefix_cache_hits{engine="0"} 50',
                'vllm:prefix_cache_hits{engine="1"} 50',
            ]), encoding="utf-8")
            after.write_text('\n'.join([
                'vllm:prefix_cache_queries{engine="0"} 150',
                'vllm:prefix_cache_queries{engine="1"} 150',
                'vllm:prefix_cache_hits{engine="0"} 50',
                'vllm:prefix_cache_hits{engine="1"} 50',
            ]), encoding="utf-8")
            code = cli_main(["analyze", "--manifest", str(paths.manifest), "--baseline", str(baseline), "--after", str(after)])
            self.assertEqual(code, 0)
            analysis = json.loads(paths.analysis.read_text(encoding="utf-8"))
            self.assertIn("ACTUAL_DEVIATION", {warning["code"] for warning in analysis["warnings"]})
            self.assertIn("raw_prometheus", analysis["runtime"]["metrics_baseline"])
            self.assertIn("raw_prometheus", analysis["runtime"]["metrics_after"])

    def test_run_rejects_stale_prepared_scenario_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenario = write_case(root)
            prepare_scenario(scenario, tokenizer_loader=lambda _: FakeTokenizer())
            data = json.loads(scenario.read_text(encoding="utf-8"))
            data["prefix_cache"]["target_hit_rate"] = 0.25
            scenario.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(PrefixCacheError, "different scenario"):
                run_scenario(scenario)


if __name__ == "__main__":
    unittest.main()
